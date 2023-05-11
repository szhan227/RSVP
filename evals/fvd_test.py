import time
import numpy as np
import torch
from utils import AverageMeter
from torchvision.utils import save_image, make_grid
from einops import rearrange
from losses.ddpm import DDPM

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
import os

import torchvision
import PIL
from PIL import Image

from src.model import get_model
from tools.dataloader import get_loaders, get_dataloader
from utils import file_name, Logger, download
from models.ddpm.unet import UNetModel, DiffusionWrapper, MosoModel
from models.ema import LitEma
import copy
import argparse
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, help='experiment name to run')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--id', type=str, default='main', help='experiment identifier')
parser.add_argument('--data', type=str, default='UCF101')

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

def get_opt_from_yaml(path):
    assert os.path.exists(path), f"{path} must exists!"
    import yaml
    with open(path, 'r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    return dict2obj(opt)

def save_image_grid(img, fname, drange, grid_size, normalize=True):
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print (f'Saving Video with {T} frames, img shape {H}, {W}')

    assert C in [3]

    if C == 3:
        torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    return img

def get_visualize_img(img): # img: [B T C H W]
    x = img[:8].detach().cpu()
    show_x = torch.clamp(x, min=0, max=1)
    b, t, c, h, w = show_x.shape
    # get output for fvd calculation
    show_x = rearrange(show_x, "b t c h w -> b t h w c").numpy()*255.
    return show_x.astype(np.uint8)

def test_fvd_moso(rank, ema_model, first_stage_model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['fvd'] = AverageMeter()
    check = time.time()

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.in_channels,
                           image_size=ema_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)
    real_embeddings = []
    fake_embeddings = []

    i3d = load_i3d_pretrained(device)
    bgs = []
    ids = []
    reals = []
    contexts = []
    fakes = []
    gts = []
    
    with torch.no_grad():        
        '''
        fake_loader = [{
            "bg_tokens": torch.rand(4, 32*32).long(),
            "id_tokens": torch.rand(4, 16*16).long(),
            "mo_tokens": torch.rand(4, 16*8*8).long()
        }]*4
        for n, inputs in enumerate(fake_loader):
        '''
        for n, inputs in enumerate(loader):
            if n >= 4:
                break
            bg_tokens = rearrange(inputs["bg_tokens"], 'B (T H W) -> B T H W', T=1, H=32, W=32).to(device)
            id_tokens = rearrange(inputs["id_tokens"], 'B (T H W) -> B T H W', T=1, H=16, W=16).to(device)
            mo_tokens = rearrange(inputs["mo_tokens"], 'B (T H W) -> B T H W', H=8, W=8).to(device)
            
            B = bg_tokens.shape[0]
            vq_bg = first_stage_model._vq_ema.quantize_code(bg_tokens)
            vq_id = first_stage_model._vq_ema.quantize_code(id_tokens)
            vq_mo = first_stage_model._vq_ema.quantize_code(mo_tokens)

            quantize_bg = first_stage_model._suf_vq_bg(vq_bg)
            quantize_id = first_stage_model._suf_vq_id(vq_id)
            quantize_mo = first_stage_model._suf_vq_mo(vq_mo)
            xbg = rearrange(quantize_bg, "b c h w -> b (h w) c").detach()
            xid = rearrange(quantize_id, "b c h w -> b (h w) c").detach()
            xmo = rearrange(quantize_mo, "(b t) c h w -> b c t h w", b=xbg.shape[0]).detach()
            context = torch.cat([xbg, xid], dim=1)
            bgs.append(rearrange(xbg, "b (t h w) c -> b t c h w", t=1, h=32, w=32))
            ids.append(rearrange(xid, "b (t h w) c -> b t c h w", t=1, h=16, w=16))
            # Replace ground truth here
            # xmo[:,:,xmo.shape[2]//2:,:,:] = xmo[:,:,:xmo.shape[2]//2, :,:] # xmo shape [1, 256, 16, 8, 8]
            gts.append(xmo)   
            contexts.append(context)

        for i in range(4): 
            
            print(i)
            print(gts[i].shape)
            mask = torch.zeros(gts[i].shape[0], gts[i].shape[2], gts[i].shape[3], gts[i].shape[4]).long()
            mask[:, :gts[i].shape[2]//2, ...] = 1
            
            # z = diffusion_model.sample_moso(gts[i].clone().to(device), mask.to(device), contexts[i].to(device))

            x_rec, _, _ = first_stage_model._decoder(bgs[i], ids[i], gts[i].permute(0, 2, 1, 3, 4))
            # x_rec_fake, _, _ = first_stage_model._decoder(bgs[i], ids[i], z.permute(0, 2, 1, 3, 4))
            x_rec_fake, _, _ = first_stage_model._decoder(bgs[i], ids[i], torch.randn(gts[i].shape).to(device).permute(0, 2, 1, 3, 4))
            x_rec = get_visualize_img(x_rec) # b 
            x_rec_fake = get_visualize_img(x_rec_fake)
            real_embeddings.append(get_fvd_logits(x_rec, i3d=i3d, device=device))
            fake_embeddings.append(get_fvd_logits(x_rec_fake, i3d=i3d, device=device))

            reals.append(x_rec)
            fakes.append(x_rec_fake)
    
    # need to translate N T H W C ->  N, C, T, H, W 
    reals = np.transpose(np.concatenate(reals), (0, 4, 1, 2, 3))
    fakes = np.transpose(np.concatenate(fakes), (0, 4, 1, 2, 3))

    real_embeddings = torch.cat(real_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)

    if rank == 0:
        real_vid = save_image_grid(reals, os.path.join(logger.logdir, f'real_{it}.gif'), drange=[0, 255], grid_size=(4,B)) # change 4 to 1
        real_vid = np.expand_dims(real_vid,0).transpose(0, 1, 4, 2, 3)
            
        fake_vid = save_image_grid(fakes, os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255], grid_size=(4,B)) # change 4 to 1
        fake_vid = np.expand_dims(fake_vid,0).transpose(0, 1, 4, 2, 3)
        
        logger.video_summary('real', real_vid, it)
        logger.video_summary('prediction', fake_vid, it)

    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()

if __name__ == '__main__':
    args = parser.parse_args()
    fn = file_name(args)
    logger = Logger(fn)
    logger.log(args)
    logger.log(f'Log path: {logger.logdir}')
    rootdir = logger.logdir
    
    device = 'cuda'
    
    config = OmegaConf.load("configs/moso-diffusion/base.yaml")
    unet_config = config.model.params.unet_config
    moso_ddpm_model = MosoModel(**unet_config).to(device)
    
    ddpm_wrapper_state = torch.load('results/moso_ddpm_main_UCF-101_42/model_760000.pth', map_location='cpu')
    moso_ddpm_model.load_state_dict(ddpm_wrapper_state, strict=False)
    
    criterion = DDPM(moso_ddpm_model, channels=unet_config.in_channels,
                            image_size=unet_config.image_size,
                            linear_start=config.model.params.linear_start,
                            linear_end=config.model.params.linear_end,
                            log_every_t=config.model.params.log_every_t,
                            w=config.model.params.w,
                            ).to(device)
    
    moso_ddpm_model.eval()
    
    ema_model = copy.deepcopy(moso_ddpm_model)
    ema = LitEma(ema_model)
    ema_model.eval()
    
    moso_opt = get_opt_from_yaml("configs/moso-vqvae/test_UCF.yaml")
    train_loader, test_loader = get_dataloader(moso_opt)
    
    rank = 0
    it = 0
    
    ema.copy_to(ema_model)
    
    first_stage_model, _ = get_model(moso_opt)
    first_stage_model = first_stage_model.to(device)
    test_fvd_moso(rank, ema_model, first_stage_model, test_loader, 0, logger=logger)