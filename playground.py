import cv2
import torch
import numpy as np
import yaml
from torchvision import transforms
from PIL import Image
from model.vq_vae.VQVAE import VQVAEModel
from model.ldm.unet import UNetModel, DiffusionWrapper
from model.ema import LitEma
from losses.ddpm import DDPM
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
import time
from utils import AverageMeter
import copy
from einops import rearrange
import random


def preprocess():
    video_cap = cv2.VideoCapture('testvideo1.avi')

    xt = []
    while True:
        success, frame = video_cap.read()
        if not success:
            break
        xt.append(frame)
    xt = np.array(xt)
    xt = torch.tensor(xt) / 127.5 - 1
    xt = xt.permute(0, 3, 1, 2)
    print('xt.shape:', xt.shape, xt.dtype)

    num_frames = 10
    start = 0

    transform = transforms.CenterCrop(64)

    lower_bound = 0.2
    upper_bound = 0.8
    pre_img = transform(xt[start])
    nxt_img = transform(xt[start + 1])

    print('pre_img.shape:', pre_img.shape)
    print('nxt_img.shape:', nxt_img.shape)

    imgs, imgs_bg, imgs_id, imgs_mo = [], [], [], []

    for i in range(start + 2, start + 2 + num_frames):
        cur_img = nxt_img
        nxt_img = transform(xt[i])

        cur_diff = cur_img * 2 - pre_img - nxt_img
        max_diff = torch.max(torch.abs(cur_diff), dim=0)[0]
        id_mask = (max_diff > lower_bound) & (max_diff <= upper_bound)
        img_id = id_mask[None, :, :] * cur_img
        img_bg = cur_img - img_id

        imgs.append(cur_img.unsqueeze(0))
        imgs_bg.append(img_bg.unsqueeze(0))
        imgs_id.append(img_id.unsqueeze(0))
        imgs_mo.append(cur_diff.unsqueeze(0))

    ret_img = torch.cat(imgs, dim=0)
    ret_img_bg = torch.cat(imgs_bg, dim=0)
    ret_img_id = torch.cat(imgs_id, dim=0)
    ret_img_mo = torch.cat(imgs_mo, dim=0)

    ret_img = torch.unsqueeze(ret_img, dim=0)
    ret_img_bg = torch.unsqueeze(ret_img_bg, dim=0)
    ret_img_id = torch.unsqueeze(ret_img_id, dim=0)
    ret_img_mo = torch.unsqueeze(ret_img_mo, dim=0)

    return [ret_img, ret_img_bg, ret_img_id, ret_img_mo]

def play_with_MOSO_VAE():
    video_cap = cv2.VideoCapture('testvideo1.avi')

    xt = []
    while True:
        success, frame = video_cap.read()
        if not success:
            break
        xt.append(frame)
    xt = np.array(xt)
    xt = torch.tensor(xt) / 255.0
    xt = xt.permute(0, 3, 1, 2)
    print('xt.shape:', xt.shape, xt.dtype)

    num_frames = 5
    start = 0

    transform = transforms.CenterCrop(64)

    lower_bound = 0.2
    upper_bound = 0.8
    pre_img = transform(xt[start])
    nxt_img = transform(xt[start + 1])

    print('pre_img.shape:', pre_img.shape)
    print('nxt_img.shape:', nxt_img.shape)
    imgs, imgs_bg, imgs_id, imgs_mo = [], [], [], []

    for i in range(start + 2, start + 2 + num_frames):
        cur_img = nxt_img
        nxt_img = transform(xt[i])

        cur_diff = cur_img * 2 - pre_img - nxt_img
        max_diff = torch.max(torch.abs(cur_diff), dim=0)[0]
        id_mask = (max_diff > lower_bound) & (max_diff <= upper_bound)
        img_id = id_mask[None, :, :] * cur_img
        img_bg = cur_img - img_id

        imgs.append(cur_img.unsqueeze(0))
        imgs_bg.append(img_bg.unsqueeze(0))
        imgs_id.append(img_id.unsqueeze(0))
        imgs_mo.append(cur_diff.unsqueeze(0))

    ret_img = torch.cat(imgs, dim=0)
    ret_img_bg = torch.cat(imgs_bg, dim=0)
    ret_img_id = torch.cat(imgs_id, dim=0)
    ret_img_mo = torch.cat(imgs_mo, dim=0)

    ret_img = torch.unsqueeze(ret_img, dim=0)
    ret_img_bg = torch.unsqueeze(ret_img_bg, dim=0)
    ret_img_id = torch.unsqueeze(ret_img_id, dim=0)
    ret_img_mo = torch.unsqueeze(ret_img_mo, dim=0)

    print('ret_img.shape:', ret_img.shape)
    print('ret_img_bg.shape:', ret_img_bg.shape)
    print('ret_img_id.shape:', ret_img_id.shape)
    print('ret_img_mo.shape:', ret_img_mo.shape)

    validate_inputs = [ret_img, ret_img, ret_img, ret_img_mo]

    opt = yaml.load(open('config/vqvae.yaml', 'r'), Loader=yaml.FullLoader)
    model_opt = opt['model']
    model = VQVAEModel(model_opt, opt)

    bg_toks, id_toks, mo_toks = model.my_encode([ret_img, ret_img_bg, ret_img_id, ret_img_mo], is_training=False)
    print('bg_toks.shape:', bg_toks.shape)
    print('id_toks.shape:', id_toks.shape)
    print('mo_toks.shape:', mo_toks.shape)
    outputs = model._decoder(bg_toks, id_toks, mo_toks)[0]
    print('output.shape:', outputs.shape)

    # (5, 4, 2048)
    # outputs = model(validate_inputs, is_training=False, writer=None)
    # dict_keys(['loss', 'x_rec', 'quantize_bg', 'quantize_id', 'quantize_mo', 'ssim_metric', 'rec_loss', 'lpips_loss', 'record_logs', 'optimizer_idx'])
    # print('outputs.shape:', outputs['x_rec'].shape)


def play_with_PVDM_Diffuser():

    unet_path = './config/small_unet.yaml'
    unet_config = OmegaConf.load(unet_path).unet_config
    print(unet_config)
    print('cuda: ', torch.torch.cuda.device_count())
    num_frames = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('in this project, use device: ', device)
    video_x = torch.randn(5, 4, 2048).to(device)
    timesteps = torch.tensor([0, 1, 2, 3, 4]).to(device)
    unet = UNetModel(**unet_config).to(device)
    ddpm = DiffusionWrapper(model=unet, conditioning_key=None)

    # 'b t c h w -> b c t h w'
    # x = [b t c//2 h w], cond = [b t c//2 h w]
    # evenly split input videos into two parts in the time dimension
    # ddpm takes in latent representations z

    output = ddpm(x=video_x, cond=None, t=timesteps)
    print('show output shape: ', output.shape)



def play_with_all_process():
    scaler = GradScaler()

    logger = None
    rank = 0
    ema_model = None
    cond_prob = 0.3

    unet_path = './config/small_unet.yaml'
    unet_config = OmegaConf.load(unet_path).unet_config
    print(unet_config)
    print('cuda: ', torch.torch.cuda.device_count())
    num_frames = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('in this project, use device: ', device)
    video_x = torch.randn(5, 4, 2048).to(device)
    timesteps = torch.tensor([0, 1, 2, 3, 4]).to(device)
    unet = UNetModel(**unet_config).to(device)
    model = DiffusionWrapper(model=unet, conditioning_key=None).to(device)

    moso_opt = yaml.load(open('config/vqvae.yaml', 'r'), Loader=yaml.FullLoader)
    moso_model_opt = moso_opt['model']
    first_stage_model = VQVAEModel(moso_model_opt, moso_opt).to(device)

    linear_start = 0.0015
    linear_end = 0.0195
    log_every_t = 200
    w = 0.0

    criterion = DDPM(model,
         channels=unet_config.in_channels,
         image_size=unet_config.image_size,
         linear_start=linear_start,
         linear_end=linear_end,
         log_every_t=log_every_t,
         w=w,
         ).to(device)

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    # if rank == 0:
    #     rootdir = logger.logdir


    # device = torch.device('cuda', rank)

    losses = dict()
    losses['diffusion_loss'] = AverageMeter()
    check = time.time()

    # lr_scheduler = LambdaLR(opt, scheduler)
    if ema_model == None:
        ema_model = copy.deepcopy(model)
        ema = LitEma(ema_model)
        ema_model.eval()
    else:
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(11200, dtype=torch.int)
        ema_model.eval()

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    first_stage_model.eval()
    model.train()

    video_x = preprocess()
    video_x = torch.stack(video_x, dim=0)

    # d: number of decomposition, to be 4
    video_x = rearrange(video_x, 'd b t c h w -> b d t c h w')
    print('video_x.shape: ', video_x.shape)
    train_loader = [(video_x, None)]

    for it, (x, _) in enumerate(train_loader):
        x = x.to(device)

        x = rearrange(x, 'b d t c h w -> b d t c h w')
        print('show x.shape: ', x.shape)
        # x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w')  # videos
        # x_bg = rearrange(x_bg / 127.5 - 1, 'b t c h w -> b c t h w')  # videos
        # x_id = rearrange(x_id / 127.5 - 1, 'b t c h w -> b c t h w')  # videos
        # x_mo = rearrange(x_mo / 127.5 - 1, 'b t c h w -> b c t h w')  # videos
        #
        # print('x.shape: ', x.shape)
        # print('x_bg.shape: ', x_bg.shape)
        # print('x_id.shape: ', x_id.shape)
        # print('x_mo.shape: ', x_mo.shape)

        c = None

        # conditional free guidance training
        model.zero_grad()

        if model.diffusion_model.cond_model:
            # p = np.random.random()
            p = 0.2

            if p < cond_prob:
                # split in time dimension
                c, x = torch.chunk(x, 2, dim=2)
                mask = (c + 1).contiguous().view(c.size(0), -1) ** 2
                mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1)

                print('c.shape: ', c.shape)
                print('x.shape: ', x.shape)
                with autocast():
                    with torch.no_grad():
                        x_img, x_bg, x_id, x_mo = x[:, 0, :, :, :, :], x[:, 1, :, :, :, :], x[:, 2, :, :, :, :], x[:, 3, :, :, :, :]
                        c_img, c_bg, c_id, c_mo = c[:, 0, :, :, :, :], c[:, 1, :, :, :, :], c[:, 2, :, :, :, :], c[:, 3, :, :, :, :]

                        print('x dtype: ', x.dtype)
                        print('x_img.shape: ', x_img.shape)
                        print('x_bg.shape: ', x_bg.shape)
                        print('x_id.shape: ', x_id.shape)
                        print('x_mo.shape: ', x_mo.shape)
                        # bg_toks, id_toks, mo_toks = first_stage_model.my_encode([ret_img, ret_img_bg, ret_img_id, ret_img_mo], is_training=False)

                        # Keep number of frames of x and c the same, now 5
                        xbg_toks, xid_toks, xmo_toks = first_stage_model.my_encode([x_img, x_bg, x_id, x_mo], is_training=False)
                        cbg_toks, cid_toks, cmo_toks = first_stage_model.my_encode([c_img, c_bg, c_id, c_mo], is_training=False)


                        print('xbg_toks.shape: ', xbg_toks.shape)
                        print('xid_toks.shape: ', xid_toks.shape)
                        print('xmo_toks.shape: ', xmo_toks.shape)
                        print('cbg_toks.shape: ', cbg_toks.shape)
                        print('cid_toks.shape: ', cid_toks.shape)
                        print('cmo_toks.shape: ', cmo_toks.shape)
                        # z = first_stage_model.module.extract(x).detach()
                        # c = first_stage_model.module.extract(c).detach()
                        z = torch.concat([xbg_toks, xid_toks, xmo_toks], dim=-1)
                        c = torch.concat([cbg_toks, cid_toks, cmo_toks], dim=-1)


                        c = c * mask + torch.zeros_like(c).to(c.device) * (1 - mask)

            else:
                c, x_tmp = torch.chunk(x, 2, dim=2)
                mask = (c + 1).contiguous().view(c.size(0), -1) ** 2
                mask = torch.where(mask.sum(dim=-1) > 0, 1, 0).view(-1, 1, 1, 1, 1)

                clip_length = x.size(2) // 2
                prefix = random.randint(0, clip_length)
                x = x[:, :, prefix:prefix + clip_length, :, :] * mask + x_tmp * (1 - mask)
                with autocast():
                    with torch.no_grad():
                        z = first_stage_model.module.extract(x).detach()
                        c = torch.zeros_like(z).to(device)
            print('show z.shape: ', z.shape)
            print('show c.shape: ', c.shape)
            (loss, t), loss_dict = criterion(z.float(), c.float())

        else:
            if it == 0:
                print("Unconditional model")
            with autocast():
                with torch.no_grad():
                    z = first_stage_model.module.extract(x).detach()

            (loss, t), loss_dict = criterion(z.float())

        """
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        """
        loss.backward()
        opt.step()

        losses['diffusion_loss'].update(loss.item(), 1)

        # ema model
        if it % 25 == 0 and it > 0:
            ema(model)

        if it % 500 == 0:
            # psnr = test_psnr(rank, model, test_loader, it, logger)
            if logger is not None and rank == 0:
                logger.scalar_summary('train/diffusion_loss', losses['diffusion_loss'].average, it)

                log_('[Time %.3f] [Diffusion %f]' %
                     (time.time() - check, losses['diffusion_loss'].average))

            losses = dict()
            losses['diffusion_loss'] = AverageMeter()

        # if it % 10000 == 0 and rank == 0:
        #     torch.save(model.module.state_dict(), rootdir + f'model_{it}.pth')
        #     ema.copy_to(ema_model)
        #     torch.save(ema_model.module.state_dict(), rootdir + f'ema_model_{it}.pth')
        #     fvd = test_fvd_ddpm(rank, ema_model, first_stage_model, test_loader, it, logger)
        #
        #     if logger is not None and rank == 0:
        #         logger.scalar_summary('test/fvd', fvd, it)
        #
        #         log_('[Time %.3f] [FVD %f]' %
        #              (time.time() - check, fvd))


if __name__ == '__main__':
    # play_with_MOSO_VAE()

    # background token shape: [batch, 1, 128, 8, 8]
    #     object token shape: [batch, 1, 128, 4, 4]
    #     motion token shape: [batch, n_frames, 128, 2, 2]

    # bg_toks = torch.randn(1, 1, 128, 8, 8)
    # id_toks = torch.randn(1, 1, 128, 4, 4)
    # mo_toks = torch.randn(1, 5, 128, 2, 2)

    # play_with_PVDM_Diffuser()

    play_with_all_process()



