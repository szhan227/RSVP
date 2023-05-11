import time
import numpy as np
import torch
from utils import AverageMeter
from torchvision.utils import save_image, make_grid
from einops import rearrange
from losses.ddpm import DDPM
import cv2

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
import os

import torchvision
import PIL
from PIL import Image

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

def test_psnr(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['psnr'] = AverageMeter()
    check = time.time()

    model.eval()
    with torch.no_grad():
        for n, (x, _) in enumerate(loader):
            if n > 100:
                break
            batch_size = x.size(0)
            clip_length = x.size(1)
            x = x.to(device) / 127.5 - 1
            recon, _ = model(rearrange(x, 'b t c h w -> b c t h w'))

            x = x.view(batch_size, -1)
            recon = recon.view(batch_size, -1)

            mse = ((x * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
            psnr = (-10 * torch.log10(mse)).mean()

            losses['psnr'].update(psnr.item(), batch_size)


    model.train()
    return losses['psnr'].average

def test_ifvd(rank, model, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['fvd'] = AverageMeter()
    check = time.time()

    real_embeddings = []
    fake_embeddings = []
    fakes = []
    reals = []

    model.eval()
    i3d = load_i3d_pretrained(device)

    with torch.no_grad():
        for n, (real, idx) in enumerate(loader):
            if n > 512:
                break
            batch_size = real.size(0)
            clip_length = real.size(1)
            real = real.to(device)
            fake, _ = model(rearrange(real / 127.5 - 1, 'b t c h w -> b c t h w'))

            real = rearrange(real, 'b t c h w -> b t h w c') # videos
            fake = rearrange((fake.clamp(-1,1) + 1) * 127.5, '(b t) c h w -> b t h w c', b=real.size(0))

            real = real.type(torch.uint8).cpu()
            fake = fake.type(torch.uint8)

            real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))
            fake_embeddings.append(get_fvd_logits(fake.cpu().numpy(), i3d=i3d, device=device))
            if len(fakes) < 16:
                reals.append(rearrange(real[0:1], 'b t h w c -> b c t h w'))
                fakes.append(rearrange(fake[0:1], 'b t h w c -> b c t h w'))

    model.train()

    reals = torch.cat(reals)
    fakes = torch.cat(fakes)

    if rank == 0:
        real_vid = save_image_grid(reals.cpu().numpy(), os.path.join(logger.logdir, "real.gif"), drange=[0, 255], grid_size=(4,4))
        fake_vid = save_image_grid(fakes.cpu().numpy(), os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255], grid_size=(4,4))

        if it == 0:
            real_vid = np.expand_dims(real_vid,0).transpose(0, 1, 4, 2, 3)
            logger.video_summary('real', real_vid, it)

        fake_vid = np.expand_dims(fake_vid,0).transpose(0, 1, 4, 2, 3)
        logger.video_summary('recon', fake_vid, it)

    real_embeddings = torch.cat(real_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)
    
    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()


def test_fvd_ddpm(rank, ema_model, decoder, loader, it, logger=None):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['fvd'] = AverageMeter()
    check = time.time()

    cond_model = ema_model.diffusion_model.cond_model

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           w=0.).to(device)
    real_embeddings = []
    fake_embeddings = []
    pred_embeddings = []

    reals = []
    fakes = []
    predictions = []

    i3d = load_i3d_pretrained(device)

    if cond_model:
        with torch.no_grad():        
            for n, (x, _) in enumerate(loader):
                k = min(4, x.size(0))
                if n >= 4:
                    break
                    x = torch.cat([x,zeros], dim=0)
                c, real = torch.chunk(x[:k], 2, dim=1)
                c = decoder.extract(rearrange(c / 127.5 - 1, 'b t c h w -> b c t h w').to(device).detach())
                z = diffusion_model.sample(batch_size=k, cond=c)
                pred = decoder.decode_from_sample(z).clamp(-1,1).cpu()
                pred = (1 + rearrange(pred, '(b t) c h w -> b t h w c', b=k)) * 127.5
                pred = pred.type(torch.uint8)
                pred_embeddings.append(get_fvd_logits(pred.numpy(), i3d=i3d, device=device))

                real = rearrange(real, 'b t c h w -> b t h w c')
                real = real.type(torch.uint8)
                real_embeddings.append(get_fvd_logits(real.numpy(), i3d=i3d, device=device))

                if len(predictions) < 4:
                    reals.append(rearrange(x[:k].type(torch.uint8), 'b t c h w -> b c t h w'))
                    predictions.append(torch.cat([rearrange(x[:k,:x.size(1)//2].type(torch.uint8), 'b t c h w -> b c t h w').type(torch.uint8), 
                                                  rearrange(pred, 'b t h w c -> b c t h w')], dim=2))


            for i in range(4):
                print(i)
                z = diffusion_model.sample(batch_size=k)
                fake = decoder.decode_from_sample(z).clamp(-1,1).cpu()
                fake = (rearrange(fake, '(b t) c h w -> b t h w c', b=k)+1) * 127.5
                fake = fake.type(torch.uint8)
                fake_embeddings.append(get_fvd_logits(fake.numpy(), i3d=i3d, device=device))

                if len(fakes) < 4:
                    fakes.append(rearrange(fake, 'b t h w c -> b c t h w'))

        reals = torch.cat(reals)
        fakes = torch.cat(fakes)
        predictions = torch.cat(predictions)

        real_embeddings = torch.cat(real_embeddings)
        fake_embeddings = torch.cat(fake_embeddings)

        if rank == 0:
            real_vid = save_image_grid(reals.cpu().numpy(), os.path.join(logger.logdir, f'real_{it}.gif'), drange=[0, 255], grid_size=(k,4))
            real_vid = np.expand_dims(real_vid,0).transpose(0, 1, 4, 2, 3)
            fake_vid = save_image_grid(fakes.cpu().numpy(), os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255], grid_size=(k,4))
            fake_vid = np.expand_dims(fake_vid,0).transpose(0, 1, 4, 2, 3)
            pred_vid = save_image_grid(predictions.cpu().numpy(), os.path.join(logger.logdir, f'predicted_{it}.gif'), drange=[0, 255], grid_size=(k,4))
            pred_vid = np.expand_dims(pred_vid,0).transpose(0, 1, 4, 2, 3)

            logger.video_summary('real', real_vid, it)
            logger.video_summary('unconditional', fake_vid, it)
            logger.video_summary('prediction', pred_vid, it)
    else:
        with torch.no_grad():        
            for n, (real, _) in enumerate(loader):
                if n >= 4:
                    break
                real = rearrange(real, 'b t c h w -> b t h w c')
                real = real.type(torch.uint8).numpy()
                real_embeddings.append(get_fvd_logits(real, i3d=i3d, device=device))

            for i in range(4):
                print(i)
                z = diffusion_model.sample(batch_size=4)
                fake = decoder.decode_from_sample(z).clamp(-1,1).cpu()
                fake = (1+rearrange(fake, '(b t) c h w -> b t h w c', b=4)) * 127.5
                fake = fake.type(torch.uint8)
                fake_embeddings.append(get_fvd_logits(fake.numpy(), i3d=i3d, device=device))

                if len(fakes) < 4:
                    fakes.append(rearrange(fake, 'b t h w c -> b c t h w'))

        fakes = torch.cat(fakes)

        real_embeddings = torch.cat(real_embeddings)
        fake_embeddings = torch.cat(fake_embeddings)

        if rank == 0:
            fake_vid = save_image_grid(fakes.cpu().numpy(), os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255], grid_size=(4,4))
            fake_vid = np.expand_dims(fake_vid,0).transpose(0, 1, 4, 2, 3)
            logger.video_summary('unconditional', fake_vid, it)

    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()

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
            gts.append(xmo)    
            contexts.append(context)

        for i in range(4):         
            
            print(i)
            mask = torch.zeros(gts[i].shape[0], gts[i].shape[2], gts[i].shape[3], gts[i].shape[4]).long()
            mask[:, :gts[i].shape[2]//2, ...] = 1
            z = diffusion_model.sample_moso(gts[i].clone().to(device), mask.to(device), contexts[i].to(device))
            # above output: b, c, t, h, w

            x_rec, _, _ = first_stage_model._decoder(bgs[i], ids[i], gts[i].permute(0, 2, 1, 3, 4))
            x_rec_fake, _, _ = first_stage_model._decoder(bgs[i], ids[i], z.permute(0, 2, 1, 3, 4))
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
        real_vid = save_image_grid(reals, os.path.join(logger.logdir, f'real_{it}.gif'), drange=[0, 255], grid_size=(4,B))
        real_vid = np.expand_dims(real_vid,0).transpose(0, 1, 4, 2, 3)
            
        fake_vid = save_image_grid(fakes, os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255], grid_size=(4,B))
        fake_vid = np.expand_dims(fake_vid,0).transpose(0, 1, 4, 2, 3)
        
        logger.video_summary('real', real_vid, it)
        logger.video_summary('prediction', fake_vid, it)

    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()

