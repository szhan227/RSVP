import time
import sys; sys.path.extend(['.', 'src'])
import numpy as np
import torch
from utils import AverageMeter
from torchvision.utils import save_image, make_grid
from einops import rearrange
from model.ldm.ddpm import DDPM

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
import os

import torchvision
import PIL

def save_image_grid(img, fname, drange, grid_size, normalize=True):
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    # _N, C, T, H, W = img.shape
    _N, B, T, H, W, C = img.shape
    # img = rearrange(img, "n b t h w c -> n b t c h w")
    # img = img.reshape(gh, gw, C, T, H, W)
    # img = img.transpose(3, 0, 4, 1, 5, 2)
    # img = img.reshape(T, gh * H, gw * W, C)
    img = img[0][0] # shape (t, h, w, c)

    print (f'Saving Video [{fname}] with {T} frames, img shape {H}, {W}')

    assert C in [3]

    if C == 3:
        # torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], 'RGB') for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    return img


def get_visualize_img(img): # img: [B T C H W]
    # mean = torch.tensor([0.485, 0.456, 0.406])
    # std = torch.tensor([0.229, 0.224, 0.225])
    # x = img[:8].detach().cpu() * std[None, None, :, None, None] + \
    #     mean[None, None, :, None, None]
    x = img[:8].detach().cpu()
    show_x = torch.clamp(x, min=0.0, max=1.0)
    b, t, c, h, w = show_x.shape
    # get output for fvd calculation
    show_x = rearrange(show_x, "b t c h w -> b t h w c").numpy() * 255.
    return show_x.astype(np.uint8)

    # below is old moso visualize
    show_x = show_x.permute((0, 3, 1, 4, 2)).numpy()
    show_x = show_x.reshape((b * h, t * w, c)) * 255.
    show_x = Image.fromarray(show_x.astype(np.uint8)).convert('RGB')
    return show_x


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

    cond_model = ema_model.module.diffusion_model.cond_model

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.module.diffusion_model.in_channels,
                           image_size=ema_model.module.diffusion_model.image_size,
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
                c = decoder.module.extract(rearrange(c / 127.5 - 1, 'b t c h w -> b c t h w').to(device).detach())
                z = diffusion_model.sample(batch_size=k, cond=c)
                pred = decoder.module.decode_from_sample(z).clamp(-1,1).cpu()
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
                fake = decoder.module.decode_from_sample(z).clamp(-1,1).cpu()
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
                fake = decoder.module.decode_from_sample(z).clamp(-1,1).cpu()
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


def test_fvd_moso(rank, ema_model, vqvae, loader, it, logger=None, num_loop=1):
    device = torch.device('cuda', rank)

    losses = dict()
    losses['fvd'] = AverageMeter()
    check = time.time()

    diffusion_model = DDPM(ema_model,
                           channels=ema_model.diffusion_model.in_channels,
                           image_size=ema_model.diffusion_model.image_size,
                           sampling_timesteps=100,
                           # parameterization='x0',
                           w=0.).to(device)
    real_embeddings = []
    pred_embeddings = []

    i3d = load_i3d_pretrained(device)
    xbgs = []
    xids = []
    reals = []
    preds = []
    xmos = []

    cbgs = []
    cids = []
    cmos = []

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
            if n > num_loop:
                break

            c_toks, x_toks = inputs

            cbg_tokens, cid_tokens, cmo_tokens = c_toks
            xbg_tokens, xid_tokens, xmo_tokens = x_toks

            B = xmo_tokens.shape[0]

            xbg_quantized, xid_quantized, xmo_quantized = vqvae.get_quantized_by_tokens_with_rearrange(
                                                                                                 xbg_tokens,
                                                                                                 xid_tokens,
                                                                                                 xmo_tokens)

            cbg_quantized, cid_quantized, cmo_quantized = vqvae.get_quantized_by_tokens_with_rearrange(
                                                                                                    cbg_tokens,
                                                                                                    cid_tokens,
                                                                                                    cmo_tokens)

            xbgs.append(xbg_quantized)
            xids.append(xid_quantized)
            xmos.append(xmo_quantized)
            cbgs.append(cbg_quantized)
            cids.append(cid_quantized)
            cmos.append(cmo_quantized)

        for i in range(min(num_loop, len(cbgs))):

            cbg, cid, cmo = cbgs[i], cids[i], cmos[i]
            xbg, xid, xmo = xbgs[i], xids[i], xmos[i]
            z = diffusion_model.sample_moso(batch_size=B, cond=(cbg, cid, cmo))

            cbg = rearrange(cbg, 'b c t h w -> b t c h w')
            cid = rearrange(cid, 'b c t h w -> b t c h w')
            # cmo = rearrange(cmo, 'b c t h w -> b t c h w')
            xbg = rearrange(xbg, 'b c t h w -> b t c h w')
            xid = rearrange(xid, 'b c t h w -> b t c h w')
            xmo = rearrange(xmo, 'b c t h w -> b t c h w')
            z = rearrange(z, 'b c t h w -> b t c h w')

            pred_bg_quantized, pred_id_quantized, pred_mo_quantized = cbg, cid, z

            true_bg_quantized, true_id_quantized, true_mo_quantized = xbg, xid, xmo

            pred_rec, _, _ = vqvae._decoder(pred_bg_quantized, pred_id_quantized, pred_mo_quantized)
            real_rec, _, _ = vqvae._decoder(true_bg_quantized, true_id_quantized, true_mo_quantized)

            pred_rec = get_visualize_img(pred_rec)
            real_rec = get_visualize_img(real_rec)

            real_embeddings.append(get_fvd_logits(real_rec, i3d=i3d, device=device))
            pred_embeddings.append(get_fvd_logits(pred_rec, i3d=i3d, device=device))

            reals.append(real_rec)
            preds.append(pred_rec)

    reals = np.array(reals)

    real_embeddings = torch.cat(real_embeddings)
    pred_embeddings = torch.cat(pred_embeddings)

    if rank == 0:
        real_vid = save_image_grid(reals, os.path.join(logger.logdir, f'real_{it}.gif'), drange=[0, 255],
                                   grid_size=(4, 4))
        real_vid = np.expand_dims(real_vid, 0).transpose(0, 1, 4, 2, 3)

        pred_vid = save_image_grid(preds, os.path.join(logger.logdir, f'generated_{it}.gif'), drange=[0, 255],
                                   grid_size=(4, 4))
        pred_vid = np.expand_dims(pred_vid, 0).transpose(0, 1, 4, 2, 3)

    fvd = frechet_distance(pred_embeddings.clone().detach(), real_embeddings.clone().detach())
    return fvd.item()


