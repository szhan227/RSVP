import cv2
import torch
import numpy as np
import yaml
from torchvision import transforms
from PIL import Image
import utils
from model.vq_vae.VQVAE import VQVAEModel
from model.ldm.unet import UNetModel, DiffusionWrapper
from model.ema import LitEma
from losses.ddpm import DDPM
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
import time
from utils import AverageMeter, Logger
import copy
from einops import rearrange
import random
from tools.token_dataloader import TokenLoader

logger = utils.logger


def train(frozen_vqvae, unet, train_data_path, num_epochs=100, device='cuda'):

    rank = 0
    ema_model = None
    cond_prob = 0.3


    moso_opt = yaml.load(open('config/vqvae_raw.yaml', 'r'), Loader=yaml.FullLoader)
    moso_model_opt = moso_opt['model']
    if frozen_vqvae is None:
        frozen_vqvae = VQVAEModel(moso_model_opt, moso_opt).to(device)

    linear_start = 0.0015
    linear_end = 0.0195
    log_every_t = 200
    w = 0.0

    unet_path = './config/small_unet.yaml'
    ldm_path = './config/ldm_base.yaml'
    unet_config = OmegaConf.load(unet_path).unet_config
    unet_config.cond_model = False
    # unet_config = OmegaConf.load(ldm_path).model.params.unet_config
    logger.debug(unet_config)
    if unet is None:
        unet = UNetModel(**unet_config).to(device)

    diffusion_wrapper = DiffusionWrapper(model=unet, conditioning_key=None).to(device)

    diffusion_criterion = DDPM(diffusion_wrapper,
                     channels=unet_config.in_channels,
                     image_size=unet_config.image_size,
                     linear_start=linear_start,
                     linear_end=linear_end,
                     log_every_t=log_every_t,
                     w=w,
                     ).to(device)


    losses = dict()
    losses['diffusion_loss'] = AverageMeter()
    check = time.time()

    # lr_scheduler = LambdaLR(opt, scheduler)
    if ema_model == None:
        ema_model = copy.deepcopy(diffusion_wrapper)
        ema = LitEma(ema_model)
        ema_model.eval()
    else:
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(11200, dtype=torch.int)
        ema_model.eval()

    optimizer = torch.optim.Adam(diffusion_wrapper.parameters(), lr=1e-4)

    # freeze vqvae during training for diffusion
    frozen_vqvae.eval()
    diffusion_wrapper.train()

    train_loader = TokenLoader(train_data_path, batch_size=2)

    for epoch in range(num_epochs):

        for it, inputs in enumerate(train_loader):
            # if it > 0:
            #     break

            bg_tokens, id_tokens, mo_tokens = inputs
            bg_tokens = bg_tokens.to(device)
            id_tokens = id_tokens.to(device)
            mo_tokens = mo_tokens.to(device)
            B, T, _, _ = mo_tokens.shape

            # shape: (B, T, C, H, W)
            bg_quantized, id_quantized, mo_quantized = frozen_vqvae.get_quantized_by_tokens(bg_tokens, id_tokens, mo_tokens)

            logger.debug('bg_quantized', bg_quantized.shape)
            logger.debug('id_quantized', id_quantized.shape)
            logger.debug('mo_quantized', mo_quantized.shape)

            # cbg_quantized, xbg_quantized = torch.clone(bg_quantized), torch.clone(bg_quantized)
            # cid_quantized, xid_quantized = torch.clone(id_quantized), torch.clone(id_quantized)
            # cmo_quantized, xmo_quantized = torch.chunk(mo_quantized, 2, dim=1)

            bg_quantized = rearrange(bg_quantized, 'b t c h w -> b c (t h w)')
            id_quantized = rearrange(id_quantized, 'b t c h w -> b c (t h w)')
            mo_quantized = rearrange(mo_quantized, 'b t c h w -> b c (t h w)')

            # cbg_quantized = rearrange(cbg_quantized, 'b t c h w -> b c (t h w)')
            # cid_quantized = rearrange(cid_quantized, 'b t c h w -> b c (t h w)')
            # cmo_quantized = rearrange(cmo_quantized, 'b t c h w -> b c (t h w)')
            # xbg_quantized = rearrange(xbg_quantized, 'b t c h w -> b c (t h w)')
            # xid_quantized = rearrange(xid_quantized, 'b t c h w -> b c (t h w)')
            # xmo_quantized = rearrange(xmo_quantized, 'b t c h w -> b c (t h w)')

            logger.debug('after rearrange quantized')
            logger.debug('xbg_quantized', bg_quantized.shape)
            logger.debug('xid_quantized', id_quantized.shape)
            logger.debug('xmo_quantized', mo_quantized.shape)

            z = torch.concat([bg_quantized, id_quantized, mo_quantized], dim=-1)
            # c = torch.concat([cbg_quantized, cid_quantized, cmo_quantized], dim=-1)

            logger.debug('show z shape', z.shape)

            # Unconditional Training
            (loss, t, output), loss_dict = diffusion_criterion(z.float())

            loss.backward()
            optimizer.step()

            if it % 25 == 0 and it > 0:
                ema(diffusion_wrapper)

            if it % 500 == 0:
                # psnr = test_psnr(rank, model, test_loader, it, logger)
                if logger is not None and rank == 0:
                    logger.scalar_summary('train/diffusion_loss', losses['diffusion_loss'].average, it)

                    logger.log('[Time %.3f] [Diffusion %f]' %
                         (time.time() - check, losses['diffusion_loss'].average))

                losses = dict()
                losses['diffusion_loss'] = AverageMeter()




if __name__ == '__main__':
    train(frozen_vqvae=None, unet=None, train_data_path='./data', num_epochs=1, device='cuda')

