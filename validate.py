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
from model.ldm.ddpm import DDPM
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
import time
from utils import AverageMeter, Logger
import copy
from einops import rearrange
import random
from tools.token_dataloader import UncondTokenLoader, CondTokenLoader

logger = utils.logger

def validate(input_batch, vqvae=None, unet=None, device='cpu'):
    """
    :param input_batch: raw input video in shape (D, B, T, C, H, W) after preprocessing
           D: number of decomposition, here to be 4: x, bg, id, mo respectively
           B: batch size
           T: number of frames, here to be 16
           C: color channels, here to be 3
           H: height, here to be 256
           W: width, here to be 256
    :param vqvae: The pretained vqvae model. If None, initialize a new one from config(only for testing)
    :param unet: The pretrained unet model. If None, initialize a new one from config(only for testing)
    :param device: cpu or cuda
    :return: the output video in shape (B, T, C, H, W)
    """

    linear_start = 0.0015
    linear_end = 0.0195
    log_every_t = 200
    w = 0.0
    _, B, T, _, _, _ = input_batch.shape

    if vqvae is None:
        moso_opt = yaml.load(open('config/vqvae_raw.yaml', 'r'), Loader=yaml.FullLoader)
        moso_model_opt = moso_opt['model']
        logger.debug('load vqvae and show config:', moso_model_opt)
        vqvae = VQVAEModel(moso_model_opt, moso_opt).to(device)

    vqvae.eval()

    if unet is None:
        unet_path = './config/small_unet.yaml'
        ldm_path = './config/ldm_base.yaml'
        unet_config = OmegaConf.load(unet_path).unet_config
        # unet_config = OmegaConf.load(ldm_path).model.params.unet_config

        unet_config.ds_bg = moso_model_opt['ds_background']
        unet_config.ds_id = moso_model_opt['ds_identity']
        unet_config.ds_mo = moso_model_opt['ds_motion']
        unet_config.vae_hidden = moso_model_opt['num_hiddens']

        logger.debug(unet_config)
        unet = UNetModel(**unet_config).to(device)

    diffusion_wrapper = DiffusionWrapper(unet)
    diffusion_wrapper.eval()

    ddpm_criterion = DDPM(diffusion_wrapper,
                          channels=unet.in_channels,
                          image_size=256,
                          linear_start=linear_start,
                          linear_end=linear_end,
                          log_every_t=log_every_t,
                          w=w,
                          ).to(device)

    input_batch = rearrange(input_batch, 'd b t c h w -> b d t c h w')
    x = input_batch[:, 0, :, :, :, :]
    bg = input_batch[:, 1, :, :, :, :]
    id = input_batch[:, 2, :, :, :, :]
    mo = input_batch[:, 3, :, :, :, :]

    bg_toks, id_toks, mo_toks = vqvae.extract_tokens([x, bg, id, mo], is_training=False)

    bg_quantized, id_quantized, mo_quantized = vqvae.get_quantized_by_tokens(bg_toks, id_toks, mo_toks)

    bg_quantized = rearrange(bg_quantized, 'b t c h w -> b c (t h w)')
    id_quantized = rearrange(id_quantized, 'b t c h w -> b c (t h w)')
    mo_quantized = rearrange(mo_quantized, 'b t c h w -> b c (t h w)')

    z = torch.cat([bg_quantized, id_quantized, mo_quantized], dim=-1)

    (loss, t, z_output), loss_dict = ddpm_criterion(z)

    ds_bg = unet.ds_bg
    ds_id = unet.ds_id
    ds_mo = unet.ds_mo
    hidden_size = unet.vae_hidden
    n_frames = T

    out_bg, out_id, out_mo = vqvae.convert_latent_to_quantized(z_output, ds_bg, ds_id, ds_mo, hidden_size, n_frames)

    x_output = vqvae._decode(out_bg, out_id, out_mo)
    print('show final output.shape:', x_output.shape)
    return x_output


if __name__ == '__main__':
    #input_batch: raw input video in shape (D, B, T, C, H, W)
    input_batch = torch.randn(4, 1, 16, 3, 256, 256)
    output = validate(input_batch, vqvae=None, unet=None, device='cpu')
    print(output.shape)

