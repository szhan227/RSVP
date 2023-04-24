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

def validate(vqvae, unet, input_batch, device='cpu'):
    """
    :param vqvae: The pretained vqvae model.
    :param unet: The pretrained unet model.
    :param input_batch: raw input video in shape (D, B, T, C, H, W), B = 1
    :param device: cpu or cuda
    :return: the output video in shape (B, T, C, H, W)
    """

    linear_start = 0.0015
    linear_end = 0.0195
    log_every_t = 200
    w = 0.0

    vqvae.eval()

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
    x, bg, id, mo = torch.chunk(input_batch, 4, dim=1)

    bg_toks, id_toks, mo_toks = vqvae.extract_tokens([x, bg, id, mo], is_training=False)

    bg_quantized, id_quantized, mo_quantized = vqvae.get_quantized_by_tokens(bg_toks, id_toks, mo_toks)

    bg_quantized = rearrange(bg_quantized, 'b t c h w -> b c (t h w)')
    id_quantized = rearrange(id_quantized, 'b t c h w -> b c (t h w)')
    mo_quantized = rearrange(mo_quantized, 'b t c h w -> b c (t h w)')

    z = torch.cat([bg_quantized, id_quantized, mo_quantized], dim=-1)

    (loss, t, output), loss_dict = ddpm_criterion(z, x)

    output = vqvae.decode(z, is_training=False)

