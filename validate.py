import cv2
import torch
import numpy as np
import yaml
from torchvision import transforms
from PIL import Image
import utils
from utils import dict2obj
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
from tools.token_dataloader import UncondTokenLoader, CondTokenDataset, get_dataloader
from evals.eval import test_fvd_moso

logger = utils.logger

def validate(input_batch, condition_batch=None, vqvae=None, diffusion_wrapper=None, device='cpu'):
    """
    :param input_batch: raw input video in shape (D, B, T, C, H, W) after preprocessing
           D: number of decomposition, here to be 4: x, bg, id, mo respectively
           B: batch size
           T: number of frames, here to be 16
           C: color channels, here to be 3
           H: height, here to be 256
           W: width, here to be 256
    :param condition_batch: condition video in shape (B, T, C, H, W), same structure as input_batch
           if none, set all to zero
    :param vqvae: The pretained vqvae model. If None, initialize a new one from config(only for testing)
    :param unet: The pretrained unet model. If None, initialize a new one from config(only for testing)
    :param device: cpu or cuda
    :return: the output video in shape (B, T, C, H, W)
    """

    if condition_batch is None:
        condition_batch = torch.zeros_like(input_batch)

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

    if diffusion_wrapper is None:
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
        diffusion_wrapper = DiffusionWrapper(unet).to(device)

    diffusion_wrapper.eval()

    ddpm_criterion = DDPM(diffusion_wrapper,
                          channels=unet.in_channels,
                          image_size=256,
                          linear_start=linear_start,
                          linear_end=linear_end,
                          log_every_t=log_every_t,
                          w=w,
                          ).to(device)

    ddpm_criterion.eval()

    print('betas:', ddpm_criterion.betas)
    print('alphas:', 1.0 - ddpm_criterion.betas)
    print('sqrt_alphas_cumprod:', ddpm_criterion.sqrt_alphas_cumprod)
    print('sqrt_one_minus_alphas_cumprod:', ddpm_criterion.sqrt_one_minus_alphas_cumprod)

    return

    ds_bg = unet.ds_bg
    ds_id = unet.ds_id
    ds_mo = unet.ds_mo
    hidden_size = diffusion_wrapper.diffusion_model.vae_hidden
    n_frames = T

    input_batch = rearrange(input_batch, 'd b t c h w -> b d t c h w')
    x = input_batch[:, 0, :, :, :, :]
    xbg = input_batch[:, 1, :, :, :, :]
    xid = input_batch[:, 2, :, :, :, :]
    xmo = input_batch[:, 3, :, :, :, :]

    condition_batch = rearrange(condition_batch, 'd b t c h w -> b d t c h w')
    c = condition_batch[:, 0, :, :, :, :]
    cbg = condition_batch[:, 1, :, :, :, :]
    cid = condition_batch[:, 2, :, :, :, :]
    cmo = condition_batch[:, 3, :, :, :, :]

    with torch.no_grad():

        xbg_toks, xid_toks, xmo_toks = vqvae.extract_tokens([x, xbg, xid, xmo], is_training=False)
        xbg_quantized, xid_quantized, xmo_quantized = vqvae.get_quantized_by_tokens(xbg_toks, xid_toks, xmo_toks)

        cbg_toks, cid_toks, cmo_toks = vqvae.extract_tokens([c, cbg, cid, cmo], is_training=False)
        logger.debug('xbg_toks shape:', xbg_toks.shape)
        logger.debug('xid_toks shape:', xid_toks.shape)
        logger.debug('xmo_toks shape:', xmo_toks.shape)
        cbg_quantized, cid_quantized, cmo_quantized = vqvae.get_quantized_by_tokens(cbg_toks, cid_toks, cmo_toks)

        xbg_quantized = rearrange(xbg_quantized, 'b t c h w -> b c (t h w)')
        xid_quantized = rearrange(xid_quantized, 'b t c h w -> b c (t h w)')
        xmo_quantized = rearrange(xmo_quantized, 'b t c h w -> b c (t h w)')

        cbg_quantized = rearrange(cbg_quantized, 'b t c h w -> b c (t h w)')
        cid_quantized = rearrange(cid_quantized, 'b t c h w -> b c (t h w)')
        cmo_quantized = rearrange(cmo_quantized, 'b t c h w -> b c (t h w)')

        zx = torch.cat([xbg_quantized, xid_quantized, xmo_quantized], dim=-1)
        zc = torch.cat([cbg_quantized, cid_quantized, cmo_quantized], dim=-1)

        (loss, t, z_output), loss_dict = ddpm_criterion(zx, zc)

        logger.debug('z_output.shape:', z_output.shape)
        out_bg, out_id, out_mo = vqvae.convert_latent_to_quantized(z_output, ds_bg, ds_id, ds_mo, hidden_size, n_frames)

    logger.debug('out_bg.shape:', out_bg.shape)
    logger.debug('out_id.shape:', out_id.shape)
    logger.debug('out_mo.shape:', out_mo.shape)
    x_output, _, _ = vqvae._decoder(out_bg, out_id, out_mo)
    logger.debug('show final output.shape:', x_output.shape)
    return x_output


def validate_fvd():

    linear_start = 0.0015
    linear_end = 0.0195
    log_every_t = 200
    w = 0.0

    batch_size = 1

    device = 'cuda'

    moso_opt = dict2obj(yaml.load(open('config/vqvae_raw.yaml', 'r'), Loader=yaml.FullLoader))
    moso_model_opt = moso_opt['model']
    logger.debug('show vqvae config:', moso_model_opt)
    frozen_vqvae = VQVAEModel(moso_model_opt, moso_opt).to(device)
    if moso_model_opt['checkpoint_path'] is not None:
        state = torch.load(moso_model_opt['checkpoint_path'], map_location='cpu')
        start_step = state['steps']

        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state['state'].items():
            if 'total_ops' in k or 'total_params' in k:
                continue
            if 'perceptual_loss' in k or '_discriminator' in k:
                # if 'perceptual_loss' in k:
                continue
            if k[:7] == 'module.':
                new_state_dict[k[7:]] = v

        # model.load_state_dict(new_state_dict, strict=False)
        frozen_vqvae.load_state_dict(new_state_dict, strict=moso_model_opt['load_strict'])
        logger.info("Successfully load state {} with step {}.".format(moso_model_opt['checkpoint_path'], start_step))

        unet_path = './config/small_unet.yaml'
        unet_config = OmegaConf.load(unet_path).unet_config

        unet_config.ds_bg = moso_model_opt['ds_background']
        unet_config.ds_id = moso_model_opt['ds_identity']
        unet_config.ds_mo = moso_model_opt['ds_motion']
        unet_config.vae_hidden = moso_model_opt['num_hiddens']

        logger.debug(unet_config)
        unet = UNetModel(**unet_config).to(device)

    diffusion_wrapper = DiffusionWrapper(model=unet, conditioning_key=None).to(device)
    ddpm_wrapper_state = torch.load('chkpt/ddpm/ddpm_wrapper_model.pt', map_location='cpu')
    diffusion_wrapper.load_state_dict(ddpm_wrapper_state, strict=False)

    ddpm_criterion = DDPM(diffusion_wrapper,
                          channels=unet_config.in_channels,
                          image_size=unet_config.image_size,
                          linear_start=linear_start,
                          linear_end=linear_end,
                          log_every_t=log_every_t,
                          w=w,
                          ).to(device)


    unet.eval()
    diffusion_wrapper.eval()

    ema_model = copy.deepcopy(diffusion_wrapper)
    ema = LitEma(ema_model)
    ema_model.eval()

    rank = 0

    logger.info('start to eval fvd')
    check = time.time()
    # torch.save(diffusion_wrapper.state_dict(), f'chkpt/ddpm/ddpm_wrapper_model_{epoch}_{it}.pt')
    ema.copy_to(ema_model)
    # torch.save(ema_model.state_dict(), f'chkpt/ddpm/ema_{epoch}_{it}.pt')
    # assert False, "a new test_fvd_ddpm that uses new first_stage_model decoder"

    path = '/export2/xu1201/MOSO/merged_Token/UCF101/img256_16frames/valid'

    valid_loader = get_dataloader(data_folder_path=path, batch_size=batch_size, device=device)

    fvd = test_fvd_moso(rank, ema_model, frozen_vqvae, valid_loader, it=0, logger=logger, num_loop=2)

    # if logger is not None and rank == 0:
    logger.scalar_summary('test/fvd', fvd, 0)

    logger.info('[Time %.3f] [FVD %f]' %
                (time.time() - check, fvd))


if __name__ == '__main__':

    logger.set_level('info')
    validate_fvd()
