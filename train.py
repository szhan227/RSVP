# import cv2
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
from utils import AverageMeter, Logger,dict2obj
import copy
from einops import rearrange
import random
from evals.eval import test_psnr, test_ifvd, test_fvd_ddpm, test_fvd_moso
from torch.utils.data import DataLoader
from tools.token_dataloader import UncondTokenLoader, CondTokenDataset, get_train_valid_loader
import argparse
import os, csv

logger = utils.logger


def write_csv(record, path):
    header = ['iteration', 'loss']
    file_exists = os.path.isfile(path)

    with open(path, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(record)


def train(frozen_vqvae,
          unet,
          train_data_path,
          num_epochs=100,
          batch_size=2,
          save_every_n_epoch=None,
          device='cuda',
          local_test=False):

    rank = 0
    ema_model = None
    cond_prob = 0.3


    # # TODO: should load a pretrained vqvae model, but now we just use a random initialized one.
    if local_test:
        if frozen_vqvae is None:
            moso_opt = yaml.load(open('config/vqvae_raw.yaml', 'r'), Loader=yaml.FullLoader)
            moso_model_opt = moso_opt['model']
            logger.debug('show vqvae config:', moso_model_opt)
            frozen_vqvae = VQVAEModel(moso_model_opt, moso_opt).to(device)
            logger.info('Local Test: initialize a vqvae model.')

    else:
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


    linear_start = 0.0015
    linear_end = 0.0195
    log_every_t = 200
    w = 0.0

    # intialize DDPM model from scratch
    if unet is None:
        unet_path = './config/small_unet.yaml'
        unet_config = OmegaConf.load(unet_path).unet_config

        # change model channels for local test to a small number
        if local_test:
            unet_config.model_channels = 32

        unet_config.ds_bg = moso_model_opt['ds_background']
        unet_config.ds_id = moso_model_opt['ds_identity']
        unet_config.ds_mo = moso_model_opt['ds_motion']
        unet_config.vae_hidden = moso_model_opt['num_hiddens']

        logger.debug(unet_config)
        unet = UNetModel(**unet_config).to(device)

    diffusion_wrapper = DiffusionWrapper(model=unet, conditioning_key=None).to(device)

    ddpm_criterion = DDPM(diffusion_wrapper,
                     channels=unet_config.in_channels,
                     image_size=unet_config.image_size,
                     linear_start=linear_start,
                     linear_end=linear_end,
                     log_every_t=log_every_t,
                     # parameterization='x0',
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

    if unet.cond_model:
        train_loader, valid_loader = get_train_valid_loader(train_data_path,
                                                            batch_size=batch_size,
                                                            device=device,
                                                            train_valid_split=1)
        logger.info('Load conditional token dataset.')
    else:
        train_loader = UncondTokenLoader(train_data_path, batch_size=batch_size, device=device)
        logger.info('Load unconditional token dataset.')

    # start train loop
    for epoch in range(num_epochs):
        # if args.local_test and epoch > 0:
        #     break
        total_length = len(train_loader)

        for it, inputs in enumerate(train_loader):
            if local_test and it > 0:
                break
            diffusion_wrapper.zero_grad()

            c_toks, x_toks = inputs
            cbg_toks, cid_toks, cmo_toks = c_toks
            xbg_toks, xid_toks, xmo_toks = x_toks

            logger.debug('xbg_toks', xbg_toks.shape)
            logger.debug('xid_toks', xid_toks.shape)
            logger.debug('xmo_toks', xmo_toks.shape)

            B, T, _, _ = cmo_toks.shape

            with torch.no_grad():
                cbg_quantized, cid_quantized, cmo_quantized = frozen_vqvae.get_quantized_by_tokens_with_rearrange(
                    cbg_toks, cid_toks, cmo_toks)
                xbg_quantized, xid_quantized, xmo_quantized = frozen_vqvae.get_quantized_by_tokens_with_rearrange(
                    xbg_toks, xid_toks, xmo_toks)

            logger.debug('right before calling denoising module')
            logger.debug('bg_quantized', xbg_quantized.shape)
            logger.debug('id_quantized', xid_quantized.shape)
            logger.debug('mo_quantized', xmo_quantized.shape)

            zc = cbg_quantized.float(), cid_quantized.float(), cmo_quantized.float()
            zx = xbg_quantized.float(), xid_quantized.float(), xmo_quantized.float()

            (loss, t, output), loss_dict = ddpm_criterion(zx, zc)

            loss.backward()
            optimizer.step()
            losses['diffusion_loss'].update(loss.item(), 1)

            if it % 25 == 0 and it > 0:
                ema(diffusion_wrapper)

            if it % 500 == 0:
                # psnr = test_psnr(rank, model, test_loader, it, logger)
                if logger is not None and rank == 0:
                    logger.scalar_summary('train/diffusion_loss', losses['diffusion_loss'].average, it)

                losses = dict()
                losses['diffusion_loss'] = AverageMeter()

            # Currently disable fvd eval during training
            if False and (it % 10000 == 0 and rank == 0):
                # logger.set_level('debug')
                logger.debug('start to eval fvd')
                torch.save(diffusion_wrapper.state_dict(), f'chkpt/ddpm/ddpm_wrapper_model_{epoch}_{it}.pt')
                ema.copy_to(ema_model)
                torch.save(ema_model.state_dict(), f'chkpt/ddpm/ema_{epoch}_{it}.pt')
                # assert False, "a new test_fvd_ddpm that uses new first_stage_model decoder"
                fvd = test_fvd_moso(rank, ema_model, frozen_vqvae, valid_loader, it, logger)

                if logger is not None and rank == 0:
                    logger.scalar_summary('test/fvd', fvd, it)

                    logger.info('[Time %.3f] [FVD %f]' %
                         (time.time() - check, fvd))
                # logger.set_level('info')
            logger.info(f'\r[Epoch {epoch}] [{it + 1}/{total_length}] [Diffusion Loss {loss.item()}]', end='')
        print()

        # save model to checkpoint every n epoch
        if save_every_n_epoch and epoch > 0 and epoch % save_every_n_epoch == 0:
            torch.save(diffusion_wrapper.state_dict(), f'chkpt/ddpm/ddpm_wrapper_model_ep_{epoch}.pt')

        write_csv([str(epoch), str(loss.item())], './results/training_loss.csv')

    torch.save(diffusion_wrapper.state_dict(), 'chkpt/ddpm/ddpm_wrapper_model.pt')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',     type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--save_n',     type=int, default=1)
    parser.add_argument('--msg_level',  type=str, default='info')
    parser.add_argument('--device',     type=str, default='cuda')
    parser.add_argument('--local_test', type=bool, default=False)

    args = parser.parse_args()
    # # change message level of the logger.
    logger.set_level(args.msg_level)
    if args.local_test:
        logger.set_level('info')

    # TODO: load your pretrained vqvae model here. Unet = None means to train DDPM from scratch.
    frozen_vqvae = None

    train_data_path = './data2' if args.local_test else '/export2/xu1201/MOSO/merged_Token/UCF101/img256_16frames/train'

    train(frozen_vqvae=frozen_vqvae,
          unet=None,
          train_data_path=train_data_path,
          num_epochs=args.epochs,
          batch_size=args.batch_size,
          save_every_n_epoch=args.save_n,
          device=args.device,
          local_test=args.local_test
          )
