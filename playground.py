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
import datetime

logger = utils.logger
def preprocess(num_frames=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    logger.debug('xt.shape:', xt.shape, xt.dtype)
    assert False, 'I am here'
    # TODO: give 32 frames to encode, first half as condition, second half as inputs
    start = 0

    transform = transforms.Resize((256, 256))

    lower_bound = 0.2
    upper_bound = 0.8
    pre_img = transform(xt[start])
    nxt_img = transform(xt[start + 1])

    logger.debug('pre_img.shape:', pre_img.shape)
    logger.debug('nxt_img.shape:', nxt_img.shape)

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

    #change dtype to float16
    dtype = torch.float32
    ret_img = ret_img.to(device, dtype=dtype)
    ret_img_bg = ret_img_bg.to(device, dtype=dtype)
    ret_img_id = ret_img_id.to(device, dtype=dtype)
    ret_img_mo = ret_img_mo.to(device, dtype=dtype)

    return [ret_img, ret_img_bg, ret_img_id, ret_img_mo]

def play_with_MOSO_VAE():
    logger.debug('cuda: ', torch.torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug('in this project, use device: ', device)

    # video_cap = cv2.VideoCapture('testvideo1.avi')
    #
    # xt = []
    # while True:
    #     success, frame = video_cap.read()
    #     if not success:
    #         break
    #     xt.append(frame)
    # xt = np.array(xt)
    # xt = torch.tensor(xt) / 255.0
    # xt = xt.permute(0, 3, 1, 2).to(device)
    # logger.debug('xt.shape:', xt.shape, xt.dtype)
    #
    # num_frames = 16
    # start = 0
    #
    # transform = transforms.CenterCrop(256)
    #
    # lower_bound = 0.2
    # upper_bound = 0.8
    # pre_img = transform(xt[start])
    # nxt_img = transform(xt[start + 1])
    #
    # logger.debug('pre_img.shape:', pre_img.shape)
    # logger.debug('nxt_img.shape:', nxt_img.shape)
    # imgs, imgs_bg, imgs_id, imgs_mo = [], [], [], []
    #
    # for i in range(start + 2, start + 2 + num_frames):
    #     cur_img = nxt_img
    #     nxt_img = transform(xt[i])
    #
    #     cur_diff = cur_img * 2 - pre_img - nxt_img
    #     max_diff = torch.max(torch.abs(cur_diff), dim=0)[0]
    #     id_mask = (max_diff > lower_bound) & (max_diff <= upper_bound)
    #     img_id = id_mask[None, :, :] * cur_img
    #     img_bg = cur_img - img_id
    #
    #     imgs.append(cur_img.unsqueeze(0))
    #     imgs_bg.append(img_bg.unsqueeze(0))
    #     imgs_id.append(img_id.unsqueeze(0))
    #     imgs_mo.append(cur_diff.unsqueeze(0))
    #
    # ret_img = torch.cat(imgs, dim=0)
    # ret_img_bg = torch.cat(imgs_bg, dim=0)
    # ret_img_id = torch.cat(imgs_id, dim=0)
    # ret_img_mo = torch.cat(imgs_mo, dim=0)
    #
    # ret_img = torch.unsqueeze(ret_img, dim=0).to(device)
    # ret_img_bg = torch.unsqueeze(ret_img_bg, dim=0).to(device)
    # ret_img_id = torch.unsqueeze(ret_img_id, dim=0).to(device)
    # ret_img_mo = torch.unsqueeze(ret_img_mo, dim=0).to(device)
    ret_img, ret_img_bg, ret_img_id, ret_img_mo = preprocess(16)

    logger.debug('ret_img.shape:', ret_img.shape)
    logger.debug('ret_img_bg.shape:', ret_img_bg.shape)
    logger.debug('ret_img_id.shape:', ret_img_id.shape)
    logger.debug('ret_img_mo.shape:', ret_img_mo.shape)

    validate_inputs = [ret_img, ret_img, ret_img, ret_img_mo]

    opt = yaml.load(open('config/vqvae_raw.yaml', 'r'), Loader=yaml.FullLoader)
    model_opt = opt['model']
    model = VQVAEModel(model_opt, opt).to(device)

    logger.debug('before my_encode')
    bg_toks, id_toks, mo_toks = model.my_encode([ret_img, ret_img_bg, ret_img_id, ret_img_mo], is_training=False)
    logger.debug('bg_toks.shape:', bg_toks.shape)
    logger.debug('id_toks.shape:', id_toks.shape)
    logger.debug('mo_toks.shape:', mo_toks.shape)
    outputs = model._decoder(bg_toks, id_toks, mo_toks)[0]
    logger.debug('output.shape:', outputs.shape)

    # (5, 4, 2048)
    # outputs = model(validate_inputs, is_training=False, writer=None)
    # dict_keys(['loss', 'x_rec', 'quantize_bg', 'quantize_id', 'quantize_mo', 'ssim_metric', 'rec_loss', 'lpips_loss', 'record_logs', 'optimizer_idx'])
    logger.debug('outputs.shape:', outputs['x_rec'].shape)


def play_with_PVDM_Diffuser():

    unet_path = './config/small_unet.yaml'
    unet_config = OmegaConf.load(unet_path).unet_config
    print(unet_config)
    logger.debug('cuda: ', torch.torch.cuda.device_count())
    num_frames = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug('in this project, use device: ', device)
    video_x = torch.randn(1, 32, 4, 2048).to(device)
    timesteps = torch.tensor([0, 1, 2, 3, 4]).to(device)
    unet = UNetModel(**unet_config).to(device)
    ddpm = DiffusionWrapper(model=unet, conditioning_key=None)

    # 'b t c h w -> b c t h w'
    # x = [b t c//2 h w], cond = [b t c//2 h w]
    # evenly split input videos into two parts in the time dimension
    # ddpm takes in latent representations z

    output = ddpm(x=video_x, cond=None, t=timesteps)
    logger.debug('show output shape: ', output.shape)



def play_with_all_process():


    scaler = GradScaler()

    rank = 0
    ema_model = None
    cond_prob = 0.3

    unet_path = './config/small_unet.yaml'
    ldm_path = './config/ldm_base.yaml'
    unet_config = OmegaConf.load(unet_path).unet_config
    unet_config.ds_bg = 3
    unet_config.ds_id = 4
    unet_config.ds_mo = 5
    unet_config.vae_hidden = 256
    # unet_config = OmegaConf.load(ldm_path).model.params.unet_config
    logger.debug(unet_config)
    logger.debug('cuda: ', torch.torch.cuda.device_count())

    device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.debug('in this project, use device: ', device)
    video_x = torch.randn(5, 4, 2048).to(device)
    timesteps = torch.tensor([0, 1, 2, 3, 4]).to(device)
    unet = UNetModel(**unet_config).to(device)
    model = DiffusionWrapper(model=unet, conditioning_key=None).to(device)

    moso_opt = yaml.load(open('config/vqvae_raw.yaml', 'r'), Loader=yaml.FullLoader)
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
    logger.debug('video_x.shape: ', video_x.shape)
    train_loader = [(video_x, None)]

    for it, (x, _) in enumerate(train_loader):

        # TODO: input mode(options): train_loader should contains [tokens, quantized]
        # pretrained vqvae model: batch_size = 2
        #                         num_batch =
        # Now for quantized

        x = x.to(device)
        # x = x.repeat(3, 1, 1, 1, 1, 1)

        batch_size = x.shape[0]
        x = rearrange(x, 'b d t c h w -> b d t c h w')
        logger.debug('show x.shape: ', x.shape)
        # x = rearrange(x / 127.5 - 1, 'b t c h w -> b c t h w')  # videos
        # x_bg = rearrange(x_bg / 127.5 - 1, 'b t c h w -> b c t h w')  # videos
        # x_id = rearrange(x_id / 127.5 - 1, 'b t c h w -> b c t h w')  # videos
        # x_mo = rearrange(x_mo / 127.5 - 1, 'b t c h w -> b c t h w')  # videos
        #
        # logger.debug('x.shape: ', x.shape)
        # logger.debug('x_bg.shape: ', x_bg.shape)
        # logger.debug('x_id.shape: ', x_id.shape)
        # logger.debug('x_mo.shape: ', x_mo.shape)

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

                logger.debug('c.shape: ', c.shape)
                logger.debug('x.shape: ', x.shape)
                with autocast():
                    with torch.no_grad():
                        x_img, x_bg, x_id, x_mo = x[:, 0, :, :, :, :], x[:, 1, :, :, :, :], x[:, 2, :, :, :, :], x[:, 3, :, :, :, :]
                        c_img, c_bg, c_id, c_mo = c[:, 0, :, :, :, :], c[:, 1, :, :, :, :], c[:, 2, :, :, :, :], c[:, 3, :, :, :, :]

                        logger.debug('x dtype: ', x.dtype)
                        logger.debug('x_img.shape: ', x_img.shape, x_img.dtype)
                        logger.debug('x_bg.shape: ', x_bg.shape, x_bg.dtype)
                        logger.debug('x_id.shape: ', x_id.shape, x_id.dtype)
                        logger.debug('x_mo.shape: ', x_mo.shape, x_mo.dtype)
                        # bg_toks, id_toks, mo_toks = first_stage_model.my_encode([ret_img, ret_img_bg, ret_img_id, ret_img_mo], is_training=False)

                        # test extract token here
                        ex_tok = True
                        # if ex_tok:
                        #     inputs = x_img, x_bg, x_id, x_mo
                        #     first_stage_model._generater(inputs, is_training=False)
                        #
                        #     return

                        # Keep number of frames of x and c the same, now 5

                        xbg_toks, xid_toks, xmo_toks = first_stage_model.extract_tokens([x_img, x_bg, x_id, x_mo], is_training=False)
                        cbg_toks, cid_toks, cmo_toks = first_stage_model.extract_tokens([c_img, c_bg, c_id, c_mo], is_training=False)
                        logger.debug('xbg_toks.shape: ', xbg_toks.shape)
                        logger.debug('xid_toks.shape: ', xid_toks.shape)
                        logger.debug('xmo_toks.shape: ', xmo_toks.shape)
                        #
                        xbg_quantized, xid_quantized, xmo_quantized = first_stage_model.get_quantized_by_tokens(xbg_toks, xid_toks, xmo_toks)
                        cbg_quantized, cid_quantized, cmo_quantized = first_stage_model.get_quantized_by_tokens(cbg_toks, cid_toks, cmo_toks)

                        # xbg_quantized, xid_quantized, xmo_quantized = first_stage_model.my_encode([x_img, x_bg, x_id, x_mo], is_training=False)
                        # cbg_quantized, cid_quantized, cmo_quantized = first_stage_model.my_encode([c_img, c_bg, c_id, c_mo], is_training=False)
                        # xbg_quantized, xid_quantized, xmo_quantized = first_stage_model.extract_tokens([x_img, x_bg, x_id, x_mo], is_training=False)
                        # cbg_quantized, cid_quantized, cmo_quantized = first_stage_model.extract_tokens([c_img, c_bg, c_id, c_mo], is_training=False)



                        logger.debug('xbg_quantized.shape: ', xbg_quantized.shape, xbg_quantized.dtype)
                        logger.debug('xid_quantized.shape: ', xid_quantized.shape, xbg_quantized.dtype)
                        logger.debug('xmo_quantized.shape: ', xmo_quantized.shape, xbg_quantized.dtype)

                        # output = first_stage_model._decoder(xbg_quantized, xid_quantized, xmo_quantized)[0]
                        # if output != None:
                        #     print('decode the output: ', output.shape)
                        #     return


                        # t = 1

                        # change 16 to num_frames
                        xbg_quantized = rearrange(xbg_quantized, 'b t c h w -> b c (t h w)')
                        xid_quantized = rearrange(xid_quantized, 'b t c h w -> b c (t h w)')
                        xmo_quantized = rearrange(xmo_quantized, 'b t c h w -> b c (t h w)')

                        cbg_quantized = rearrange(cbg_quantized, 'b t c h w -> b c (t h w)')
                        cid_quantized = rearrange(cid_quantized, 'b t c h w -> b c (t h w)')
                        cmo_quantized = rearrange(cmo_quantized, 'b t c h w -> b c (t h w)')


                        logger.debug('xbg_quantized.shape: ', xbg_quantized.shape)
                        logger.debug('xid_quantized.shape: ', xid_quantized.shape)
                        logger.debug('xmo_quantized.shape: ', xmo_quantized.shape)
                        # z = first_stage_model.module.extract(x).detach()
                        # c = first_stage_model.module.extract(c).detach()

                        # here bg_quantized, id_quantized only have 1 frame in time dimension
                        # but mo_quantized have num_frame many frames in time dimension
                        # option1: repeat bg_quantized, id_quantized to have num_frame many frames in time dimension
                        #          waste memory
                        # option2: concat bg_quantized, id_quantized, mo_quantized in time dimension.
                        concat_dim = -1 # concat in dimension: -1 for latent, 1 for time
                        z = torch.concat([xbg_quantized, xid_quantized, xmo_quantized], dim=concat_dim)
                        c = torch.concat([cbg_quantized, cid_quantized, cmo_quantized], dim=concat_dim)


                        c = c * mask + torch.zeros_like(c).to(c.device) * (1 - mask)

                        logger.debug('show z.shape: ', z.shape)
                        logger.debug('show c.shape: ', c.shape)

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
            logger.debug('show z.shape: ', z.shape)
            logger.debug('show c.shape: ', c.shape)
            (loss, t, output), loss_dict = criterion(z.float(), c.float())

        else:
            if it == 0:
                print("Unconditional model")
            with autocast():
                with torch.no_grad():
                    z = first_stage_model.module.extract(x).detach()

            (loss, t, output), loss_dict = criterion(z.float())

        """
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        """
        loss.backward()
        opt.step()

        losses['diffusion_loss'].update(loss.item(), 1)
        logger.debug('show diffusion model output: ', output.shape)
        out_bg, out_id, out_mo = first_stage_model.convert_latent_to_quantized(output, 3, 4, 5, 256, 16)
        #
        logger.debug('show out_bg: ', out_bg.shape)
        logger.debug('show out_id: ', out_id.shape)
        logger.debug('show out_mo: ', out_mo.shape)

        # out_bg = rearrange(out_bg, 'b c (t h w) -> b t c h w', t=1, h=32, w=32)
        # out_id = rearrange(out_id, 'b c (t h w) -> b t c h w', t=1, h=16, w=16)
        # out_mo = rearrange(out_mo, 'b c (t h w) -> b t c h w', t=16, h=8, w=8)
        #
        # print('after rearrange out_bg: ', out_bg.shape)
        # print('after rearrange out_id: ', out_id.shape)
        # print('after rearrange out_mo: ', out_mo.shape)

        recon = first_stage_model._decoder(out_bg, out_id, out_mo)[0]
        print('show vae recon: ', recon.shape)
        # output = first_stage_model._decoder(xbg_quantized, xid_quantized, xmo_quantized)[0]
        # if output != None:
        #     print('decode the output: ', output.shape)
        #     return

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


def play_with_diffusion_train():



    scaler = GradScaler()

    rank = 0
    ema_model = None
    cond_prob = 0.3

    unet_path = './config/small_unet.yaml'
    ldm_path = './config/ldm_base.yaml'
    unet_config = OmegaConf.load(unet_path).unet_config
    # unet_config = OmegaConf.load(ldm_path).model.params.unet_config
    logger.debug(unet_config)
    logger.debug('cuda: ', torch.torch.cuda.device_count())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug('in this project, use device: ', device)
    video_x = torch.randn(5, 4, 2048).to(device)
    timesteps = torch.tensor([0, 1, 2, 3, 4]).to(device)
    unet = UNetModel(**unet_config).to(device)
    model = DiffusionWrapper(model=unet, conditioning_key=None).to(device)

    moso_opt = yaml.load(open('config/vqvae_raw.yaml', 'r'), Loader=yaml.FullLoader)
    moso_model_opt = moso_opt['model']

    # TODO: load pretrained vqvae from checkpoint
    vqvae = VQVAEModel(moso_model_opt, moso_opt).to(device)

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
    vqvae.eval()
    model.train()


    # TODO: load the tokens from the dataset generated by the vqvae
    # each piece of data is a list of batched bg, id, mo tokens
    # bg tok shape: [batch, 32, 32]
    # id tok shape: [batch, 16, 16]
    # mo tok shape: [(batch * n_frames), 8, 8]
    bg_shape = (3, 32, 32)
    id_shape = (3, 16, 16)
    mo_shape = (48, 8, 8)
    train_loader = [
        [
            torch.randn(bg_shape),
            torch.randn(id_shape),
            torch.randn(mo_shape)
        ]
                ]

    # TODO: finish the training code
    num_epochs = 100
    for epoch in range(num_epochs):

        for it, data in enumerate(train_loader):
            bg_toks, id_toks, mo_toks = data
            bg_toks = bg_toks.to(device)
            id_toks = id_toks.to(device)
            mo_toks = mo_toks.to(device)

            with torch.no_grad():
                bg_quantized, id_quantized, mo_quantized = vqvae.get_quantized_by_tokens(bg_toks, id_toks, mo_toks)




if __name__ == '__main__':
    # play_with_MOSO_VAE()

    # play_with_PVDM_Diffuser()

    # play_with_all_process()
    # import glob
    preprocess()
    # data_list = glob.glob('./data/*.npy')
    # for data_path in data_list:
    #     dt = np.load(data_path, allow_pickle=True).item()
    #     for key in dt:
    #         print(key+':', torch.from_numpy(dt[key]).shape)
    #     print('---------------------')
