import os
import json

import torch
from torchvision.io import read_video

from tools.trainer import mosoDDPM
from tools.dataloader import get_loaders
from tools.scheduler import LambdaLinearScheduler
from models.autoencoder.autoencoder_vit import ViTAutoencoder
from models.ddpm.unet import UNetModel, DiffusionWrapper, MosoModel
from losses.ddpm import DDPM

from src.model import get_model

import copy
from utils import file_name, Logger, download
from tools.data_utils import *
from tqdm import tqdm
import torchvision
#----------------------------------------------------------------------------

_num_moments    = 3             # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype   = torch.float32 # Data type to use for initial per-tensor reduction.
_counter_dtype  = torch.float64 # Data type to use for the internal counters.
_rank           = 0             # Rank of the current process.
_sync_device    = None          # Device to use for multiprocess communication. None = single-process.
_sync_called    = False         # Has _sync() been called yet?
_counters       = dict()        # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative     = dict()        # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor

#----------------------------------------------------------------------------

def init_multiprocessing(rank, sync_device):
    r"""Initializes `torch_utils.training_stats` for collecting statistics
    across multiple processes.
    This function must be called after
    `torch.distributed.init_process_group()` and before `Collector.update()`.
    The call is not necessary if multi-process collection is not needed.
    Args:
        rank:           Rank of the current process.
        sync_device:    PyTorch device to use for inter-process
                        communication, or None to disable multi-process
                        collection. Typically `torch.device('cuda', rank)`.
    """
    global _rank, _sync_device
    assert not _sync_called
    _rank = rank
    _sync_device = sync_device

#----------------------------------------------------------------------------

def validation(rank, args):
    device = torch.device('cuda', rank)
    
    temp_dir = './'
    if args.n_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.n_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.n_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.n_gpus > 1 else None
    init_multiprocessing(rank=rank, sync_device=sync_device)
    

    """ ROOT DIRECTORY """
    '''
    if rank == 0:
        fn = file_name(args)
        logger = Logger(fn)
        logger.log(args)
        logger.log(f'Log path: {logger.logdir}')
        rootdir = logger.logdir
    else:
        logger = None
   

    if logger is None:
        log_ = print
    else:
        log_ = logger.log
    '''
    log_ = print

    """ Get Image """
    if rank == 0:
        log_(f"Loading dataset {args.data} with resolution {args.res}")
    root_dir = os.path.join("..", "MOSO", "Token", "UCF101", "img256_16frames", "valid")
    data = sorted(os.listdir(root_dir))
    save_dir = os.path.join('results', "validation")
    os.makedirs(save_dir, exist_ok=True)

    """ Get Model """
    if rank == 0:
        log_(f"Generating model")    

    torch.cuda.set_device(rank)
    '''
    first_stage_model = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)

    if rank == 0:
        first_stage_model_ckpt = torch.load(args.first_model)
        first_stage_model.load_state_dict(first_stage_model_ckpt)
    '''
    first_stage_model, _ = get_model(args.opt)
    first_stage_model = first_stage_model.to(device)

    '''
    unet = UNetModel(**args.unetconfig)
    model = DiffusionWrapper(unet).to(device)
    '''

    model = MosoModel(**args.unetconfig).to(device)

    if rank == 0:
        model.load_state_dict(torch.load(args.final_ckpt))

    ema_model = None
    if args.n_gpus > 1:
        first_stage_model = torch.nn.parallel.DistributedDataParallel(
                                                          first_stage_model,
                                                          device_ids=[device],
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=False)

        model             = torch.nn.parallel.DistributedDataParallel(
                                                          model, 
                                                          device_ids=[device], 
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=False)

    diffusion_model = DDPM(model, channels=args.unetconfig.in_channels,
                            image_size=args.unetconfig.image_size,
                            sampling_timesteps=100,
                            w=0.
                            ).to(device)

    if args.n_gpus > 1:
        diffusion_model         = torch.nn.parallel.DistributedDataParallel(
                                                          diffusion_model,
                                                          device_ids=[device],
                                                          broadcast_buffers=False,
                                                          find_unused_parameters=False)

  

    log_(f"Start validating...")
    scenes = {}
    pbar = tqdm(total=len(data))
    batch_size = 4
    for scene in data:
        os.makedirs(os.path.join(root_dir, scene), exist_ok=True)
        for idxx, subdir in enumerate(os.listdir(os.path.join(root_dir, scene))):
            if "_g01_c01.avi" in subdir:
                instance = str(subdir)
                break
        
        os.makedirs(os.path.join(save_dir, scene, instance), exist_ok=True)
        basedir = os.path.join(save_dir, scene, instance)
        
        start_frame = 16
        start_nid = 24
        bgs = None
        ids = None
        mos = None
        while True: 
            ctoken = np.load(os.path.join(root_dir, scene, instance, "start_%03d_rank0.npy" % start_frame),
               allow_pickle=True)
            bg_tokens = ctoken.item()['bg_tokens'] # H, W
            id_tokens = ctoken.item()['id_tokens'] # H, W
            mo_tokens = ctoken.item()['mo_tokens'] # T, H, W

            bg_tokens = bg_tokens.flatten().astype(np.int64)
            id_tokens = id_tokens.flatten().astype(np.int64)
            mo_tokens = mo_tokens.flatten().astype(np.int64)
            
            if bgs is None:
                bgs = torch.from_numpy(bg_tokens).unsqueeze(0)
                ids = torch.from_numpy(id_tokens).unsqueeze(0)
                mos = torch.from_numpy(mo_tokens).unsqueeze(0)
            elif len(bgs) < batch_size:
                bgs = torch.cat([bgs,
                    torch.from_numpy(bg_tokens).unsqueeze(0)
                ], dim=0)
                ids = torch.cat([ids,
                    torch.from_numpy(id_tokens).unsqueeze(0)
                ], dim=0)
                mos = torch.cat([mos,
                    torch.from_numpy(mo_tokens).unsqueeze(0)
                ], dim=0)
            if len(bgs) == batch_size:
                with torch.no_grad():
                    bgs = rearrange(bgs, 'B (T H W) -> B T H W', T=1, H=32, W=32).to(device)
                    ids = rearrange(ids, 'B (T H W) -> B T H W', T=1, H=16, W=16).to(device)
                    mos = rearrange(mos, 'B (T H W) -> B T H W', H=8, W=8).to(device)
                    vq_bg = first_stage_model._vq_ema.quantize_code(bgs)
                    vq_id = first_stage_model._vq_ema.quantize_code(ids)
                    vq_mo = first_stage_model._vq_ema.quantize_code(mos)
                    quantize_bg = first_stage_model._suf_vq_bg(vq_bg)
                    quantize_id = first_stage_model._suf_vq_id(vq_id)
                    quantize_mo = first_stage_model._suf_vq_mo(vq_mo)
                    xbg = rearrange(quantize_bg, "b c h w -> b (h w) c").detach()
                    xid = rearrange(quantize_id, "b c h w -> b (h w) c").detach()
                    xmo = rearrange(quantize_mo, "(b t) c h w -> b c t h w", b=xbg.shape[0]).detach()
                    context = torch.cat([xbg, xid], dim=1)
                    mask = torch.zeros(xmo.shape[0], xmo.shape[2], xmo.shape[3], xmo.shape[4]).long()
                    mask[:, :xmo.shape[2]//2, ...] = 1
                    z = diffusion_model.ddim_sample_moso(xmo.clone().to(device), mask.to(device), context.to(device))
                    bgs = (rearrange(xbg, "b (t h w) c -> b t c h w", t=1, h=32, w=32))
                    ids = (rearrange(xid, "b (t h w) c -> b t c h w", t=1, h=16, w=16))
            
                    x_rec, _, _ = first_stage_model._decoder(bgs, ids, z.permute(0, 2, 1, 3, 4))

                    for nid in range(x_rec.shape[0]):
                        os.makedirs(os.path.join(basedir, str(start_nid)), exist_ok=True)
                        for tid in range(x_rec.shape[1]):
                            torchvision.utils.save_image(x_rec[nid, tid].cpu().float(), os.path.join(basedir, str(start_nid), "%02d.png" % tid))
                        start_nid += 16
                    
                    bgs = None
                    ids = None
                    mos = None

            start_frame += 16
            if not os.path.exists(os.path.join(root_dir, scene, instance, "start_%03d_rank0.npy" % start_frame)):
                break
        if bgs is not None:
            with torch.no_grad():
                bgs = rearrange(bgs, 'B (T H W) -> B T H W', T=1, H=32, W=32).to(device)
                ids = rearrange(ids, 'B (T H W) -> B T H W', T=1, H=16, W=16).to(device)
                mos = rearrange(mos, 'B (T H W) -> B T H W', H=8, W=8).to(device)
                vq_bg = first_stage_model._vq_ema.quantize_code(bgs)
                vq_id = first_stage_model._vq_ema.quantize_code(ids)
                vq_mo = first_stage_model._vq_ema.quantize_code(mos)
                quantize_bg = first_stage_model._suf_vq_bg(vq_bg)
                quantize_id = first_stage_model._suf_vq_id(vq_id)
                quantize_mo = first_stage_model._suf_vq_mo(vq_mo)
                xbg = rearrange(quantize_bg, "b c h w -> b (h w) c").detach()
                xid = rearrange(quantize_id, "b c h w -> b (h w) c").detach()
                xmo = rearrange(quantize_mo, "(b t) c h w -> b c t h w", b=xbg.shape[0]).detach()
                context = torch.cat([xbg, xid], dim=1)
                mask = torch.zeros(xmo.shape[0], xmo.shape[2], xmo.shape[3], xmo.shape[4]).long()
                mask[:, :xmo.shape[2]//2, ...] = 1
                z = diffusion_model.ddim_sample_moso(xmo.clone().to(device), mask.to(device), context.to(device))
                bgs = (rearrange(xbg, "b (t h w) c -> b t c h w", t=1, h=32, w=32))
                ids = (rearrange(xid, "b (t h w) c -> b t c h w", t=1, h=16, w=16))
        
                x_rec, _, _ = first_stage_model._decoder(bgs, ids, z.permute(0, 2, 1, 3, 4))

                for nid in range(x_rec.shape[0]):
                    os.makedirs(os.path.join(basedir, str(start_nid)), exist_ok=True)
                    for tid in range(x_rec.shape[1]):
                        torchvision.utils.save_image(x_rec[nid, tid].cpu().float(), os.path.join(basedir, str(start_nid), "%02d.png" % tid))
                    start_nid += 16
                bgs = None
                ids = None
                mos = None

        
        pbar.update(1)
    prefix = np.random.randint(len(video)-self.nframes+1)
    video = video[prefix:prefix+self.nframes].float().permute(3,0,1,2)
    assert False, data

