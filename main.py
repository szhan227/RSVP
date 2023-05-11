import sys; sys.path.extend(['.'])

import os
import argparse

import torch
from omegaconf import OmegaConf

from exps.moso import moso_diffusion
from exps.diffusion import diffusion
from exps.first_stage import first_stage
from exps.validation import validation

from utils import set_random_seed



parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, help='experiment name to run')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--id', type=str, default='main', help='experiment identifier')

""" Args about Data """
parser.add_argument('--data', type=str, default='UCF101')
parser.add_argument('--ds', type=int, default=4)

""" Args about Model """
parser.add_argument('--pretrain_config', type=str, default='configs/autoencoder/autoencoder_kl_f4d6_res128.yaml')
parser.add_argument('--diffusion_config', type=str, default='configs/latent-diffusion/ucf101-ldm-kl-3_res128.yaml')

# for GAN resume
parser.add_argument('--first_stage_folder', type=str, default='', help='the folder of first stage experiment before GAN')

# for diffusion model path specification
parser.add_argument('--first_model', type=str, default='', help='the path of pretrained model')
parser.add_argument('--scale_lr', action='store_true')

# for MOSO
parser.add_argument('--moso_config', type=str, help='the path to MOSO VQVAE config file')
parser.add_argument('--moso_checkpoint', type=str, help='the path to MOSO checkpoint')

parser.add_argument("--final_ckpt", type=str, default="")

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

def get_opt_from_yaml(path):
    assert os.path.exists(path), f"{path} must exists!"
    import yaml
    with open(path, 'r') as f:
        opt = yaml.load(f, Loader=yaml.FullLoader)
    return dict2obj(opt)

def main():
    """ Additional args ends here. """
    args = parser.parse_args()
    """ FIX THE RANDOMNESS """
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    args.n_gpus = torch.cuda.device_count()

    # init and save configs
    
    """ RUN THE EXP """
    if args.exp == 'moso_ddpm':
        
        #assert False, args.moso_dir
        config = OmegaConf.load(args.diffusion_config)
        
        args.opt = get_opt_from_yaml(args.moso_config)
        args.opt["checkpoint_path"] = args.moso_checkpoint
        ''' 
        # no need to load old first stage model
        first_stage_config = OmegaConf.load(args.pretrain_config)
        # instead, read from opt
        args.res        = first_stage_config.model.params.ddconfig.resolution
        args.timesteps  = first_stage_config.model.params.ddconfig.timesteps
        args.skip       = first_stage_config.model.params.ddconfig.skip
        args.ddconfig   = first_stage_config.model.params.ddconfig
        args.embed_dim  = first_stage_config.model.params.embed_dim
        '''

        args.res = args.opt["dataset"]["img_size"]
        '''
        # other arguments not set anymore
        args.timesteps  = config.model.params.timesteps
        args.skip       = config.model.params.ddconfig.skip
        args.ddconfig   = config.model.params.ddconfig
        args.embed_dim  = config.model.params.embed_dim
        '''

        args.unetconfig = config.model.params.unet_config
        args.lr         = config.model.base_learning_rate
        args.scheduler  = config.model.params.scheduler_config        
        args.ddpmconfig = config.model.params
        args.cond_model = config.model.cond_model

        if args.n_gpus == 1:
            moso_diffusion(rank=0, args=args)
        else:
            torch.multiprocessing.spawn(fn=moso_diffusion, args=(args, ), nprocs=args.n_gpus)

    elif args.exp == 'validation':
        
        config = OmegaConf.load(args.diffusion_config)
        
        args.opt = get_opt_from_yaml(args.moso_config)
        args.opt["checkpoint_path"] = args.moso_checkpoint
        ''' 
        # no need to load old first stage model
        first_stage_config = OmegaConf.load(args.pretrain_config)
        # instead, read from opt
        args.res        = first_stage_config.model.params.ddconfig.resolution
        args.timesteps  = first_stage_config.model.params.ddconfig.timesteps
        args.skip       = first_stage_config.model.params.ddconfig.skip
        args.ddconfig   = first_stage_config.model.params.ddconfig
        args.embed_dim  = first_stage_config.model.params.embed_dim
        '''

        args.res = args.opt["dataset"]["img_size"]
        '''
        # other arguments not set anymore
        args.timesteps  = config.model.params.timesteps
        args.skip       = config.model.params.ddconfig.skip
        args.ddconfig   = config.model.params.ddconfig
        args.embed_dim  = config.model.params.embed_dim
        '''

        args.unetconfig = config.model.params.unet_config
        args.lr         = config.model.base_learning_rate
        args.scheduler  = config.model.params.scheduler_config        
        args.ddpmconfig = config.model.params
        args.cond_model = config.model.cond_model
        assert args.final_ckpt != "", "must have a final ckpt for validation"
        if args.n_gpus == 1:
            validation(rank=0, args=args)
        else:
            torch.multiprocessing.spawn(fn=validation, args=(args, ), nprocs=args.n_gpus)


    elif args.exp == 'ddpm':
        config = OmegaConf.load(args.diffusion_config)
        first_stage_config = OmegaConf.load(args.pretrain_config)

        args.unetconfig = config.model.params.unet_config
        args.lr         = config.model.base_learning_rate
        args.scheduler  = config.model.params.scheduler_config
        args.res        = first_stage_config.model.params.ddconfig.resolution
        args.timesteps  = first_stage_config.model.params.ddconfig.timesteps
        args.skip       = first_stage_config.model.params.ddconfig.skip
        args.ddconfig   = first_stage_config.model.params.ddconfig
        args.embed_dim  = first_stage_config.model.params.embed_dim
        args.ddpmconfig = config.model.params
        args.cond_model = config.model.cond_model

        if args.n_gpus == 1:
            diffusion(rank=0, args=args)
        else:
            torch.multiprocessing.spawn(fn=diffusion, args=(args, ), nprocs=args.n_gpus)

    elif args.exp == 'first_stage':
        config = OmegaConf.load(args.pretrain_config)
        args.ddconfig   = config.model.params.ddconfig
        args.embed_dim  = config.model.params.embed_dim
        args.lossconfig = config.model.params.lossconfig
        args.lr         = config.model.base_learning_rate
        args.res        = config.model.params.ddconfig.resolution
        args.timesteps  = config.model.params.ddconfig.timesteps
        args.skip       = config.model.params.ddconfig.skip
        args.resume     = config.model.resume
        args.amp        = config.model.amp
        if args.n_gpus == 1:
            first_stage(rank=0, args=args)
        else:
            torch.multiprocessing.spawn(fn=first_stage, args=(args, ), nprocs=args.n_gpus)

    else:
        raise ValueError("Unknown experiment.")

if __name__ == '__main__':
    main()