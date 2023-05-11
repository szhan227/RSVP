import argparse
import os
import torch
from tqdm import tqdm
import torchvision

from fvd.fvd import get_fvd_logits, frechet_distance
from fvd.download import load_i3d_pretrained


import sys
sys.path.append("../tools")

from data_utils import *
from torchvision.io import read_video

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--reversed", action="store_true")
    # reversed: default to be RSVP: only read 08-15
    # if set to True: only 0-7
    args = parser.parse_args()
    scenes = sorted(os.listdir(args.dir))
    scenes = [os.path.join(scene, os.listdir(os.path.join(args.dir, scene))[0]) for scene in scenes]
    
    real_embeddings = []
    fake_embeddings = []
    device = torch.device("cuda")
    i3d = load_i3d_pretrained(device)
    losses = dict()
    losses['psnr'] = AverageMeter()
    
    pbar = tqdm(total=len(scenes))
    for scene in scenes:
        video = read_video(os.path.join("../data/UCF-101", scene))[0]
        video = resize_crop(video.float().permute(3,0,1,2), 256) #0-255.
        instances = sorted(["%03d" % int(instance) for instance in os.listdir(os.path.join(args.dir, scene))])
        ext = len(instances) % 4
        if ext > 0:
            instances = instances[:-ext]
        instances = [str(int(instance)) for instance in instances]
        reals = []
        fakes = []
        for instance in instances:
            if args.reversed:
                real = video[int(instance)-8:int(instance)+8]
            else:
                real = video[int(instance):int(instance)+16]
            if len(real) < 16:
                print(scene, instance)
                continue
            fake = torch.stack([torchvision.io.read_image(os.path.join(args.dir, scene, instance, "%02d.png" % idxx)) for idxx in range(16)], dim=0)
            reals.append(real)
            fakes.append(fake)
        if not reals:
            print(scene)
            continue
        reals = torch.stack(reals, dim=0).permute(0, 1, 3, 4, 2)
        fakes = torch.stack(fakes, dim=0).permute(0, 1, 3, 4, 2)
        # B, T, H, W, C
        with torch.no_grad():
            real_embeddings.append(get_fvd_logits(reals.type(torch.uint8).numpy(), i3d=i3d, device=device))
            fake_embeddings.append(get_fvd_logits(fakes.type(torch.uint8).numpy(), i3d=i3d, device=device))
        reals = reals / 127.5 - 1.
        fakes = fakes / 127.5 - 1.
        reals = reals.contiguous().view(reals.shape[0], -1)
        fakes = fakes.contiguous().view(fakes.shape[0], -1)
        mse = ((reals * 0.5 - fakes * 0.5) ** 2).mean(dim=-1)
        psnr = (-10 * torch.log10(mse)).mean()

        losses['psnr'].update(psnr.item(), reals.shape[0])

        pbar.update(1)
    real_embeddings = torch.cat(real_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)
    
    fvd = frechet_distance(fake_embeddings.clone().detach(), real_embeddings.clone().detach())
    print(fvd.item())
    print(losses['psnr'].average)