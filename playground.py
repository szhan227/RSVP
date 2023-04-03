import cv2
import torch
import numpy as np
import yaml
from torchvision import transforms
from PIL import Image
from model.vq_vae.VQVAE import VQVAEModel


video_cap = cv2.VideoCapture('testvideo1.avi')


it = 0
xt = []
while True:
    success, frame = video_cap.read()
    if not success:
        break
    xt.append(frame)
xt = np.array(xt)
xt = torch.tensor(xt) / 255.0
xt = xt.permute(0, 3, 1, 2)
print('xt.shape:', xt.shape, xt.dtype)

num_frames = 5
start = 0

transform = transforms.CenterCrop(64)

lower_bound = 0.2
upper_bound = 0.8
pre_img = transform(xt[start])
nxt_img = transform(xt[start + 1])

print('pre_img.shape:', pre_img.shape)
print('nxt_img.shape:', nxt_img.shape)
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

print('ret_img.shape:', ret_img.shape)
print('ret_img_bg.shape:', ret_img_bg.shape)
print('ret_img_id.shape:', ret_img_id.shape)
print('ret_img_mo.shape:', ret_img_mo.shape)

validate_inputs = [ret_img, ret_img, ret_img, ret_img_mo]

opt = yaml.load(open('config/vqvae.yaml', 'r'), Loader=yaml.FullLoader)
model_opt = opt['model']
model = VQVAEModel(model_opt, opt)

bg_toks, id_toks, mo_toks = model.my_encode([ret_img, ret_img, ret_img, ret_img_mo], is_training=False)
print('bg_toks.shape:', bg_toks.shape)
print('id_toks.shape:', id_toks.shape)
print('mo_toks.shape:', mo_toks.shape)
outputs = model._decoder(bg_toks, id_toks, mo_toks)[0]
print('output.shape:', outputs.shape)

# outputs = model(validate_inputs, is_training=False, writer=None)
# dict_keys(['loss', 'x_rec', 'quantize_bg', 'quantize_id', 'quantize_mo', 'ssim_metric', 'rec_loss', 'lpips_loss', 'record_logs', 'optimizer_idx'])
# print('outputs.shape:', outputs['x_rec'].shape)

