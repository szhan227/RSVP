## RSVP: Revealing Semantics for Latent Diffusion based Video Prediction

CSCI 2952N Advanced Topics in Deep Learning

Yiqing Liang, 
Lingyu Ma, 
Siyang Zhang, 
Pinyuan Feng.  

Brown University

<img src="./README_images/teaser.png" alt="teaser" style="zoom:50%;" />




### Abstract
Diffusion models have recently shown their promising potential in the video prediction domain, as they can capture the temporal dynamics of videos and synthesize video frames with good quality. However, challenges still remain in the computational and memory requirements of these models. Recent work utilized latent diffusion to mitigate the issue by projecting original video into a lower-dimensional latent space and training diffusion models on projected space, but no semantic prior is used during latent encoding and decoding, which compromises the consistency of predicted video frames. In this paper, we propose a two-stage framework, RSVP, that reveals meaningful semantic representations for latent diffusion to achieve inexpensive and efficient video prediction. Specifically, in the first stage, we learn to decompose videos into scene, object, and motion components; in the second stage, we train a latent diffusion model over the semantic latent space for video prediction. RSVP can fit into a single NVIDIA A40 GPU, and experimental results on the UCF-101 video dataset demonstrate the effectiveness of integrating semantic knowledge into the diffusion-based video prediction framework. We believe RSVP can serve as a promising direction to inspire and encourage further advancements in the field of video diffusion.

### Environment setup
```bash
conda create -n pvdm python=3.8 -y
conda activate pvdm
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install natsort tqdm gdown omegaconf einops lpips pyspng tensorboard imageio av moviepy
```

### Dataset 

#### Dataset download
Currently, we provide experiments for the following the dataset: [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php). The dataset should be placed in `/data` with the following structures below; you may change the data location directory in `tools/dataloadet.py` by adjusting the variable `data_location`.

#### UCF-101
```
UCF-101
|-- class1
    |-- video1.avi
    |-- video2.avi
    |-- ...
|-- class2
    |-- video1.avi
    |-- video2.avi
    |-- ...
    |-- ...
```

### Training

#### Autoencoder

First, execute the following script:
```bash
 python main.py 
 --exp first_stage \
 --id [EXP_NAME] \
 --pretrain_config configs/autoencoder/base.yaml \
 --data [DATASET_NAME] \
 --batch_size [BATCH_SIZE]
```
Then the script will automatically create the folder in `./results` to save logs and checkpoints.

If the loss converges, then execute the following script:
```bash
 python main.py 
 --exp first_stage \
 --id [EXP_NAME]_gan \
 --pretrain_config configs/autoencoder/base_gan.yaml \
 --data [DATASET] \
 --batch_size [BATCH_SIZE] \
 --first_stage_folder [DIRECTOTY OF PREVIOUS EXP]
```

Here, `[EXP_NAME]` is an experiment name you want to specifiy (string), `[DATASET]` is either `UCF101` or `SKY`, and `[DIRECTOTY OF PREVIOUS EXP]` is a directory for the previous script. For instance, the entire scripts for training the model on UCF-101 becomes: 
```bash
 python main.py \
 --exp first_stage \
 --id main \
 --pretrain_config configs/autoencoder/base.yaml \
 --data UCF101 \
 --batch_size 8

 python main.py \
 --exp first_stage \ 
 --id main_gan \
 --pretrain_config configs/autoencoder/base_gan.yaml \
 --data UCF101 \
 --batch_size 8 \
 --first_stage_folder 'results/first_stage_main_UCF101_42/'
```

You may change the model configs via modifying `configs/autoencoder`. Moreover, one needs early-stopping to further train the model with the GAN loss (typically 8k-14k iterations with a batch size of 8).

#### Diffusion model

```bash
 python main.py \
 --exp ddpm \
 --id [EXP_NAME] \
 --pretrain_config configs/latent-diffusion/base.yaml \
 --data [DATASET] \
 --first_model [AUTOENCODER DIRECTORY] 
 --diffusion_config configs/latent-diffusion/base.yaml \
 --batch_size [BATCH_SIZE]
```

Here, `[EXP_NAME]` is an experiment name you want to specifiy (string), `[DATASET]` is either `UCF101` or `SKY`, and `[DIRECTOTY OF PREVIOUS EXP]` is a directory of the autoencoder to be used. For instance, the entire scripts for training the model on UCF-101 becomes: 
```bash
 python main.py \
 --exp ddpm \
 --id main \
 --pretrain_config configs/latent-diffusion/base.yaml \
 --data UCF101 \
 --first_model 'results/first_stage_main_gan_UCF101_42/model_last.pth'  
 --diffusion_config configs/latent-diffusion/base.yaml \
 --batch_size 48
```

### Evaluation
We will provide checkpoints with the evaluation scripts as soon as possible, once the refactoring is done.

### Citation
```bibtex
@inproceedings{yu2023video,
  title={Video Probabilistic Diffusion Models in Projected Latent Space},
  author={Yu, Sihyun and Sohn, Kihyuk and Kim, Subin and Shin, Jinwoo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

### Note

It's possible that this code may not accurately replicate the results outlined in the paper due to potential human errors during the preparation and cleaning of the code for release. If you encounter any difficulties in reproducing our findings, please don't hesitate to inform us. Additionally, we'll make an effort to carry out sanity-check experiments in the near future.

### Reference
This code is mainly built upon [SiMT](https://github.com/jihoontack/simt), [latent-diffusion](https://github.com/CompVis/latent-diffusionn), and [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) repositories.\
We also used the code from following repositories: [StyleGAN-V](https://github.com/universome/stylegan-v), [VideoGPT](https://github.com/wilson1yan/VideoGPT), and [MDGAN](https://github.com/weixiong-ur/mdgan).

