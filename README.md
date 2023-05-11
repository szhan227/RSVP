# RSVP-S2

Revealing Semantics for Latent Diffusion based Video Prediction



Before starting to train the DDPM model, we need to prepare vector quantized latent data by MOSO-VQVAE.



In the RSVP-S(1) introduced in the report, both up- and down-sampling blocks in the U-Net were based on 2D convolution, assuming that, in VQ-VAE, both the scene and object are stationary and, therefore, have no temporal dimension. However, 2D convolution and reshaping alter the temporal structure of the motion vector, and therefore, lead to the failure that only noisy output of the resampling is generated shown in the report.



Considering the weakness of RSVP-S1, I re-design the RSVP-S2 as the new U-Net model, it has two major differences and improvements:

1. The use of 3D convolution in up- and down-sampling blocks could probably remedy this issue. Additionally, expanding both the scene and object vectors in the temporal dimension would not enable the 3D convolution filter boxes to slide on the temporal dimension, as these two vectors have a size of only 1 in the temporal dimension. Therefore, the extra computation would be applied only to the motion vector, whose latent space has much smaller dimensions than those of the scene and object.
2. Another possible improvement is that RSVP-S2 performs diffusion and defines the learning objective solely on the motion vectors. Assuming that VQ-VAE effectively decomposes the scene, object, and motion, the motion vector contains all the dynamic information, meaning that the model would only need to learn diffusion on the motion vector to generate new video frames from static scenes, static objects, and newly predicted dynamic motion. 



After failure of RSVP-S1, we change the focus of exploiting GPU on training RSVP-M(AE), experiments and ablation studies, due to lack of time and device.

RSVP-S2 is theoretically better than RSVP-S1 because the former has the structure that fits video task better. 

The first difference between RSVP-S2 and RSVP-M(AE) is that the former does not require a random masking as condition, and only predict next frames. This difference can be an analogy to the difference between "language modeling" and "masked language modeling". In fact, when we trained RSVP-M(AE) the mask was only applied on the second half latent vectors on temporal dimension, basically did the same thing as future video frame prediction, other than video frame interpolation or inpainting. RSVP-S2 also has a self-attention mechanism in each up- or down-sampling block of the U-Net, and is supposed to make connection among scene, object, and motion vectors.

We did not train the RSVP-S2 on the GPU due to lack of time and resource. If future device support is available, we could check the feasibility and performance of RSVP-S2.



Training:

Run train.py with corresponding run configuration, see details in train.py. During training, some checkpoints will be saved into the corresponding directory.



Validation:

Run validate.py, the program will load the DDPM from checkpoint and sample an image, by given image condition, with a random noise motion vector for timestep T.

