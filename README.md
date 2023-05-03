# RSVP
Revealing Semantics for Latent Diffusion based Video Prediction



Important Notice:

1. Unconditional DDPM and Conditional DDPM DOES NOT share same parameters and structures, because the in_channels for Conditional DDPM is twice as much as Unconditional Ones.
2. If you do not want to see debug message when running, set message level to at least "info".
3. For Conditional DDPM Training, first prepare x_tokens and c_tokens as .npy files, also load the pretrained VQVAE model. See details in train.py
4. After training of DDPM, the trained model will be saved in the folder chkpt.
5. For validation, please do the preprocess of both batched input and conditional video decomposition. See details in validate.py