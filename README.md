# DDPM_pytorch
Denoising Diffusion Probabilistic Models (DDPM)

## Introduction
--------------

Unofficial DDPM implementation by Mingtao Guo

Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)

## Training
```
cd DDPM_pytorch
wget https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz
tar -zxvf cifar-10-matlab.tar.gz
python train.py --data_type cifar10 --path ./cifar-10-batches-mat/ --batchsize 128 --epoch 500
```
## Diffusion process
```
python diffusion_process.py --timesteps 1000 --t 100,200,300,400,500,600,700,800,999 --img_path resources/face.png
```
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/diffusion1000.png)
## Reverse diffusion process
```
python reverse_diffusion_process.py --data_type cifar10 --timesteps 1000 --weights ./saved_models/model500.pth
```
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/rev_diff.png)

## Read code
### difussion process
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/intro_diff.png)
```python
    x_0 = torch.tensor(x_0).to(device)
    betas = linear_beta_schedule(beta_start, beta_end, timesteps)
    alphas = 1 - betas 
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device) # [a0, a0*a1, a0*a1*a2,...,a0*a1*a2*...*at]
    alpha_bar = alphas_cumprod.gather(-1, torch.tensor(t)).  # look up the element in alphas_cumprod
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps
```

## Requirements
1. python3
2. torch
3. torchvision
4. pillow
5. scipy
6. numpy
7. tqdm

## 

