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
python train.py --data_type cifar10 --path ./cifar-10-matlab/ --batchsize 128 --epoch 500
```
## Diffusion process
```
python diffusion_process.py --timesteps 1000 --t 100,200,300,400,500,600,700,800,999 --img_path resources/face.png
```

## Reverse diffusion process
```
python reverse_diffusion_process.py --data_type cifar10 --timesteps 1000 --weights ./saved_models/model500.pth
```
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/intro_diff.png)



### Diffusion results
-------------

![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/diffusion.png)

## Requirements
1. python3
2. torch
3. torchvision
4. pillow
5. scipy
6. numpy
7. tqdm

## 

