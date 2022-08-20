import torch 
import torch.nn as nn
from torch.optim import Adam
from models import UNet
from PIL import Image 
import numpy as np 
from Dataset import Dataset_cifar, Dataset_celeba
from reverse_diffusion_process import reverse_diffusion
from torch.utils.data import DataLoader
import argparse 
import os 


def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def train(batchsize, epoch, data_type, path, beta_start, beta_end, timesteps, device):
    model = UNet(ch=128, droprate=0.5, groups=8).to(device)
    model.train()
    if data_type == "cifar10":
        dataset = Dataset_cifar(path=path)
        h, w = 32, 32
    elif data_type == "celeba":
        dataset = Dataset_celeba(path=path)
        h, w = 192, 160
    opt = Adam(model.parameters(), lr=2e-4)
    dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=8, drop_last=True)

    betas = linear_beta_schedule(beta_start, beta_end, timesteps).to(device)
    alphas = 1 - betas 
    alphas_cumprod = torch.cumprod(alphas, dim=0) # return [a0, a0*a1, a0*a1*a2, ..., a0*a1*...*at]
    smoothl1 = nn.SmoothL1Loss()
    if not os.path.exists("./results"):
        os.mkdir("./results")
    if not os.path.exists("./saved_models"):
        os.mkdir("./saved_models")
    f = open("loss.txt", "w")
    for epoch in range(epoch):
        for itr, batch_x0 in enumerate(dataloader):
            batch_x0 = batch_x0.to(device)
            opt.zero_grad()
            t = torch.randint(0, timesteps, [batchsize]).to(device)
            alphas_bar = alphas_cumprod.gather(-1, t).view(batchsize, 1, 1, 1)
            eps = torch.randn_like(batch_x0)        
            x_t = torch.sqrt(alphas_bar) * batch_x0 + torch.sqrt(1 - alphas_bar) * eps
            pred_noise = model(x_t, t)
            loss = smoothl1(pred_noise, eps)
            loss.backward()
            opt.step()
            if itr % 100 == 0:
                print(f"Epoch: {epoch}, Iteration: {itr}, Loss: {loss.item()}")
                f.write(f"Epoch: {epoch}, Iteration: {itr}, Loss: {loss.item()}\n")
                f.flush()
            if epoch % 10 == 0 and itr == 0:
                torch.save(model.state_dict(), f"./saved_models/model{str(epoch)}.pth") 
            if epoch % 1 == 0 and itr == 0:
                print("reverse diffusion: ")
                x_t = torch.randn(1, 3, h, w).to(device)
                result = reverse_diffusion(model, x_t, beta_start, beta_end, timesteps, device)
                result = np.concatenate(result, axis=1)
                result = np.clip(result, 0, 255)
                Image.fromarray(np.uint8(result)).save(f"./results/{str(epoch)}_{str(itr)}.png") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/Data_2/gmt/Dataset/img_align_celeba/")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--data_type", type=str, default="celeba", help="cifar10 | celeba")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args.batchsize, args.epoch, args.data_type, args.path, args.beta_start, args.beta_end, args.timesteps, args.device)
    # CUDA_VISIBLE_DEVICES=1 python train.py --data_type celeba --path /Data_2/gmt/Dataset/img_align_celeba/ --batchsize 16