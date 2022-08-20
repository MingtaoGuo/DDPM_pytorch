import torch 
import torch.nn.functional as F 
from models import UNet
import numpy as np 
from PIL import Image 
import argparse
from tqdm import tqdm 

def map_to_0_255(x):
    min_x, max_x = x.min(), x.max()
    k = 255 / (max_x - min_x)
    y = -k * min_x + k * x
    return y 

def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def reverse_diffusion(model, x_t, beta_start, beta_end, timesteps, device):
    betas = linear_beta_schedule(beta_start, beta_end, timesteps).to(device)
    alphas = 1 - betas                                                     # DDPM paper section 2, equation (4)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], pad=(1, 0), value=1.) # More convenient calculate alpha_t_1 
    res = []
    for t in tqdm(range(timesteps - 1, -1, -1)):
        t = torch.tensor([t]).to(device)
        alphas_bar_t_1 = alphas_cumprod_prev.gather(-1, t).view(-1, 1, 1, 1)
        alphas_bar_t = alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        beta_t = betas.gather(-1, t).view(-1, 1, 1, 1)
        alpha_t = 1 - beta_t           
        sigma = torch.sqrt((1 - alphas_bar_t_1) / (1 - alphas_bar_t) * beta_t) # DDPM paper section 3.2
        z = torch.randn_like(x_t).to(device)
        with torch.no_grad():
            mu_pred = (1 - alpha_t) / torch.sqrt(1 - alphas_bar_t) * model(x_t, t)
        if t.cpu().numpy() > 0:
            x_t_1 = 1 / torch.sqrt(alpha_t) * (x_t - mu_pred) + sigma * z      # DDPM paper Algorithm 2 line 4
        else:
            x_t_1 = 1 / torch.sqrt(alpha_t) * (x_t - mu_pred)
        x_t = x_t_1
        if t.cpu().numpy() % 100 == 0:
            res.append((x_t.permute(0, 2, 3, 1).cpu().numpy()[0] + 1) * 127.5)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="cifar10", help="cifar10 | celeba")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--weights", type=str, default="./saved_models/model400.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    model = UNet(ch=128, droprate=0.5, groups=8, device=args.device).to(args.device)
    model.eval()
    model.load_state_dict(torch.load(args.weights, map_location=args.device))
    if args.data_type == "cifar10":
        h, w = 32, 32
    elif args.data_type == "celeba":
        h, w = 192, 160
    x_t = torch.randn(1, 3, h, w).to(args.device)
    result = reverse_diffusion(model, x_t, args.beta_start, args.beta_end, args.timesteps, args.device)
    result = np.concatenate(result, axis=1)
    result = np.clip(result, 0, 255)
    Image.fromarray(np.uint8(result)).save("reverse_diffusion.png")
    
    # python reverse_diffusion_process.py --data_type cifar10 --timesteps 1000 --weights ./saved_models/model500.pth