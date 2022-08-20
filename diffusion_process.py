import argparse 
import torch 
import numpy as np 
from PIL import Image 


def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def diffusion(x_0, beta_start, beta_end, timesteps, t, device):
    x_0 = torch.tensor(x_0).to(device)
    betas = linear_beta_schedule(beta_start, beta_end, timesteps)
    alphas = 1 - betas 
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    alpha_bar = alphas_cumprod.gather(-1, torch.tensor(t))
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps
    return x_t 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, default="resources/face.png")
    parser.add_argument("--t", type=str, default="199")
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    x_0 = np.array(Image.open(args.img_path).resize([128, 128]), dtype=np.float32)[..., :3] / 127.5 - 1.0 
    noiseds = [(x_0 + 1.0) * 127.5]
    for t in args.t.split(","):
        noised = diffusion(x_0, args.beta_start, args.beta_end, args.timesteps, int(t), device=args.device)
        noised = np.clip((noised.numpy() + 1.0) * 127.5, 0, 255)
        noiseds.append(noised)
    noiseds = np.concatenate(noiseds, axis=1)
    Image.fromarray(np.uint8(noiseds)).save(f"diffusion{args.t}.png")
    
    # python diffusion_process.py --timesteps 1000 --t 100,200,300,400,500,600,700,800,999