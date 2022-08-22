import torch 
import torch.nn.functional as F 
from diffusion_process import diffusion
from models import UNet
import numpy as np 
from PIL import Image 
import argparse
from tqdm import tqdm 

def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def reverse_diffusion(model, x_t, interp_step, beta_start, beta_end, timesteps, device):
    betas = linear_beta_schedule(beta_start, beta_end, timesteps).to(device)
    alphas = 1 - betas                                                     # DDPM paper section 2, equation (4)
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], pad=(1, 0), value=1.) # More convenient calculate alpha_t_1 
    res = []
    for t in tqdm(range(interp_step - 1, -1, -1)):
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

def interpolate(model, img1, img2, interp_steps, beta_start, beta_end, timesteps, device):
    x_t1 = diffusion(img1, beta_start, beta_end, timesteps, interp_steps, device=device)
    x_t2 = diffusion(img2, beta_start, beta_end, timesteps, interp_steps, device=device)
    rev_diff_interp = []
    for i in range(11):
        lambd = 1 / 10 * i 
        print("lambd: ", lambd)
        x_t = x_t2* lambd + x_t1 * (1 - lambd)
        result = reverse_diffusion(model, x_t, interp_steps, beta_start, beta_end, timesteps, device)
        # result = np.concatenate(result, axis=1)
        result = np.clip(result[-1], 0, 255)
        rev_diff_interp.append(result)
    rev_diff_interp = np.concatenate(rev_diff_interp, axis=1)
    return rev_diff_interp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img1_path", type=str, default="")
    parser.add_argument("--img2_path", type=str, default="")
    parser.add_argument("--data_type", type=str, default="celeba", help="cifar10 | celeba")
    parser.add_argument("--interp_steps", type=int, default=500)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--weights", type=str, default="./saved_models/model20.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    model = UNet(ch=128, droprate=0.5, groups=8, device=args.device).to(args.device)
    model.eval()
    model.load_state_dict(torch.load(args.weights, map_location=args.device))
    if args.data_type == "cifar10":
        h, w = 32, 32
    elif args.data_type == "celeba":
        h, w = 192, 160
    img1 = np.array(Image.open(args.img1_path).resize([160, 192]), dtype=np.float32)[..., :3] / 127.5 - 1.0 
    img2 = np.array(Image.open(args.img2_path).resize([160, 192]), dtype=np.float32)[..., :3] / 127.5 - 1.0 
    img1 = np.transpose(img1[None], axes=[0, 3, 1, 2])
    img2 = np.transpose(img2[None], axes=[0, 3, 1, 2])
    rev_diff_interp = interpolate(model, img1, img2, args.interp_steps, args.beta_start, args.beta_end, args.timesteps, args.device)
    Image.fromarray(np.uint8(rev_diff_interp)).save(f"reverse_diffusion_interp_{args.interp_steps}.png")
    
    # python interpolate.py --data_type celeba  --timesteps 1000 --weights ./saved_models/model20.pth --interp_step 500 --img1_path resources/000001.jpg --img2_path resources/000002.jpg