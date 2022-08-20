# Denoising diffusion model
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
Download the cifar10 pretrained model from [GoogleDrive](https://drive.google.com/file/d/1-fFUkAsGi1uHQxWXmkHtt7LwnDzm7odN/view?usp=sharing), and then put the model into the folder saved_models
```
python reverse_diffusion_process.py --data_type cifar10 --timesteps 1000 --weights ./saved_models/model500.pth
```
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/rev_diff.png)

## Read code
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
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/intro_tra.png)
```python
    t = torch.randint(0, timesteps, [batchsize]).to(device)
    alphas_bar = alphas_cumprod.gather(-1, t).view(batchsize, 1, 1, 1)
    eps = torch.randn_like(batch_x0)        
    x_t = torch.sqrt(alphas_bar) * batch_x0 + torch.sqrt(1 - alphas_bar) * eps
    pred_noise = model(x_t, t)
    loss = smoothl1(pred_noise, eps)
```
![](https://github.com/MingtaoGuo/DDPM_pytorch/raw/main/resources/intro_rev.png)
```python
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
```
## Embedding timestep into UNet
```python
class ResBlock(nn.Module):
    def __init__(self, ch_in, ch_out, temb_ch, droprate=0.5, groups=32) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=ch_in)
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

        self.linear1 = nn.Linear(temb_ch, ch_out)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=ch_out)
        self.dropout = nn.Dropout(droprate)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)

        if ch_in != ch_out:
            self.shortcut = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x, temb):
        h = x 
        h = F.silu(self.norm1(h))
        h = self.conv1(h)

        # add in timestep embedding
        temb_proj = self.linear1(F.silu(temb))[:, :, None, None]
        h += temb_proj
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h) 

        x = self.shortcut(x)
        return x + h 
```
## Acknowledgements
* Official tensorflow implementation [ddpm](https://github.com/hojonathanho/diffusion)
* Unofficial pytorch implementation [ddpm](https://github.com/lucidrains/denoising-diffusion-pytorch)
## Author 
Mingtao Guo
E-mail: gmt798714378@hotmail.com

