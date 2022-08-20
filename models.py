import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F 

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

class AttnBlock(nn.Module):
    def __init__(self, ch_in, ch_out, groups=32) -> None:
       super().__init__()
       self.conv_q = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
       self.conv_k = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
       self.conv_v = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
       self.conv_out = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True)
       self.norm = nn.GroupNorm(groups, ch_in)

    def forward(self, x):
        B, C, H, W = x.shape

        h = self.norm(x)
        q = self.conv_q(h) # B x C x H x W
        k = self.conv_k(h)
        v = self.conv_v(h)
        
        q = torch.permute(q.view(B, C, H * W), dims=[0, 2, 1]) # B x H*W x C
        k = k.view(B, C, H * W)                                # B x C x H*W
        attn = torch.matmul(q, k) / math.sqrt(C)               # B x H*W x H*W
        attn = F.softmax(attn, dim=-1)

        v = torch.permute(v.view(B, C, H * W), dims=[0, 2, 1]) # B x H*W x C
        o = torch.matmul(attn, v)
        
        o = torch.permute(o, dims=[0, 2, 1]).view(B, C, H, W)
        o = self.conv_out(o)

        return o + x 

class TimestepEmbedding(nn.Module):
    def __init__(self, ch, device) -> None:
        super().__init__()
        self.ch = ch
        self.device = device

    def forward(self, timesteps):
        half_dim = self.ch // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.range(0, half_dim - 1) * -emb).to(self.device)
        emb = timesteps[:, None] * emb[None, :] # b x 1 * 1 x n
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
        
        return emb

class Encoder(nn.Module):
    def __init__(self, ch=128, droprate=0.5, groups=32) -> None:
        super().__init__()
        # Downsample
        self.conv = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1, bias=True)            # 128 x 128 x 128
        self.resblock1_1 = ResBlock(ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock1_2 = ResBlock(ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.downsample1 = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1, bias=True)     # 64 x 64 x 128
        self.resblock2_1 = ResBlock(ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock2_2 = ResBlock(ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.downsample2 = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1, bias=True)     # 32 x 32 x 128
        self.resblock3_1 = ResBlock(ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock3_2 = ResBlock(2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.downsample3 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=2, padding=1, bias=True) # 16 x 16 x 256
        self.resblock4_1 = ResBlock(2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.attnblock4_1 = AttnBlock(2*ch, 2*ch, groups=groups)
        self.resblock4_2 = ResBlock(2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.attnblock4_2 = AttnBlock(2*ch, 2*ch, groups=groups)
        self.downsample4 = nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=2, padding=1, bias=True) # 8 x 8 x 256
        self.resblock5_1 = ResBlock(2*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock5_2 = ResBlock(4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.downsample5 = nn.Conv2d(4*ch, 4*ch, kernel_size=3, stride=2, padding=1, bias=True) # 4 x 4 x 512
        self.resblock6_1 = ResBlock(4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock6_2 = ResBlock(4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
    
    def forward(self, x, temb):
        hs = []
        # Downsample
        x = self.conv(x)
        hs.append(x)
        x = self.resblock1_1(x, temb)
        hs.append(x)
        x = self.resblock1_2(x, temb)
        hs.append(x)
        x = self.downsample1(x)
        hs.append(x)
        x = self.resblock2_1(x, temb)
        hs.append(x)
        x = self.resblock2_2(x, temb)
        hs.append(x)
        x = self.downsample2(x)
        hs.append(x)
        x = self.resblock3_1(x, temb)
        hs.append(x)
        x = self.resblock3_2(x, temb)
        hs.append(x)
        x = self.downsample3(x)
        hs.append(x)
        x = self.resblock4_1(x, temb)
        x = self.attnblock4_1(x)
        hs.append(x)
        x = self.resblock4_2(x, temb)
        x = self.attnblock4_2(x)
        hs.append(x)
        x = self.downsample4(x)
        hs.append(x)
        x = self.resblock5_1(x, temb)
        hs.append(x)
        x = self.resblock5_2(x, temb)
        hs.append(x)
        x = self.downsample5(x)
        hs.append(x)
        x = self.resblock6_1(x, temb)
        hs.append(x)
        x = self.resblock6_2(x, temb)
        hs.append(x)
        return x, hs 

class Decoder(nn.Module):
    def __init__(self, ch=128, droprate=0.5, groups=32) -> None:
        super().__init__()
        self.resblock1_1 = ResBlock(4*ch+4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock1_2 = ResBlock(4*ch+4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock1_3 = ResBlock(4*ch+4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.upsample1 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(4*ch, 4*ch, kernel_size=3, stride=1, padding=1, bias=False))
        self.resblock2_1 = ResBlock(4*ch+4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock2_2 = ResBlock(4*ch+4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock2_3 = ResBlock(2*ch+4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.upsample2 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(4*ch, 4*ch, kernel_size=3, stride=1, padding=1, bias=False))
        self.resblock3_1 = ResBlock(2*ch+4*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.attnblock3_1 = AttnBlock(2*ch, 2*ch, groups=groups)
        self.resblock3_2 = ResBlock(2*ch+2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.attnblock3_2 = AttnBlock(2*ch, 2*ch, groups=groups)
        self.resblock3_3 = ResBlock(2*ch+2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.attnblock3_3 = AttnBlock(2*ch, 2*ch, groups=groups)
        self.upsample3 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1, bias=False))
        self.resblock4_1 = ResBlock(2*ch+2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock4_2 = ResBlock(2*ch+2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock4_3 = ResBlock(ch+2*ch, 2*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.upsample4 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(2*ch, 2*ch, kernel_size=3, stride=1, padding=1, bias=False))
        self.resblock5_1 = ResBlock(ch+2*ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock5_2 = ResBlock(ch+ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock5_3 = ResBlock(ch+ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.upsample5 = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                       nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False))
        self.resblock6_1 = ResBlock(ch+ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock6_2 = ResBlock(ch+ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.resblock6_3 = ResBlock(ch+ch, ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.to_rgb = nn.Sequential(nn.GroupNorm(groups, ch), 
                                    nn.SiLU(),
                                    nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x, temb, hs):
        x = self.resblock1_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock1_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock1_3(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.upsample1(x)
        x = self.resblock2_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock2_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock2_3(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.upsample2(x)
        x = self.resblock3_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.attnblock3_1(x)
        x = self.resblock3_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.attnblock3_2(x)
        x = self.resblock3_3(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.attnblock3_3(x)
        x = self.upsample3(x)
        x = self.resblock4_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock4_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock4_3(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.upsample4(x)
        x = self.resblock5_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock5_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock5_3(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.upsample5(x)
        x = self.resblock6_1(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock6_2(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.resblock6_3(torch.cat([x, hs.pop()], dim=1), temb)
        x = self.to_rgb(x)
        return x

class UNet(nn.Module):
    def __init__(self, ch=128, droprate=0.5, groups=32, device="cuda") -> None:
        super().__init__()
        # Time Embedding
        self.Proj = nn.Sequential(TimestepEmbedding(ch, device),
                                  nn.Linear(ch, 4*ch), 
                                  nn.SiLU(),
                                  nn.Linear(4*ch, 4*ch))
        # Downsample
        self.encoder = Encoder(ch, droprate, groups)
        # Middle
        self.resblock1 = ResBlock(4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        self.attnblock = AttnBlock(4*ch, 4*ch, groups=groups)
        self.resblock2 = ResBlock(4*ch, 4*ch, temb_ch=ch * 4, droprate=droprate, groups=groups)
        # Upsample
        self.decoder = Decoder(ch, droprate, groups)
        
    def forward(self, x, t):
        hs = []
        # Time embedding
        temb = self.Proj(t)
        # Downsample
        x, hs = self.encoder(x, temb)
        # Middle
        x = self.resblock1(x, temb)
        x = self.attnblock(x)
        x = self.resblock2(x, temb)
        # Upsample
        x = self.decoder(x, temb, hs)
        return x


# temb = torch.randn(16, 512)  
# x = torch.randn(8, 3, 192, 160).cuda()
# t = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).cuda()
# model = UNet().cuda()
# with torch.no_grad():
#     out = model(x, t)
    # print(out.shape)
# torch.save(model, "./model.pth")
# print(out.shape)
# resblock = ResBlock(128, 128, 512, droprate=0.5) 
# out = resblock(x, temb)
# print(out.shape)
# attn = AttnBlock(256, 256)
# out = attn(x)
# timesteps = torch.range(0, 60)
# emb = get_timestep_embedding(timesteps, embedding_dim=120)
# import matplotlib.pyplot as plt 
# plt.imshow(emb.numpy())
# plt.show()
