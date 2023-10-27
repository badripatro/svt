import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import numpy as np
from pytorch_wavelets import DTCWTForward, DTCWTInverse

class SVT_channel_mixing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 64: #[b, 64,56,56]
            self.hidden_size = dim
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 56, 56, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)

        if dim ==128: #[b, 128,28,28]
            self.hidden_size = dim
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 28, 28, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)

        if dim == 96: #96 for large model, 64 for small and base model
            self.hidden_size = dim
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 56, 56, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
        if dim ==192:
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 28, 28, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink =0.0 

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, H, W):
        B, N, C = x.shape 
        x = x.view(B, H, W, C)
        x=torch.permute(x, (0, 3, 1, 2))
        B, C, H, W = x.shape 
        x = x.to(torch.float32) 
        
        xl,xh = self.xfm(x)
        xl = xl * self.complex_weight_ll

        xh[0]=torch.permute(xh[0], (5, 0, 2, 3, 4, 1))
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4], self.num_blocks, self.block_size)
        
        x_real=xh[0][0]
        x_imag=xh[0][1]
        
        x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) + self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) + self.complex_weight_lh_b1[1])
        
        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + self.complex_weight_lh_b2[1]

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6])
        xh[0]=torch.permute(xh[0], (0, 4, 1, 2, 3, 5))

        x = self.ifm((xl,xh))
        x=torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(B, N, C)# permute is not same as reshape or view
        return x



class SVT_channel_token_mixing(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 64: #[b, 64,56,56]
            self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2, bias=True)

            self.hidden_size = dim // 2
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.token_blocks = 28
            
            self.complex_weight_ll = nn.Parameter(torch.randn(dim // 2, 56, 56, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)

            self.complex_weight_lh_1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks,  dtype=torch.float32) * 0.02)

        if dim ==128: #[b, 128,28,28]
            self.conv = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, groups=dim // 2, bias=True)

            self.hidden_size = dim // 2
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.token_blocks = 14

            self.complex_weight_ll = nn.Parameter(torch.randn(dim // 2, 28, 28, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)

            self.complex_weight_lh_1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks,  dtype=torch.float32) * 0.02)
        
        if dim == 96: #96 for large model, 64 for small and base model
            self.hidden_size = dim
            self.num_blocks = 4 
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 56, 56, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
        
            self.complex_weight_lh_1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks,  dtype=torch.float32) * 0.02)
        
        if dim ==192:
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            assert self.hidden_size % self.num_blocks == 0
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 28, 28, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(torch.randn(2, self.num_blocks, self.block_size,  dtype=torch.float32) * 0.02)
            
            self.complex_weight_lh_1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks, self.token_blocks, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks,  dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2_t = nn.Parameter(torch.randn(2, self.token_blocks, self.token_blocks,  dtype=torch.float32) * 0.02)
        

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink =0.0 #args.fno_softshrink

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x, H, W):
        B, N, C = x.shape 

        x = x.view(B, H, W, C)
        x=torch.permute(x, (0, 3, 1, 2)).contiguous() # (N, H, W, C) -> (N, C, H, W)
        B, C, H, W = x.shape # this shape is required for dwt

        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.conv(x1)

        x2 = x2.to(torch.float32)
        B, C1, a, b = x2.shape

        xl,xh = self.xfm(x2)
        xl = xl * self.complex_weight_ll
        xh[0]=torch.permute(xh[0], (5, 0, 2, 3, 4, 1)).contiguous()
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4], self.num_blocks, self.block_size)
        
        ###########################################################################################
        # This is for Channel mixing: 
        x_real=xh[0][0]
        x_imag=xh[0][1]

        x_real_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) + self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) + self.complex_weight_lh_b1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1, self.complex_weight_lh_2[1]) + self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1, self.complex_weight_lh_2[0]) + self.complex_weight_lh_b2[1]

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6])
        ###########################################################################################
        # This is for Token mixing: 
        xh[0]=torch.permute(xh[0], (5, 0, 4, 1, 2, 3)).contiguous() #2, B, 64, 6,28,28
        x_real_t=xh[0][0]
        x_imag_t=xh[0][1]

        x_real_1_t = F.relu(self.multiply(x_real_t, self.complex_weight_lh_1_t[0]) - self.multiply(x_imag_t, self.complex_weight_lh_1_t[1]) + self.complex_weight_lh_b1_t[0])
        x_imag_1_t = F.relu(self.multiply(x_real_t, self.complex_weight_lh_1_t[1]) + self.multiply(x_imag_t, self.complex_weight_lh_1_t[0]) + self.complex_weight_lh_b1_t[1])
        x_real_2_t = self.multiply(x_real_1_t, self.complex_weight_lh_2_t[0]) - self.multiply(x_imag_1_t, self.complex_weight_lh_2_t[1]) + self.complex_weight_lh_b2_t[0]
        x_imag_2_t = self.multiply(x_real_1_t, self.complex_weight_lh_2_t[1]) + self.multiply(x_imag_1_t, self.complex_weight_lh_2_t[0]) + self.complex_weight_lh_b2_t[1]

        xh[0] = torch.stack([x_real_2_t, x_imag_2_t], dim=-1).float() #B, 64, 6,24,28, 2

        x2 = self.ifm((xl,xh))

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C1, a, b)
        x=torch.permute(x, (0, 2, 3, 1)).contiguous() # (N, C, H, W) -> (N, H, W, C)
        x = x.reshape(B, N, C)# permute is not same as reshape or view
        return x

def rand_bbox(size, lam, scale=1):
    W = size[1] // scale
    H = size[2] // scale
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int_(W * cut_rat)
    cut_h = np.int_(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.kv = nn.Linear(dim, dim * 2)
        self.q = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
        cls_embed = self.proj(cls_embed)
        return cls_embed

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class ClassBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = ClassAttention(dim, num_heads)
        self.mlp = FFN(dim, int(dim * mlp_ratio))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        cls_embed = x[:, :1]
        cls_embed = cls_embed + self.attn(self.norm1(x))
        cls_embed = cls_embed + self.mlp(self.norm2(cls_embed))
        return torch.cat([cls_embed, x[:, 1:]], dim=1)

class PVT2FFN(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, 
        dim, 
        num_heads, 
        mlp_ratio,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        sr_ratio=1, 
        block_type = 'scatter'
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        if block_type == 'std_att':
            self.attn = Attention(dim, num_heads)
        else:
            self.attn = SVT_channel_mixing (dim)
            # self.attn = SVT_token_mixing (dim)
            # self.attn = SVT_channel_token_mixing (dim)
        self.mlp = PVT2FFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class DownSamples(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_channels)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class Stem(nn.Module):
    def __init__(self, in_channels, stem_hidden_dim, out_channels):
        super().__init__()
        hidden_dim = stem_hidden_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=7, stride=2,
                      padding=3, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1,
                      padding=1, bias=False),  # 112x112
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(hidden_dim,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.norm = nn.LayerNorm(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class SVT(nn.Module):
    def __init__(self, 
        in_chans=3, 
        num_classes=1000, 
        stem_hidden_dim = 32,
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14], 
        mlp_ratios=[8, 8, 4, 4], 
        drop_path_rate=0., 
        norm_layer=nn.LayerNorm,
        depths=[3, 4, 6, 3], 
        sr_ratios=[4, 2, 1, 1], 
        num_stages=4,
        token_label=True,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        alpha = 1
        
        for i in range(num_stages):
            if i == 0:
                patch_embed = Stem(in_chans, stem_hidden_dim, embed_dims[i])
            else:
                patch_embed = DownSamples(embed_dims[i - 1], embed_dims[i])

            block = nn.ModuleList([Block(
                dim = embed_dims[i], 
                num_heads = num_heads[i], 
                mlp_ratio = mlp_ratios[i], 
                drop_path=dpr[cur + j], 
                norm_layer=norm_layer,
                sr_ratio = sr_ratios[i],
                block_type='scatter' if i < alpha else 'std_att')
            for j in range(depths[i])])

            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        post_layers = ['ca']
        self.post_network = nn.ModuleList([
            ClassBlock(
                dim = embed_dims[-1], 
                num_heads = num_heads[-1], 
                mlp_ratio = mlp_ratios[-1],
                norm_layer=norm_layer)
            for _ in range(len(post_layers))
        ])

        # classification head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        ##################################### token_label #####################################
        self.return_dense = token_label
        self.mix_token = token_label
        self.beta = 1.0
        self.pooling_scale = 8
        if self.return_dense:
            self.aux_head = nn.Linear(
                embed_dims[-1],
                num_classes) if num_classes > 0 else nn.Identity()
        ##################################### token_label #####################################

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward_cls(self, x):
        B, N, C = x.shape
        cls_tokens = x.mean(dim=1, keepdim=True)
        x = torch.cat((cls_tokens, x), dim=1)
        for block in self.post_network:
            x = block(x)
        return x

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            
            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.forward_cls(x)[:, 0]
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)
        return x

    def forward(self, x):
        if not self.return_dense:
            x = self.forward_features(x)
            x = self.head(x)
            return x
        else:
            x, H, W = self.forward_embeddings(x)
            # mix token, see token labeling for details.
            if self.mix_token and self.training:
                lam = np.random.beta(self.beta, self.beta)
                patch_h, patch_w = x.shape[1] // self.pooling_scale, x.shape[
                    2] // self.pooling_scale
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam, scale=self.pooling_scale)
                temp_x = x.clone()
                sbbx1,sbby1,sbbx2,sbby2=self.pooling_scale*bbx1,self.pooling_scale*bby1,\
                                        self.pooling_scale*bbx2,self.pooling_scale*bby2
                temp_x[:, sbbx1:sbbx2, sbby1:sbby2, :] = x.flip(0)[:, sbbx1:sbbx2, sbby1:sbby2, :]
                x = temp_x
            else:
                bbx1, bby1, bbx2, bby2 = 0, 0, 0, 0
            
            x = self.forward_tokens(x, H, W)
            x_cls = self.head(x[:, 0])
            x_aux = self.aux_head(
                x[:, 1:]
            )  # generate classes in all feature tokens, see token labeling

            if not self.training:
                return x_cls + 0.5 * x_aux.max(1)[0]

            if self.mix_token and self.training:  # reverse "mix token", see token labeling for details.
                x_aux = x_aux.reshape(x_aux.shape[0], patch_h, patch_w, x_aux.shape[-1])

                temp_x = x_aux.clone()
                temp_x[:, bbx1:bbx2, bby1:bby2, :] = x_aux.flip(0)[:, bbx1:bbx2, bby1:bby2, :]
                x_aux = temp_x

                x_aux = x_aux.reshape(x_aux.shape[0], patch_h * patch_w, x_aux.shape[-1])

            return x_cls, x_aux, (bbx1, bby1, bbx2, bby2)

    def forward_tokens(self, x, H, W):
        B = x.shape[0]
        x = x.view(B, -1, x.size(-1))

        for i in range(self.num_stages):
            if i != 0:
                patch_embed = getattr(self, f"patch_embed{i + 1}")
                x, H, W = patch_embed(x)

            block = getattr(self, f"block{i + 1}")
            for blk in block:
                x = blk(x, H, W)

            if i != self.num_stages - 1:
                norm = getattr(self, f"norm{i + 1}")
                x = norm(x)
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x = self.forward_cls(x)
        norm = getattr(self, f"norm{self.num_stages}")
        x = norm(x)    
        return x

    def forward_embeddings(self, x):
        patch_embed = getattr(self, f"patch_embed{0 + 1}")
        x, H, W = patch_embed(x)
        x = x.view(x.size(0), H, W, -1)
        return x, H, W


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

@register_model
def svt_s(pretrained=False, **kwargs):
    model = SVT(
        stem_hidden_dim = 32,
        embed_dims = [64, 128, 320, 448], 
        num_heads = [2, 4, 10, 14], 
        mlp_ratios = [8, 8, 4, 4],
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 4, 6, 3], 
        sr_ratios = [4, 2, 1, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def svt_b(pretrained=False, **kwargs):
    model = SVT(
        stem_hidden_dim = 64,
        embed_dims = [64, 128, 320, 512], 
        num_heads = [2, 4, 10, 16], 
        mlp_ratios = [8, 8, 4, 4], 
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 4, 12, 3], 
        sr_ratios = [4, 2, 1, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def svt_l(pretrained=False, **kwargs):
    model = SVT(
        stem_hidden_dim = 64,
        embed_dims = [96, 192, 384, 512],
        num_heads = [3, 6, 12, 16], 
        mlp_ratios = [8, 8, 4, 4],
        norm_layer = partial(nn.LayerNorm, eps=1e-6), 
        depths = [3, 6, 18, 3], 
        sr_ratios = [4, 2, 1, 1], 
        **kwargs)
    model.default_cfg = _cfg()
    return model