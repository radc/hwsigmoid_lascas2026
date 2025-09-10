# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from .cuda_inference import CUSTOMIZED_CUDA_INFERENCE
if CUSTOMIZED_CUDA_INFERENCE:
    from .cuda_inference import DepthConvProxy, SubpelConv2xProxy




import torch
import numpy as np
import struct
from typing import List, Tuple


##COLOQUE A WSILU AQUI

class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()
        # Quebras dos intervalos [bk[i], bk[i+1]) (17 pontos -> 16 intervalos)
        bk = [-2.000, -1.500, -1.000, -0.750, -0.500, -0.250,  0.000,
               0.250,  0.500,  0.750,  1.000,  1.250,  1.312,  1.375,
               1.438,  1.500,  2.000]
        # Coeficientes (a, b, c) para cada intervalo, na mesma ordem
        a = [-0.00947, -0.03964, -0.07245, -0.01180,  0.31836,  0.87061,
              0.87061,  0.31787, -0.01367, -0.07178, -0.07483,  0.27051,
              0.26294,  0.24866,  0.22717,  0.01075]
        b = [-0.03897, -0.12683, -0.19702, -0.11218,  0.20410,  0.48315,
              0.51709,  0.79639,  1.11426,  1.19531,  1.20508,  0.33130,
              0.33179,  0.33203,  0.33252,  0.96826]
        c = [-0.04077, -0.10498, -0.14258, -0.11292, -0.03668, -0.00039,
             -0.00039, -0.03674, -0.11359, -0.14172, -0.14819,  0.40454,
              0.41650,  0.44238,  0.48633,  0.02046]

        # Buffers em float16 para evitar alocações no forward
        self.register_buffer("bk", torch.tensor(bk, dtype=torch.float16), persistent=False)  # (17,)
        self.register_buffer("a",  torch.tensor(a,  dtype=torch.float16), persistent=False)  # (16,)
        self.register_buffer("b",  torch.tensor(b,  dtype=torch.float16), persistent=False)  # (16,)
        self.register_buffer("c",  torch.tensor(c,  dtype=torch.float16), persistent=False)  # (16,)

    @torch.no_grad()  # remova se precisar de gradiente (treino). Mantém mais rápido em inferência.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computa em float16 e retorna no dtype original
        out_dtype = x.dtype
        xh = x.to(torch.float16)

        # Saída base: identidade (para x >= 2.0)
        y = xh.clone()

        # Máscaras de regiões
        mask_low  = xh <  self.bk[0]      # x < -2.0
        mask_mid  = (xh >= self.bk[0]) & (xh < self.bk[-1])  # [-2.0, 2.0)

        # x < -2.0 -> 0.0
        if mask_low.any():
            y[mask_low] = 0.0

        # Para [-2, 2): localizar intervalo com searchsorted (O(log N)) e aplicar polinômio
        if mask_mid.any():
            xm = xh[mask_mid]
            # idx_intervalo em [0, 15]; para bk[idx] <= x < bk[idx+1]
            idx = torch.searchsorted(self.bk, xm, right=False) - 1
            idx.clamp_(0, self.a.numel() - 1)

            a = self.a.index_select(0, idx)
            b = self.b.index_select(0, idx)
            c = self.c.index_select(0, idx)

            y_mid = a * xm * xm + b * xm + c
            y[mask_mid] = y_mid

        return y.to(out_dtype)


##FIM DA WSILU AQUI

class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, 1)
        return x1 + x2

class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding),
            nn.PixelShuffle(2),
        )
        self.padding = padding

        self.proxy = None

    def forward(self, x, to_cat=None, cat_at_front=True):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, to_cat, cat_at_front)

        return self.forward_cuda(x, to_cat, cat_at_front)

    def forward_torch(self, x, to_cat=None, cat_at_front=True):
        out = self.conv(x)
        if to_cat is None:
            return out
        if cat_at_front:
            return torch.cat((to_cat, out), dim=1)
        return torch.cat((out, to_cat), dim=1)

    def forward_cuda(self, x, to_cat=None, cat_at_front=True):
        if self.proxy is None:
            self.proxy = SubpelConv2xProxy()
            self.proxy.set_param(self.conv[0].weight, self.conv[0].bias, self.padding)

        if to_cat is None:
            return self.proxy.forward(x)

        return self.proxy.forward_with_cat(x, to_cat, cat_at_front)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch * 2, out_ch, 1),
        )

        self.proxy = None

    def forward(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step, to_cat, cat_at_front)

        return self.forward_cuda(x, quant_step, to_cat, cat_at_front)

    def forward_torch(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        if quant_step is not None:
            out = out * quant_step
        if to_cat is not None:
            if cat_at_front:
                out = torch.cat((to_cat, out), dim=1)
            else:
                out = torch.cat((out, to_cat), dim=1)
        return out

    def forward_cuda(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if self.proxy is None:
            self.proxy = DepthConvProxy()
            if self.adaptor is not None:
                self.proxy.set_param_with_adaptor(self.dc[0].weight, self.dc[0].bias,
                                                  self.dc[2].weight, self.dc[2].bias,
                                                  self.dc[3].weight, self.dc[3].bias,
                                                  self.ffn[0].weight, self.ffn[0].bias,
                                                  self.ffn[2].weight, self.ffn[2].bias,
                                                  self.adaptor.weight, self.adaptor.bias,
                                                  self.shortcut)
            else:
                self.proxy.set_param(self.dc[0].weight, self.dc[0].bias,
                                     self.dc[2].weight, self.dc[2].bias,
                                     self.dc[3].weight, self.dc[3].bias,
                                     self.ffn[0].weight, self.ffn[0].bias,
                                     self.ffn[2].weight, self.ffn[2].bias,
                                     self.shortcut)

        if quant_step is not None:
            return self.proxy.forward_with_quant_step(x, quant_step)
        if to_cat is not None:
            return self.proxy.forward_with_cat(x, to_cat, cat_at_front)

        return self.proxy.forward(x)


class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        x = self.down(x)
        out = self.conv(x)
        return out


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out
