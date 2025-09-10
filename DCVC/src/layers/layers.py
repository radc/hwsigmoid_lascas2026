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
    """
    Simulação em float16 (fp16) com LUT 2^16 em [-2.5, +2.5].
    Regras:
      - x < -2.5  -> 0
      - x > +2.5  -> identidade (x)
      - caso contrário -> LUT[index(x)]
    Todas as operações de ponto flutuante internas usam fp16.
    Sempre roda em CUDA.
    """

    def __init__(self, return_fp16: bool = True):
        super().__init__()
        self.return_fp16 = return_fp16

        # força CUDA
        device = torch.device("cuda")
        dtype = torch.float16

        # ----- Constantes -----
        xmin = torch.tensor(-2.5, dtype=dtype, device=device)
        xmax = torch.tensor( 2.5, dtype=dtype, device=device)
        scale = torch.tensor(65535.0/5.0, dtype=dtype, device=device)

        self.register_buffer("XMIN", xmin, persistent=False)
        self.register_buffer("XMAX", xmax, persistent=False)
        self.register_buffer("SCALE", scale, persistent=False)
        self.register_buffer("ZERO", torch.tensor(0.0, dtype=dtype, device=device), persistent=False)

        # ----- Geração automática da LUT -----
        with torch.no_grad():
            grid = torch.linspace(float(xmin), float(xmax), steps=65536,
                                  dtype=dtype, device=device)
            vals = torch.sigmoid(torch.tensor(4.0, dtype=dtype, device=device) * grid) * grid
            lut_fp16 = vals.to(dtype)
        self.register_buffer("lut", lut_fp16, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # força CUDA + fp16
        xh = x.to(dtype=torch.float16, device="cuda")

        # Máscaras
        m_low  = xh < self.XMIN
        m_high = xh > self.XMAX
        m_mid  = ~(m_low | m_high)

        # Índices
        idx_f16 = torch.floor((xh - self.XMIN) * self.SCALE)
        idx = idx_f16.to(torch.int32).clamp_(0, self.lut.numel() - 1).to(torch.long)

        # Valores da LUT
        y_lut = self.lut[idx]

        # Base: identidade
        y = xh
        y = torch.where(m_mid,  y_lut, y)
        y = torch.where(m_low, self.ZERO, y)

        return y if self.return_fp16 else y.to(x.dtype, device=x.device)

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
