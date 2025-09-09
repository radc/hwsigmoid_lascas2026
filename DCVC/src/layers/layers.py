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

        # ----- LUT em float16 (mesma lista hex do seu código) -----
        lut_hex = [
            0x3800, 0x3804, 0x3808, 0x380c, 0x3810, 0x3814, 0x3818, 0x381c, 0x3820, 0x3824, 0x3828, 0x382c, 0x3830, 0x3834, 0x3838, 0x383c, 0x3840, 0x3844, 0x3848, 0x384c, 0x3850, 0x3854, 0x3858, 0x385c, 0x3860, 0x3864, 0x3868, 0x386c, 0x3870, 0x3874, 0x3877, 0x387b, 0x387f, 0x3883, 0x3887, 0x388b, 0x388f, 0x3893, 0x3897, 0x389b, 0x389f, 0x38a3, 0x38a7, 0x38aa, 0x38ae, 0x38b2, 0x38b6, 0x38ba, 0x38be, 0x38c2, 0x38c5, 0x38c9, 0x38cd, 0x38d1, 0x38d5, 0x38d9, 0x38dc, 0x38e0, 0x38e4, 0x38e8, 0x38ec, 0x38ef, 0x38f3, 0x38f7, 0x38fb, 0x38ff, 0x3902, 0x3906, 0x390a, 0x390e, 0x3911, 0x3915, 0x3919, 0x391c, 0x3920, 0x3924, 0x3927, 0x392b, 0x392f, 0x3932, 0x3936, 0x393a, 0x393d, 0x3941, 0x3944, 0x3948, 0x394c, 0x394f, 0x3953, 0x3956, 0x395a, 0x395d, 0x3961, 0x3964, 0x3968, 0x396b, 0x396f, 0x3972, 0x3976, 0x3979, 0x397d, 0x3980, 0x3984, 0x3987, 0x398b, 0x398e, 0x3991, 0x3995, 0x3998, 0x399b, 0x399f, 0x39a2, 0x39a5, 0x39a9, 0x39ac, 0x39af, 0x39b3, 0x39b6, 0x39b9, 0x39bc, 0x39c0, 0x39c3, 0x39c6, 0x39c9, 0x39cd, 0x39d0, 0x39d3, 0x39d6, 0x39d9, 0x39dc, 0x39df, 0x39e3, 0x39e6, 0x39e9, 0x39ec, 0x39ef, 0x39f2, 0x39f5, 0x39f8, 0x39fb, 0x39fe, 0x3a01, 0x3a04, 0x3a07, 0x3a0a, 0x3a0d, 0x3a10, 0x3a13, 0x3a16, 0x3a19, 0x3a1c, 0x3a1e, 0x3a21, 0x3a24, 0x3a27, 0x3a2a, 0x3a2d, 0x3a30, 0x3a32, 0x3a35, 0x3a38, 0x3a3b, 0x3a3d, 0x3a40, 0x3a43, 0x3a46, 0x3a48, 0x3a4b, 0x3a4e, 0x3a50, 0x3a53, 0x3a56, 0x3a58, 0x3a5b, 0x3a5e, 0x3a60, 0x3a63, 0x3a65, 0x3a68, 0x3a6a, 0x3a6d, 0x3a6f, 0x3a72, 0x3a74, 0x3a77, 0x3a79, 0x3a7c, 0x3a7e, 0x3a81, 0x3a83, 0x3a86, 0x3a88, 0x3a8a, 0x3a8d, 0x3a8f, 0x3a91, 0x3a94, 0x3a96, 0x3a98, 0x3a9b, 0x3a9d, 0x3a9f, 0x3aa2, 0x3aa4, 0x3aa6, 0x3aa8, 0x3aab, 0x3aad, 0x3aaf, 0x3ab1, 0x3ab3, 0x3ab6, 0x3ab8, 0x3aba, 0x3abc, 0x3abe, 0x3ac0, 0x3ac2, 0x3ac4, 0x3ac7, 0x3ac9, 0x3acb, 0x3acd, 0x3acf, 0x3ad1, 0x3ad3, 0x3ad5, 0x3ad7, 0x3ad9, 0x3adb, 0x3add, 0x3adf, 0x3ae1, 0x3ae3, 0x3ae4, 0x3ae6, 0x3ae8, 0x3aea, 0x3aec, 0x3aee, 0x3af0, 0x3af2, 0x3af3, 0x3af5, 0x3af7, 0x3af9, 0x3afb, 0x3afc, 0x3afe, 0x3b00, 0x3b02, 0x3b03, 0x3b05, 0x3b07, 0x3b08, 0x3b0a, 0x3b0c, 0x3b0f, 0x3b13, 0x3b16, 0x3b19, 0x3b1c, 0x3b1f, 0x3b22, 0x3b25, 0x3b29, 0x3b2c, 0x3b2e, 0x3b31, 0x3b34, 0x3b37, 0x3b3a, 0x3b3d, 0x3b3f, 0x3b42, 0x3b45, 0x3b47, 0x3b4a, 0x3b4d, 0x3b4f, 0x3b52, 0x3b54, 0x3b57, 0x3b59, 0x3b5b, 0x3b5e, 0x3b60, 0x3b62, 0x3b65, 0x3b67, 0x3b69, 0x3b6b, 0x3b6d, 0x3b6f, 0x3b72, 0x3b74, 0x3b76, 0x3b78, 0x3b7a, 0x3b7c, 0x3b7e, 0x3b7f, 0x3b81, 0x3b83, 0x3b85, 0x3b87, 0x3b89, 0x3b8a, 0x3b8c, 0x3b8e, 0x3b8f, 0x3b91, 0x3b93, 0x3b94, 0x3b96, 0x3b97, 0x3b99, 0x3b9a, 0x3b9c, 0x3b9d, 0x3b9f, 0x3ba0, 0x3ba2, 0x3ba3, 0x3ba4, 0x3ba6, 0x3ba7, 0x3ba9, 0x3baa, 0x3bab, 0x3bac, 0x3bae, 0x3baf, 0x3bb0, 0x3bb1, 0x3bb2, 0x3bb4, 0x3bb5, 0x3bb6, 0x3bb7, 0x3bb8, 0x3bb9, 0x3bba, 0x3bbb, 0x3bbc, 0x3bbd, 0x3bbe, 0x3bbf, 0x3bc0, 0x3bc1, 0x3bc2, 0x3bc3, 0x3bc4, 0x3bc5, 0x3bc6, 0x3bc7, 0x3bc8, 0x3bc8, 0x3bc9, 0x3bca, 0x3bcb, 0x3bcc, 0x3bcc, 0x3bcd, 0x3bce, 0x3bcf, 0x3bcf, 0x3bd0, 0x3bd1, 0x3bd2, 0x3bd2, 0x3bd3, 0x3bd4, 0x3bd4, 0x3bd5, 0x3bd6, 0x3bd6, 0x3bd7, 0x3bd8, 0x3bd8, 0x3bd9, 0x3bd9, 0x3bda, 0x3bdb, 0x3bdb, 0x3bdc, 0x3bdd, 0x3bde, 0x3bdf, 0x3be0, 0x3be1, 0x3be2, 0x3be3, 0x3be4, 0x3be5, 0x3be6, 0x3be7, 0x3be7, 0x3be8, 0x3be9, 0x3be9, 0x3bea, 0x3beb, 0x3beb, 0x3bec, 0x3bed, 0x3bed, 0x3bee, 0x3bee, 0x3bef, 0x3bef, 0x3bf0, 0x3bf0, 0x3bf1, 0x3bf1, 0x3bf2, 0x3bf2, 0x3bf3, 0x3bf3, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf5, 0x3bf5, 0x3bf5, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3c00,
        ]

        lut_float16 = np.array(
            [struct.unpack('>e', struct.pack('>H', v))[0] for v in lut_hex],
            dtype=np.float16
        )
        # registra como buffer para mover com .to(device) e salvar no state_dict
        self.register_buffer("lut", torch.from_numpy(lut_float16),persistent=False)  # float16

        # Constantes (iguais ao seu VHDL)
        self.LIMIT_1 = 0.5
        self.LIMIT_2 = 1.0
        self.LIMIT_3 = 1.5
        self.LIMIT_4 = 2.0
        self.N_1 = 256
        self.N_2 = 128
        self.N_3 = 64
        self.N_4 = 64

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype

        # LUT no mesmo device/dtype do input
        lut = self.lut.to(device=device, dtype=dtype)              # (512,)
        lut_size = lut.numel()

        xq = x  # já está no dtype desejado
        absx = xq.abs()

        # índices inteiros (0 .. lut_size-1)
        idx = torch.empty_like(xq, dtype=torch.long)

        # máscaras por intervalo (mesma lógica do seu código)
        m1 = absx <= self.LIMIT_1
        m2 = (absx > self.LIMIT_1) & (absx <= self.LIMIT_2)
        m3 = (absx > self.LIMIT_2) & (absx <= self.LIMIT_3)
        m4 = (absx > self.LIMIT_3) & (absx <= self.LIMIT_4)
        m5 = absx > self.LIMIT_4

        # int() em Python trunca para baixo para números positivos -> use floor
        if m1.any():
            idx[m1] = torch.floor((absx[m1] / self.LIMIT_1) * self.N_1).to(torch.long)

        if m2.any():
            idx[m2] = ( self.N_1
                        + torch.floor(((absx[m2] - self.LIMIT_1) / (self.LIMIT_2 - self.LIMIT_1)) * self.N_2).to(torch.long) )

        if m3.any():
            idx[m3] = ( self.N_1 + self.N_2
                        + torch.floor(((absx[m3] - self.LIMIT_2) / (self.LIMIT_3 - self.LIMIT_2)) * self.N_3).to(torch.long) )

        if m4.any():
            idx[m4] = ( self.N_1 + self.N_2 + self.N_3
                        + torch.floor(((absx[m4] - self.LIMIT_3) / (self.LIMIT_4 - self.LIMIT_3)) * self.N_4).to(torch.long) )

        if m5.any():
            idx[m5] = lut_size - 1

        # garante 0..lut_size-1 (cobre bordas como 0.5, 1.0, 1.5, 2.0)
        idx = idx.clamp_(min=0, max=lut_size - 1)

        # valor de sigmoid a partir da LUT
        sig_val = lut[idx]

        # simetria para x < 0 (e zerar para x < -2)
        one = torch.tensor(1.0, dtype=dtype, device=device)
        y_sigmoid = torch.where(xq < 0, one - sig_val, sig_val)
        y_sigmoid = torch.where(xq < -self.LIMIT_4, torch.zeros_like(y_sigmoid), y_sigmoid)

        return xq * y_sigmoid
  
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
