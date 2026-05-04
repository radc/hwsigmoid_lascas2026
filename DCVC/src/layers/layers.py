# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from .cuda_inference import CUSTOMIZED_CUDA_INFERENCE
if CUSTOMIZED_CUDA_INFERENCE:
    from .cuda_inference import DepthConvProxy, SubpelConv2xProxy


from .layers_improved import TracedConv2d




##COLOQUE A WSILU AQUI

# class WSiLU(nn.Module):
#     def forward(self, x):
#         y = torch.relu(x)
#         return y


#DENIS COM 1 INTERVALOS E GRAU 11 (poly_1int_deg11_16)


class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("xmin", torch.tensor(-2.0, dtype=torch.float16), persistent=False)
        self.register_buffer("xmax", torch.tensor( 2.0, dtype=torch.float16), persistent=False)
        self.register_buffer("zero", torch.tensor(0.0, dtype=torch.float16), persistent=False)

        # Coeficientes em float16
        self.register_buffer("c11", torch.tensor( 2.771615982055664e-05,   dtype=torch.float16), persistent=False)
        self.register_buffer("c10", torch.tensor( 0.004848480224609375,    dtype=torch.float16), persistent=False)
        self.register_buffer("c9",  torch.tensor(-0.00024890899658203125,  dtype=torch.float16), persistent=False)
        self.register_buffer("c8",  torch.tensor(-0.05645751953125,        dtype=torch.float16), persistent=False)
        self.register_buffer("c7",  torch.tensor( 0.0007786750793457031,   dtype=torch.float16), persistent=False)
        self.register_buffer("c6",  torch.tensor( 0.25341796875,           dtype=torch.float16), persistent=False)
        self.register_buffer("c5",  torch.tensor(-0.0009975433349609375,   dtype=torch.float16), persistent=False)
        self.register_buffer("c4",  torch.tensor(-0.56982421875,           dtype=torch.float16), persistent=False)
        self.register_buffer("c3",  torch.tensor( 0.0004525184631347656,   dtype=torch.float16), persistent=False)
        self.register_buffer("c2",  torch.tensor( 0.84716796875,           dtype=torch.float16), persistent=False)
        self.register_buffer("c1",  torch.tensor( 0.5,                     dtype=torch.float16), persistent=False)
        self.register_buffer("c0",  torch.tensor( 0.005859375,             dtype=torch.float16), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype
        xh = x.to(torch.float16)

        # Base:
        # x < -2  -> 0
        # x >= 2  -> x
        y = torch.where(xh < self.xmin, self.zero, xh)

        # Região polinomial: -2 <= x < 2
        mask_mid = (xh >= self.xmin) & (xh < self.xmax)

        if mask_mid.any():
            xm = xh[mask_mid]

            # Horner manual:
            y_mid = self.c11
            y_mid = y_mid * xm + self.c10
            y_mid = y_mid * xm + self.c9
            y_mid = y_mid * xm + self.c8
            y_mid = y_mid * xm + self.c7
            y_mid = y_mid * xm + self.c6
            y_mid = y_mid * xm + self.c5
            y_mid = y_mid * xm + self.c4
            y_mid = y_mid * xm + self.c3
            y_mid = y_mid * xm + self.c2
            y_mid = y_mid * xm + self.c1
            y_mid = y_mid * xm + self.c0

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

# class DepthConvBlock(nn.Module):
#     _id = 1

#     def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False, track=True):
#         super().__init__()

#         self.track = track
#         self.counter = 0
#         self.id = DepthConvBlock._id
#         DepthConvBlock._id += 1

#         self.adaptor = None
#         if in_ch != out_ch or force_adaptor:
#             self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
#         self.shortcut = shortcut

#         if self.id == 38 :
#             self.dc = nn.Sequential(
#                 TracedConv2d(
#                     out_ch, out_ch, 1,
#                     track=True,
#                     log_dir="/home/ruhan/hwsigmoid_lascas2026/coding_outputs/trace_logs/dc0",
#                     index_name="index.jsonl",
#                     save_every=1,
#                     compress=False, 
#                     dtype_on_save=None, 
#                     flush_every=20,
#                     name="id38_dc_0"
#                 ),
#                 WSiLU(),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
#                 nn.Conv2d(out_ch, out_ch, 1),
#             )

#         else:
#             self.dc = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch, 1),
#             WSiLU(),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
#             nn.Conv2d(out_ch, out_ch, 1),
#         )

#         self.ffn = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch * 4, 1),
#             WSiLUChunkAdd(),
#             nn.Conv2d(out_ch * 2, out_ch, 1),
#         )

#         self.proxy = None

#     def forward(self, x, quant_step=None, to_cat=None, cat_at_front=True):
#         if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
#             return self.forward_torch(x, quant_step, to_cat, cat_at_front)

#         return self.forward_cuda(x, quant_step, to_cat, cat_at_front)

#     def forward_torch(self, x, quant_step=None, to_cat=None, cat_at_front=True):        
        
#         if self.adaptor is not None:
#             x = self.adaptor(x)       
        
#         if (self.track):
#             self.counter += 1
#             k = self.dc[0].kernel_size
            
#             n = self.dc[0].out_channels
#             # print(f"id{self.id}",x.shape[1], x.shape[2], x.shape[3], n, self.counter, sep=",")

#         out = self.dc(x) + x
#         out = self.ffn(out) + out
#         if self.shortcut:
#             out = out + x
#         if quant_step is not None:
#             out = out * quant_step
#         if to_cat is not None:
#             if cat_at_front:
#                 out = torch.cat((to_cat, out), dim=1)
#             else:
#                 out = torch.cat((out, to_cat), dim=1)
#         return out

#     def forward_cuda(self, x, quant_step=None, to_cat=None, cat_at_front=True):
#         if self.proxy is None:
#             self.proxy = DepthConvProxy()
#             if self.adaptor is not None:
#                 self.proxy.set_param_with_adaptor(self.dc[0].weight, self.dc[0].bias,
#                                                   self.dc[2].weight, self.dc[2].bias,
#                                                   self.dc[3].weight, self.dc[3].bias,
#                                                   self.ffn[0].weight, self.ffn[0].bias,
#                                                   self.ffn[2].weight, self.ffn[2].bias,
#                                                   self.adaptor.weight, self.adaptor.bias,
#                                                   self.shortcut)
#             else:
#                 self.proxy.set_param(self.dc[0].weight, self.dc[0].bias,
#                                      self.dc[2].weight, self.dc[2].bias,
#                                      self.dc[3].weight, self.dc[3].bias,
#                                      self.ffn[0].weight, self.ffn[0].bias,
#                                      self.ffn[2].weight, self.ffn[2].bias,
#                                      self.shortcut)

#         if quant_step is not None:
#             return self.proxy.forward_with_quant_step(x, quant_step)
#         if to_cat is not None:
#             return self.proxy.forward_with_cat(x, to_cat, cat_at_front)

#         return self.proxy.forward(x)


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
