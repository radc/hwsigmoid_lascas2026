# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from .cuda_inference import CUSTOMIZED_CUDA_INFERENCE
if CUSTOMIZED_CUDA_INFERENCE:
    from .cuda_inference import DepthConvProxy, SubpelConv2xProxy


from .layers_improved import TracedConv2d




import json
import os

from .wsilu_variants import (
    WSiLULUTAsyn4Int1024Entries,
    WSiLULUTAsyn4Int32Entries,
    WSiLULUTAsyn4Int64Entries,
    WSiLULUTAsyn4Int128Entries,
    WSiLULUTAsyn4Int256Entries,
    WSiLULUTAsyn4Int512Entries,
    WSiLUPoly1IntDeg11_16,
    WSiLUPoly25IntDeg2_32,
)

WSILU_TYPE = os.getenv("DCVC_WSILU_TYPE", "wsilu4").strip().lower()
WSILU_CONFIG_PATH = os.getenv("DCVC_WSILU_CONFIG", "").strip()


def _load_wsilu_config():
    if not WSILU_CONFIG_PATH:
        return {}
    with open(WSILU_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("DCVC_WSILU_CONFIG must point to a JSON object mapping module names to WSiLU types.")
    return {str(k): str(v).strip().lower() for k, v in cfg.items()}


WSILU_CONFIG = _load_wsilu_config()


class _WSiLUReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class _WSiLUSigmoid4(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(4.0 * x) * x


class _WSiLUSiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)


WSILU_IMPLS = {
    "relu": _WSiLUReLU,
    "wsilu": _WSiLUSigmoid4,
    "wsilu4": _WSiLUSigmoid4,
    "silu": _WSiLUSiLU,
    "lut_asyn_4int_32entries": WSiLULUTAsyn4Int32Entries,
    "lut_asyn_4int_64entries": WSiLULUTAsyn4Int64Entries,
    "lut_asyn_4int_128entries": WSiLULUTAsyn4Int128Entries,
    "lut_asyn_4int_256entries": WSiLULUTAsyn4Int256Entries,
    "lut_asyn_4int_512entries": WSiLULUTAsyn4Int512Entries,
    "lut_asyn_4int_1024entries": WSiLULUTAsyn4Int1024Entries,
    "poly_25int_deg2_32": WSiLUPoly25IntDeg2_32,
    "poly_1int_deg11_16": WSiLUPoly1IntDeg11_16,
}


class WSiLU(nn.Module):
    def __init__(self, module_name=None):
        super().__init__()
        wsilu_type = WSILU_TYPE
        if module_name:
            if module_name in WSILU_CONFIG:
                wsilu_type = WSILU_CONFIG[module_name]
            elif "default" in WSILU_CONFIG:
                wsilu_type = WSILU_CONFIG["default"]

        impl_cls = WSILU_IMPLS.get(wsilu_type)
        if impl_cls is None:
            raise ValueError(
                f"Unsupported WSiLU type={wsilu_type!r}. "
                f"Use one of: {', '.join(sorted(WSILU_IMPLS))}."
            )
        self.impl = impl_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.impl(x)


class WSiLUChunkAdd(nn.Module):
    def __init__(self, module_name=None):
        super().__init__()
        self.silu = WSiLU(module_name=module_name)

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
#                 WSiLU(module_name=f"{module_name}.dc" if module_name else None),
#                 nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
#                 nn.Conv2d(out_ch, out_ch, 1),
#             )

#         else:
#             self.dc = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch, 1),
#             WSiLU(module_name=f"{module_name}.dc" if module_name else None),
#             nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
#             nn.Conv2d(out_ch, out_ch, 1),
#         )

#         self.ffn = nn.Sequential(
#             nn.Conv2d(out_ch, out_ch * 4, 1),
#             WSiLUChunkAdd(module_name=f"{module_name}.ffn" if module_name else None),
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
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False, module_name=None):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(module_name=f"{module_name}.dc" if module_name else None),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1),
            WSiLUChunkAdd(module_name=f"{module_name}.ffn" if module_name else None),
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
    def __init__(self, in_ch, out_ch, module_name=None):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True, module_name=f"{module_name}.conv" if module_name else None)

    def forward(self, x):
        x = self.down(x)
        out = self.conv(x)
        return out


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, module_name=None):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True, module_name=f"{module_name}.conv" if module_name else None)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out
