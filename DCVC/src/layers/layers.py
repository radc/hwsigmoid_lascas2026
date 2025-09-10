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

import atexit
import threading
import torch
import torch.nn as nn
import math


class WSiLU(nn.Module):
    def __init__(
        self,
        alpha: float = 4.0,
        log_path: str = "wsilu_inputs.log",
        buffer_capacity: int = 4,
        buffer_device: str | torch.device = "cuda:0",
        enable_logging: bool = True,
        sample_ratio: float = 0.01,     # <<< fração amostrada por flush (p.ex., 5%)
        min_samples: int = 1,           # <<< garante pelo menos N amostras por flush
        random_seed: int | None = None  # <<< opcional: reprodutibilidade
    ):
        """
        y = x * sigmoid(alpha * x)

        buffer_capacity: quantidade de batches acumulados antes de escrever
        buffer_device: dispositivo do buffer (ex.: 'cuda:0')
        sample_ratio: fração do buffer a ser gravada no disco a cada flush (0..1)
        min_samples: mínimo de amostras por flush (>=0)
        random_seed: se definido, torna a amostragem reprodutível
        """
        super().__init__()
        self.alpha = alpha
        self.log_path = log_path
        self.buffer_capacity = int(buffer_capacity)
        self.buffer_device = torch.device(buffer_device)
        self.enable_logging = enable_logging
        self.sample_ratio = float(sample_ratio)
        self.min_samples = int(min_samples)
        self.random_seed = random_seed

        self._buffer: list[torch.Tensor] = []  # batches residentes no buffer_device
        self._lock = threading.Lock()
        atexit.register(self.flush)

        # Gerador RNG opcional para amostragem reprodutível
        self._g = None
        if self.random_seed is not None:
            self._g = torch.Generator(device=self.buffer_device)
            self._g.manual_seed(self.random_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_logging:
            self._append_to_buffer(x)
        return x * torch.sigmoid(self.alpha * x)

    @torch.no_grad()
    def _append_to_buffer(self, x: torch.Tensor):
        x_buf = x.detach()
        if x_buf.device != self.buffer_device:
            x_buf = x_buf.to(self.buffer_device, non_blocking=True)
        with self._lock:
            self._buffer.append(x_buf)
            if len(self._buffer) >= self.buffer_capacity:
                self._flush_locked()

    @torch.no_grad()
    def flush(self):
        with self._lock:
            self._flush_locked()

    def _flush_locked(self):
        if not self._buffer:
            return

        # Empilha os batches do buffer na GPU (ou device escolhido)
        chunk = torch.stack(self._buffer, dim=0)  # shape: (N, *input_shape)
        N = chunk.shape[0]

        # Calcula quantas amostras salvar
        k = max(math.ceil(N * max(0.0, min(1.0, self.sample_ratio))), self.min_samples)
        k = min(k, N)

        # Amostragem sem reposição
        if k == N:
            sampled = chunk
        else:
            # Permutação/índices no mesmo device do buffer
            if self._g is not None:
                idx = torch.randperm(N, generator=self._g, device=self.buffer_device)[:k]
            else:
                idx = torch.randperm(N, device=self.buffer_device)[:k]
            sampled = chunk.index_select(0, idx)

        # Grava amostra no arquivo único (append binário)
        with open(self.log_path, "ab") as f:
            torch.save(sampled, f)

        # Limpa o buffer
        self._buffer.clear()

    # Utilidades
    def set_logging(self, enabled: bool):
        self.enable_logging = bool(enabled)

    def set_sample_ratio(self, ratio: float):
        self.sample_ratio = float(ratio)

    def set_min_samples(self, n: int):
        self.min_samples = int(n)

    def set_seed(self, seed: int | None):
        self.random_seed = seed
        if seed is None:
            self._g = None
        else:
            self._g = torch.Generator(device=self.buffer_device)
            self._g.manual_seed(seed)

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
