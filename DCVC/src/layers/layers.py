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
import math
import json
import time
import os
from typing import Optional, Union

class WSiLU(nn.Module):
    def __init__(
        self,
        alpha: float = 4.0,
        # NOVOS caminhos:
        txt_pairs_path: str = "/home/ruhan/hwsigmoid_lascas2026/coding_outputs/wsilu_pairs.txt",
        txt_bins_path: str = "/home/ruhan/hwsigmoid_lascas2026/coding_outputs/wsilu_pairs_bin.txt",
        # Buffer / amostragem:
        buffer_capacity: int = 4,
        buffer_device: Union[str, torch.device] = "cuda:0",
        enable_logging: bool = True,
        sample_ratio: float = 0.01,
        min_samples: int = 1,
        log_prob: float = 0.001,  # prob. de armazenar um batch no buffer
        random_seed: Optional[int] = None,
        # Formatação:
        float_precision: Optional[int] = None,  # ex.: 6 -> 6 casas no TXT
        binary_format: str = "float16",  # "float16" | "float32" | "float64"
    ):
        """
        y = x * sigmoid(alpha * x)

        txt_pairs_path: arquivo texto com linhas "x y"
        txt_bins_path:  arquivo texto com linhas "<bin(x)> <bin(y)>"
        buffer_capacity: nº de batches acumulados antes de flush
        buffer_device: dispositivo do buffer (ex.: 'cuda:0')
        sample_ratio: fração do buffer gravada em cada flush (0..1)
        min_samples: mínimo de batches gravados por flush (>=0)
        log_prob: probabilidade de armazenar um batch no buffer (0..1)
        float_precision: se definido, arredonda/formatará os floats no TXT
        binary_format: precisão IEEE-754 para o arquivo binário (16/32/64)
        """
        super().__init__()
        self.alpha = float(alpha)

        self.txt_pairs_path = str(txt_pairs_path)
        self.txt_bins_path = str(txt_bins_path)

        self.buffer_capacity = int(buffer_capacity)
        self.buffer_device = torch.device(buffer_device)
        self.enable_logging = bool(enable_logging)
        self.sample_ratio = float(sample_ratio)
        self.min_samples = int(min_samples)
        self.random_seed = random_seed
        self.log_prob = float(log_prob)

        self.float_precision = float_precision
        self.binary_format = binary_format.lower()
        assert self.binary_format in {"float16", "float32", "float64"}

        self._buffer = []  # lista de tensores no buffer_device
        self._lock = threading.Lock()
        atexit.register(self.flush)

        # RNG opcional para reprodutibilidade
        self._g = None
        if self.random_seed is not None:
            self._g = torch.Generator(device=self.buffer_device)
            self._g.manual_seed(self.random_seed)

        # Cria pastas dos logs se necessário
        os.makedirs(os.path.dirname(self.txt_pairs_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.txt_bins_path), exist_ok=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.enable_logging and self._rand() < self.log_prob:
            self._append_to_buffer(x)
        return x * torch.sigmoid(self.alpha * x)

    # ---------- utilidades internas ----------

    def _rand(self) -> float:
        if self._g is not None:
            return torch.rand(1, generator=self._g, device=self.buffer_device).item()
        else:
            return torch.rand(1, device=self.buffer_device).item()

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

    # ---------- binários ----------

    @staticmethod
    def _float_to_bin(val: float, binary_format: str) -> str:
        """Converte float Python para string binária IEEE-754 (sem '0b')."""
        import struct

        if binary_format == "float16":
            # usar float32 -> float16 via numpy para manter IEEE-754 half
            import numpy as np
            arr = np.float16(val)
            # .tobytes() -> 2 bytes; formatar MSB..LSB
            b = arr.tobytes()
            # garantir ordem consistente (little-endian na maioria)
            u = int.from_bytes(b, byteorder="little", signed=False)
            return format(u, "016b")
        elif binary_format == "float32":
            b = struct.pack("<f", float(val))     # little-endian 32-bit float
            u = struct.unpack("<I", b)[0]
            return format(u, "032b")
        elif binary_format == "float64":
            b = struct.pack("<d", float(val))     # little-endian 64-bit double
            u = struct.unpack("<Q", b)[0]
            return format(u, "064b")
        else:
            raise ValueError("binary_format inválido")

    def _fmt_float(self, v: float) -> str:
        if self.float_precision is None:
            return str(float(v))
        return f"{float(v):.{int(self.float_precision)}f}"

    # ---------- flush lógico ----------

    def _flush_locked(self):
        if not self._buffer:
            return

        # Empilha (permanece no device do buffer)
        chunk = torch.stack(self._buffer, dim=0)  # (N, *shape)
        N = chunk.shape[0]

        # Amostragem de batches: seleciona k entre N
        k = max(math.ceil(N * max(0.0, min(1.0, self.sample_ratio))), self.min_samples)
        k = min(k, N)

        if k == N:
            sampled = chunk
        else:
            if self._g is not None:
                idx = torch.randperm(N, generator=self._g, device=self.buffer_device)[:k]
            else:
                idx = torch.randperm(N, device=self.buffer_device)[:k]
            sampled = chunk.index_select(0, idx)

        # Move para CPU para serialização e computa a saída correspondente
        x_cpu = sampled.detach().to("cpu")  # (k, *shape)
        # y = x * sigmoid(alpha*x) (no CPU, sem grad)
        y_cpu = x_cpu * torch.sigmoid(self.alpha * x_cpu)

        # Flattens (valores em sequência, independente da forma original)
        x_flat = x_cpu.reshape(-1).tolist()
        y_flat = y_cpu.reshape(-1).tolist()

        # (opcional) arredonda apenas para o TXT de pares
        if self.float_precision is not None:
            # a formatação é aplicada na escrita; listas mantêm float “cheio”
            pass

        # Abre arquivos em modo append texto
        with open(self.txt_pairs_path, "a", encoding="utf-8") as f_pairs, \
             open(self.txt_bins_path, "a", encoding="utf-8") as f_bins:

            # Escreve linha a linha: "x y" e "<bin(x)> <bin(y)>"
            for xv, yv in zip(x_flat, y_flat):
                # TXT “legível”:
                f_pairs.write(f"{self._fmt_float(xv)} {self._fmt_float(yv)}\n")

                # TXT “binários”:
                bx = self._float_to_bin(xv, self.binary_format)
                by = self._float_to_bin(yv, self.binary_format)
                f_bins.write(f"{bx} {by}\n")

        # Limpa buffer
        self._buffer.clear()

    # ---------- setters dinâmicos ----------
    def set_logging(self, enabled: bool):
        self.enable_logging = bool(enabled)

    def set_sample_ratio(self, ratio: float):
        self.sample_ratio = float(ratio)

    def set_min_samples(self, n: int):
        self.min_samples = int(n)

    def set_seed(self, seed: Optional[int]):
        self.random_seed = seed
        if seed is None:
            self._g = None
        else:
            self._g = torch.Generator(device=self.buffer_device)
            self._g.manual_seed(seed)

    def set_log_prob(self, p: float):
        self.log_prob = float(p)

    def set_binary_format(self, fmt: str):
        fmt = fmt.lower()
        assert fmt in {"float16", "float32", "float64"}
        self.binary_format = fmt



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
