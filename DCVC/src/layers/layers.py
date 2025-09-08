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

import torch
import numpy as np
import struct

from typing import List, Tuple

#LUT SYMETRIC
# class WSiLU(torch.nn.Module):
#     """
#     wsilu_lut_un: usa LUT uniforme em |x|<=2.0 e clampa fora.
#     Saída = x * sigmoid_aprox(|x|), com simetria para x<0.
#     A LUT é registrada como buffer não-persistente (não entra no state_dict).
#     """
#     def __init__(self, device=None, dtype=torch.float16):
#         super().__init__()

#         # ---- monta LUT (float16) a partir dos hex fornecidos ----
#         lut_hex = [
#             0x3800, 0x380c, 0x3818, 0x3824, 0x3830, 0x383c, 0x3848, 0x3854, 0x3860, 0x386c, 0x3878, 0x3884, 0x388f, 0x389b, 0x38a7,
#             0x38b3, 0x38be, 0x38ca, 0x38d5, 0x38e1, 0x38ec, 0x38f7, 0x3903, 0x390e, 0x3919, 0x3924, 0x392f, 0x393a, 0x3945, 0x3950,
#             0x395a, 0x3965, 0x3970, 0x397a, 0x3984, 0x398f, 0x3999, 0x39a3, 0x39ad, 0x39b7, 0x39c0, 0x39ca, 0x39d4, 0x39dd, 0x39e7,
#             0x39f0, 0x39f9, 0x3a02, 0x3a0b, 0x3a14, 0x3a1c, 0x3a25, 0x3a2e, 0x3a36, 0x3a3e, 0x3a46, 0x3a4f, 0x3a57, 0x3a5e, 0x3a66,
#             0x3a6e, 0x3a75, 0x3a7d, 0x3a84, 0x3a8b, 0x3a92, 0x3a99, 0x3aa0, 0x3aa7, 0x3aae, 0x3ab4, 0x3abb, 0x3ac1, 0x3ac7, 0x3ace,
#             0x3ad4, 0x3ada, 0x3ae0, 0x3ae5, 0x3aeb, 0x3af1, 0x3af6, 0x3afb, 0x3b01, 0x3b06, 0x3b0b, 0x3b10, 0x3b15, 0x3b1a, 0x3b1f,
#             0x3b23, 0x3b28, 0x3b2c, 0x3b31, 0x3b35, 0x3b39, 0x3b3e, 0x3b42, 0x3b46, 0x3b4a, 0x3b4d, 0x3b51, 0x3b55, 0x3b59, 0x3b5c,
#             0x3b60, 0x3b63, 0x3b66, 0x3b6a, 0x3b6d, 0x3b70, 0x3b73, 0x3b76, 0x3b79, 0x3b7c, 0x3b7f, 0x3b82, 0x3b85, 0x3b87, 0x3b8a,
#             0x3b8d, 0x3b8f, 0x3b92, 0x3b94, 0x3b96, 0x3b99, 0x3b9b, 0x3b9d, 0x3b9f, 0x3ba2, 0x3ba4, 0x3ba6, 0x3ba8, 0x3baa, 0x3bac,
#             0x3bad, 0x3baf, 0x3bb1, 0x3bb3, 0x3bb5, 0x3bb6, 0x3bb8, 0x3bba, 0x3bbb, 0x3bbd, 0x3bbe, 0x3bc0, 0x3bc1, 0x3bc3, 0x3bc4,
#             0x3bc5, 0x3bc7, 0x3bc8, 0x3bc9, 0x3bca, 0x3bcc, 0x3bcd, 0x3bce, 0x3bcf, 0x3bd0, 0x3bd1, 0x3bd2, 0x3bd3, 0x3bd4, 0x3bd5,
#             0x3bd6, 0x3bd7, 0x3bd8, 0x3bd9, 0x3bda, 0x3bdb, 0x3bdc, 0x3bdd, 0x3bdd, 0x3bde, 0x3bdf, 0x3be0, 0x3be0, 0x3be1, 0x3be2,
#             0x3be3, 0x3be3, 0x3be4, 0x3be5, 0x3be5, 0x3be6, 0x3be6, 0x3be7, 0x3be8, 0x3be8, 0x3be9, 0x3be9, 0x3bea, 0x3bea, 0x3beb,
#             0x3beb, 0x3bec, 0x3bec, 0x3bed, 0x3bed, 0x3bed, 0x3bee, 0x3bee, 0x3bef, 0x3bef, 0x3bf0, 0x3bf0, 0x3bf0, 0x3bf1, 0x3bf1,
#             0x3bf1, 0x3bf2, 0x3bf2, 0x3bf2, 0x3bf3, 0x3bf3, 0x3bf3, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf5, 0x3bf5, 0x3bf5, 0x3bf5,
#             0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf9,
#             0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfb, 0x3bfb, 0x3bfb,
#             0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc,
#             0x3bfc, 0x3bfc, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd,
#             0x3bfd, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe,
#             0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#             0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#             0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#             0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00, 0x3c00,
#             0x3c00, 0x3c00,
#         ]
#         lut_f16 = np.array([struct.unpack('>e', struct.pack('>H', v))[0] for v in lut_hex], dtype=np.float16)
#         lut = torch.from_numpy(lut_f16)

#         # buffer NÃO persistente (não entra no state_dict)
#         self.register_buffer("lut", lut, persistent=False)

#         # ---- constantes ----
#         self.LIMIT = torch.tensor(2.0, dtype=dtype)   # limite uniforme
#         self.LUT_SIZE = lut.numel()
#         self.out_dtype = dtype

#         if device is not None:
#             self.to(device)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # garante device compatível (por segurança)
#         if self.lut.device != x.device:
#             self.lut = self.lut.to(x.device)

#         x_q = x.to(self.out_dtype)
#         abs_x = x_q.abs()

#         # mapeamento linear de |x| em [0, LIMIT] para índice [0, LUT_SIZE-1]
#         # fora do limite, clampa em LUT_SIZE-1
#         # idx = round_down( (|x|/LIMIT) * (LUT_SIZE-1) )
#         scale = (abs_x.to(torch.float32) / self.LIMIT.to(torch.float32)) * float(self.LUT_SIZE - 1)
#         idx = scale.to(torch.long).clamp(0, self.LUT_SIZE - 1)

#         # pega valor da sigmoide aproximada e aplica simetria
#         sig = self.lut[idx].to(self.out_dtype)         # \in (0.5..1] ~ metade superior
#         sig = torch.where(x_q < 0, 1.0 - sig, sig)     # simetria para x<0

#         return (x_q * sig).to(self.out_dtype)



# LUT ASYMETRIC
# class WSiLU(torch.nn.Module):
#     def __init__(self, device=None, dtype=torch.float16):
#         super().__init__()
#         # --- LUT em meia precisão (float16) a partir dos códigos hex fornecidos ---
#         lut_hex = [
#             0x3800, 0x3806, 0x380d, 0x3813, 0x381a, 0x3820, 0x3826, 0x382d, 0x3833, 0x383a, 0x3840, 0x3846,
#             0x384d, 0x3853, 0x3859, 0x3860, 0x3866, 0x386c, 0x3873, 0x3879, 0x387f, 0x3886, 0x388c, 0x3892,
#             0x3898, 0x389f, 0x38a5, 0x38ab, 0x38b1, 0x38b8, 0x38be, 0x38c4, 0x38ca, 0x38d0, 0x38d6, 0x38dc,
#             0x38e3, 0x38e9, 0x38ef, 0x38f5, 0x38fb, 0x3901, 0x3907, 0x390d, 0x3913, 0x3919, 0x391f, 0x3924,
#             0x392a, 0x3930, 0x3936, 0x393c, 0x3942, 0x3947, 0x394d, 0x3953, 0x3958, 0x395e, 0x3964, 0x3969,
#             0x396f, 0x3975, 0x397a, 0x3980, 0x3985, 0x398b, 0x3990, 0x3995, 0x399b, 0x39a0, 0x39a5, 0x39ab,
#             0x39b0, 0x39b5, 0x39ba, 0x39c0, 0x39c5, 0x39ca, 0x39cf, 0x39d4, 0x39d9, 0x39de, 0x39e3, 0x39e8,
#             0x39ed, 0x39f2, 0x39f7, 0x39fc, 0x3a01, 0x3a05, 0x3a0a, 0x3a0f, 0x3a13, 0x3a18, 0x3a1d, 0x3a21,
#             0x3a26, 0x3a2a, 0x3a2f, 0x3a33, 0x3a38, 0x3a3c, 0x3a41, 0x3a45, 0x3a49, 0x3a4e, 0x3a52, 0x3a56,
#             0x3a5a, 0x3a5f, 0x3a63, 0x3a67, 0x3a6b, 0x3a6f, 0x3a73, 0x3a77, 0x3a7b, 0x3a7f, 0x3a83, 0x3a87,
#             0x3a8a, 0x3a8e, 0x3a92, 0x3a96, 0x3a99, 0x3a9d, 0x3aa1, 0x3aa4, 0x3aa8, 0x3aac, 0x3aaf, 0x3ab3,
#             0x3ab6, 0x3ab9, 0x3abd, 0x3ac0, 0x3ac4, 0x3ac7, 0x3aca, 0x3ace, 0x3ad1, 0x3ad4, 0x3ad7, 0x3ada,
#             0x3add, 0x3ae1, 0x3ae4, 0x3ae7, 0x3aea, 0x3aed, 0x3af0, 0x3af3, 0x3af6, 0x3af8, 0x3afb, 0x3afe,
#             0x3b01, 0x3b04, 0x3b06, 0x3b09, 0x3b0c, 0x3b0f, 0x3b11, 0x3b14, 0x3b16, 0x3b19, 0x3b1c, 0x3b1e,
#             0x3b21, 0x3b23, 0x3b25, 0x3b28, 0x3b2a, 0x3b2d, 0x3b2f, 0x3b31, 0x3b34, 0x3b36, 0x3b38, 0x3b3b,
#             0x3b3d, 0x3b3f, 0x3b41, 0x3b43, 0x3b45, 0x3b47, 0x3b4a, 0x3b4c, 0x3b4e, 0x3b50, 0x3b52, 0x3b54,
#             0x3b56, 0x3b58, 0x3b5a, 0x3b5b, 0x3b5d, 0x3b5f, 0x3b61, 0x3b63, 0x3b65, 0x3b66, 0x3b68, 0x3b6a,
#             0x3b6c, 0x3b6d, 0x3b6f, 0x3b71, 0x3b72, 0x3b74, 0x3b76, 0x3b77, 0x3b79, 0x3b7a, 0x3b7c, 0x3b7e,
#             0x3b7f, 0x3b81, 0x3b82, 0x3b83, 0x3b85, 0x3b86, 0x3b88, 0x3b89, 0x3b8b, 0x3b8c, 0x3b8d, 0x3b8f,
#             0x3b90, 0x3b91, 0x3b93, 0x3b94, 0x3b95, 0x3b96, 0x3b98, 0x3b99, 0x3b9a, 0x3b9b, 0x3b9d, 0x3b9e,
#             0x3b9f, 0x3ba0, 0x3ba1, 0x3ba2, 0x3ba3, 0x3ba4, 0x3ba6, 0x3ba7, 0x3ba8, 0x3ba9, 0x3baa, 0x3bab,
#             0x3bac, 0x3bad, 0x3bae, 0x3baf, 0x3bb0, 0x3bb1, 0x3bb2, 0x3bb3, 0x3bb4, 0x3bb4, 0x3bb5, 0x3bb6,
#             0x3bb7, 0x3bb8, 0x3bb9, 0x3bba, 0x3bbb, 0x3bbb, 0x3bbc, 0x3bbd, 0x3bbe, 0x3bbf, 0x3bbf, 0x3bc0,
#             0x3bc1, 0x3bc2, 0x3bc2, 0x3bc3, 0x3bc4, 0x3bc5, 0x3bc5, 0x3bc6, 0x3bc7, 0x3bc8, 0x3bc8, 0x3bc9,
#             0x3bca, 0x3bca, 0x3bcb, 0x3bcb, 0x3bcc, 0x3bcd, 0x3bcd, 0x3bce, 0x3bcf, 0x3bcf, 0x3bd0, 0x3bd0,
#             0x3bd1, 0x3bd2, 0x3bd2, 0x3bd3, 0x3bd3, 0x3bd4, 0x3bd4, 0x3bd5, 0x3bd5, 0x3bd6, 0x3bd6, 0x3bd7,
#             0x3bd7, 0x3bd8, 0x3bd8, 0x3bd9, 0x3bd9, 0x3bda, 0x3bda, 0x3bdb, 0x3bdb, 0x3bdc, 0x3bdc, 0x3bdc,
#             0x3bdd, 0x3bdd, 0x3bde, 0x3bde, 0x3bdf, 0x3bdf, 0x3bdf, 0x3be0, 0x3be0, 0x3be1, 0x3be1, 0x3be1,
#             0x3be2, 0x3be2, 0x3be2, 0x3be3, 0x3be3, 0x3be4, 0x3be4, 0x3be4, 0x3be5, 0x3be5, 0x3be5, 0x3be6,
#             0x3be6, 0x3be6, 0x3be7, 0x3be7, 0x3be7, 0x3be7, 0x3be8, 0x3be8, 0x3be8, 0x3be9, 0x3be9, 0x3be9,
#             0x3be9, 0x3bea, 0x3bea, 0x3bea, 0x3beb, 0x3beb, 0x3beb, 0x3beb, 0x3bec, 0x3bec, 0x3bec, 0x3bec,
#             0x3bed, 0x3bed, 0x3bed, 0x3bed, 0x3bee, 0x3bee, 0x3bee, 0x3bee, 0x3bee, 0x3bef, 0x3bef, 0x3bef,
#             0x3bef, 0x3bef, 0x3bf0, 0x3bf0, 0x3bf0, 0x3bf0, 0x3bf0, 0x3bf1, 0x3bf1, 0x3bf1, 0x3bf1, 0x3bf1,
#             0x3bf2, 0x3bf2, 0x3bf2, 0x3bf2, 0x3bf2, 0x3bf2, 0x3bf3, 0x3bf3, 0x3bf3, 0x3bf3, 0x3bf3, 0x3bf3,
#             0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf4, 0x3bf5, 0x3bf5, 0x3bf5, 0x3bf5, 0x3bf5,
#             0x3bf5, 0x3bf5, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf6, 0x3bf7, 0x3bf7,
#             0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf7, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8,
#             0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf8, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9,
#             0x3bf9, 0x3bf9, 0x3bf9, 0x3bf9, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa,
#             0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfa, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb, 0x3bfb,
#             0x3bfb, 0x3bfb, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfc, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfd, 0x3bfe,
#             0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bfe, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#             0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff, 0x3bff,
#         ]
#         lut_f16 = np.array([struct.unpack('>e', struct.pack('>H', v))[0] for v in lut_hex], dtype=np.float16)
#         lut = torch.from_numpy(lut_f16).to(device=device)
#         self.register_buffer("lut", lut, persistent=False)  # fica no device certo e vai junto no state_dict

#         # --- Constantes ---
#         self.DENSE_LIMIT = torch.tensor(1.5, dtype=dtype, device=device)
#         self.SPARSE_LIMIT = torch.tensor(2.0, dtype=dtype, device=device)
#         self.N_DENSE = torch.tensor(448.0, dtype=torch.float32, device=device)   # usar float p/ aritmética
#         self.N_SPARSE = torch.tensor(64.0, dtype=torch.float32, device=device)
#         self.LUT_SIZE = len(lut)
#         self.NEG_LIMIT = torch.tensor(-2.0, dtype=dtype, device=device)
#         self.out_dtype = dtype

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # Tudo elemento a elemento, vetorizado
#         x_q = x.to(self.out_dtype)
#         abs_x = x_q.abs()

#         # Mapear |x| -> índice da LUT (região densa, depois esparsa; senão clampa no fim)
#         mask_dense = abs_x <= self.DENSE_LIMIT
#         mask_sparse = (abs_x > self.DENSE_LIMIT) & (abs_x <= self.SPARSE_LIMIT)

#         # Começa assumindo "fora do limite superior"
#         scaled = torch.full_like(abs_x, float(self.LUT_SIZE - 1), dtype=torch.float32)

#         # Região densa: [0, 1.5]
#         scaled = torch.where(
#             mask_dense,
#             (abs_x.to(torch.float32) / self.DENSE_LIMIT.to(torch.float32)) * self.N_DENSE,
#             scaled,
#         )

#         # Região esparsa: (1.5, 2.0]
#         scaled = torch.where(
#             mask_sparse,
#             self.N_DENSE + ((abs_x.to(torch.float32) - self.DENSE_LIMIT.to(torch.float32)) /
#                             (self.SPARSE_LIMIT.to(torch.float32) - self.DENSE_LIMIT.to(torch.float32))) * self.N_SPARSE,
#             scaled,
#         )

#         idx = scaled.to(torch.long).clamp(0, self.LUT_SIZE - 1)

#         # Sigmóide aproximada pela LUT (metade superior), com simetria para x<0
#         sig = self.lut[idx]                      # float16
#         sig = torch.where(x_q < 0, 1.0 - sig, sig)

#         # Regra especial: x < -2.0 -> 0
#         y_sigmoid = torch.where(x_q < self.NEG_LIMIT, torch.zeros_like(x_q), sig)

#         # wsilu: x * sigmoid_aproximada
#         return (x_q * y_sigmoid).to(self.out_dtype)


# class WSiLU(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         y = torch.sigmoid(4.0 * x) * x
#         return y

# Uniform-noise addition
class WSiLU(nn.Module):
    # ===== Class-level (static) attributes =====
    _noise_enabled: bool = False
    _noise_amp: float = 1e-8
    _noise_intervals: List[Tuple[float, float]] = []  # closed intervals [low, high]

    def __init__(self):
        super().__init__()

    # ===== Static methods (enable/disable) =====
    @staticmethod
    def enable_noise() -> None:
        """Globally enable noise injection."""
        WSiLU._noise_enabled = True

    @staticmethod
    def disable_noise() -> None:
        """Globally disable noise injection."""
        WSiLU._noise_enabled = False

    @staticmethod
    def is_noise_enabled() -> bool:
        """Check if noise is globally enabled."""
        return WSiLU._noise_enabled

    # ===== Static methods (getters/setters) =====
    @staticmethod
    def set_noise_amp(amp: float) -> None:
        """Set the global noise amplitude."""
        if amp < 0:
            raise ValueError("noise_amp must be >= 0.")
        WSiLU._noise_amp = float(amp)

    @staticmethod
    def get_noise_amp() -> float:
        """Return the global noise amplitude."""
        return WSiLU._noise_amp

    @staticmethod
    def set_noise_intervals(intervals: List[Tuple[float, float]]) -> None:
        """Set the global list of intervals [low, high] where noise will be applied."""
        checked = []
        for low, high in intervals:
            low = float(low); high = float(high)
            if high < low:
                raise ValueError(f"Invalid interval: ({low}, {high})")
            checked.append((low, high))
        WSiLU._noise_intervals = checked

    @staticmethod
    def add_noise_interval(low: float, high: float) -> None:
        """Add a single interval [low, high] to the global list of intervals."""
        if high < low:
            raise ValueError(f"Invalid interval: ({low}, {high})")
        WSiLU._noise_intervals.append((float(low), float(high)))

    @staticmethod
    def clear_noise_intervals() -> None:
        """Remove all noise intervals."""
        WSiLU._noise_intervals = []

    @staticmethod
    def get_noise_intervals() -> List[Tuple[float, float]]:
        """Return the current global list of noise intervals."""
        return list(WSiLU._noise_intervals)

    # ===== Forward pass =====
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base SiLU-like activation
        y = torch.sigmoid(4.0 * x) * x

        # Conditions where no noise is applied
        if (not WSiLU._noise_enabled) or (WSiLU._noise_amp == 0.0) or (len(WSiLU._noise_intervals) == 0):
            return y

        # Boolean mask: True where x falls into any defined interval
        mask = torch.zeros_like(x, dtype=torch.bool)
        for low, high in WSiLU._noise_intervals:
            mask |= (x >= low) & (x <= high)

        if mask.any():
            noise = torch.zeros_like(y)
            noise[mask] = torch.empty_like(y[mask]).uniform_(-WSiLU._noise_amp, WSiLU._noise_amp)
            y = y + noise

        return y
    
    # def forward_polinomial(self, x): #GRAU 6
    # def forward(self, x):
    #     # parte polinomial
    #     poly = (
    #         0.004546
    #         + 0.5 * x
    #         + 0.859845 * x**2
    #         - 0.553798 * x**4
    #         + 0.168839 * x**6
    #     )

    #     # aplica as condições em cada ponto
    #     return torch.where(
    #         x < -1.2,
    #         torch.zeros_like(x),  # se x < -1.2
    #         torch.where(x > 1.2, x, poly)  # se x > 1.2 → x, senão → poly
    #     )

    # def wsilu_polinomial_forward(x: torch.Tensor) -> torch.Tensor: #GRAU 16
    # def forward(self, x) :
    #     x2 = x * x
    #     poly_even = (
    #         (((((((( -0.00047333 * x2 + 0.0083775) * x2 - 0.06236471) * x2
    #                 + 0.25503359) * x2 - 0.63088907) * x2 + 0.99181597) * x2
    #             - 1.04954556) * x2 + 0.96917012) * x2 + 0.00059508
    #         )
    #     )
    #     poly = poly_even + 0.5 * x
    #     return torch.where(x < -2, torch.zeros_like(x),
    #         torch.where(x > 2, x, poly))
        
    


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
