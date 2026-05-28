#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, time, json, threading, csv, gc
import torch
import torch.nn.functional as F

# ---- NVML import com fallback ----
# Pacote recomendado: nvidia-ml-py (expõe o módulo pynvml)
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage
    )
except Exception:
    # fallback explícito para o mesmo nome de símbolos
    import importlib
    _nv = importlib.import_module("nvidia_ml_py3")  # em alguns envs o nome é esse
    nvmlInit = _nv.nvmlInit
    nvmlShutdown = _nv.nvmlShutdown
    nvmlDeviceGetHandleByIndex = _nv.nvmlDeviceGetHandleByIndex
    nvmlDeviceGetPowerUsage = _nv.nvmlDeviceGetPowerUsage

PATTERN = re.compile(r"id38_dc_0_step(\d+)_input\.pt$")


# -------- util --------
def find_input_files(input_dir):
    files = []
    for fn in os.listdir(input_dir):
        m = PATTERN.match(fn)
        if m:
            n = int(m.group(1))
            files.append((n, os.path.join(input_dir, fn)))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


# -------- NVML sampler --------
class GPUPowerSampler:
    def __init__(self, gpu_index=0, interval=0.2):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = None
        self.handle = None

    def start(self):
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._t0 = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        t_prev = time.time()
        while not self._stop.is_set():
            p_mw = nvmlDeviceGetPowerUsage(self.handle)  # mW
            now = time.time()
            self.samples.append((now - self._t0, p_mw / 1000.0))
            sleep_left = self.interval - (now - t_prev)
            t_prev = now
            if sleep_left > 0:
                time.sleep(sleep_left)

    def stop(self):
        # amostra final
        p_mw = nvmlDeviceGetPowerUsage(self.handle)
        now = time.time()
        self.samples.append((now - self._t0, p_mw / 1000.0))
        self._stop.set()
        if self._thread:
            self._thread.join()
        nvmlShutdown()

    def energy_joules(self):
        if len(self.samples) < 2:
            return 0.0
        e = 0.0
        for i in range(len(self.samples) - 1):
            t0, p0 = self.samples[i]
            t1, p1 = self.samples[i + 1]
            dt = t1 - t0
            e += (p0 + p1) * 0.5 * dt
        return e


# -------- pré-carregamento --------
def try_preload_tensors(file_list, device, dtype, verbose=True):
    """
    Tenta mover o máximo possível dos tensores para a VRAM.
    Retorna:
      gpu_inputs: lista [(name, tensor_cuda), ...]
      cpu_inputs: lista dos caminhos restantes (não couberam)
    """
    gpu_inputs = []
    remaining_paths = []
    for i, path in enumerate(file_list):
        try:
            x = torch.load(path, map_location="cpu")
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"{path} não contém um tensor PyTorch")
            # normaliza para (N,C,H,W)
            if x.dim() == 3:  # (C,H,W)
                x = x.unsqueeze(0)
            elif x.dim() != 4:
                raise ValueError(f"Entrada deve ser (N,C,H,W) ou (C,H,W), veio {tuple(x.shape)}")

            # move para GPU com dtype pedido
            x = x.to(device=device, dtype=dtype, non_blocking=True)
            torch.cuda.synchronize(device=device)
            gpu_inputs.append((os.path.basename(path), x))

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if verbose:
                    print(f"[PRELOAD] Sem VRAM suficiente após {len(gpu_inputs)} tensores. "
                          f"Demais ficarão em streaming.")
                # limpa resquícios e adiciona o atual+restante como streaming
                try:
                    del x
                except Exception:
                    pass
                torch.cuda.empty_cache()
                remaining_paths = file_list[i:]  # inclui o atual que falhou
                break
            else:
                raise

    # Se coube tudo, remaining_paths fica vazio mesmo.
    if verbose:
        total = len(file_list)
        print(f"[PRELOAD] {len(gpu_inputs)}/{total} tensores pré-carregados na VRAM.")
    return gpu_inputs, remaining_paths


# -------- processamento: pointwise 1x1 conv --------
def process_pointwise(inputs_gpu, inputs_cpu_paths, W, b, device, keep_outputs=False, out_dir=None):
    """
    Executa conv2d 1x1 com pesos/bias dados, sobre:
      - 'inputs_gpu': lista [(name, tensor_cuda)]
      - 'inputs_cpu_paths': lista [path, ...] que serão streamados
    """
    if keep_outputs and out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    meta = []

    # 1) Tensores já na GPU
    for name, x in inputs_gpu:
        inC = W.shape[1]
        if x.shape[1] != inC:
            raise ValueError(f"Canais não batem (GPU): input C={x.shape[1]} vs pesos inC={inC} ({name})")
        y = F.conv2d(x, W, b, stride=1, padding=0, dilation=1, groups=1)
        torch.cuda.synchronize(device=device)
        if keep_outputs:
            base = name.replace("_input.pt", "_conv.pt")
            torch.save(y.detach().cpu(), os.path.join(out_dir, base))
        meta.append({"input": name, "output_shape": tuple(y.shape)})
        del y
        torch.cuda.empty_cache()

    # 2) Streaming dos que não couberam
    for path in inputs_cpu_paths:
        name = os.path.basename(path)
        x = torch.load(path, map_location="cpu")
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Entrada deve ser (N,C,H,W) ou (C,H,W), veio {tuple(x.shape)}")
        x = x.to(device=device, dtype=W.dtype, non_blocking=True)

        inC = W.shape[1]
        if x.shape[1] != inC:
            raise ValueError(f"Canais não batem (CPU->GPU): input C={x.shape[1]} vs pesos inC={inC} ({name})")
        y = F.conv2d(x, W, b, stride=1, padding=0, dilation=1, groups=1)
        torch.cuda.synchronize(device=device)
        if keep_outputs:
            base = name.replace("_input.pt", "_conv.pt")
            torch.save(y.detach().cpu(), os.path.join(out_dir, base))
        meta.append({"input": name, "output_shape": tuple(y.shape)})

        del x, y
        torch.cuda.empty_cache()
    return meta


def main():
    ap = argparse.ArgumentParser(
        description="Pointwise 1x1 Conv2d (DCVC-RT) com pré-carregamento de tensores e medição de energia via NVML."
    )
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--weights", required=True, help="weights.pt (outC,inC,1,1) ou (outC,inC)")
    ap.add_argument("--bias", required=True, help="bias.pt (outC,)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=0.2, help="Amostragem de potência (s)")
    ap.add_argument("--dtype", choices=["float16","float32","bfloat16"], default="float16")
    ap.add_argument("--keep_outputs", action="store_true")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--power_csv", default="gpu_power_log.csv")
    ap.add_argument("--no_warmup", action="store_true", help="Desativa warm-up (por padrão faz warm-up fora da medição)")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)

    torch_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }[args.dtype]

    # --- carregar pesos/bias (pointwise 1x1) ---
    W = torch.load(args.weights, map_location="cpu")
    b = torch.load(args.bias, map_location="cpu")
    if not (isinstance(W, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise ValueError("weights.pt e bias.pt devem conter tensores.")
    # Aceita (outC,inC,1,1) – ou (outC,inC) e expande
    if W.dim() == 2:
        W = W.unsqueeze(-1).unsqueeze(-1)
    if W.dim() != 4 or W.shape[-2:] != (1, 1):
        raise ValueError(
            f"Esperado weights (outC,inC,1,1) (ou (outC,inC) que será expandido), recebido {tuple(W.shape)}"
        )
    if b.dim() != 1 or b.shape[0] != W.shape[0]:
        raise ValueError(f"bias shape inválido: {tuple(b.shape)} vs outC={W.shape[0]}")

    W = W.to(device=device, dtype=torch_dtype, non_blocking=True)
    b = b.to(device=device, dtype=torch_dtype, non_blocking=True)

    # --- localizar entradas ---
    files = find_input_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"Nenhum id38_dc_0_stepN_input.pt em {args.input_dir}")

    # --- PRÉ-CARREGAMENTO (fora da medição) ---
    gpu_inputs, remaining_paths = try_preload_tensors(files, device=device, dtype=torch_dtype, verbose=True)

    # --- WARM-UP (fora da medição) ---
    if not args.no_warmup:
        inC = W.shape[1]
        dummy = torch.zeros((1, inC, 8, 8), device=device, dtype=torch_dtype)
        _ = F.conv2d(dummy, W, b, stride=1, padding=0, dilation=1, groups=1)
        torch.cuda.synchronize(device=device)
        del dummy
        torch.cuda.empty_cache()

    # --- MEDIÇÃO: somente o loop de convolução ---
    sampler = GPUPowerSampler(gpu_index=args.gpu, interval=args.interval)
    sampler.start()
    t0 = time.time()

    meta = process_pointwise(
        inputs_gpu=gpu_inputs,
        inputs_cpu_paths=remaining_paths,
        W=W, b=b, device=device,
        keep_outputs=args.keep_outputs,
        out_dir=args.out_dir if args.keep_outputs else None
    )

    torch.cuda.synchronize(device=device)
    t1 = time.time()
    sampler.stop()

    energia_J = sampler.energy_joules()
    energia_Wh = energia_J / 3600.0
    duracao_s = t1 - t0
    pot_media_W = energia_J / max(duracao_s, 1e-9)

    # salvar CSV
    with open(args.power_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tempo_s", "potencia_W"])
        w.writerows(sampler.samples)

    resumo = {
        "arquivos_total": len(files),
        "preload_gpu": len(gpu_inputs),
        "streaming_cpu": len(remaining_paths),
        "conv": "pointwise_1x1_conv2d",
        "inC": int(W.shape[1]),
        "outC": int(W.shape[0]),
        "dtype": str(W.dtype).replace("torch.", ""),
        "duracao_s": duracao_s,
        "energia_J": energia_J,
        "energia_Wh": energia_Wh,
        "pot_media_W": pot_media_W,
        "saidas": meta[:5] + ([{"...": "..."}] if len(meta) > 5 else [])
    }
    print(json.dumps(resumo, indent=2))
    print(f"\nAmostras de potência salvas em: {args.power_csv}")

    # limpeza
    del gpu_inputs, W, b
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
