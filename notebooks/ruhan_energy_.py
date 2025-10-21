#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Exemplo de setup:
# conda create -n energy python=3.10 -y
# conda activate energy
# pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
# pip install nvidia-ml-py

import argparse, os, re, time, json, threading, csv, gc
import torch
import torch.nn.functional as F

# ---- NVML import (nvidia-ml-py expõe pynvml) ----
try:
    from pynvml import (
        nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage
    )
except Exception:
    import importlib
    _nv = importlib.import_module("nvidia_ml_py3")
    nvmlInit = _nv.nvmlInit
    nvmlShutdown = _nv.nvmlShutdown
    nvmlDeviceGetHandleByIndex = _nv.nvmlDeviceGetHandleByIndex
    nvmlDeviceGetPowerUsage = _nv.nvmlDeviceGetPowerUsage

PATTERN = re.compile(r"id38_dc_0_step(\d+)_input\.pt$")


def find_input_files(input_dir):
    """Retorna lista ordenada de paths que batem com id38_dc_0_stepN_input.pt."""
    files = []
    for fn in os.listdir(input_dir):
        m = PATTERN.match(fn)
        if m:
            n = int(m.group(1))
            files.append((n, os.path.join(input_dir, fn)))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def measure_baseline_power(gpu_index=0, seconds=2.0, interval=0.05):
    """Mede potência média (W) da GPU por alguns segundos para servir de baseline."""
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_index)
    samples = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        p_mw = nvmlDeviceGetPowerUsage(handle)
        samples.append(p_mw / 1000.0)
        time.sleep(interval)
    nvmlShutdown()
    return (sum(samples) / len(samples)) if samples else 0.0


class GPUPowerSampler:
    """
    Sampler por intervalo que pode ser start/stop múltiplas vezes.
    Acumula amostras apenas das janelas medidas (ex.: convoluções) e “cola” as timelines.
    """
    def __init__(self, gpu_index=0, interval=0.2):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []   # lista global de (t_rel_s, W) só dos trechos medidos
        self._thread = None
        self._stop = threading.Event()
        self._t0 = None
        self._t_offset = 0.0
        self._handle = None
        self._running = False

    def _run(self):
        t_prev = time.time()
        while not self._stop.is_set():
            p_mw = nvmlDeviceGetPowerUsage(self._handle)  # mW
            now = time.time()
            self._current.append((now - self._t0, p_mw / 1000.0))
            sleep_left = self.interval - (now - t_prev)
            t_prev = now
            if sleep_left > 0:
                time.sleep(sleep_left)

    def start(self):
        if self._running:
            return
        if self._handle is None:
            nvmlInit()
            self._handle = nvmlDeviceGetHandleByIndex(self.gpu_index)
        self._stop.clear()
        self._t0 = time.time()
        self._current = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._running = True

    def stop(self):
        if not self._running:
            return
        # amostra final
        p_mw = nvmlDeviceGetPowerUsage(self._handle)
        now = time.time()
        self._current.append((now - self._t0, p_mw / 1000.0))
        # para thread
        self._stop.set()
        self._thread.join()
        self._thread = None
        self._running = False
        # cola na timeline global
        if self._current:
            t0_local = self._current[0][0]
            base = self._t_offset - t0_local  # t0_local ~ 0
            for t, w in self._current:
                self.samples.append((t + base, w))
            self._t_offset = self.samples[-1][0]
        self._current = None

    def shutdown(self):
        if self._running:
            self.stop()
        if self._handle is not None:
            nvmlShutdown()
            self._handle = None

    def energy_joules(self, baseline_w=0.0):
        """Integra potência (com subtração de baseline se fornecido) pela regra do trapézio."""
        if len(self.samples) < 2:
            return 0.0
        e = 0.0
        for i in range(len(self.samples) - 1):
            t0, p0 = self.samples[i]
            t1, p1 = self.samples[i + 1]
            dt = t1 - t0
            p0c = max(0.0, p0 - baseline_w)
            p1c = max(0.0, p1 - baseline_w)
            e += (p0c + p1c) * 0.5 * dt
        return e


def load_inputs_cpu(paths):
    """
    Carrega tensores do disco (CPU), normalizando para (N,C,H,W).
    Retorna lista [tensor_cpu, ...].
    """
    batch = []
    for p in paths:
        x = torch.load(p, map_location="cpu")
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"{p} não contém tensor PyTorch")
        if x.dim() == 3:
            x = x.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)
        elif x.dim() != 4:
            raise ValueError(f"Entrada deve ser (N,C,H,W) ou (C,H,W); veio {tuple(x.shape)}")
        batch.append(x)
    return batch


def cat_microbatch_to_device(tensors_cpu, device, dtype, max_count):
    """
    Concatena até 'max_count' tensores CPU (mesmo shape) em um batch e move para GPU.
    Retorna (x_gpu, usados). Se shapes divergirem, cai para usados=1.
    """
    if not tensors_cpu:
        return None, 0
    first = tensors_cpu[0]
    group = [first]
    used = 1
    for i in range(1, min(max_count, len(tensors_cpu))):
        t = tensors_cpu[i]
        if t.shape == first.shape:
            group.append(t)
            used += 1
        else:
            break
    x_cpu = torch.cat(group, dim=0)
    x_gpu = x_cpu.to(device=device, dtype=dtype, non_blocking=True)
    return x_gpu, used


def main():
    ap = argparse.ArgumentParser(
        description="Pointwise 1x1 Conv2d (DCVC-RT) em chunks fixos, medindo energia só durante as convs, com baseline opcional."
    )
    ap.add_argument("--input_dir", required=True, help="Pasta com id38_dc_0_stepN_input.pt")
    ap.add_argument("--weights", required=True, help="weights.pt (outC,inC,1,1) ou (outC,inC) para expandir")
    ap.add_argument("--bias", required=True, help="bias.pt (outC,)")
    ap.add_argument("--gpu", type=int, default=0, help="Índice da GPU para CUDA/NVML")
    ap.add_argument("--interval", type=float, default=0.05, help="Intervalo de amostragem de potência (s) nas convs")
    ap.add_argument("--dtype", choices=["float16","float32","bfloat16"], default="float16", help="dtype de compute")
    ap.add_argument("--chunk_size", type=int, default=100, help="Qtd de tensores lidos do disco por iteração")
    ap.add_argument("--microbatch", type=int, default=10, help="Qtd de tensores por conv (controle de VRAM)")
    ap.add_argument("--power_csv", default="gpu_power_log.csv", help="CSV com (tempo_s,potencia_W) das convs")
    # Baseline flags
    ap.add_argument("--baseline_seconds", type=float, default=2.0, help="Segundos para medir baseline em idle")
    ap.add_argument("--no_baseline", action="store_true", help="Não subtrair baseline; integrar potência bruta")
    args = ap.parse_args()

    # --- baseline (opcional) ---
    baseline_w = 0.0
    if not args.no_baseline and args.baseline_seconds > 0:
        baseline_w = measure_baseline_power(
            gpu_index=args.gpu,
            seconds=args.baseline_seconds,
            interval=args.interval
        )
        print(json.dumps({"baseline_W": baseline_w, "baseline_seconds": args.baseline_seconds}, indent=2))

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(args.gpu)
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    # --- pesos/bias ---
    W = torch.load(args.weights, map_location="cpu")
    b = torch.load(args.bias, map_location="cpu")
    if not (isinstance(W, torch.Tensor) and isinstance(b, torch.Tensor)):
        raise ValueError("weights.pt e bias.pt devem conter tensores.")
    if W.dim() == 2:
        W = W.unsqueeze(-1).unsqueeze(-1)
    if W.dim() != 4 or (W.size(-2) != 1 or W.size(-1) != 1):
        raise ValueError(f"Esperado weights (outC,inC,1,1) ou (outC,inC); recebido {tuple(W.shape)}")
    if b.dim() != 1 or b.shape[0] != W.shape[0]:
        raise ValueError(f"bias inválido: {tuple(b.shape)} vs outC={W.shape[0]}")
    W = W.to(device=device, dtype=torch_dtype, non_blocking=True)
    b = b.to(device=device, dtype=torch_dtype, non_blocking=True)

    

    # --- lista de arquivos ---
    files = find_input_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"Nenhum id38_dc_0_stepN_input.pt em {args.input_dir}")

    # --- sampler de energia (apenas nas convs) ---
    sampler = GPUPowerSampler(gpu_index=args.gpu, interval=args.interval)

    # Métricas
    total_cnt = 0
    conv_wall_time = 0.0
    t_all_start = time.time()

    # Varre em chunks
    for start in range(0, len(files), args.chunk_size):
        chunk_paths = files[start:start + args.chunk_size]

        # 1) I/O CPU (fora da medição)
        tensors_cpu = load_inputs_cpu(chunk_paths)

        # 2) Microbatches
        idx = 0
        while idx < len(tensors_cpu):
            x_gpu, used = cat_microbatch_to_device(
                tensors_cpu[idx:], device=device, dtype=torch_dtype, max_count=args.microbatch
            )
            if used == 0:
                break

            # 3) Medir SÓ a convolução
            torch.cuda.synchronize(device=device)
            sampler.start()
            t0 = time.time()

            y = F.conv2d(x_gpu, W, b, stride=1, padding=0, dilation=1, groups=1)
            torch.cuda.synchronize(device=device)

            t1 = time.time()
            sampler.stop()
            conv_wall_time += (t1 - t0)

            # 4) Liberar GPU; não salvamos saídas
            del x_gpu, y
            torch.cuda.empty_cache()

            total_cnt += used
            idx += used

        # 5) Liberar CPU
        del tensors_cpu
        gc.collect()

    sampler.shutdown()
    t_all_end = time.time()

    # Energia integrada (com baseline se habilitado)
    energy_J = sampler.energy_joules(baseline_w=baseline_w)
    energy_Wh = energy_J / 3600.0
    avg_power_W = energy_J / max(conv_wall_time, 1e-9)

    # Salva CSV (timeline só das convs)
    with open(args.power_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "power_W"])
        w.writerows(sampler.samples)

    summary = {
        "total_files": len(files),
        "processed": total_cnt,
        "chunk_size": args.chunk_size,
        "microbatch": args.microbatch,
        "conv_dtype": str(W.dtype).replace("torch.", ""),
        "inC": int(W.shape[1]),
        "outC": int(W.shape[0]),
        "total_wall_time_s": t_all_end - t_all_start,
        "convs_time_s": conv_wall_time,
        "baseline_W": baseline_w,
        "energy_J_convs_only": energy_J,
        "energy_Wh_convs_only": energy_Wh,
        "avg_power_W_convs_only": avg_power_W,
        "baseline_subtracted": not args.no_baseline
    }
    print(json.dumps(summary, indent=2))
    print(f"Power samples (convs only) saved to: {args.power_csv}")


if __name__ == "__main__":
    main()
