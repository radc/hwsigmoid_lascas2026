#!/usr/bin/env python3
import os, argparse, gzip, csv, sys
from typing import Optional, Tuple
import torch

def load_any(path: str):
    """Carrega .pt ou .pt.gz. Retorna objeto Python (tensor ou dict)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(f, map_location="cpu")
    return torch.load(path, map_location="cpu")

def get_weight_tensor(obj) -> torch.Tensor:
    """
    Tenta extrair um tensor de pesos de 'obj'.
    Aceita:
      - Tensor direto
      - Dict com chaves comuns: 'weight', 'conv.weight', 'W'
    """
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict):
        for k in ("weight", "conv.weight", "W", "param", "tensor"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                return obj[k]
    raise TypeError("Arquivo não contém um tensor de pesos reconhecido.")

def parse_slice(spec: Optional[str], max_n: int) -> slice:
    """
    Converte 'inicio:fim' em slice. Aceita vazio, ':', '5:', ':10', '3:7'.
    Indexação como Python (fim exclusivo).
    """
    if not spec or spec == ":":
        return slice(None, None)
    if ":" not in spec:
        # número único => único índice
        i = int(spec)
        if i < 0: i += max_n
        return slice(i, i+1)
    a, b = spec.split(":", 1)
    a = int(a) if a.strip() != "" else None
    b = int(b) if b.strip() != "" else None
    if a is not None and a < 0: a += max_n
    if b is not None and b < 0: b += max_n
    return slice(a, b)

def coerce_dtype(t: torch.Tensor, dtype: str) -> torch.Tensor:
    d = dtype.lower()
    if d in ("float32", "fp32"): return t.float()
    if d in ("float64", "double"): return t.double()
    if d in ("float16", "fp16"): return t.half()
    if d in ("bfloat16", "bf16"): return t.bfloat16()
    raise ValueError(f"dtype não suportado: {dtype}")

def main():
    ap = argparse.ArgumentParser(
        description=("Gera CSV com uma linha por canal de saída, contendo todos os pesos que "
                     "multiplicam os canais de entrada (pixel-wise conv = 1x1)."))
    ap.add_argument("--weights", required=True, help="Caminho do arquivo de pesos (.pt|.pt.gz).")
    ap.add_argument("--out", required=True, help="Caminho do CSV de saída.")
    ap.add_argument("--dtype", default="float32",
                    help="float32|float64|float16|bfloat16 para exportação.")
    ap.add_argument("--no_header", action="store_true",
                    help="Não escrever cabeçalho no CSV (padrão: escreve).")
    ap.add_argument("--allow_spatial", action="store_true",
                    help="Permitir kh×kw > 1 achatando (ordem: por canal de entrada, depois varredura do kernel).")
    ap.add_argument("--out_slice", default=":",
                    help="Faixa de canais de saída no formato 'ini:fim' (fim exclusivo). Ex.: ':64', '128:256', '5'.")
    ap.add_argument("--progress_every", type=int, default=1000,
                    help="Imprimir progresso a cada N linhas (padrão: 1000).")
    args = ap.parse_args()

    obj = load_any(args.weights)
    W = get_weight_tensor(obj)  # esperado [out_ch, in_ch, kh, kw] ou [out_ch, in_ch]
    if W.dim() == 2:
        out_ch, in_ch = W.shape
        kh = kw = 1
    elif W.dim() == 4:
        out_ch, in_ch, kh, kw = W.shape
    else:
        raise ValueError(f"Pesos devem ser 2D [out,in] ou 4D [out,in,kh,kw]; recebido shape={tuple(W.shape)}")

    if (kh, kw) != (1, 1) and not args.allow_spatial:
        raise ValueError(f"Kernel {kh}x{kw} detectado, mas --allow_spatial não foi passado. "
                         "Use --allow_spatial para achatar o kernel.")

    # Converte dtype para exportação
    W = coerce_dtype(W, args.dtype)

    # Se 4D e allow_spatial, achata para [out, in * kh * kw]
    if W.dim() == 4:
        # ordem: para cada in, varre kernel em raster (y, x)
        # => rearranjo [out, in, kh, kw] -> [out, in*kh*kw]
        W = W.reshape(out_ch, in_ch, kh*kw)
        W = W.flatten(start_dim=1)  # [out, in*kh*kw]

    # Seleção de faixa de canais de saída
    s = parse_slice(args.out_slice, W.shape[0])
    W = W[s, ...]
    out_rows = W.shape[0]
    cols = W.shape[1]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not args.no_header:
            # Cabeçalho com índices de coluna (c0..c{cols-1})
            writer.writerow([f"c{i}" for i in range(cols)])

        written = 0
        for o in range(out_rows):
            row = W[o].tolist()
            writer.writerow(row)
            written += 1
            if args.progress_every and (written % args.progress_every == 0):
                # progresso simples por linhas (cada linha = um canal de saída)
                pct = (written / out_rows) * 100.0 if out_rows > 0 else 100.0
                print(f"[progresso] linhas={written}/{out_rows}  ~{pct:.1f}%")
                sys.stdout.flush()

    print(f"[OK] Gerado CSV: {args.out}  ({written} linhas; {cols} colunas por linha)")

if __name__ == "__main__":
    main()
