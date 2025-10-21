#!/usr/bin/env python3
import os, argparse, gzip, csv, sys
from typing import Optional
import torch

def load_any(path: str):
    """Carrega .pt ou .pt.gz e retorna o objeto Python armazenado (tensor/dict/etc)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(f, map_location="cpu")
    return torch.load(path, map_location="cpu")

def get_bias_tensor(obj) -> torch.Tensor:
    """
    Extrai o tensor de bias de 'obj'.
    Aceita:
      - Tensor direto
      - Dict com chaves comuns: 'bias', 'conv.bias', 'b'
    Ajusta shapes típicos para 1D [out_ch] (ex.: [out_ch,1,1] -> [out_ch]).
    """
    if isinstance(obj, torch.Tensor):
        bias = obj
    elif isinstance(obj, dict):
        for k in ("bias", "conv.bias", "b", "param", "tensor"):
            v = obj.get(k, None)
            if isinstance(v, torch.Tensor):
                bias = v
                break
        else:
            raise TypeError("Dict não contém um tensor de bias reconhecido (ex.: 'bias').")
    else:
        raise TypeError("Objeto carregado não é Tensor nem Dict com bias.")

    # Normaliza para [out_ch]
    if bias.dim() == 1:
        return bias
    if bias.dim() == 3 and bias.shape[1:] == (1, 1):
        return bias.view(-1)
    if bias.dim() == 2 and bias.shape[1] == 1:
        return bias.view(-1)
    # Outros formatos: tenta achatar mantendo primeira dimensão como out_ch
    if bias.dim() > 1:
        if all(s == 1 for s in bias.shape[1:]):
            return bias.view(bias.shape[0])
        raise ValueError(f"Formato de bias inesperado: shape={tuple(bias.shape)} (esperado [out], [out,1], [out,1,1]).")
    return bias

def parse_slice(spec: Optional[str], max_n: int) -> slice:
    """
    Converte 'ini:fim' em slice. Também aceita número único (ex.: '5').
    Fim é exclusivo, como no Python.
    """
    if not spec or spec == ":":
        return slice(None, None)
    if ":" not in spec:
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
    if d in ("float32","fp32"): return t.float()
    if d in ("float64","double"): return t.double()
    if d in ("float16","fp16"): return t.half()
    if d in ("bfloat16","bf16"): return t.bfloat16()
    raise ValueError(f"dtype não suportado: {dtype}")

def main():
    ap = argparse.ArgumentParser(
        description="Exporta bias de uma conv: um valor por linha (um por canal de saída)."
    )
    ap.add_argument("--bias", required=True, help="Caminho do arquivo de bias (.pt ou .pt.gz).")
    ap.add_argument("--out", required=True, help="Caminho do CSV de saída.")
    ap.add_argument("--dtype", default="float32",
                    help="float32|float64|float16|bfloat16 para exportação.")
    ap.add_argument("--no_header", action="store_true",
                    help="Não escrever cabeçalho no CSV.")
    ap.add_argument("--out_slice", default=":",
                    help="Faixa de canais de saída: 'ini:fim' (fim exclusivo) ou único índice (ex.: ':64', '128:256', '5').")
    ap.add_argument("--progress_every", type=int, default=1000,
                    help="Imprimir progresso a cada N linhas (padrão: 1000). 0 desativa.")
    args = ap.parse_args()

    obj = load_any(args.bias)
    b = get_bias_tensor(obj)
    b = coerce_dtype(b, args.dtype)

    # Seleciona faixa de canais de saída
    sl = parse_slice(args.out_slice, b.shape[0])
    b = b[sl]
    n = b.shape[0]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not args.no_header:
            writer.writerow(["bias"])
        for i in range(n):
            writer.writerow([b[i].item()])
            if args.progress_every and (i+1) % args.progress_every == 0:
                pct = ((i+1) / n) * 100.0 if n > 0 else 100.0
                print(f"[progresso] linhas={i+1}/{n}  ~{pct:.1f}%")
                sys.stdout.flush()

    print(f"[OK] Gerado CSV: {args.out}  ({n} linhas)")

if __name__ == "__main__":
    main()
