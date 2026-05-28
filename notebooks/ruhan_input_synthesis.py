#!/usr/bin/env python3
import os, json, gzip, argparse, csv, sys
from typing import List, Dict, Optional
import torch

def load_tensor(path: str) -> torch.Tensor:
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(f, map_location="cpu")
    return torch.load(path, map_location="cpu")

def read_index(index_path: str) -> List[dict]:
    recs = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return recs

def collect_records(
    recs: List[dict],
    target: str,                    # "input" | "output"
    only_saved: bool = True,
    name_filter: Optional[str] = None,
) -> List[dict]:
    key = "input_file" if target == "input" else "output_file"
    out = []
    for r in recs:
        if only_saved and not r.get("saved", False):
            continue
        if name_filter and r.get("name") != name_filter:
            continue
        if key in r:
            out.append({
                "name": r.get("name", "TracedConv2d"),
                "step": r.get("step", None),
                "file_path": r[key],
            })
    out.sort(key=lambda d: (d["name"], d["step"] if d["step"] is not None else -1))
    return out

def ensure_4d_ncwh(t: torch.Tensor) -> torch.Tensor:
    if t.dim() == 4: return t
    if t.dim() == 3: return t.unsqueeze(0)
    raise ValueError(f"Tensor deve ser 3D [C,H,W] ou 4D [N,C,H,W], recebido {tuple(t.shape)}")

def write_raster_csv(
    items: List[dict],
    out_path: str,
    float_dtype: str = "float32",
    limit_samples: Optional[int] = None,
    limit_pixels: Optional[int] = None,
    max_lines: Optional[int] = None,
    mode: str = "compact",   # "compact" (apenas canais) | "verbose" (metadados + canais)
):
    assert mode in ("compact", "verbose")
    total_items = len(items)
    files_done = 0
    total_lines = 0  # conta somente linhas de dados

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "verbose":
            writer.writerow(["name","step","sample_idx","y","x","channels..."])

        for it in items:
            name, step, fpath = it["name"], it["step"], it["file_path"]
            t = load_tensor(fpath)

            # dtype para CSV
            d = float_dtype.lower()
            if d in ("float32","fp32"): t = t.float()
            elif d in ("float64","double"): t = t.double()
            elif d in ("float16","fp16"): t = t.half()
            elif d in ("bfloat16","bf16"): t = t.bfloat16()
            else: raise ValueError(f"dtype não suportado para CSV: {float_dtype}")

            t = ensure_4d_ncwh(t)
            N, C, H, W = t.shape

            nmax = N if limit_samples is None else min(N, limit_samples)
            stop_early = False

            for n in range(nmax):
                chw = t[n]                 # [C,H,W]
                hwc = chw.permute(1,2,0)   # [H,W,C]
                flat = hwc.reshape(H*W, C) # [P,C]
                pmax = H*W if limit_pixels is None else min(H*W, limit_pixels)

                for p in range(pmax):
                    if max_lines is not None and total_lines >= max_lines:
                        stop_early = True
                        break

                    y = p // W
                    x = p %  W
                    row = flat[p].tolist() if mode == "compact" else [name, step, n, y, x] + flat[p].tolist()
                    writer.writerow(row)
                    total_lines += 1

                    if total_lines % 1000 == 0:
                        percent = (files_done / total_items) * 100.0 if total_items > 0 else 100.0
                        print(f"[progresso] linhas={total_lines:,}  arquivos_concluídos={files_done}/{total_items}  ~{percent:.1f}%")
                        sys.stdout.flush()

                if stop_early:
                    break

            if stop_early:
                percent = (files_done / total_items) * 100.0 if total_items > 0 else 100.0
                print(f"[interrompido] Limite de linhas atingido ({total_lines:,}). Progresso ~{percent:.1f}% (por arquivos).")
                sys.stdout.flush()
                return total_lines

            files_done += 1
            percent = (files_done / total_items) * 100.0 if total_items > 0 else 100.0
            print(f"[arquivo ok] {fpath}  ({files_done}/{total_items})  ~{percent:.1f}%")
            sys.stdout.flush()

    return total_lines

def main():
    ap = argparse.ArgumentParser(
        description="Exporta inputs/outputs salvos pelo TracedConv2d em formato raster. Modo 'compact' escreve apenas canais; 'verbose' inclui metadados."
    )
    ap.add_argument("--log_dir", required=True, help="Diretório com index.jsonl e tensores.")
    ap.add_argument("--index_name", default="index.jsonl", help="Nome do índice (padrão: index.jsonl).")
    ap.add_argument("--name", default=None, help="Filtrar por 'name' da camada (ex.: id38_dc_0).")
    ap.add_argument("--target", choices=["input","output"], default="input",
                    help="Exportar 'input' (padrão) ou 'output'.")
    ap.add_argument("--out", required=True, help="Caminho do arquivo de saída (ex.: /tmp/raster.csv).")
    ap.add_argument("--dtype", default="float32", help="float32|float64|float16|bfloat16 para valores no CSV.")
    ap.add_argument("--limit_samples", type=int, default=None, help="Limitar N por tensor (depuração).")
    ap.add_argument("--limit_pixels", type=int, default=None, help="Limitar #linhas por amostra (depuração).")
    ap.add_argument("--max_lines", type=int, default=None, help="Limitar o total de linhas de dados (global, sem cabeçalho).")
    ap.add_argument("--mode", choices=["compact","verbose"], default="compact",
                    help="compact = somente canais (sem cabeçalho). verbose = inclui name,step,sample_idx,y,x e cabeçalho.")
    args = ap.parse_args()

    if args.max_lines is not None and args.max_lines <= 0:
        raise ValueError("--max_lines deve ser > 0.")

    index_path = os.path.join(args.log_dir, args.index_name)
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Não encontrei índice: {index_path}")

    recs = read_index(index_path)
    items = collect_records(recs, args.target, only_saved=True, name_filter=args.name)
    if not items:
        hint = f" com name='{args.name}'" if args.name else ""
        raise RuntimeError(f"Nenhum registro salvo de {args.target}{hint} encontrado.")

    total = write_raster_csv(
        items=items,
        out_path=args.out,
        float_dtype=args.dtype,
        limit_samples=args.limit_samples,
        limit_pixels=args.limit_pixels,
        max_lines=args.max_lines,
        mode=args.mode,
    )
    print(f"[OK] Escrevi {total:,} linhas em: {args.out}")

if __name__ == "__main__":
    main()
