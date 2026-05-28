#!/usr/bin/env python3
import os, json, gzip, argparse
from typing import List, Tuple, Dict, Optional
import torch

def load_tensor(path: str) -> torch.Tensor:
    """Carrega .pt ou .pt.gz sem dor de cabeça."""
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

def collect_files(
    recs: List[dict],
    only_saved: bool = True,
    name_filter: Optional[str] = None
) -> Dict[str, Dict[str, List[str]]]:
    """
    Agrupa por 'name' (camada) e coleta listas de arquivos de peso e bias.
    Retorna: { name: {"weights":[...], "biases":[...]} }
    """
    out: Dict[str, Dict[str, List[str]]] = {}
    for r in recs:
        if only_saved and not r.get("saved", False):
            continue
        nm = r.get("name", "TracedConv2d")
        if name_filter and nm != name_filter:
            continue
        grp = out.setdefault(nm, {"weights": [], "biases": []})
        if "weight_file" in r: grp["weights"].append(r["weight_file"])
        if "bias_file"   in r: grp["biases"].append(r["bias_file"])
    return out

def all_equal(tensors: List[torch.Tensor]) -> bool:
    """Igualdade estrita (valores e dtype)."""
    if len(tensors) <= 1: return True
    ref = tensors[0]
    return all(torch.equal(ref, t) for t in tensors[1:])

def all_allclose(
    tensors: List[torch.Tensor],
    rtol: float = 1e-5,
    atol: float = 1e-8
) -> bool:
    """Igualdade com tolerância numérica."""
    if len(tensors) <= 1: return True
    ref = tensors[0]
    # Converte dtypes para evitar falsos negativos (p.ex., bf16 vs f32)
    ref = ref.float()
    for t in tensors[1:]:
        if not torch.allclose(ref, t.float(), rtol=rtol, atol=atol):
            return False
    return True

def check_group(
    file_list: List[str],
    mode: str,
    rtol: float,
    atol: float
) -> Tuple[bool, List[int]]:
    """
    Carrega todos os arquivos e verifica igualdade.
    Retorna (ok, indices_diferentes)
    """
    if len(file_list) <= 1:
        return True, []

    tensors = [load_tensor(p) for p in file_list]
    if mode == "equal":
        ok = all_equal(tensors)
    else:
        ok = all_allclose(tensors, rtol=rtol, atol=atol)

    if ok:
        return True, []

    # Identifica quais divergem do primeiro
    diffs = []
    ref = tensors[0]
    ref_f = ref.float()
    for i, t in enumerate(tensors[1:], start=1):
        if mode == "equal":
            same = torch.equal(ref, t)
        else:
            same = torch.allclose(ref_f, t.float(), rtol=rtol, atol=atol)
        if not same:
            diffs.append(i)
    return False, diffs

def main():
    ap = argparse.ArgumentParser(description="Verifica se todos os weights/bias salvos pelo TracedConv2d são iguais.")
    ap.add_argument("--log_dir", required=True, help="Diretório do log (onde está o index.jsonl).")
    ap.add_argument("--index_name", default="index.jsonl", help="Nome do arquivo de índice (padrão: index.jsonl).")
    ap.add_argument("--name", default=None, help="Filtrar por 'name' específico (ex.: id38_dc_0).")
    ap.add_argument("--mode", choices=["equal","allclose"], default="equal",
                    help="Comparação estrita (equal) ou com tolerância (allclose).")
    ap.add_argument("--rtol", type=float, default=1e-5, help="rtol para allclose.")
    ap.add_argument("--atol", type=float, default=1e-8, help="atol para allclose.")
    args = ap.parse_args()

    index_path = os.path.join(args.log_dir, args.index_name)
    if not os.path.isfile(index_path):
        print(f"[ERRO] Não encontrei: {index_path}")
        return

    recs = read_index(index_path)
    groups = collect_files(recs, only_saved=True, name_filter=args.name)

    if not groups:
        if args.name:
            print(f"[AVISO] Nenhuma entrada salva para name='{args.name}'. Verifique save_every e saved=True.")
        else:
            print("[AVISO] Nenhuma entrada salva encontrada (saved=True).")
        return

    ok_global = True
    for name, files in groups.items():
        weights = files["weights"]
        biases  = files["biases"]

        print(f"\n=== Camada: {name} ===")
        print(f"  #weights salvos: {len(weights)}")
        print(f"  #biases  salvos: {len(biases)}")

        if len(weights) >= 1:
            ok_w, diff_w_idx = check_group(weights, args.mode, args.rtol, args.atol)
            if ok_w:
                print("  Weights: OK (todos iguais)")
            else:
                ok_global = False
                print("  Weights: DIFERENTES")
                for i in diff_w_idx:
                    print(f"    - Diferente do primeiro: {weights[i]}")
        else:
            print("  Weights: nenhum arquivo salvo encontrado.")

        if len(biases) >= 1:
            ok_b, diff_b_idx = check_group(biases, args.mode, args.rtol, args.atol)
            if ok_b:
                print("  Biases: OK (todos iguais)")
            else:
                ok_global = False
                print("  Biases: DIFERENTES")
                for i in diff_b_idx:
                    print(f"    - Diferente do primeiro: {biases[i]}")
        else:
            print("  Biases: nenhum arquivo salvo encontrado (pode ser que sua conv não tenha bias).")

    print("\n=== RESULTADO GERAL ===")
    if ok_global:
        print("Todos os weights/bias verificados são iguais dentro do critério escolhido.")
    else:
        print("Foram encontradas diferenças. Veja os caminhos listados acima.")

if __name__ == "__main__":
    main()
