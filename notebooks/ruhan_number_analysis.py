import pandas as pd

# ajuste o caminho do arquivo
arquivo = "/home/ruhan/hwsigmoid_lascas2026/DCVC/saidadcblock.txt"

# ler sem cabeçalho e dar nomes
df = pd.read_csv(arquivo, header=None, names=["id", "c1", "c2", "c3", "cout", "ultimo"])

# pegar, para cada id, o índice da linha onde 'ultimo' é máximo
idx = df.groupby("id")["ultimo"].idxmax()

# filtrar e (opcional) ordenar por id
res = df.loc[idx].sort_values("id")

# salvar se quiser
res.to_csv("/home/ruhan/hwsigmoid_lascas2026/DCVC/saidadcblock_v2.csv", index=False, header=True)
