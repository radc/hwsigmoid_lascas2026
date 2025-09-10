#!/bin/sh
set -eu pipefail

read -rp "Nome do experimento: " EXP_NAME

# Se quiser permitir vazio, remova este bloco
if [[ -z "${EXP_NAME// }" ]]; then
  echo "Erro: nome do experimento não pode ser vazio."
  exit 1
fi

# Opcional: sanitiza para virar um nome de arquivo seguro (minúsculas, números, . _ -)
EXP_SAFE=$(printf "%s" "$EXP_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_')

# Garante que a pasta de saída exista
OUT_DIR="../coding_outputs"
# OUT_DIR="../coding_outputs/noise_analysis"
mkdir -p "$OUT_DIR"

OUT_PATH="${OUT_DIR}/${EXP_SAFE}.json"
echo "Saída: $OUT_PATH"

python test_video.py \
  --model_path_i ./checkpoints/cvpr2025_image.pth.tar \
  --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
  --rate_num 4 --test_config ./dataset_config_tcsvt_uvg.json \
  --cuda 1 -w 1 --write_stream 1 --force_zero_thres 0.12 \
  --output_path "$OUT_PATH" \
  --force_intra_period -1 --reset_interval 64 --force_frame_num -1 --check_existing 0

