#!/bin/bash
set -eo pipefail

source /home/ruhan625/miniconda3/etc/profile.d/conda.sh
set +u
conda activate dcvc_qatkd
set -u

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/home/ruhan625/miniconda3/lib:${LD_LIBRARY_PATH:-}"


# Configure seu experimento aqui (sem prompt interativo)
# EXP_NAME="baseline_wsilu4"
# WSILU_TYPE="wsilu4"

# EXP_NAME="lut_asyn_4int_128"
# WSILU_TYPE="lut_asyn_4int_256entries"



# Se quiser permitir vazio, remova este bloco
if [[ -z "${EXP_NAME// }" ]]; then
  echo "Erro: nome do experimento não pode ser vazio."
  exit 1
fi

# Opcional: sanitiza para virar um nome de arquivo seguro (minúsculas, números, . _ -)
EXP_SAFE=$(printf "%s" "$EXP_NAME" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_')

# Garante que a pasta de saída exista

# OUT_DIR="../coding_outputs/noise_analysis"
mkdir -p "$OUT_DIR"

OUT_PATH="${OUT_DIR}/${EXP_SAFE}.json"
echo "Saída: $OUT_PATH"

# DCVC_WSILU_TYPE="$WSILU_TYPE" python test_video.py \
#   --model_path_i ./checkpoints/cvpr2025_image.pth.tar \
#   --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
#   --rate_num 4 --test_config ./dataset_fast.json \
#   --cuda 1 -w 1 --write_stream 1 --force_zero_thres 0.12 \
#   --output_path "$OUT_PATH" \
#   --force_intra_period -1 --reset_interval 64 --force_frame_num -1 --check_existing 0


python test_video.py \
  --model_path_i ./checkpoints/cvpr2025_image.pth.tar \
  --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
  --rate_num 4 --test_config ./dataset_test.json \
  --cuda 1 -w 1 --write_stream 1 --force_zero_thres 0.12 \
  --output_path "$OUT_PATH" \
  --force_intra_period -1 --reset_interval 64 --force_frame_num -1 --check_existing 0