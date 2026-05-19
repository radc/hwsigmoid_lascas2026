#!/bin/bash
#SBATCH --job-name=dcvc_wsilu_exp
#SBATCH --output=/work/ruhan625/logs/dcvc_wsilu_exp_%j.out
#SBATCH --error=/work/ruhan625/logs/dcvc_wsilu_exp_%j.err
#SBATCH --account=acd114139
#SBATCH --partition=gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=48:00:00

set -euo pipefail

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=true

# Opcional: habilite se quiser usar Comet
# export COMET_API_KEY="<SUA_CHAVE>"

PROJECT_DIR="/home/ruhan625/hwsigmoid_lascas2026/DCVC"
CONTAINER_IMG="${PROJECT_DIR}/aimet_ubuntu2204.sif"

cd "$PROJECT_DIR" || exit 1
chmod +x launch.sh

apptainer exec --nv \
  --bind /home/ruhan625:/home/ruhan625 \
  --bind /work/ruhan625:/work/ruhan625 \
  "$CONTAINER_IMG" \
  bash -lc "./launch.sh"
