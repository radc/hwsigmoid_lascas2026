#!/bin/bash
#SBATCH --job-name=dcvc_wsilu_fullbaseline
#SBATCH --output=/work/ruhan625/logs/dcvc_wsilu_exp_%j.out
#SBATCH --error=/work/ruhan625/logs/dcvc_wsilu_exp_%j.err
#SBATCH --account=mst108318
#SBATCH --partition=gp1d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00

set -euo pipefail

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=true

# Opcional: habilite se quiser usar Comet
# export COMET_API_KEY="<SUA_CHAVE>"

PROJECT_DIR="/home/ruhan625/hwsigmoid_lascas2026/DCVC"
export CONTAINER_IMG="${PROJECT_DIR}/aimet_ubuntu2204.sif"



export DCVC_WSILU_TYPE="wsilu4"
export EXP_NAME="multiple_precision"
export OUT_DIR="../coding_outputs/complete"
export DCVC_WSILU_CONFIG="/home/ruhan625/hwsigmoid_lascas2026/DCVC/wsilu_modules_config.example.json"

cd "$PROJECT_DIR" || exit 1
chmod +x launch_single.sh

apptainer exec --nv \
  --bind /home/ruhan625:/home/ruhan625 \
  --bind /work/ruhan625:/work/ruhan625 \
  "$CONTAINER_IMG" \
  bash -lc "./launch_single.sh"
