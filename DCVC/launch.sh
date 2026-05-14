#!/bin/bash
set -eo pipefail

source /home/ruhan625/miniconda3/etc/profile.d/conda.sh
set +u
conda activate dcvc_qatkd
set -u

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:/home/ruhan625/miniconda3/lib:${LD_LIBRARY_PATH:-}"

# Executa a varredura módulo-a-módulo com wsilu4 como default.
# Cada experimento expõe apenas uma GPU via CUDA_VISIBLE_DEVICES, e o
# escalonador Python mantém até 8 jobs simultâneos.
python run_wsilu_module_sweep.py \
  --gpus 0 1 2 3 4 5 6 7 \
  --default_activation wsilu4 \
  --variants \
    lut_asyn_4int_64entries \
    lut_asyn_4int_128entries \
    lut_asyn_4int_256entries \
    lut_asyn_4int_512entries \
  --include_baseline 1 \
  --worker 1 \
  --test_config ./dataset_fast_sweep.json \
  --output_dir ../coding_outputs/module_sweep \
  --config_dir generated_wsilu_configs/module_sweep \
  --log_dir ../coding_outputs/module_sweep_logs
