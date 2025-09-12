python test_video.py \
  --model_path_i ./checkpoints/cvpr2025_image.pth.tar \
  --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
  --rate_num 4 --test_config ./dataset_config_tcsvt_uvg_2f.json \
  --cuda 1 -w 1 --write_stream 1 --force_zero_thres 0.12 \
  --output_path coding_outputs/apagar.json \
  --force_intra_period -1 --reset_interval 64 --force_frame_num -1 --check_existing 0