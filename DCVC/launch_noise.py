import subprocess
import os

# Base command (everything except the two parameters we will sweep)
BASE_PATH = "/home/ruhan/hwsigmoid_lascas2026/coding_outputs/interval_noise_analysis"

base_cmd = [
    "python", "test_video.py",
    "--model_path_i", "./checkpoints/cvpr2025_image.pth.tar",
    "--model_path_p", "./checkpoints/cvpr2025_video.pth.tar",
    "--rate_num", "4", "--test_config", "./dataset_config_tcsvt_uvg.json",
    "--cuda", "1", "--write_stream", "1", "--force_zero_thres", "0.12",
    "--force_intra_period", "-1", "--reset_interval", "64",
    "--force_frame_num", "-1", "--check_existing", "0",
    "--wsilu_inject_noise", "1"
]

# Sweep parameters
noise_amp_values = range(-3, -10, -3)         # -3, -6, -9
noise_low_interval_values = range(-5, 6, 2)  # -5, -3, -1, 1, 3, 5

for amp in noise_amp_values:
    for interval in noise_low_interval_values:
        output_dir = f"{BASE_PATH}/noise_amp_10E{amp}_interval_{interval}.json"

        # Skip if path already exists
        if os.path.exists(output_dir):
            print(f"Skipping: output path already exists -> {output_dir}")
            continue
        
        if (interval <= -1 and interval+2 <= -1 or interval >= 1 and interval+1 >= 1):
            workers = 7
        else:
            workers = 2

        print("Number of workers:", workers)
        
        cmd = base_cmd + [
            "--wsilu_noise_amp", str(amp),
            "--wsilu_noise_low_interval", str(interval),
            "--wsilu_noise_high_interval", str(interval + 2),
            "--output_path", output_dir,
            "-w", f"{workers}"
        ]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)
