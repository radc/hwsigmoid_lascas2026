#!/usr/bin/env python3
import subprocess
from pathlib import Path

def main():
    # pasta de saída
    out_dir = Path("../coding_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pasta de saída: {out_dir.resolve()}")

    # variação de wsilu_bw: 16, 14, 12, 10, 8, 6, 4
    for bw in range(16, 3, -2):
        out_path = out_dir / f"integer_lut_{bw}.json"
        print(f"\n>>> Rodando wsilu_bw={bw}")
        print(f"Saída: {out_path}")

        cmd = [
            "python", "test_video.py",
            "--model_path_i", "./checkpoints/cvpr2025_image.pth.tar",
            "--model_path_p", "./checkpoints/cvpr2025_video.pth.tar",
            "--rate_num", "4",
            "--test_config", "./dataset_config_tcsvt_uvg.json",
            "--cuda", "1",
            "-w", "1",
            "--write_stream", "1",
            "--force_zero_thres", "0.12",
            "--output_path", str(out_path),
            "--force_intra_period", "-1",
            "--reset_interval", "64",
            "--force_frame_num", "-1",
            "--check_existing", "0",
            "--wsilu_bw", str(bw),
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Execução falhou para wsilu_bw={bw} (retcode {e.returncode}). Abortando.")
            raise SystemExit(e.returncode)

    print("\nTodas as execuções concluídas com sucesso.")

if __name__ == "__main__":
    main()
