#!/usr/bin/env python3
"""Run a WSiLU module-by-module activation sweep across multiple GPUs.

The sweep keeps every activation at the default implementation (wsilu4) except
one logical module at a time. For the selected module, both depth-convolution
(``.dc``) and feed-forward (``.ffn``) activations are assigned the same variant.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_VARIANTS = (
    "lut_asyn_4int_128entries",
    "lut_asyn_4int_256entries",
    "lut_asyn_4int_512entries",
)

# Logical modules that contain WSiLU activations in src/models/*.
# For each logical module, the generated JSON includes both the direct
# DepthConvBlock names (<module>.dc/.ffn) and the names produced by residual
# blocks (<module>.conv.dc/.ffn). Unused keys are harmless because the runtime
# selector matches names exactly.
MODULES = (
    "i.enc",
    "i.dec",
    "i.hyper_enc",
    "i.hyper_dec",
    "i.y_prior_fusion",
    "i.y_spatial_prior_adaptor_1",
    "i.y_spatial_prior_adaptor_2",
    "i.y_spatial_prior_adaptor_3",
    "i.y_spatial_prior",
    "p.feature_extractor",
    "p.encoder",
    "p.decoder",
    "p.recon_generation_net",
    "p.hyper_encoder",
    "p.hyper_decoder",
    "p.y_prior_fusion",
    "p.y_spatial_prior",
    "p.feature_adaptor_i",
    "p.temporal_prior_encoder",
)


@dataclass(frozen=True)
class Job:
    name: str
    module: str | None
    variant: str
    config_path: Path
    output_path: Path
    stdout_path: Path
    stderr_path: Path


RunningJob = tuple[Job, int, float]


def sanitize(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
    value = value.lower().replace("/", "_")
    return "".join(ch if ch in allowed else "_" for ch in value).strip("_")


def activation_keys(module: str) -> list[str]:
    return [
        f"{module}.dc",
        f"{module}.ffn",
        f"{module}.conv.dc",
        f"{module}.conv.ffn",
    ]


def write_config(path: Path, module: str | None, variant: str, default: str) -> dict[str, str]:
    config = {"default": default}
    if module is not None:
        for key in activation_keys(module):
            config[key] = variant

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config


def build_jobs(args: argparse.Namespace, project_dir: Path) -> list[Job]:
    output_dir = (project_dir / args.output_dir).resolve()
    config_dir = (project_dir / args.config_dir).resolve()
    log_dir = (project_dir / args.log_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[Job] = []

    if args.include_baseline:
        name = sanitize(f"baseline_{args.default_activation}")
        cfg_path = config_dir / f"{name}.json"
        write_config(cfg_path, None, args.default_activation, args.default_activation)
        jobs.append(
            Job(
                name=name,
                module=None,
                variant=args.default_activation,
                config_path=cfg_path,
                output_path=output_dir / f"{name}.json",
                stdout_path=log_dir / f"{name}.out",
                stderr_path=log_dir / f"{name}.err",
            )
        )

    for module in args.modules:
        for variant in args.variants:
            name = sanitize(f"{module}__{variant}")
            cfg_path = config_dir / f"{name}.json"
            write_config(cfg_path, module, variant, args.default_activation)
            jobs.append(
                Job(
                    name=name,
                    module=module,
                    variant=variant,
                    config_path=cfg_path,
                    output_path=output_dir / f"{name}.json",
                    stdout_path=log_dir / f"{name}.out",
                    stderr_path=log_dir / f"{name}.err",
                )
            )

    manifest = {
        "default_activation": args.default_activation,
        "variants": args.variants,
        "modules": args.modules,
        "jobs": [
            {
                "name": job.name,
                "module": job.module,
                "variant": job.variant,
                "config_path": str(job.config_path),
                "output_path": str(job.output_path),
                "stdout_path": str(job.stdout_path),
                "stderr_path": str(job.stderr_path),
            }
            for job in jobs
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return jobs


def base_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "test_video.py",
        "--model_path_i",
        args.model_path_i,
        "--model_path_p",
        args.model_path_p,
        "--rate_num",
        str(args.rate_num),
        "--test_config",
        args.test_config,
        "--cuda",
        "1",
        "-w",
        str(args.worker),
        "--write_stream",
        str(int(args.write_stream)),
        "--force_zero_thres",
        str(args.force_zero_thres),
        "--force_intra_period",
        str(args.force_intra_period),
        "--reset_interval",
        str(args.reset_interval),
        "--force_frame_num",
        str(args.force_frame_num),
        "--check_existing",
        str(int(args.check_existing)),
    ]


def launch_job(job: Job, gpu: int, args: argparse.Namespace, project_dir: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["DCVC_WSILU_CONFIG"] = str(job.config_path)
    env["PYTHONUNBUFFERED"] = "1"

    # test_video.py also manages CUDA_VISIBLE_DEVICES internally when --cuda_idx
    # is present. Passing the selected GPU here prevents its worker initializer
    # from overwriting every subprocess back to CUDA_VISIBLE_DEVICES=0.
    cmd = base_command(args) + ["--cuda_idx", str(gpu), "--output_path", str(job.output_path)]
    stdout_f = job.stdout_path.open("w", encoding="utf-8")
    stderr_f = job.stderr_path.open("w", encoding="utf-8")
    process = subprocess.Popen(cmd, cwd=project_dir, env=env, stdout=stdout_f, stderr=stderr_f)
    print(
        f"[ALLOC] GPU {gpu} -> pid={process.pid} job={job.name} "
        f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} --cuda_idx={gpu} "
        f"stdout={job.stdout_path} stderr={job.stderr_path}",
        flush=True,
    )
    # Keep file objects alive so Popen can write to them; close after wait.
    process._wsilu_stdout_f = stdout_f  # type: ignore[attr-defined]
    process._wsilu_stderr_f = stderr_f  # type: ignore[attr-defined]
    return process


def close_process_logs(process: subprocess.Popen) -> None:
    for attr in ("_wsilu_stdout_f", "_wsilu_stderr_f"):
        file_obj = getattr(process, attr, None)
        if file_obj is not None:
            file_obj.close()


def format_gpu_allocations(running: dict[subprocess.Popen, RunningJob]) -> str:
    if not running:
        return "no running jobs"

    allocations = []
    for process, (job, gpu, start_time) in sorted(running.items(), key=lambda item: item[1][1]):
        elapsed_min = (time.time() - start_time) / 60.0
        allocations.append(f"gpu={gpu}:pid={process.pid}:{job.name}:{elapsed_min:.1f}min")
    return "; ".join(allocations)


def print_progress(
    label: str,
    total_jobs: int,
    completed_jobs: int,
    pending: list[Job],
    running: dict[subprocess.Popen, RunningJob],
) -> None:
    if total_jobs == 0:
        print(f"[PROGRESS] {label}: no jobs", flush=True)
        return

    running_jobs = len(running)
    pending_jobs = len(pending)
    completed_pct = completed_jobs / total_jobs * 100.0
    running_pct = running_jobs / total_jobs * 100.0
    pending_pct = pending_jobs / total_jobs * 100.0
    print(
        f"[PROGRESS] {label}: "
        f"executed={completed_jobs}/{total_jobs} ({completed_pct:.1f}%) | "
        f"running={running_jobs}/{total_jobs} ({running_pct:.1f}%) | "
        f"waiting={pending_jobs}/{total_jobs} ({pending_pct:.1f}%) | "
        f"allocations=[{format_gpu_allocations(running)}]",
        flush=True,
    )


def run_queue(jobs: Iterable[Job], gpus: list[int], args: argparse.Namespace, project_dir: Path) -> int:
    pending = list(jobs)
    total_jobs = len(pending)
    completed_jobs = 0
    running: dict[subprocess.Popen, RunningJob] = {}
    failures: list[tuple[Job, int]] = []
    next_progress_at = time.time()

    print_progress("initial", total_jobs, completed_jobs, pending, running)

    while pending or running:
        launched_job = False
        while pending and len(running) < len(gpus) and not (failures and args.fail_fast):
            used = {gpu for _, gpu, _ in running.values()}
            gpu = next(gpu for gpu in gpus if gpu not in used)
            job = pending.pop(0)
            running[launch_job(job, gpu, args, project_dir)] = (job, gpu, time.time())
            launched_job = True

        if launched_job:
            print_progress("after-launch", total_jobs, completed_jobs, pending, running)
            next_progress_at = time.time() + args.progress_interval

        time.sleep(args.poll_interval)

        finished_job = False
        for process, (job, gpu, start_time) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            process.wait()
            close_process_logs(process)
            elapsed = time.time() - start_time
            status = "OK" if return_code == 0 else f"FAIL retcode={return_code}"
            completed_jobs += 1
            finished_job = True
            print(
                f"[DONE] GPU {gpu} pid={process.pid} job={job.name}: "
                f"{status} ({elapsed / 60:.1f} min)",
                flush=True,
            )
            del running[process]
            if return_code != 0:
                failures.append((job, return_code))

        if finished_job:
            print_progress("after-finish", total_jobs, completed_jobs, pending, running)
            next_progress_at = time.time() + args.progress_interval
        elif running and time.time() >= next_progress_at:
            print_progress("periodic", total_jobs, completed_jobs, pending, running)
            next_progress_at = time.time() + args.progress_interval

        if failures and args.fail_fast and pending:
            print(f"Fail-fast enabled; {len(pending)} pending job(s) will not be launched.", flush=True)
            pending.clear()
            print_progress("fail-fast", total_jobs, completed_jobs, pending, running)

    if failures:
        print("\nFailed jobs:", flush=True)
        for job, return_code in failures:
            print(f"  - {job.name}: retcode={return_code}; stderr={job.stderr_path}", flush=True)
        return 1

    print("\nAll WSiLU module-sweep jobs completed successfully.", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)), help="GPU IDs to use as an execution pool.")
    parser.add_argument("--default_activation", default="wsilu4", help="Activation used by all non-selected modules.")
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS), help="Activation variants tested for each selected module.")
    parser.add_argument("--modules", nargs="+", default=list(MODULES), help="Logical modules to sweep.")
    parser.add_argument("--include_baseline", type=int, default=1, help="Also run one all-default wsilu4 baseline job.")
    parser.add_argument("--fail_fast", type=int, default=0, help="Stop launching new jobs after the first failure.")
    parser.add_argument("--dry_run", type=int, default=0, help="Generate configs/manifest and print the queue without launching test_video.py.")
    parser.add_argument("--poll_interval", type=float, default=5.0)
    parser.add_argument("--progress_interval", type=float, default=60.0, help="Seconds between periodic stdout progress updates while jobs are running.")

    parser.add_argument("--output_dir", default="../coding_outputs/module_sweep")
    parser.add_argument("--config_dir", default="generated_wsilu_configs/module_sweep")
    parser.add_argument("--log_dir", default="../coding_outputs/module_sweep_logs")

    parser.add_argument("--model_path_i", default="./checkpoints/cvpr2025_image.pth.tar")
    parser.add_argument("--model_path_p", default="./checkpoints/cvpr2025_video.pth.tar")
    parser.add_argument("--rate_num", type=int, default=4)
    parser.add_argument("--test_config", default="./dataset_fast.json")
    parser.add_argument("--worker", type=int, default=1, help="Workers inside each test_video.py process; keep 1 for one GPU per job.")
    parser.add_argument("--write_stream", type=int, default=1)
    parser.add_argument("--force_zero_thres", type=float, default=0.12)
    parser.add_argument("--force_intra_period", type=int, default=-1)
    parser.add_argument("--reset_interval", type=int, default=64)
    parser.add_argument("--force_frame_num", type=int, default=-1)
    parser.add_argument("--check_existing", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.gpus:
        raise SystemExit("At least one GPU must be supplied through --gpus.")
    if args.worker != 1:
        print("Warning: --worker > 1 can make a single job use more than one visible GPU if more are exposed.")

    project_dir = Path(__file__).resolve().parent
    jobs = build_jobs(args, project_dir)
    print(f"Prepared {len(jobs)} job(s); running up to {len(args.gpus)} in parallel on GPUs {args.gpus}.")
    if args.dry_run:
        for job in jobs:
            target = "baseline" if job.module is None else f"{job.module} -> {job.variant}"
            print(f"DRY-RUN {job.name}: {target}; config={job.config_path}; output={job.output_path}")
        return 0
    return run_queue(jobs, args.gpus, args, project_dir)


if __name__ == "__main__":
    raise SystemExit(main())
