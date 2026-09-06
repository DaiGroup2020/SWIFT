#!/usr/bin/env python
"""Run the released top64 model on the bundled compact prediction demo."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PREDICT_OUTPUT = ROOT / "outputs" / "demo_prediction"


def run(command: list[str]) -> None:
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--calibration-time-end", type=float, default=200.0)
    parser.add_argument("--time-start", type=float, default=None, help="Optional first prediction endpoint; omitted for the complete replay.")
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--skip-evaluation", action="store_true", help="Do not read voltage ground truth.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing outputs for the bundled demo session.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PREDICT_OUTPUT,
        help="Directory for predictions, metrics, and figures (default: outputs/demo_prediction).",
    )
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    command = [
            PYTHON, "scripts/predict_future_day.py",
            "--model-path", "models/top64_local_peak/manifold_lstm_decoder.pt",
            "--spike-csv", "data/demo/Online_demo_20260202.csv",
            "--voltage-dat", "data/demo/M06-2026-02-02_voltage.dat",
            "--day-name", "2026-02-02",
            "--calibration-time-end", str(args.calibration_time_end),
            "--output-dir", str(args.output_dir),
            "--device", args.device,
        ]
    for option, value in (("--time-start", args.time_start), ("--time-end", args.time_end)):
        if value is not None:
            command.extend([option, str(value)])
    if args.skip_evaluation:
        command.append("--skip-evaluation")
    if args.overwrite:
        command.append("--overwrite")
    run(command)
    print(f"\nPrediction demo complete. Inspect {args.output_dir}")


if __name__ == "__main__":
    main()
