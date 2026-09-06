"""Verify the complete bundled prediction demo without third-party dependencies."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "data/demo/Online_demo_20260202.csv": "4658468b2ef0ef2038626bbf53cdd71510785946a7cc18a761fb43166438b936",
    "data/demo/M06-2026-02-02_voltage.dat": "64dc94713994799a1b39a91bd19923784d37943b1b9ef9ec51e0eeace63fd9e5",
    "models/top64_local_peak/manifold_lstm_decoder.pt": "4f87529ee8250dab814387f77e72e2d349d70771f47cf31491a048836b2c3565",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        first = stream.read(1024 * 1024)
        require(not first.startswith(b"version https://git-lfs.github.com/spec/v1"),
                f"Git LFS pointer, not data: {path}. Run git lfs pull.")
        digest.update(first)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(rows: list[dict]) -> dict[str, float]:
    residuals = [row[f"pred_v{axis}"] - row[f"true_v{axis}"] for row in rows for axis in "xy"]
    squared = sum(value * value for value in residuals)
    mse = squared / len(residuals)
    means = {axis: sum(row[f"true_v{axis}"] for row in rows) / len(rows) for axis in "xy"}
    total = sum((row[f"true_v{axis}"] - means[axis]) ** 2 for row in rows for axis in "xy")
    return {"mse": mse, "rmse": math.sqrt(mse), "mae": sum(map(abs, residuals)) / len(residuals),
            "r2": 1.0 - squared / (total + 1e-12)}


def compare(actual: dict, expected: dict, label: str, tolerance: float = 2e-6) -> None:
    for key, value in expected.items():
        require(math.isclose(actual[key], value, rel_tol=tolerance, abs_tol=tolerance),
                f"{label}.{key}: {actual[key]} != {value}")


def verify(output_dir: Path | None) -> dict:
    for name, expected in ASSETS.items():
        require(sha256(ROOT / name) == expected, f"Asset checksum mismatch: {name}")
    report = {"assets": "PASS", "asset_count": len(ASSETS)}
    if output_dir is None:
        return report
    with (output_dir / "predictions_2026-02-02.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == ["day_name", "base_day_name", "t", "pred_vx", "pred_vy", "true_vx", "true_vy"],
                "Unexpected prediction CSV columns; run the complete demo with evaluation enabled.")
        rows = []
        for raw in reader:
            require(raw["day_name"] == raw["base_day_name"] == "2026-02-02", "Unexpected session name")
            row = {key: float(raw[key]) for key in ("t", "pred_vx", "pred_vy", "true_vx", "true_vy")}
            require(all(math.isfinite(value) for value in row.values()), "Non-finite prediction or ground truth")
            rows.append(row)
    require(len(rows) == 7950, f"Expected 7950 predictions, found {len(rows)}")
    require(all(math.isclose(row["t"], 1.515 + index * 0.03, abs_tol=1e-5)
                for index, row in enumerate(rows)), "Unexpected prediction time grid")
    summary = json.loads((output_dir / "prediction_summary_2026-02-02.json").read_text(encoding="utf-8"))
    expected_inputs = {
        "model_checkpoint": ASSETS["models/top64_local_peak/manifold_lstm_decoder.pt"],
        "spike_csv": ASSETS["data/demo/Online_demo_20260202.csv"],
        "calibration_spike_csv": ASSETS["data/demo/Online_demo_20260202.csv"],
        "voltage_dat": ASSETS["data/demo/M06-2026-02-02_voltage.dat"],
    }
    require(summary["input_sha256"] == expected_inputs, "Unexpected summary input fingerprints")
    require(summary["prediction_window"]["prediction_count"] == len(rows), "Incorrect summary prediction count")
    require(summary["calibration_window"]["time_end"] == 200.0, "Expected 200-second spike-only calibration")
    evaluation = summary["evaluation"]
    require(evaluation["valid_ground_truth_count"] == len(rows), "Incomplete ground truth")
    require(evaluation["ground_truth_used_for_calibration"] is False, "Unexpected calibration protocol")
    all_metrics = metrics(rows)
    compare(all_metrics, evaluation["metrics"], "CSV versus summary")
    # This permits small differences between CPU math libraries, not arbitrary outputs.
    compare(all_metrics, {"rmse": 0.39822554, "mae": 0.30646890, "r2": 0.88841956}, "reference demo", 1e-4)
    post = [row for row in rows if row["t"] - 50 * 0.03 > 200.0 + 0.03e-9]
    post_summary = evaluation["post_calibration"]
    require(post_summary["prediction_count"] == post_summary["valid_ground_truth_count"] == len(post),
            "Incorrect post-calibration count")
    compare(metrics(post), post_summary["metrics"], "post-calibration CSV versus summary")
    require(len(post) == 1283, "Unexpected complete-demo post-calibration count")
    compare(metrics(post), {"rmse": 0.44549288, "mae": 0.34432911, "r2": 0.82825918},
            "post-calibration reference demo", 1e-4)
    with (output_dir / "predictions_2026-02-02.png").open("rb") as stream:
        require(stream.read(8) == b"\x89PNG\r\n\x1a\n", "Missing or invalid prediction plot")
    report.update({"outputs": "PASS", "prediction_count": len(rows), "metrics": all_metrics,
                   "post_calibration_count": len(post), "post_calibration_metrics": metrics(post)})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "demo_prediction")
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(None if args.assets_only else args.output_dir), indent=2))
    except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
        parser.exit(1, f"Demo verification FAILED: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
