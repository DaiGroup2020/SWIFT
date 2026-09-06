#!/usr/bin/env python
from __future__ import annotations

"""
Spike-only manifold calibration and decoding for future M06 sessions.

Workflow:
1. Load a trained manifold decoder checkpoint.
2. Read a future day's spike CSV only.
3. Fit that day's FA manifold on an initial spike-only calibration window and align it
   to the saved reference manifold.
4. Reuse that fixed aligned manifold to project later latent windows and decode (vx, vy).
5. Optionally compare predictions against the original joystick/voltage file when
   it is available.
6. Save predictions plus an updated predictor state so the reference manifold can
   keep adapting day by day.
"""

import argparse
import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .manifold_decoder import (
    EncodedLatentLSTMRegressor,
    FAModel,
    StandardScaler,
    align_factor_model,
    bin_spike_times_to_counts,
    fit_factor_analysis,
    interpolate_columns,
    json_ready,
    project_latents,
    read_voltage_dat,
    regression_metrics,
    unique_preserve_order,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICT_DATA_DIR = SCRIPT_DIR / "data" / "predict"
DEFAULT_OUTPUT_DIR = Path("outputs") / "prediction"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "predict_config.json"

# Edit only this block for double-click usage.
# Leave a value empty to keep config/auto-discovery fallback.
USER_EDITABLE_SETTINGS: dict[str, object] = {
    "model_path": "",
    "spike_csv": "",
    "voltage_dat": "",
    "output_dir": "",
    "day_name": "",
    "no_daily_calibration": False,
    "calibration_time_start": 0.0,
    "calibration_time_end": 200.0,
    "time_start": None,
    "time_end": None,
    "plot_time_start": None,
    "plot_time_end": None,
    "plot_max_points": 3000,
    "fa_restarts": None,
    "fa_max_iter": None,
    "align_n": None,
    "align_th": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict future control signals using spike-only manifold calibration."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Training checkpoint or updated predictor state. If omitted, auto-discover the best available local model.",
    )
    parser.add_argument(
        "--spike-csv",
        type=Path,
        default=None,
        help="Future day's spike CSV. If omitted, auto-discover the most recent local spike CSV.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Optional JSON config used when running by double-click without CLI arguments.",
    )
    parser.add_argument(
        "--voltage-dat",
        type=Path,
        default=None,
        help="Optional raw joystick/voltage dat file used only for post-hoc evaluation.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--day-name", type=str, default=None, help="Optional label for the future session.")
    parser.add_argument("--calibration-spike-csv", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--calibration-day-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-daily-calibration",
        action="store_true",
        help="Skip fitting a future-day FA manifold and directly use the checkpoint reference manifold.",
    )
    parser.add_argument(
        "--calibration-time-start",
        type=float,
        default=None,
        help="Start time of the spike-only calibration window used to fit the future day's manifold.",
    )
    parser.add_argument(
        "--calibration-time-end",
        type=float,
        default=None,
        help="End time of the spike-only calibration window used to fit the future day's manifold.",
    )
    parser.add_argument("--time-start", type=float, default=None, help="Optional start time of the decoded prediction window.")
    parser.add_argument("--time-end", type=float, default=None, help="Optional end time of the decoded prediction window.")
    parser.add_argument("--plot-time-start", type=float, default=None, help="Optional plot window start time.")
    parser.add_argument("--plot-time-end", type=float, default=None, help="Optional plot window end time.")
    parser.add_argument(
        "--plot-max-points",
        type=int,
        default=None,
        help="Maximum number of plotted points after time filtering. Use 0 or a negative value to disable downsampling.",
    )
    parser.add_argument("--fa-restarts", type=int, default=None, help="Optional override for future-day FA restarts.")
    parser.add_argument("--fa-max-iter", type=int, default=None, help="Optional override for future-day FA iterations.")
    parser.add_argument("--align-n", type=int, default=None, help="Optional override for stable-unit count.")
    parser.add_argument("--align-th", type=float, default=None, help="Optional override for loading threshold.")
    parser.add_argument("--skip-reference-update", action="store_true", help="Do not save an updated predictor state.")
    parser.add_argument(
        "--update-state-path",
        type=Path,
        default=None,
        help="Optional path for the updated predictor state. Defaults to output_dir/predictor_state_after_<day>.pt",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalize_day_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    match_yyyymmdd = re.fullmatch(r"(\d{8})", text)
    if match_yyyymmdd:
        raw_full = match_yyyymmdd.group(1)
        return f"{raw_full[:4]}-{raw_full[4:6]}-{raw_full[6:8]}"
    match_compact = re.fullmatch(r"(\d{6})", text)
    if match_compact:
        raw_compact = match_compact.group(1)
        return f"20{raw_compact[:2]}-{raw_compact[2:4]}-{raw_compact[4:6]}"
    match_iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match_iso:
        return text
    raise ValueError(f"Unsupported day format: {raw}. Use YYYY-MM-DD or YYMMDD.")


def extract_base_day_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        direct = normalize_day_name(text)
    except Exception:
        direct = None
    if direct is not None:
        return direct
    match_iso = re.search(r"(20\d{2}-\d{2}-\d{2})", text)
    if match_iso:
        return normalize_day_name(match_iso.group(1))
    match_yyyymmdd = re.search(r"(20\d{6})", text)
    if match_yyyymmdd:
        return normalize_day_name(match_yyyymmdd.group(1))
    match_yymmdd = re.search(r"(?<!\d)(\d{6})(?!\d)", text)
    if match_yymmdd:
        return normalize_day_name(match_yymmdd.group(1))
    return None


def day_code_from_name(day_name: str) -> str:
    normalized = extract_base_day_name(day_name)
    if normalized is None:
        raise ValueError("day_name is empty")
    return normalized[2:4] + normalized[5:7] + normalized[8:10]


def infer_day_name(spike_csv: Path) -> str:
    match_after_template = re.search(r"tmpl\d{6}_(\d{8})", spike_csv.stem)
    if match_after_template:
        raw_full = match_after_template.group(1)
        return f"{raw_full[:4]}-{raw_full[4:6]}-{raw_full[6:8]}"
    match_yyyymmdd = re.search(r"(20\d{6})", spike_csv.stem)
    if match_yyyymmdd:
        raw_full = match_yyyymmdd.group(1)
        return f"{raw_full[:4]}-{raw_full[4:6]}-{raw_full[6:8]}"
    match_compact = re.search(r"(\d{6})(?!.*\d)", spike_csv.stem)
    if match_compact:
        raw = match_compact.group(1)
        return f"20{raw[:2]}-{raw[2:4]}-{raw[4:6]}"
    return spike_csv.stem


def load_predict_config(config_path: Path) -> dict[str, object]:
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Predict config must be a JSON object: {config_path}")
    return payload


def is_empty_value(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def resolve_optional_float(value: object) -> float | None:
    if is_empty_value(value):
        return None
    return float(value)


def resolve_optional_int(value: object) -> int | None:
    if is_empty_value(value):
        return None
    return int(value)


def resolve_optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def resolve_config_path(value: object, base_dir: Path) -> Path | None:
    if is_empty_value(value):
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def resolve_inline_settings(base_dir: Path) -> dict[str, object]:
    raw = USER_EDITABLE_SETTINGS
    day_name = str(raw.get("day_name")).strip() if not is_empty_value(raw.get("day_name")) else None
    return {
        "model_path": resolve_config_path(raw.get("model_path"), base_dir),
        "spike_csv": resolve_config_path(raw.get("spike_csv"), base_dir),
        "voltage_dat": resolve_config_path(raw.get("voltage_dat"), base_dir),
        "output_dir": resolve_config_path(raw.get("output_dir"), base_dir),
        "day_name": day_name,
        "no_daily_calibration": resolve_optional_bool(raw.get("no_daily_calibration")),
        "calibration_time_start": resolve_optional_float(raw.get("calibration_time_start")),
        "calibration_time_end": resolve_optional_float(raw.get("calibration_time_end")),
        "time_start": resolve_optional_float(raw.get("time_start")),
        "time_end": resolve_optional_float(raw.get("time_end")),
        "plot_time_start": resolve_optional_float(raw.get("plot_time_start")),
        "plot_time_end": resolve_optional_float(raw.get("plot_time_end")),
        "plot_max_points": resolve_optional_int(raw.get("plot_max_points")),
        "fa_restarts": resolve_optional_int(raw.get("fa_restarts")),
        "fa_max_iter": resolve_optional_int(raw.get("fa_max_iter")),
        "align_n": resolve_optional_int(raw.get("align_n")),
        "align_th": resolve_optional_float(raw.get("align_th")),
    }


def contains_smoke_path(path: Path) -> bool:
    return "smoke" in str(path).lower()


def read_summary_eval_rmse(summary_path: Path) -> float | None:
    if not summary_path.exists():
        return None
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    eval_metrics = payload.get("eval_metrics")
    if not isinstance(eval_metrics, dict):
        return None
    rmse = eval_metrics.get("rmse")
    return float(rmse) if rmse is not None else None


def predictor_state_day(path: Path) -> str | None:
    match = re.search(r"predictor_state_after_(\d{4}-\d{2}-\d{2})\.pt$", path.name)
    if not match:
        return None
    return match.group(1)


def auto_discover_model_path(target_day: str | None = None) -> Path:
    predictor_states = [
        path
        for path in SCRIPT_DIR.rglob("predictor_state_after_*.pt")
        if path.is_file() and not contains_smoke_path(path)
    ]
    if predictor_states:
        if target_day is not None:
            eligible = [
                path for path in predictor_states
                if predictor_state_day(path) is None or predictor_state_day(path) < target_day
            ]
            if eligible:
                eligible.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return eligible[0]
        else:
            predictor_states.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return predictor_states[0]

    candidates = [
        path
        for path in SCRIPT_DIR.rglob("manifold_lstm_decoder.pt")
        if path.is_file() and not contains_smoke_path(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            "No usable model checkpoint was found automatically. "
            "Please provide --model-path explicitly."
        )

    scored = []
    unscored = []
    for path in candidates:
        rmse = read_summary_eval_rmse(path.parent / "summary.json")
        if rmse is None:
            unscored.append(path)
        else:
            scored.append((rmse, -path.stat().st_mtime, path))

    if scored:
        scored.sort(key=lambda item: (item[0], item[1]))
        return scored[0][2]

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def spike_csv_sort_key(path: Path) -> tuple[int, float]:
    match = re.search(r"(\d{6})", path.stem)
    day_code = int(match.group(1)) if match else -1
    return day_code, path.stat().st_mtime


def auto_discover_spike_csv(day_name: str | None = None) -> Path:
    candidates = [path for path in DEFAULT_PREDICT_DATA_DIR.glob("Online_*.csv") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(
            "No spike CSV was found automatically in the predict data directory. "
            "Please provide --spike-csv explicitly."
        )
    if day_name is not None:
        target_code = day_code_from_name(day_name)
        matched = [path for path in candidates if target_code in path.stem]
        if not matched:
            raise FileNotFoundError(
                f"No spike CSV matching day {day_name} was found automatically. "
                "Please update predict_config.json or provide --spike-csv explicitly."
            )
        candidates = matched
    candidates.sort(key=spike_csv_sort_key, reverse=True)
    return candidates[0]


def load_checkpoint(model_path: Path, device: str) -> dict:
    require_materialized_file(model_path)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict) or not any(
        name in checkpoint for name in ("model_state_dict", "decoder_state_dict")
    ):
        raise ValueError(f"{model_path} is missing model_state_dict/decoder_state_dict")
    return checkpoint


def require_materialized_file(path: Path) -> None:
    """Fail clearly when a required asset is missing or still a Git LFS pointer."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input file does not exist: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(128)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(f"{path} is a Git LFS pointer. Run 'git lfs pull' in the repository first.")


def input_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fa_from_state(state: dict, fallback_score: float = float("nan")) -> FAModel:
    return FAModel(
        mean=np.asarray(state["mean"], dtype=np.float64),
        C=np.asarray(state["C"], dtype=np.float64),
        psi=np.asarray(state["psi"], dtype=np.float64),
        score=float(state.get("score", fallback_score)),
    )


def fa_to_state(fa: FAModel) -> dict:
    return {
        "mean": fa.mean,
        "C": fa.C,
        "psi": fa.psi,
        "score": fa.score,
    }


def restore_scaler(mean: np.ndarray, std: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(mean, dtype=np.float32)
    scaler.std_ = np.asarray(std, dtype=np.float32)
    return scaler


def load_spike_csv_aligned(
    spike_csv: Path,
    expected_unit_cols: list[str],
    *,
    allow_missing_units: bool = False,
    allow_positional_mapping: bool = False,
) -> tuple[list[np.ndarray], dict[str, object]]:
    require_materialized_file(spike_csv)
    df = pd.read_csv(spike_csv)
    available_unit_cols = [str(c) for c in df.columns if str(c).startswith("Unit")]
    if not available_unit_cols:
        raise ValueError(f"No Unit columns found in {spike_csv}")

    lookup = {col: df[col].dropna().to_numpy(dtype=np.float64) for col in available_unit_cols}
    for col, times in lookup.items():
        if not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError(f"{spike_csv}: {col} must contain finite, non-negative spike times in seconds.")
    matched_count = sum(1 for col in expected_unit_cols if col in lookup)

    if matched_count == 0 and allow_positional_mapping and len(available_unit_cols) >= len(expected_unit_cols):
        mapped_unit_cols = available_unit_cols[: len(expected_unit_cols)]
        unit_times = [lookup[col] for col in mapped_unit_cols]
        metadata = {
            "mapping_mode": "positional_fallback",
            "available_unit_cols": available_unit_cols,
            "mapped_unit_cols": mapped_unit_cols,
            "matched_unit_count": 0,
            "missing_unit_cols": [],
            "extra_unit_cols": available_unit_cols[len(expected_unit_cols) :],
        }
        return unit_times, metadata

    missing_unit_cols = [col for col in expected_unit_cols if col not in lookup]
    if matched_count == 0:
        raise ValueError(
            f"{spike_csv}: no checkpoint Unit identities match the input. "
            "Supply correctly identified units; use --allow-positional-unit-mapping only if the column order is known to match."
        )
    if missing_unit_cols and not allow_missing_units:
        preview = ", ".join(missing_unit_cols[:10])
        raise ValueError(
            f"{spike_csv}: missing {len(missing_unit_cols)} checkpoint Unit identities ({preview}). "
            "Supply the expected units or explicitly pass --allow-missing-units to assume zero spikes for missing units."
        )
    extra_unit_cols = [col for col in available_unit_cols if col not in expected_unit_cols]
    unit_times = [lookup.get(col, np.array([], dtype=np.float64)) for col in expected_unit_cols]
    metadata = {
        "mapping_mode": "name_match",
        "available_unit_cols": available_unit_cols,
        "mapped_unit_cols": list(expected_unit_cols),
        "matched_unit_count": matched_count,
        "missing_unit_cols": missing_unit_cols,
        "extra_unit_cols": extra_unit_cols,
    }
    return unit_times, metadata


def build_time_grid_from_spikes(
    unit_times: list[np.ndarray],
    bin_size_s: float,
    time_start: float | None = None,
    time_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    non_empty = [spikes for spikes in unit_times if spikes.size > 0]
    if not non_empty:
        raise ValueError("All spike units are empty; cannot build a time grid")

    spike_min = min(float(spikes.min()) for spikes in non_empty)
    spike_max = max(float(spikes.max()) for spikes in non_empty)
    t_start = float(time_start) if time_start is not None else math.floor(spike_min / bin_size_s) * bin_size_s
    t_end = float(time_end) if time_end is not None else math.ceil(spike_max / bin_size_s) * bin_size_s
    if t_end <= t_start:
        t_end = t_start + bin_size_s
    edges = np.arange(t_start, t_end + 0.5 * bin_size_s, bin_size_s, dtype=np.float64)
    if edges.size < 2:
        edges = np.array([t_start, t_start + bin_size_s], dtype=np.float64)
    centers = edges[:-1] + 0.5 * bin_size_s
    return edges, centers


def find_voltage_dat(day_name: str, spike_csv: Path, explicit_voltage_dat: Path | None) -> Path | None:
    if explicit_voltage_dat is not None:
        return explicit_voltage_dat

    def normalize_variant_suffix(raw: str | None) -> str:
        text = str(raw or "").strip().lower()
        text = re.sub(r"^[^0-9a-z]+", "", text)
        text = re.sub(r"[^0-9a-z]+", "_", text)
        return text.strip("_")

    def infer_spike_variant_suffix(path: Path) -> str:
        compact_day = day_name.replace("-", "")
        stem = path.stem
        idx = stem.rfind(compact_day)
        if idx < 0:
            return ""
        return normalize_variant_suffix(stem[idx + len(compact_day) :])

    def infer_voltage_variant_suffix(path: Path) -> str:
        stem = path.stem
        canonical_prefix = f"M06-{day_name}_voltage"
        if stem.startswith(canonical_prefix):
            return normalize_variant_suffix(stem[len(canonical_prefix) :])
        match = re.search(rf"{re.escape(day_name)}_voltage(?P<suffix>.*)$", stem)
        if match:
            return normalize_variant_suffix(match.group("suffix"))
        return ""

    def suffix_match_rank(target_suffix: str, candidate_suffix: str) -> int:
        if target_suffix:
            if candidate_suffix == target_suffix:
                return 0
            if candidate_suffix and (
                candidate_suffix.endswith(target_suffix) or target_suffix.endswith(candidate_suffix)
            ):
                return 1
            if not candidate_suffix:
                return 2
            return 3
        if not candidate_suffix:
            return 0
        return 1

    search_dirs = unique_preserve_order([str(spike_csv.parent), str(DEFAULT_PREDICT_DATA_DIR), str(SCRIPT_DIR / "data")])
    spike_variant_suffix = infer_spike_variant_suffix(spike_csv)
    candidates: list[tuple[int, Path]] = []
    for dir_index, raw_dir in enumerate(search_dirs):
        search_dir = Path(raw_dir)
        if not search_dir.exists():
            continue
        seen: dict[Path, None] = {}
        for pattern in (f"M06-{day_name}_voltage*.dat", f"*{day_name}*_voltage*.dat"):
            for path in sorted(search_dir.glob(pattern)):
                if path.is_file():
                    seen[path.resolve()] = None
        candidates.extend((dir_index, path) for path in seen.keys())
    if not candidates:
        return None

    def candidate_key(item: tuple[int, Path]) -> tuple[object, ...]:
        dir_index, path = item
        candidate_suffix = infer_voltage_variant_suffix(path)
        return (
            dir_index,
            suffix_match_rank(spike_variant_suffix, candidate_suffix),
            0 if path.name.startswith(f"M06-{day_name}_voltage") else 1,
            0 if not candidate_suffix else 1,
            -path.stat().st_mtime,
            len(path.name),
            path.name.lower(),
        )

    return min(candidates, key=candidate_key)[1]


def build_prediction_windows(latent: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    end_indices = np.arange(seq_len - 1, len(latent), dtype=np.int64)
    if len(end_indices) == 0:
        raise ValueError(
            f"Not enough spike bins ({len(latent)}) for seq_len={seq_len}. "
            "Try providing more data for the day."
        )
    X = np.empty((len(end_indices), seq_len, latent.shape[1]), dtype=np.float32)
    for i, end_idx in enumerate(end_indices):
        X[i] = latent[end_idx - seq_len + 1 : end_idx + 1]
    return X, end_indices


def infer_processing_window(
    unit_times: list[np.ndarray],
    bin_size_s: float,
    prediction_time_start: float | None,
    prediction_time_end: float | None,
    calibration_time_start: float | None,
    calibration_time_end: float | None,
) -> tuple[float, float]:
    non_empty = [spikes for spikes in unit_times if spikes.size > 0]
    if not non_empty:
        raise ValueError("All spike units are empty; cannot infer a processing window")

    spike_min = min(float(spikes.min()) for spikes in non_empty)
    spike_max = max(float(spikes.max()) for spikes in non_empty)
    default_start = math.floor(spike_min / bin_size_s) * bin_size_s
    default_end = math.ceil(spike_max / bin_size_s) * bin_size_s
    requested_start = [prediction_time_start if prediction_time_start is not None else default_start]
    requested_end = [prediction_time_end if prediction_time_end is not None else default_end]
    if calibration_time_start is not None:
        requested_start.append(calibration_time_start)
    if calibration_time_end is not None:
        requested_end.append(calibration_time_end)
    t_start = min(requested_start)
    t_end = max(requested_end)
    if t_end <= t_start:
        raise ValueError("The combined processing time window is empty. Check prediction and calibration times.")
    return float(t_start), float(t_end)


def build_range_mask(
    t: np.ndarray,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    mask = np.ones(len(t), dtype=bool)
    if start is not None:
        mask &= t >= float(start)
    if end is not None:
        mask &= t <= float(end)
    return mask


def save_prediction_plot(
    output_path: Path,
    t: np.ndarray,
    y_pred: np.ndarray,
    day_name: str,
    y_true: np.ndarray | None = None,
    plot_time_start: float | None = None,
    plot_time_end: float | None = None,
    max_points: int | None = 3000,
) -> None:
    mask = np.ones(len(t), dtype=bool)
    if plot_time_start is not None:
        mask &= t >= float(plot_time_start)
    if plot_time_end is not None:
        mask &= t <= float(plot_time_end)
    if not np.any(mask):
        mask = np.ones(len(t), dtype=bool)

    t_plot = t[mask]
    y_pred_plot = y_pred[mask]
    y_true_plot = y_true[mask] if y_true is not None else None

    if max_points is not None and max_points > 0 and len(t_plot) > max_points:
        sample_idx = np.linspace(0, len(t_plot) - 1, num=max_points, dtype=np.int64)
        sample_idx = np.unique(sample_idx)
        t_plot = t_plot[sample_idx]
        y_pred_plot = y_pred_plot[sample_idx]
        if y_true_plot is not None:
            y_true_plot = y_true_plot[sample_idx]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    if y_true_plot is not None:
        axes[0].plot(t_plot, y_true_plot[:, 0], label="true_vx", linewidth=1.0)
    axes[0].plot(t_plot, y_pred_plot[:, 0], label="pred_vx", linewidth=1.0)
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("vx")
    if y_true_plot is not None:
        axes[1].plot(t_plot, y_true_plot[:, 1], label="true_vy", linewidth=1.0)
    axes[1].plot(t_plot, y_pred_plot[:, 1], label="pred_vy", linewidth=1.0)
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("vy")
    axes[1].set_xlabel("time (s)")
    if y_true_plot is None:
        fig.suptitle(f"Predicted control voltage | {day_name}")
    else:
        fig.suptitle(f"Predicted vs true control voltage | {day_name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def update_reference_fa(
    reference_fa: FAModel,
    aligned_day_fa: FAModel,
    update_count: int,
    psi_floor: float,
) -> tuple[FAModel, int]:
    if update_count < 1:
        update_count = 1
    next_count = update_count + 1
    old_weight = update_count / next_count
    new_weight = 1.0 / next_count
    updated = FAModel(
        mean=old_weight * reference_fa.mean + new_weight * aligned_day_fa.mean,
        C=old_weight * reference_fa.C + new_weight * aligned_day_fa.C,
        psi=np.maximum(old_weight * reference_fa.psi + new_weight * aligned_day_fa.psi, float(psi_floor)),
        score=float(np.nanmean([reference_fa.score, aligned_day_fa.score])),
    )
    return updated, next_count


def make_output_paths(output_dir: Path, day_name: str) -> dict[str, Path]:
    safe_day = day_name.replace("/", "-").replace("\\", "-")
    return {
        "predictions_csv": output_dir / f"predictions_{safe_day}.csv",
        "predictions_png": output_dir / f"predictions_{safe_day}.png",
        "summary_json": output_dir / f"prediction_summary_{safe_day}.json",
        "updated_state": output_dir / f"predictor_state_after_{safe_day}.pt",
    }


def main() -> None:
    args = parse_args()
    inline_settings = resolve_inline_settings(SCRIPT_DIR)
    config_payload = load_predict_config(args.config)
    config_dir = args.config.resolve().parent

    config_day_name = str(config_payload.get("day_name")).strip() if not is_empty_value(config_payload.get("day_name")) else None
    if args.day_name is not None:
        args.day_name = str(args.day_name).strip() or None
    elif inline_settings["day_name"] is not None:
        args.day_name = inline_settings["day_name"]
    elif config_day_name is not None:
        args.day_name = config_day_name

    requested_base_day_name = extract_base_day_name(args.day_name)
    if args.model_path is None:
        args.model_path = (
            inline_settings["model_path"]
            or
            resolve_config_path(config_payload.get("model_path"), config_dir)
            or auto_discover_model_path(requested_base_day_name)
        )
    if args.spike_csv is None:
        args.spike_csv = (
            inline_settings["spike_csv"]
            or resolve_config_path(config_payload.get("spike_csv"), config_dir)
            or auto_discover_spike_csv(requested_base_day_name)
        )
    if args.voltage_dat is None:
        args.voltage_dat = inline_settings["voltage_dat"] or resolve_config_path(config_payload.get("voltage_dat"), config_dir)
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = (
            inline_settings["output_dir"]
            or resolve_config_path(config_payload.get("output_dir"), config_dir)
            or DEFAULT_OUTPUT_DIR
        )
    if not args.no_daily_calibration:
        inline_no_daily = inline_settings.get("no_daily_calibration")
        if inline_no_daily is True:
            args.no_daily_calibration = True
        elif inline_no_daily is None:
            config_no_daily = resolve_optional_bool(config_payload.get("no_daily_calibration"))
            if config_no_daily is True:
                args.no_daily_calibration = True
    if args.calibration_time_start is None:
        args.calibration_time_start = inline_settings["calibration_time_start"]
    if args.calibration_time_start is None:
        args.calibration_time_start = resolve_optional_float(config_payload.get("calibration_time_start"))
    if args.calibration_time_end is None:
        args.calibration_time_end = inline_settings["calibration_time_end"]
    if args.calibration_time_end is None:
        args.calibration_time_end = resolve_optional_float(config_payload.get("calibration_time_end"))
    if args.time_start is None:
        args.time_start = inline_settings["time_start"]
    if args.time_start is None:
        args.time_start = resolve_optional_float(config_payload.get("time_start"))
    if args.time_end is None:
        args.time_end = inline_settings["time_end"]
    if args.time_end is None:
        args.time_end = resolve_optional_float(config_payload.get("time_end"))
    if args.plot_time_start is None:
        args.plot_time_start = inline_settings["plot_time_start"]
    if args.plot_time_start is None:
        args.plot_time_start = resolve_optional_float(config_payload.get("plot_time_start"))
    if args.plot_time_end is None:
        args.plot_time_end = inline_settings["plot_time_end"]
    if args.plot_time_end is None:
        args.plot_time_end = resolve_optional_float(config_payload.get("plot_time_end"))
    if args.plot_max_points is None:
        args.plot_max_points = inline_settings["plot_max_points"]
    if args.plot_max_points is None:
        args.plot_max_points = resolve_optional_int(config_payload.get("plot_max_points"))
    if args.fa_restarts is None:
        args.fa_restarts = inline_settings["fa_restarts"]
    if args.fa_restarts is None:
        args.fa_restarts = resolve_optional_int(config_payload.get("fa_restarts"))
    if args.fa_max_iter is None:
        args.fa_max_iter = inline_settings["fa_max_iter"]
    if args.fa_max_iter is None:
        args.fa_max_iter = resolve_optional_int(config_payload.get("fa_max_iter"))
    if args.align_n is None:
        args.align_n = inline_settings["align_n"]
    if args.align_n is None:
        args.align_n = resolve_optional_int(config_payload.get("align_n"))
    if args.align_th is None:
        args.align_th = inline_settings["align_th"]
    if args.align_th is None:
        args.align_th = resolve_optional_float(config_payload.get("align_th"))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    session_name = args.day_name or infer_day_name(args.spike_csv)
    base_day_name = extract_base_day_name(session_name) or infer_day_name(args.spike_csv)
    paths = make_output_paths(output_dir, session_name)
    checkpoint = load_checkpoint(args.model_path, device="cpu")
    checkpoint_config = checkpoint.get("config", {})
    print(f"Using model checkpoint: {args.model_path.resolve()}")
    print(f"Using spike CSV: {args.spike_csv.resolve()}")
    if args.calibration_spike_csv is not None:
        print(f"Using external calibration spike CSV: {args.calibration_spike_csv.resolve()}")
    if args.calibration_spike_csv is not None and args.no_daily_calibration:
        raise ValueError("calibration_spike_csv cannot be combined with --no-daily-calibration.")
    if args.no_daily_calibration:
        print("Using direct reference mode: skip future-day FA fitting and project all bins with the checkpoint reference manifold.")
    if (not args.no_daily_calibration) and (args.calibration_time_start is not None or args.calibration_time_end is not None):
        print(
            "Using manifold calibration window: "
            f"start={args.calibration_time_start}, end={args.calibration_time_end}"
        )
    if args.time_start is not None or args.time_end is not None:
        print(f"Using prediction time window: start={args.time_start}, end={args.time_end}")
    if args.plot_time_start is not None or args.plot_time_end is not None:
        print(f"Using plot time window: start={args.plot_time_start}, end={args.plot_time_end}")

    model_kwargs = checkpoint["model_kwargs"]
    model = EncodedLatentLSTMRegressor(**model_kwargs).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    x_scaler = restore_scaler(checkpoint["x_mean"], checkpoint["x_std"])
    y_scaler = restore_scaler(checkpoint["y_mean"], checkpoint["y_std"])

    expected_unit_cols = list(checkpoint["unit_cols"])
    unit_times, unit_mapping = load_spike_csv_aligned(args.spike_csv, expected_unit_cols)
    bin_size_s = float(checkpoint_config["bin_size_s"])
    seq_len = int(checkpoint.get("sequence_length", model_kwargs["seq_len"]))
    n_latents = int(checkpoint_config["n_latents"])
    psi_floor = float(checkpoint_config["psi_floor"])
    fa_tol = float(checkpoint_config["fa_tol"])
    fa_restarts = int(args.fa_restarts if args.fa_restarts is not None else checkpoint_config.get("fa_restarts", 1))
    fa_max_iter = int(args.fa_max_iter or checkpoint_config["fa_max_iter"])
    align_n = int(args.align_n or checkpoint_config["align_n"])
    align_th = float(args.align_th or checkpoint_config["align_th"])
    reference_source_key = "reference_fa" if "reference_fa" in checkpoint else "base_fa"

    print(
        "Using FA/alignment params: "
        f"fa_restarts={fa_restarts}, fa_max_iter={fa_max_iter}, align_n={align_n}, align_th={align_th}"
    )
    print(f"Using reference manifold source: {reference_source_key}")

    processing_time_start, processing_time_end = infer_processing_window(
        unit_times=unit_times,
        bin_size_s=bin_size_s,
        prediction_time_start=args.time_start,
        prediction_time_end=args.time_end,
        calibration_time_start=None if args.no_daily_calibration or args.calibration_spike_csv is not None else args.calibration_time_start,
        calibration_time_end=None if args.no_daily_calibration or args.calibration_spike_csv is not None else args.calibration_time_end,
    )
    edges, t_centers = build_time_grid_from_spikes(
        unit_times=unit_times,
        bin_size_s=bin_size_s,
        time_start=processing_time_start,
        time_end=processing_time_end,
    )
    counts = bin_spike_times_to_counts(unit_times, edges)

    if "reference_fa" in checkpoint:
        reference_fa = fa_from_state(checkpoint["reference_fa"])
    else:
        reference_fa = fa_from_state(checkpoint["base_fa"])

    calibration_mask = np.zeros(len(t_centers), dtype=bool)
    calibration_source_day_name = session_name
    calibration_source_base_day_name = base_day_name
    calibration_source_spike_csv = args.spike_csv
    if args.no_daily_calibration:
        aligned_day_fa = reference_fa
        latent = project_latents(counts, reference_fa)
        alignment_summary = {
            "mode": "direct_reference",
            "stable_unit_count": None,
            "loading_rmse_before": None,
            "loading_rmse_after": None,
        }
    else:
        if args.calibration_spike_csv is not None:
            calibration_source_day_name = args.calibration_day_name or infer_day_name(args.calibration_spike_csv)
            calibration_source_base_day_name = extract_base_day_name(calibration_source_day_name) or infer_day_name(args.calibration_spike_csv)
            calibration_source_spike_csv = args.calibration_spike_csv
            calibration_unit_times, _calibration_mapping = load_spike_csv_aligned(args.calibration_spike_csv, expected_unit_cols)
            calibration_processing_time_start, calibration_processing_time_end = infer_processing_window(
                unit_times=calibration_unit_times,
                bin_size_s=bin_size_s,
                prediction_time_start=None,
                prediction_time_end=None,
                calibration_time_start=args.calibration_time_start,
                calibration_time_end=args.calibration_time_end,
            )
            calibration_edges, calibration_t_centers = build_time_grid_from_spikes(
                unit_times=calibration_unit_times,
                bin_size_s=bin_size_s,
                time_start=calibration_processing_time_start,
                time_end=calibration_processing_time_end,
            )
            calibration_counts_all = bin_spike_times_to_counts(calibration_unit_times, calibration_edges)
            calibration_mask = build_range_mask(
                calibration_t_centers,
                start=args.calibration_time_start,
                end=args.calibration_time_end,
            )
        else:
            calibration_counts_all = counts
            calibration_mask = build_range_mask(
                t_centers,
                start=args.calibration_time_start,
                end=args.calibration_time_end,
            )
        if not np.any(calibration_mask):
            raise ValueError(
                "Calibration time window does not contain any spike bins. "
                "Please adjust calibration_time_start/calibration_time_end."
            )
        calibration_counts = calibration_counts_all[calibration_mask]
        if len(calibration_counts) < max(8, n_latents + 1):
            raise ValueError(
                f"Calibration window is too short for FA fitting: got {len(calibration_counts)} bins, "
                f"need at least {max(8, n_latents + 1)}."
            )

        day_fa = fit_factor_analysis(
            X=calibration_counts,
            n_latents=n_latents,
            n_restarts=fa_restarts,
            max_iter=fa_max_iter,
            tol=fa_tol,
            psi_floor=psi_floor,
            seed=int(checkpoint_config.get("seed", 42)),
        )

        aligned_day_fa, alignment = align_factor_model(
            base_fa=reference_fa,
            day_fa=day_fa,
            base_unit_cols=expected_unit_cols,
            day_unit_cols=list(unit_mapping.get("mapped_unit_cols", expected_unit_cols)),
            day_name=session_name,
            align_n=align_n,
            align_th=align_th,
        )
        latent = project_latents(counts, aligned_day_fa)
        alignment_summary = {
            "mode": "previous_day_calibration" if args.calibration_spike_csv is not None else "daily_calibration",
            "stable_unit_count": int(alignment.stable_mask.sum()),
            "loading_rmse_before": alignment.loading_rmse_before,
            "loading_rmse_after": alignment.loading_rmse_after,
        }
    X_pred_all, end_indices_all = build_prediction_windows(latent, seq_len)
    t_pred_all = t_centers[end_indices_all]
    prediction_time_start = (
        args.time_start
        if args.time_start is not None
        else (None if args.no_daily_calibration or args.calibration_spike_csv is not None else args.calibration_time_end)
    )
    prediction_time_end = args.time_end if args.time_end is not None else float(t_centers[-1])
    prediction_mask = build_range_mask(
        t_pred_all,
        start=prediction_time_start,
        end=prediction_time_end,
    )
    if not np.any(prediction_mask):
        raise ValueError("Prediction time window does not contain any decodable sequence endpoints.")
    X_pred = X_pred_all[prediction_mask]
    end_indices = end_indices_all[prediction_mask]
    X_pred_scaled = x_scaler.transform(X_pred).astype(np.float32, copy=False)

    with torch.no_grad():
        y_pred_scaled = model(torch.from_numpy(X_pred_scaled).to(args.device)).cpu().numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    t_pred = t_pred_all[prediction_mask]
    voltage_dat = find_voltage_dat(base_day_name, args.spike_csv, args.voltage_dat)
    if voltage_dat is not None and voltage_dat.exists():
        print(f"Using voltage DAT for evaluation: {voltage_dat.resolve()}")
    else:
        print("No voltage DAT found automatically; prediction will run without post-hoc evaluation.")
    y_true_eval: np.ndarray | None = None
    valid_eval_mask: np.ndarray | None = None
    eval_metrics: dict[str, float] | None = None
    evaluation_source: str | None = None
    if voltage_dat is not None and voltage_dat.exists():
        voltage_df = read_voltage_dat(voltage_dat)
        y_true_full = interpolate_columns(voltage_df, t_pred, ["vx", "vy"])
        valid_eval_mask = np.isfinite(y_true_full).all(axis=1)
        if np.any(valid_eval_mask):
            y_true_eval = y_true_full[valid_eval_mask].astype(np.float32, copy=False)
            eval_metrics = regression_metrics(y_true_eval, y_pred[valid_eval_mask])
            evaluation_source = str(voltage_dat)
        else:
            y_true_eval = None
            valid_eval_mask = None
    if valid_eval_mask is None:
        valid_eval_mask = np.ones(len(y_pred), dtype=bool)

    predictions_df = pd.DataFrame(
        {
            "day_name": session_name,
            "base_day_name": base_day_name,
            "t": t_pred,
            "pred_vx": y_pred[:, 0],
            "pred_vy": y_pred[:, 1],
        }
    )
    if y_true_eval is not None:
        predictions_df["true_vx"] = np.nan
        predictions_df["true_vy"] = np.nan
        predictions_df.loc[valid_eval_mask, "true_vx"] = y_true_eval[:, 0]
        predictions_df.loc[valid_eval_mask, "true_vy"] = y_true_eval[:, 1]
    predictions_df.to_csv(paths["predictions_csv"], index=False)
    if y_true_eval is not None:
        save_prediction_plot(
            paths["predictions_png"],
            t_pred[valid_eval_mask],
            y_pred[valid_eval_mask],
            session_name,
            y_true=y_true_eval,
            plot_time_start=args.plot_time_start,
            plot_time_end=args.plot_time_end,
            max_points=args.plot_max_points,
        )
    else:
        save_prediction_plot(
            paths["predictions_png"],
            t_pred,
            y_pred,
            session_name,
            plot_time_start=args.plot_time_start,
            plot_time_end=args.plot_time_end,
            max_points=args.plot_max_points,
        )

    current_update_count = int(checkpoint.get("reference_update_count", len(checkpoint.get("train_days", [1]))))
    updated_state_path: Path | None = None
    next_update_count = current_update_count
    if not args.no_daily_calibration and args.calibration_spike_csv is None and not args.skip_reference_update:
        updated_reference_fa, next_update_count = update_reference_fa(
            reference_fa=reference_fa,
            aligned_day_fa=aligned_day_fa,
            update_count=current_update_count,
            psi_floor=psi_floor,
        )
        updated_checkpoint = deepcopy(checkpoint)
        updated_checkpoint["reference_fa"] = fa_to_state(updated_reference_fa)
        updated_checkpoint["reference_update_count"] = next_update_count
        source_days = unique_preserve_order([*checkpoint.get("reference_source_days", checkpoint.get("train_days", [])), session_name])
        updated_checkpoint["reference_source_days"] = source_days
        updated_checkpoint["last_prediction_day"] = session_name
        updated_checkpoint["last_prediction_spike_csv"] = str(args.spike_csv)
        updated_state_path = args.update_state_path or paths["updated_state"]
        torch.save(updated_checkpoint, updated_state_path)

    summary = {
        "model_path": str(args.model_path),
        "spike_csv": str(args.spike_csv),
        "voltage_dat": str(voltage_dat) if voltage_dat is not None and voltage_dat.exists() else None,
        "day_name": session_name,
        "base_day_name": base_day_name,
        "calibration_source": {
            "day_name": calibration_source_day_name,
            "base_day_name": calibration_source_base_day_name,
            "spike_csv": str(calibration_source_spike_csv),
            "same_as_prediction_day": calibration_source_day_name == session_name,
        },
        "inference_mode": "direct_reference" if args.no_daily_calibration else "daily_calibration",
        "no_daily_calibration": bool(args.no_daily_calibration),
        "bin_size_s": bin_size_s,
        "sequence_length": seq_len,
        "calibration_time_start": None if args.no_daily_calibration else args.calibration_time_start,
        "calibration_time_end": None if args.no_daily_calibration else args.calibration_time_end,
        "prediction_time_start": float(t_pred[0]) if len(t_pred) else None,
        "prediction_time_end": float(t_pred[-1]) if len(t_pred) else None,
        "processing_time_start": processing_time_start,
        "processing_time_end": processing_time_end,
        "calibration_bin_count": int(np.sum(calibration_mask)),
        "prediction_count": int(len(predictions_df)),
        "time_start": float(t_centers[0]),
        "time_end": float(t_centers[-1]),
        "plot_time_start": args.plot_time_start,
        "plot_time_end": args.plot_time_end,
        "plot_max_points": args.plot_max_points,
        "effective_params": {
            "fa_restarts": fa_restarts,
            "fa_max_iter": fa_max_iter,
            "align_n": align_n,
            "align_th": align_th,
            "reference_manifold_source": reference_source_key,
        },
        "unit_mapping": {
            "mapping_mode": unit_mapping["mapping_mode"],
            "expected_unit_count": len(expected_unit_cols),
            "available_unit_count": len(unit_mapping["available_unit_cols"]),
            "mapped_unit_count": len(unit_mapping.get("mapped_unit_cols", [])),
            "matched_unit_count": int(unit_mapping["matched_unit_count"]),
            "missing_unit_count": len(unit_mapping["missing_unit_cols"]),
            "extra_unit_count": len(unit_mapping["extra_unit_cols"]),
            "missing_unit_cols": unit_mapping["missing_unit_cols"][:20],
            "extra_unit_cols": unit_mapping["extra_unit_cols"][:20],
        },
        "alignment": alignment_summary,
        "evaluation": {
            "has_ground_truth": y_true_eval is not None,
            "valid_ground_truth_count": int(np.sum(valid_eval_mask)) if y_true_eval is not None else 0,
            "ground_truth_source": evaluation_source,
            "metrics": eval_metrics,
        },
        "reference_update": {
            "updated": False if args.no_daily_calibration or args.calibration_spike_csv is not None else not args.skip_reference_update,
            "previous_count": current_update_count,
            "next_count": next_update_count,
            "updated_state_path": str(updated_state_path) if updated_state_path is not None else None,
            "reason": (
                "disabled_in_direct_reference_mode"
                if args.no_daily_calibration
                else ("disabled_in_previous_day_calibration_mode" if args.calibration_spike_csv is not None else None)
            ),
        },
    }
    with paths["summary_json"].open("w", encoding="utf-8") as f:
        json.dump(json_ready(summary), f, indent=2, ensure_ascii=False)

    print(f"Predictions written to: {paths['predictions_csv'].resolve()}")
    print(f"Prediction plot written to: {paths['predictions_png'].resolve()}")
    if eval_metrics is not None:
        print(f"Evaluation metrics: {eval_metrics}")
    if updated_state_path is not None:
        print(f"Updated predictor state written to: {updated_state_path.resolve()}")
    elif args.no_daily_calibration:
        print("Reference manifold update skipped in direct reference mode.")
    elif args.calibration_spike_csv is not None:
        print("Reference manifold update skipped in previous-day calibration mode.")


if __name__ == "__main__":
    main()
