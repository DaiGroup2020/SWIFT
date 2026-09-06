#!/usr/bin/env python
from __future__ import annotations

"""
Spike-only prediction for NoMAD-style latent dynamics + LSTM checkpoints.

Workflow:
1. Load a trained `nomad_lstm` checkpoint.
2. Read a future day's spike CSV only.
3. Use an initial spike-only calibration window to fit a day-specific alignment net.
4. Project the whole session into the shared latent-dynamics space.
5. Decode (vx, vy) with the saved encoded LSTM decoder.
6. Optionally compare predictions against the original joystick/voltage file.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .manifold_decoder import (
    StandardScaler,
    bin_spike_times_to_counts,
    interpolate_columns,
    read_voltage_dat,
    regression_metrics,
)
from .manifold_decoder_MoMAD import (
    Config,
    ReferenceDynamicsModel,
    ResidualAlignmentNet,
    build_count_windows,
    encode_full_sequence,
    fit_day_alignment_modules,
    set_seed,
)
from .predict import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PREDICT_DATA_DIR,
    auto_discover_spike_csv,
    build_prediction_windows,
    build_range_mask,
    build_time_grid_from_spikes,
    extract_base_day_name,
    find_voltage_dat,
    infer_day_name,
    infer_processing_window,
    input_file_sha256,
    load_checkpoint,
    load_predict_config,
    load_spike_csv_aligned,
    normalize_day_name,
    require_materialized_file,
    resolve_config_path,
    resolve_optional_float,
    resolve_optional_int,
    restore_scaler,
    save_prediction_plot,
)
from .manifold_decoder import build_decoder_model_from_kwargs


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "predict_nomad_config.json"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "model_Nomad"
PREDICT_ALIGNMENT_SEED = 42

USER_EDITABLE_SETTINGS: dict[str, object] = {
    "model_path": "",
    "spike_csv": "",
    "voltage_dat": "",
    "output_dir": "",
    "day_name": "",
    "no_daily_calibration": None,
    "calibration_time_start": None,
    "calibration_time_end": None,
    "time_start": None,
    "time_end": None,
    "plot_time_start": None,
    "plot_time_end": None,
    "plot_max_points": None,
}


def is_empty_value(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


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
    }


def auto_discover_nomad_model_path() -> Path:
    if not DEFAULT_MODEL_DIR.exists():
        raise FileNotFoundError(f"No NoMAD model directory exists: {DEFAULT_MODEL_DIR}")
    candidates = [path for path in DEFAULT_MODEL_DIR.rglob("manifold_lstm_decoder.pt") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No NoMAD checkpoints were found in {DEFAULT_MODEL_DIR}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict future control signals using NoMAD-style spike-only calibration.")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--spike-csv", type=Path, default=None)
    parser.add_argument("--allow-missing-units", action="store_true", help="Explicitly assume zero spikes for missing checkpoint Unit identities (at least one identity must match).")
    parser.add_argument("--allow-positional-unit-mapping", action="store_true", help="Allow first-N column order mapping only when no Unit identities match. Use only with a verified identity mapping.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--voltage-dat", type=Path, default=None)
    parser.add_argument("--skip-evaluation", action="store_true", help="Do not read or auto-discover voltage ground truth; write spike-only predictions.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace this session's existing prediction CSV, plot, and summary.")
    parser.add_argument("--day-name", type=str, default=None)
    parser.add_argument("--calibration-spike-csv", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--calibration-day-name", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-daily-calibration",
        action="store_true",
        help="Skip day-specific NoMAD alignment and run the checkpoint shared GRU dynamics + decoder directly.",
    )
    parser.add_argument("--calibration-time-start", type=float, default=None)
    parser.add_argument("--calibration-time-end", type=float, default=None)
    parser.add_argument("--time-start", type=float, default=None, help="First decoded endpoint to retain; default keeps the whole-session replay, including calibration.")
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--plot-time-start", type=float, default=None)
    parser.add_argument("--plot-time-end", type=float, default=None)
    parser.add_argument("--plot-max-points", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-reference-update", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--update-state-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fa-restarts", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--fa-max-iter", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--align-n", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--align-th", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def make_output_paths(output_dir: Path, day_name: str) -> dict[str, Path]:
    safe_day = day_name.replace("/", "-").replace("\\", "-")
    return {
        "predictions_csv": output_dir / f"predictions_{safe_day}.csv",
        "predictions_png": output_dir / f"predictions_{safe_day}.png",
        "summary_json": output_dir / f"prediction_summary_{safe_day}.json",
    }


def prepare_output_paths(paths: dict[str, Path], *, overwrite: bool) -> None:
    """Check every output before starting the expensive calibration step."""
    directories = [path for path in paths.values() if path.is_dir()]
    if directories:
        raise IsADirectoryError(f"An output path is an existing directory: {directories[0]}")
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Prediction outputs already exist: {names}. Choose another --output-dir or pass --overwrite.")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)


def post_calibration_window_mask(
    t_pred: np.ndarray, *, sequence_length: int, bin_size_s: float, calibration_time_end: float
) -> np.ndarray:
    """Select decoder windows whose first bin is strictly after calibration.

    The shared GRU still carries causal state from earlier bins. This mask
    excludes overlapping decoder windows; it does not reset recurrent history.
    """
    first_bin_times = t_pred - (sequence_length - 1) * bin_size_s
    return first_bin_times > calibration_time_end + bin_size_s * 1e-9


def evaluate_prediction_segments(
    t_pred: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray | None,
    *,
    sequence_length: int,
    bin_size_s: float,
    same_session_calibration: bool,
    calibration_time_end: float | None,
) -> dict[str, object]:
    """Keep whole-replay metrics and report the non-overlapping future segment."""
    valid = np.zeros(len(t_pred), dtype=bool) if y_true is None else np.isfinite(y_true).all(axis=1)
    metrics = regression_metrics(y_true[valid], y_pred[valid]) if valid.any() else None
    result: dict[str, object] = {
        "metrics": metrics,
        "valid_ground_truth_count": int(valid.sum()),
        "metrics_scope": "all_decoded_endpoints_with_ground_truth",
        "ground_truth_used_for_calibration": False,
        "post_calibration": None,
    }
    if same_session_calibration and calibration_time_end is not None:
        post = post_calibration_window_mask(
            t_pred,
            sequence_length=sequence_length,
            bin_size_s=bin_size_s,
            calibration_time_end=calibration_time_end,
        )
        post_valid = post & valid
        result["post_calibration"] = {
            "selection_rule": "first_decoder_window_bin_strictly_after_calibration_time_end",
            "calibration_time_end": calibration_time_end,
            "recurrent_history": "continuous_causal_state_including_calibration",
            "prediction_count": int(post.sum()),
            "time_start": float(t_pred[post][0]) if post.any() else None,
            "time_end": float(t_pred[post][-1]) if post.any() else None,
            "valid_ground_truth_count": int(post_valid.sum()),
            "metrics": regression_metrics(y_true[post_valid], y_pred[post_valid]) if post_valid.any() else None,
        }
    return result


def train_day_alignment_for_prediction(
    calibration_counts: np.ndarray,
    reference_model: ReferenceDynamicsModel,
    checkpoint: dict,
    device: str,
    seq_len: int,
) -> tuple[ResidualAlignmentNet, nn.Module, nn.Module, dict[str, object] | None, dict[str, float]]:
    config_payload = checkpoint.get("config", {})
    end_idx = np.arange(seq_len - 1, len(calibration_counts), dtype=np.int64)
    X_day = build_count_windows(calibration_counts, seq_len, end_idx)
    if len(X_day) == 0:
        raise ValueError("Calibration window is too short for the checkpoint sequence length.")

    ref_stats = checkpoint["reference_stats"]
    align_kwargs = checkpoint["alignment_network_kwargs"]
    set_seed(PREDICT_ALIGNMENT_SEED)
    runtime_config = Config(
        seed=PREDICT_ALIGNMENT_SEED,
        bin_size_s=float(config_payload.get("bin_size_s", 0.03)),
        window_ms=max(seq_len - 1, 1) * float(config_payload.get("bin_size_s", 0.03)) * 1000.0,
        alignment_hidden_dim=int(align_kwargs["hidden_dim"]),
        alignment_batch_size=int(config_payload.get("alignment_batch_size", 256)),
        alignment_epochs=int(config_payload.get("alignment_epochs", 18)),
        alignment_lr=float(config_payload.get("alignment_lr", 5e-4)),
        alignment_patience=int(config_payload.get("alignment_patience", 5)),
        alignment_weight_decay=float(config_payload.get("alignment_weight_decay", 1e-4)),
        recon_weight=float(config_payload.get("recon_weight", 0.10)),
        align_kl_weight=float(config_payload.get("align_kl_weight", 0.10)),
        align_identity_weight=float(config_payload.get("align_identity_weight", 1e-4)),
        input_norm_sigma_ms=float(config_payload.get("input_norm_sigma_ms", 20.0)),
        disable_input_normalization=bool(config_payload.get("disable_input_normalization", False)),
        dropout=float(align_kwargs.get("dropout", config_payload.get("dropout", 0.0))),
        grad_clip=float(config_payload.get("grad_clip", 1.0)),
        device=device,
    )
    align_net, day_readin, day_rate_head, best_metrics, input_normalization = fit_day_alignment_modules(
        day_counts=calibration_counts,
        reference_model=reference_model,
        reference_mean=np.asarray(ref_stats["mean"], dtype=np.float32),
        reference_cov=np.asarray(ref_stats["cov"], dtype=np.float32),
        config=runtime_config,
    )
    return align_net, day_readin, day_rate_head, input_normalization, best_metrics


def main() -> None:
    args = parse_args()
    inline_settings = resolve_inline_settings(SCRIPT_DIR)
    config_payload = load_predict_config(args.config)
    config_dir = args.config.resolve().parent

    if args.day_name is not None:
        args.day_name = str(args.day_name).strip() or None
    elif inline_settings["day_name"] is not None:
        args.day_name = inline_settings["day_name"]
    elif "day_name" in config_payload and config_payload.get("day_name"):
        args.day_name = str(config_payload.get("day_name")).strip()

    requested_base_day_name = extract_base_day_name(args.day_name)

    if args.model_path is None:
        args.model_path = (
            inline_settings["model_path"]
            or resolve_config_path(config_payload.get("model_path"), config_dir)
            or auto_discover_nomad_model_path()
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
        args.output_dir = inline_settings["output_dir"] or resolve_config_path(config_payload.get("output_dir"), config_dir) or DEFAULT_OUTPUT_DIR
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
    if args.calibration_time_start is None:
        args.calibration_time_start = 0.0
    if args.calibration_time_end is None:
        args.calibration_time_end = 200.0
    if args.plot_max_points is None:
        args.plot_max_points = 3000
    if not args.no_daily_calibration and args.calibration_time_end <= args.calibration_time_start:
        raise ValueError("Calibration time end must be greater than calibration time start.")
    if args.time_start is not None and args.time_end is not None and args.time_end <= args.time_start:
        raise ValueError("Prediction time end must be greater than prediction time start.")
    if args.calibration_spike_csv is not None and args.no_daily_calibration:
        raise ValueError("calibration_spike_csv cannot be combined with --no-daily-calibration.")

    output_dir = args.output_dir.resolve()
    session_name = args.day_name or infer_day_name(args.spike_csv)
    base_day_name = extract_base_day_name(session_name) or infer_day_name(args.spike_csv)
    paths = make_output_paths(output_dir, session_name)
    prepare_output_paths(paths, overwrite=args.overwrite)
    if args.voltage_dat is not None and not args.skip_evaluation:
        require_materialized_file(args.voltage_dat)

    checkpoint = load_checkpoint(args.model_path, device="cpu")
    if checkpoint.get("model_family") != "nomad_lstm":
        raise ValueError(f"{args.model_path} is not a nomad_lstm checkpoint.")

    reference_model = ReferenceDynamicsModel(**checkpoint["reference_model_kwargs"]).to(args.device)
    reference_model.load_state_dict(checkpoint["reference_model_state_dict"])
    reference_model.eval()
    checkpoint_config = checkpoint.get("config", {})
    decoder_kwargs = dict(checkpoint.get("decoder_kwargs", checkpoint.get("model_kwargs")))
    if "decoder_variant" not in decoder_kwargs and checkpoint_config.get("decoder_variant"):
        decoder_kwargs["decoder_variant"] = checkpoint_config.get("decoder_variant")
    decoder = build_decoder_model_from_kwargs(decoder_kwargs).to(args.device)
    decoder_state = checkpoint["decoder_state_dict"] if "decoder_state_dict" in checkpoint else checkpoint["model_state_dict"]
    decoder.load_state_dict(decoder_state)
    decoder.eval()
    x_scaler = restore_scaler(checkpoint["x_mean"], checkpoint["x_std"])
    y_scaler = restore_scaler(checkpoint["y_mean"], checkpoint["y_std"])

    expected_unit_cols = list(checkpoint["unit_cols"])
    unit_times, unit_mapping = load_spike_csv_aligned(
        args.spike_csv, expected_unit_cols,
        allow_missing_units=args.allow_missing_units,
        allow_positional_mapping=args.allow_positional_unit_mapping,
    )
    bin_size_s = float(checkpoint_config["bin_size_s"])
    seq_len = int(checkpoint["sequence_length"] if "sequence_length" in checkpoint else decoder_kwargs["seq_len"])

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
    calibration_mask = np.zeros(len(t_centers), dtype=bool)
    calibration_source_day_name = session_name
    calibration_source_base_day_name = base_day_name
    calibration_source_spike_csv = args.spike_csv
    if args.no_daily_calibration:
        align_net = None
        day_readin = None
        day_rate_head = None
        day_input_normalization = checkpoint.get("reference_input_normalization")
        align_metrics = {
            "mode": "direct_shared_dynamics",
            "loss": None,
            "recon_loss": None,
            "kl_loss": None,
            "identity_loss": None,
        }
    else:
        if args.calibration_spike_csv is not None:
            calibration_source_day_name = args.calibration_day_name or infer_day_name(args.calibration_spike_csv)
            calibration_source_base_day_name = extract_base_day_name(calibration_source_day_name) or infer_day_name(args.calibration_spike_csv)
            calibration_source_spike_csv = args.calibration_spike_csv
            calibration_unit_times, _calibration_mapping = load_spike_csv_aligned(
                args.calibration_spike_csv, expected_unit_cols,
                allow_missing_units=args.allow_missing_units,
                allow_positional_mapping=args.allow_positional_unit_mapping,
            )
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
            calibration_mask = build_range_mask(calibration_t_centers, start=args.calibration_time_start, end=args.calibration_time_end)
        else:
            calibration_counts_all = counts
            calibration_mask = build_range_mask(t_centers, start=args.calibration_time_start, end=args.calibration_time_end)
        if not np.any(calibration_mask):
            raise ValueError("Calibration time window does not contain any spike bins.")
        calibration_counts = calibration_counts_all[calibration_mask]
        align_net, day_readin, day_rate_head, day_input_normalization, align_metrics = train_day_alignment_for_prediction(
            calibration_counts=calibration_counts,
            reference_model=reference_model,
            checkpoint=checkpoint,
            device=args.device,
            seq_len=seq_len,
        )

    latent, _rates = encode_full_sequence(
        reference_model=reference_model,
        counts=counts,
        device=args.device,
        chunk_len=int(checkpoint_config.get("chunk_len", 4096)),
        alignment_net=align_net,
        input_normalization=day_input_normalization,
        readin_layer=day_readin,
        rate_head=day_rate_head,
    )
    X_pred, end_indices = build_prediction_windows(latent, seq_len)
    t_pred = t_centers[end_indices]
    pred_mask = build_range_mask(t_pred, start=args.time_start, end=args.time_end)
    if not np.any(pred_mask):
        raise ValueError("Prediction time window does not contain any valid decoded windows.")
    X_pred = X_pred[pred_mask]
    t_pred = t_pred[pred_mask]

    pred_ds = TensorDataset(torch.from_numpy(x_scaler.transform(X_pred).astype(np.float32, copy=False)))
    pred_loader = DataLoader(pred_ds, batch_size=int(checkpoint_config.get("batch_size", 5000)), shuffle=False)
    y_pred_scaled_parts: list[np.ndarray] = []
    with torch.no_grad():
        for (xb,) in pred_loader:
            xb = xb.to(args.device)
            y_pred_scaled_parts.append(decoder(xb).cpu().numpy())
    y_pred = y_scaler.inverse_transform(np.concatenate(y_pred_scaled_parts, axis=0))

    voltage_dat = None if args.skip_evaluation else find_voltage_dat(base_day_name, args.spike_csv, args.voltage_dat)
    y_true_full = None
    if voltage_dat is not None and voltage_dat.exists():
        require_materialized_file(voltage_dat)
        voltage_df = read_voltage_dat(voltage_dat)
        y_true_full = interpolate_columns(voltage_df, t_pred, ["vx", "vy"])
    same_session_calibration = (
        not args.no_daily_calibration and calibration_source_spike_csv.resolve() == args.spike_csv.resolve()
    )
    evaluation = evaluate_prediction_segments(
        t_pred, y_pred, y_true_full,
        sequence_length=seq_len,
        bin_size_s=bin_size_s,
        same_session_calibration=same_session_calibration,
        calibration_time_end=args.calibration_time_end,
    )
    metrics = evaluation["metrics"]

    predictions_df = pd.DataFrame(
        {
            "day_name": session_name,
            "base_day_name": base_day_name,
            "t": t_pred,
            "pred_vx": y_pred[:, 0],
            "pred_vy": y_pred[:, 1],
        }
    )
    if y_true_full is not None and evaluation["valid_ground_truth_count"]:
        predictions_df["true_vx"] = y_true_full[:, 0]
        predictions_df["true_vy"] = y_true_full[:, 1]
    predictions_df.to_csv(paths["predictions_csv"], index=False)

    save_prediction_plot(
        output_path=paths["predictions_png"],
        t=t_pred,
        y_pred=y_pred,
        day_name=session_name,
        y_true=y_true_full,
        plot_time_start=args.plot_time_start,
        plot_time_end=args.plot_time_end,
        max_points=args.plot_max_points,
    )

    alignment_payload = {
        "matched_unit_count": int(unit_mapping.get("matched_unit_count", len(expected_unit_cols))),
        "missing_unit_count": len(unit_mapping.get("missing_unit_cols", [])),
        "extra_unit_count": len(unit_mapping.get("extra_unit_cols", [])),
        "mapping_mode": unit_mapping.get("mapping_mode", "unknown"),
        "calibration_bin_count": int(calibration_mask.sum()),
        "alignment_mode": (
            "direct_shared_dynamics"
            if args.no_daily_calibration
            else ("previous_day_calibration" if args.calibration_spike_csv is not None else checkpoint.get("alignment_mode", "align_net_only_legacy"))
        ),
        "input_normalization": {
            "enabled": day_input_normalization is not None,
            "sigma_ms": None if day_input_normalization is None else float(day_input_normalization.get("sigma_ms", 0.0)),
            "count": None if day_input_normalization is None else int(day_input_normalization.get("count", 0)),
        },
        **align_metrics,
    }
    summary_payload = {
        "mode": "prediction",
        "model_family": "nomad_lstm",
        "model_backend": "nomad_style_dynamics",
        "inference_mode": "direct_shared_dynamics" if args.no_daily_calibration else "daily_calibration",
        "no_daily_calibration": bool(args.no_daily_calibration),
        "day_name": session_name,
        "base_day_name": base_day_name,
        "calibration_source": {
            "day_name": calibration_source_day_name,
            "base_day_name": calibration_source_base_day_name,
            "spike_csv": calibration_source_spike_csv.name,
            "same_as_prediction_day": calibration_source_spike_csv.resolve() == args.spike_csv.resolve(),
        },
        "model_path": args.model_path.name,
        "spike_csv": args.spike_csv.name,
        "voltage_dat": voltage_dat.name if voltage_dat is not None and voltage_dat.exists() else None,
        "input_sha256": {
            "model_checkpoint": input_file_sha256(args.model_path),
            "spike_csv": input_file_sha256(args.spike_csv),
            "calibration_spike_csv": None if args.no_daily_calibration else input_file_sha256(calibration_source_spike_csv),
            "voltage_dat": input_file_sha256(voltage_dat) if voltage_dat is not None and voltage_dat.exists() else None,
        },
        "prediction_csv": paths["predictions_csv"].name,
        "prediction_plot": paths["predictions_png"].name,
        "unit_mapping": unit_mapping,
        "alignment": alignment_payload,
        "calibration_window": {
            "time_start": None if args.no_daily_calibration else args.calibration_time_start,
            "time_end": None if args.no_daily_calibration else args.calibration_time_end,
        },
        "prediction_window": {
            "time_start": args.time_start,
            "time_end": args.time_end,
            "actual_first_endpoint": float(t_pred[0]),
            "actual_last_endpoint": float(t_pred[-1]),
            "prediction_count": int(len(t_pred)),
        },
        "evaluation": evaluation,
    }
    with paths["summary_json"].open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"Using model checkpoint: {args.model_path.resolve()}")
    print(f"Using spike CSV: {args.spike_csv.resolve()}")
    if args.no_daily_calibration:
        print("Using direct shared-dynamics mode: skip day-specific alignment and run the checkpoint shared GRU + decoder directly.")
    elif args.calibration_spike_csv is not None:
        print(
            "Using previous-day calibration mode: "
            f"calibration_spike_csv={args.calibration_spike_csv.resolve()}, "
            f"calibration_day_name={calibration_source_day_name}"
        )
    else:
        print(f"Using NoMAD calibration window: start={args.calibration_time_start}, end={args.calibration_time_end}")
    if args.skip_reference_update or args.update_state_path is not None:
        print("Note: NoMAD prediction does not write predictor_state updates; calibration is per-run only.")
    print(f"Prediction CSV: {paths['predictions_csv'].resolve()}")
    print(f"Prediction plot: {paths['predictions_png'].resolve()}")
    print(f"Summary JSON: {paths['summary_json'].resolve()}")
    if metrics is not None:
        print(f"Metrics: {metrics}")
    else:
        print("Metrics: skipped (no voltage ground truth found)")


if __name__ == "__main__":
    main()
