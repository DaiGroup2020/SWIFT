#!/usr/bin/env python
from __future__ import annotations

"""
NoMAD-style latent dynamics alignment + encoded LSTM decoder for the M06 sessions.

Hybrid pipeline:
1. Read sparse spike timestamps from `Online_*.csv` and control voltages from `*_voltage.dat`.
2. Use the base day to train a reference latent-dynamics model on binned spikes.
3. For every other day, fit a light day-specific alignment network that maps spike counts
   into the frozen reference dynamics space using spike-only unsupervised losses.
4. Extract aligned latent time series for all days.
5. Train the existing encoded LSTM decoder on aligned latent windows and evaluate held-out days.
"""

import argparse
import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import chain
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .manifold_decoder import (
    StandardScaler,
    build_decoder_model_from_kwargs,
    bin_spike_times_to_counts,
    build_model_export_stem,
    canonical_data_source_name,
    build_sequences,
    build_time_grid,
    copy_directory_contents,
    discover_data_source_dirs,
    discover_day_pairs,
    ensure_dir,
    evaluate_model,
    interpolate_columns,
    json_ready,
    normalize_day_name,
    read_sparse_spike_times,
    read_voltage_dat,
    regression_metrics,
    resolve_unique_export_dir,
    resolve_data_dir_from_source,
    split_fit_train_val_indices,
    train_lstm_decoder,
    unique_preserve_order,
    window_bins_from_ms,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "manifold_decoder_nomad_outputs"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "model_Nomad"

USER_EDITABLE_SETTINGS: dict[str, object] = {
    "data_source": "data",
    "data_dir": "",
    "output_dir": "",
    "selected_days": [],
    "base_day": "2025-12-22",
    "train_days": [
        "2025-12-01",
        "2025-12-02",
        "2025-12-03",
        "2025-12-08",
        "2025-12-09",
        "2025-12-15",
        "2025-12-16",
        "2025-12-17",
        "2025-12-22",
        "2025-12-26",
        "2025-12-29",
        "2025-12-30",
    ],
    "eval_days": [
        "2026-01-04",
        "2026-01-05",
        "2026-01-06",
        "2026-01-12",
        "2026-01-26",
    ],
    "export_tag": "nomad",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_parser(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def resolve_optional_path(value: object, base_dir: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def parse_day_list_value(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = list_parser(value)
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise TypeError("Day list must be a comma-separated string or a list-like object.")
    normalized = [normalize_day_name(item) for item in items]
    return [item for item in normalized if item is not None] or None


def parse_day_time_end_value(value: str | None) -> dict[str, float]:
    """Parse comma-separated DAY=SECONDS limits used to crop individual sessions."""
    if value is None or not str(value).strip():
        return {}
    limits: dict[str, float] = {}
    for item in list_parser(str(value)):
        if "=" not in item:
            raise ValueError("--day-time-end entries must use DAY=SECONDS, for example 2026-01-29=300")
        raw_day, raw_limit = item.split("=", 1)
        day_name = normalize_day_name(raw_day)
        if day_name is None:
            raise ValueError(f"Invalid day in --day-time-end: {raw_day!r}")
        limit_s = float(raw_limit)
        if limit_s <= 0.0:
            raise ValueError(f"Time limit for {day_name} must be positive.")
        limits[day_name] = limit_s
    return limits


def resolve_inline_settings(base_dir: Path) -> dict[str, object]:
    raw = USER_EDITABLE_SETTINGS
    return {
        "data_source": str(raw.get("data_source")).strip() if raw.get("data_source") is not None else "",
        "data_dir": resolve_optional_path(raw.get("data_dir"), base_dir),
        "output_dir": resolve_optional_path(raw.get("output_dir"), base_dir),
        "selected_days": parse_day_list_value(raw.get("selected_days")),
        "base_day": normalize_day_name(raw.get("base_day")) if raw.get("base_day") else None,
        "train_days": parse_day_list_value(raw.get("train_days")),
        "eval_days": parse_day_list_value(raw.get("eval_days")),
        "export_tag": str(raw.get("export_tag")).strip() if raw.get("export_tag") is not None else "",
    }


@dataclass
class Config:
    data_source: str | None = None
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    selected_days: list[str] | None = None
    base_day: str = "2025-12-22"
    train_days: list[str] | None = None
    eval_days: list[str] | None = None
    day_time_end_s: dict[str, float] | None = None
    export_tag: str = "nomad"
    skip_model_export: bool = False
    latent_cache_output: Path | None = None
    build_latent_cache_only: bool = False
    bin_size_s: float = 0.03
    train_sample_stride_bins: int = 1
    window_ms: float = 780.0
    readin_dim: int = 64
    nomad_latent_dim: int = 32
    alignment_hidden_dim: int = 96
    alignment_batch_size: int = 256
    alignment_epochs: int = 18
    alignment_lr: float = 5e-4
    alignment_patience: int = 5
    alignment_weight_decay: float = 1e-4
    recon_weight: float = 0.10
    behavior_weight: float = 1.0
    align_kl_weight: float = 0.10
    align_identity_weight: float = 1e-4
    input_norm_sigma_ms: float = 20.0
    disable_input_normalization: bool = False
    reference_batch_size: int = 256
    reference_epochs: int = 32
    reference_lr: float = 1e-3
    reference_weight_decay: float = 1e-4
    reference_alignment_dropout: float | None = None
    reference_patience: int = 8
    reference_scheduler_patience: int = 3
    reference_scheduler_factor: float = 0.5
    chunk_len: int = 4096
    fit_val_ratio: float = 0.1
    decoder_variant: str = "encoded"
    time_dim: int = 50
    latent_embed_dim: int = 60
    hidden_dim: int = 88
    num_layers: int = 1
    dropout: float = 0.2
    batch_size: int = 5000
    epochs: int = 40
    lr: float = 1e-3
    weight_decay: float = 1e-4
    l1_lambda: float = 5e-5
    grad_clip: float = 1.0
    patience: int = 12
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def seq_len(self) -> int:
        return window_bins_from_ms(self.window_ms, self.bin_size_s, include_current_bin=True)

    def validate(self) -> None:
        if self.selected_days is None:
            self.selected_days = []
        if self.train_days is None:
            self.train_days = []
        if self.eval_days is None:
            self.eval_days = []
        if self.day_time_end_s is None:
            self.day_time_end_s = {}
        if int(self.train_sample_stride_bins) < 1:
            raise ValueError("train_sample_stride_bins must be at least 1.")
        self.train_sample_stride_bins = int(self.train_sample_stride_bins)
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.reference_alignment_dropout is not None and not 0.0 <= float(self.reference_alignment_dropout) < 1.0:
            raise ValueError("reference_alignment_dropout must be in [0, 1).")
        if not self.train_days:
            raise ValueError("At least one train_day is required.")
        if self.base_day not in self.train_days:
            raise ValueError("base_day must be included in train_days.")
        if not self.selected_days:
            self.selected_days = unique_preserve_order([self.base_day, *self.train_days, *self.eval_days])
        required = set([self.base_day, *self.train_days, *self.eval_days])
        if not required.issubset(set(self.selected_days)):
            missing = sorted(required.difference(self.selected_days))
            raise ValueError(f"selected_days is missing required days: {missing}")
        overlap = set(self.train_days).intersection(self.eval_days)
        if overlap:
            raise ValueError(f"train_days and eval_days must be disjoint: {sorted(overlap)}")
        decoder_variant = str(self.decoder_variant).strip().lower()
        if decoder_variant not in {"encoded", "v2"}:
            raise ValueError("decoder_variant must be either 'encoded' or 'v2'")
        self.decoder_variant = decoder_variant


@dataclass
class DayAlignmentSummary:
    day_name: str
    matched_unit_count: int
    missing_unit_count: int
    extra_unit_count: int
    mapping_mode: str
    calibration_windows: int
    alignment_loss: float
    reconstruction_loss: float
    kl_to_reference: float
    identity_shift: float
    input_norm_sigma_ms: float
    input_norm_enabled: bool


@dataclass
class DayData:
    day_name: str
    unit_cols: list[str]
    t: np.ndarray
    counts: np.ndarray
    target: np.ndarray
    target_columns: list[str]
    voltage_df: pd.DataFrame
    unit_mapping: dict[str, object]
    latent: np.ndarray | None = None
    alignment: DayAlignmentSummary | None = None


class ResidualAlignmentNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.fc2(self.dropout(torch.relu(self.fc1(x))))
        return x + delta


class ReferenceDynamicsModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        readin_dim: int,
        latent_dim: int,
        dropout: float,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.readin = nn.Linear(input_dim, readin_dim, bias=False)
        self.readin_norm = nn.LayerNorm(readin_dim)
        self.gru = nn.GRU(readin_dim, latent_dim, batch_first=True)
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.rate_head = nn.Sequential(nn.Linear(latent_dim, input_dim), nn.Softplus())
        head_hidden = max(latent_dim // 2, 8)
        self.behavior_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, output_dim),
        )

    def encode(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        readin_layer: nn.Module | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        readin_module = self.readin if readin_layer is None else readin_layer
        readin = self.readin_norm(readin_module(x))
        latent, hidden_out = self.gru(readin, hidden)
        latent = self.latent_norm(latent)
        return latent, hidden_out

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
        readin_layer: nn.Module | None = None,
        rate_head: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        latent, hidden_out = self.encode(x, hidden=hidden, readin_layer=readin_layer)
        rate_module = self.rate_head if rate_head is None else rate_head
        rates = rate_module(latent)
        pred = self.behavior_head(latent[:, -1, :])
        return {"latent": latent, "rates": rates, "pred": pred, "hidden": hidden_out}


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train a NoMAD-style latent alignment + LSTM decoder.")
    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        help="Named child folder under data/ (for example train, predict, data) or a directory path.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--selected-days", type=str, default=None)
    parser.add_argument("--base-day", type=str, default=None)
    parser.add_argument("--train-days", type=str, default=None)
    parser.add_argument("--eval-days", type=str, default=None)
    parser.add_argument(
        "--day-time-end",
        type=str,
        default=None,
        help="Optional per-day end time(s) in seconds: DAY=SECONDS[,DAY=SECONDS].",
    )
    parser.add_argument("--test-days", dest="eval_days_alias", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--export-tag", type=str, default=None)
    parser.add_argument("--skip-model-export", action="store_true")
    parser.add_argument("--latent-cache-output", type=Path, default=None)
    parser.add_argument("--build-latent-cache-only", action="store_true")
    parser.add_argument("--bin-size-s", type=float, default=0.03)
    parser.add_argument(
        "--train-sample-stride-bins",
        type=int,
        default=1,
        help="Keep every Nth training and fit-validation window; evaluation remains dense.",
    )
    parser.add_argument("--window-ms", type=float, default=780.0)
    parser.add_argument("--readin-dim", type=int, default=64)
    parser.add_argument("--nomad-latent-dim", type=int, default=32)
    parser.add_argument("--alignment-hidden-dim", type=int, default=96)
    parser.add_argument("--alignment-batch-size", type=int, default=256)
    parser.add_argument("--alignment-epochs", type=int, default=18)
    parser.add_argument("--alignment-lr", type=float, default=5e-4)
    parser.add_argument("--alignment-patience", type=int, default=5)
    parser.add_argument("--alignment-weight-decay", type=float, default=1e-4)
    parser.add_argument("--recon-weight", type=float, default=0.10)
    parser.add_argument("--behavior-weight", type=float, default=1.0)
    parser.add_argument("--align-kl-weight", type=float, default=0.10)
    parser.add_argument("--align-identity-weight", type=float, default=1e-4)
    parser.add_argument("--input-norm-sigma-ms", type=float, default=20.0)
    parser.add_argument("--disable-input-normalization", action="store_true")
    parser.add_argument("--reference-batch-size", type=int, default=256)
    parser.add_argument("--reference-epochs", type=int, default=32)
    parser.add_argument("--reference-lr", type=float, default=1e-3)
    parser.add_argument("--reference-weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--reference-alignment-dropout",
        type=float,
        default=None,
        help="Optional dropout for the reference dynamics model and day-alignment networks; defaults to --dropout.",
    )
    parser.add_argument("--reference-patience", type=int, default=8)
    parser.add_argument("--reference-scheduler-patience", type=int, default=3)
    parser.add_argument("--reference-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--chunk-len", type=int, default=4096)
    parser.add_argument("--fit-val-ratio", type=float, default=0.1)
    parser.add_argument("--decoder-variant", type=str, default="encoded", choices=["encoded", "v2"])
    parser.add_argument("--time-dim", type=int, default=50)
    parser.add_argument("--latent-embed-dim", type=int, default=60)
    parser.add_argument("--hidden-dim", type=int, default=88)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--l1-lambda", type=float, default=5e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    inline = resolve_inline_settings(SCRIPT_DIR)
    explicit_data_dir = args.data_dir or inline["data_dir"]
    requested_data_source = args.data_source if args.data_source is not None else inline["data_source"]
    data_dir = resolve_data_dir_from_source(
        explicit_data_dir=explicit_data_dir,
        data_source=requested_data_source,
        base_dir=SCRIPT_DIR,
        default_dir=DEFAULT_DATA_DIR,
    )
    data_source_name = canonical_data_source_name(data_dir)
    output_dir = args.output_dir or inline["output_dir"] or DEFAULT_OUTPUT_DIR
    eval_days_raw = args.eval_days_alias if args.eval_days_alias is not None else args.eval_days
    if args.eval_days_alias is not None and args.eval_days is not None and args.eval_days != args.eval_days_alias:
        parser.error("--eval-days and legacy --test-days disagree; please provide only one.")

    available_pairs = discover_day_pairs(data_dir, None)
    available_days = list(available_pairs.keys())
    if not available_days:
        raise FileNotFoundError(f"No paired spike CSV + voltage DAT files were found in {data_dir}")

    train_days = parse_day_list_value(args.train_days) if args.train_days is not None else inline["train_days"]
    if not train_days:
        train_days = available_days
    eval_days = parse_day_list_value(eval_days_raw) if eval_days_raw is not None else inline["eval_days"]
    if eval_days is None:
        eval_days = []
    base_day = normalize_day_name(args.base_day) if args.base_day is not None else inline["base_day"]
    if base_day is None:
        base_day = train_days[0]
    selected_days = parse_day_list_value(args.selected_days) if args.selected_days is not None else inline["selected_days"]
    if not selected_days:
        selected_days = unique_preserve_order([base_day, *train_days, *eval_days])
    export_tag = args.export_tag if args.export_tag is not None else inline["export_tag"]

    config = Config(
        data_source=data_source_name or str(requested_data_source or "").strip() or None,
        data_dir=data_dir,
        output_dir=output_dir,
        selected_days=selected_days,
        base_day=base_day,
        train_days=train_days,
        eval_days=eval_days,
        day_time_end_s=parse_day_time_end_value(args.day_time_end),
        export_tag=str(export_tag or "").strip(),
        skip_model_export=args.skip_model_export,
        latent_cache_output=args.latent_cache_output,
        build_latent_cache_only=args.build_latent_cache_only,
        bin_size_s=args.bin_size_s,
        train_sample_stride_bins=args.train_sample_stride_bins,
        window_ms=args.window_ms,
        readin_dim=args.readin_dim,
        nomad_latent_dim=args.nomad_latent_dim,
        alignment_hidden_dim=args.alignment_hidden_dim,
        alignment_batch_size=args.alignment_batch_size,
        alignment_epochs=args.alignment_epochs,
        alignment_lr=args.alignment_lr,
        alignment_patience=args.alignment_patience,
        alignment_weight_decay=args.alignment_weight_decay,
        recon_weight=args.recon_weight,
        behavior_weight=args.behavior_weight,
        align_kl_weight=args.align_kl_weight,
        align_identity_weight=args.align_identity_weight,
        input_norm_sigma_ms=args.input_norm_sigma_ms,
        disable_input_normalization=args.disable_input_normalization,
        reference_batch_size=args.reference_batch_size,
        reference_epochs=args.reference_epochs,
        reference_lr=args.reference_lr,
        reference_weight_decay=args.reference_weight_decay,
        reference_alignment_dropout=args.reference_alignment_dropout,
        reference_patience=args.reference_patience,
        reference_scheduler_patience=args.reference_scheduler_patience,
        reference_scheduler_factor=args.reference_scheduler_factor,
        chunk_len=args.chunk_len,
        fit_val_ratio=args.fit_val_ratio,
        decoder_variant=args.decoder_variant,
        time_dim=args.time_dim,
        latent_embed_dim=args.latent_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        l1_lambda=args.l1_lambda,
        grad_clip=args.grad_clip,
        patience=args.patience,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        min_lr=args.min_lr,
        seed=args.seed,
        device=args.device,
    )
    config.validate()
    return config


def read_sparse_spike_times_aligned(
    csv_path: Path,
    expected_unit_cols: list[str] | None,
) -> tuple[list[str], list[np.ndarray], dict[str, object]]:
    unit_cols, unit_times = read_sparse_spike_times(csv_path)
    if expected_unit_cols is None:
        return unit_cols, unit_times, {
            "mapping_mode": "native",
            "available_unit_cols": unit_cols,
            "mapped_unit_cols": unit_cols,
            "matched_unit_count": len(unit_cols),
            "missing_unit_cols": [],
            "extra_unit_cols": [],
        }

    lookup = {col: times for col, times in zip(unit_cols, unit_times)}
    matched_count = sum(1 for col in expected_unit_cols if col in lookup)
    if matched_count == 0 and len(unit_cols) >= len(expected_unit_cols):
        mapped_cols = unit_cols[: len(expected_unit_cols)]
        return list(expected_unit_cols), [lookup[col] for col in mapped_cols], {
            "mapping_mode": "positional_fallback",
            "available_unit_cols": unit_cols,
            "mapped_unit_cols": list(expected_unit_cols),
            "matched_unit_count": 0,
            "missing_unit_cols": [],
            "extra_unit_cols": unit_cols[len(expected_unit_cols) :],
        }

    missing = [col for col in expected_unit_cols if col not in lookup]
    extra = [col for col in unit_cols if col not in expected_unit_cols]
    aligned_times = [lookup.get(col, np.array([], dtype=np.float64)) for col in expected_unit_cols]
    return list(expected_unit_cols), aligned_times, {
        "mapping_mode": "name_match",
        "available_unit_cols": unit_cols,
        "mapped_unit_cols": list(expected_unit_cols),
        "matched_unit_count": matched_count,
        "missing_unit_cols": missing,
        "extra_unit_cols": extra,
    }


def load_day_data(
    day_name: str,
    csv_path: Path,
    dat_path: Path,
    bin_size_s: float,
    expected_unit_cols: list[str] | None = None,
) -> DayData:
    unit_cols, unit_times, mapping = read_sparse_spike_times_aligned(csv_path, expected_unit_cols)
    voltage_df = read_voltage_dat(dat_path)
    edges, t_centers = build_time_grid(voltage_df, bin_size_s)
    counts = bin_spike_times_to_counts(unit_times, edges)
    target_cols = ["vx", "vy"]
    target = interpolate_columns(voltage_df, t_centers, target_cols)
    valid = np.isfinite(target).all(axis=1)
    return DayData(
        day_name=day_name,
        unit_cols=unit_cols,
        t=t_centers[valid].astype(np.float64, copy=False),
        counts=counts[valid].astype(np.float32, copy=False),
        target=target[valid].astype(np.float32, copy=False),
        target_columns=target_cols,
        voltage_df=voltage_df,
        unit_mapping=mapping,
    )


def crop_day_to_time_end(day_data: DayData, time_end_s: float | None) -> DayData:
    if time_end_s is None:
        return day_data
    keep = day_data.t <= float(time_end_s)
    if int(np.sum(keep)) < 2:
        raise ValueError(
            f"{day_data.day_name}: --day-time-end={time_end_s:g}s leaves fewer than two bins "
            f"(available duration={day_data.t[-1]:.2f}s)."
        )
    return DayData(
        day_name=day_data.day_name,
        unit_cols=day_data.unit_cols,
        t=day_data.t[keep],
        counts=day_data.counts[keep],
        target=day_data.target[keep],
        target_columns=day_data.target_columns,
        voltage_df=day_data.voltage_df,
        unit_mapping=day_data.unit_mapping,
    )


def build_count_windows(counts: np.ndarray, seq_len: int, end_indices: np.ndarray) -> np.ndarray:
    X = np.empty((len(end_indices), seq_len, counts.shape[1]), dtype=np.float32)
    for i, end_idx in enumerate(end_indices):
        X[i] = counts[end_idx - seq_len + 1 : end_idx + 1]
    return X


def build_input_normalization_stats(
    counts: np.ndarray,
    bin_size_s: float,
    sigma_ms: float,
    eps: float = 1e-3,
) -> dict[str, object]:
    counts64 = counts.astype(np.float64, copy=False)
    sigma_bins = float(sigma_ms) / 1000.0 / max(float(bin_size_s), 1e-8)
    if sigma_ms > 0.0 and sigma_bins > 1e-6:
        radius = max(1, int(math.ceil(4.0 * sigma_bins)))
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
        kernel /= np.sum(kernel)
        padded = np.pad(counts64, ((radius, radius), (0, 0)), mode="edge")
        smoothed = np.empty_like(counts64, dtype=np.float64)
        for unit_idx in range(counts64.shape[1]):
            smoothed[:, unit_idx] = np.convolve(padded[:, unit_idx], kernel, mode="valid")
    else:
        smoothed = counts64
    mean = smoothed.mean(axis=0).astype(np.float32, copy=False)
    std = np.clip(smoothed.std(axis=0).astype(np.float32, copy=False), eps, None)
    return {
        "mean": mean,
        "std": std,
        "sigma_ms": float(sigma_ms),
        "sigma_bins": float(sigma_bins),
        "count": int(len(counts)),
    }


def normalize_counts_array(
    counts: np.ndarray,
    input_normalization: dict[str, object] | None,
) -> np.ndarray:
    if input_normalization is None:
        return counts.astype(np.float32, copy=False)
    mean = np.asarray(input_normalization["mean"], dtype=np.float32)
    std = np.asarray(input_normalization["std"], dtype=np.float32)
    reshape = (1,) * (counts.ndim - 1) + (counts.shape[-1],)
    normalized = (counts.astype(np.float32, copy=False) - mean.reshape(reshape)) / std.reshape(reshape)
    return normalized.astype(np.float32, copy=False)


def input_normalization_to_tensors(
    input_normalization: dict[str, object] | None,
    device: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if input_normalization is None:
        return None, None
    mean_t = torch.from_numpy(np.asarray(input_normalization["mean"], dtype=np.float32)).to(device).view(1, 1, -1)
    std_t = torch.from_numpy(np.asarray(input_normalization["std"], dtype=np.float32)).to(device).view(1, 1, -1)
    return mean_t, std_t


def apply_input_normalization_tensor(
    counts: torch.Tensor,
    mean_t: torch.Tensor | None,
    std_t: torch.Tensor | None,
) -> torch.Tensor:
    if mean_t is None or std_t is None:
        return counts
    return (counts - mean_t) / std_t


def poisson_reconstruction_loss(rates: torch.Tensor, counts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    safe_rates = torch.clamp(rates, min=eps)
    return torch.mean(safe_rates - counts * torch.log(safe_rates))


def gaussian_stats_from_latent_torch(latent: torch.Tensor, eps: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor]:
    flat = latent.reshape(-1, latent.shape[-1])
    mu = flat.mean(dim=0)
    xc = flat - mu
    denom = max(int(flat.shape[0]) - 1, 1)
    cov = (xc.T @ xc) / float(denom)
    cov = cov + eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    return mu, cov


def gaussian_kl_to_reference(
    latent: torch.Tensor,
    ref_mean: torch.Tensor,
    ref_cov: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    mu, cov = gaussian_stats_from_latent_torch(latent, eps=eps)
    dim = mu.shape[0]
    ref_cov = ref_cov + eps * torch.eye(dim, device=ref_cov.device, dtype=ref_cov.dtype)
    cov = cov + eps * torch.eye(dim, device=cov.device, dtype=cov.dtype)
    inv_ref_cov = torch.linalg.inv(ref_cov)
    delta = (ref_mean - mu).unsqueeze(0)
    trace_term = torch.trace(inv_ref_cov @ cov)
    mahal = torch.sum((delta @ inv_ref_cov) * delta)
    logdet_ref = torch.logdet(ref_cov)
    logdet_cov = torch.logdet(cov)
    return 0.5 * (trace_term + mahal - dim + logdet_ref - logdet_cov)


def encode_full_sequence(
    reference_model: ReferenceDynamicsModel,
    counts: np.ndarray,
    device: str,
    chunk_len: int,
    alignment_net: nn.Module | None = None,
    input_normalization: dict[str, object] | None = None,
    readin_layer: nn.Module | None = None,
    rate_head: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reference_model.eval()
    if alignment_net is not None:
        alignment_net.eval()
    if readin_layer is not None:
        readin_layer.eval()
    if rate_head is not None:
        rate_head.eval()
    hidden = None
    latent_parts: list[np.ndarray] = []
    rate_parts: list[np.ndarray] = []
    mean_t, std_t = input_normalization_to_tensors(input_normalization, device)
    with torch.no_grad():
        counts_tensor = torch.from_numpy(counts.astype(np.float32, copy=False)).to(device)
        for start in range(0, counts_tensor.shape[0], chunk_len):
            end = min(start + chunk_len, counts_tensor.shape[0])
            chunk = counts_tensor[start:end].unsqueeze(0)
            chunk = apply_input_normalization_tensor(chunk, mean_t, std_t)
            if alignment_net is not None:
                chunk = alignment_net(chunk)
            out = reference_model(chunk, hidden=hidden, readin_layer=readin_layer, rate_head=rate_head)
            hidden = out["hidden"]
            if hidden is not None:
                hidden = hidden.detach()
            latent_parts.append(out["latent"].squeeze(0).cpu().numpy())
            rate_parts.append(out["rates"].squeeze(0).cpu().numpy())
    latent = np.concatenate(latent_parts, axis=0).astype(np.float32, copy=False)
    rates = np.concatenate(rate_parts, axis=0).astype(np.float32, copy=False)
    return latent, rates


def train_reference_model(
    base_day: DayData,
    config: Config,
) -> tuple[
    ReferenceDynamicsModel,
    list[dict[str, float]],
    dict[str, float],
    dict[str, float],
    dict[str, object] | None,
]:
    seq_len = config.seq_len
    train_end, fit_val_end = split_fit_train_val_indices(len(base_day.counts), seq_len, config.fit_val_ratio)
    X_train = build_count_windows(base_day.counts, seq_len, train_end)
    X_fit_val = build_count_windows(base_day.counts, seq_len, fit_val_end)
    Y_train = base_day.target[train_end]
    Y_fit_val = base_day.target[fit_val_end]
    input_normalization = None
    if not config.disable_input_normalization:
        input_normalization = build_input_normalization_stats(
            counts=base_day.counts,
            bin_size_s=config.bin_size_s,
            sigma_ms=config.input_norm_sigma_ms,
        )
    X_train_norm = normalize_counts_array(X_train, input_normalization)
    X_fit_val_norm = normalize_counts_array(X_fit_val, input_normalization)

    y_scaler = StandardScaler().fit(Y_train)
    Y_train_scaled = y_scaler.transform(Y_train)
    Y_fit_val_scaled = y_scaler.transform(Y_fit_val)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train.astype(np.float32, copy=False)),
            torch.from_numpy(X_train_norm),
            torch.from_numpy(Y_train_scaled.astype(np.float32, copy=False)),
        ),
        batch_size=config.reference_batch_size,
        shuffle=True,
    )
    fit_val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_fit_val.astype(np.float32, copy=False)),
            torch.from_numpy(X_fit_val_norm),
            torch.from_numpy(Y_fit_val_scaled.astype(np.float32, copy=False)),
        ),
        batch_size=config.reference_batch_size,
        shuffle=False,
    )

    model = ReferenceDynamicsModel(
        input_dim=base_day.counts.shape[1],
        readin_dim=config.readin_dim,
        latent_dim=config.nomad_latent_dim,
        dropout=(config.reference_alignment_dropout if config.reference_alignment_dropout is not None else config.dropout),
        output_dim=Y_train.shape[1],
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.reference_lr,
        weight_decay=config.reference_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.reference_scheduler_factor,
        patience=config.reference_scheduler_patience,
        min_lr=config.min_lr,
        threshold=1e-5,
    )
    mse_loss = nn.MSELoss()
    best_state = None
    best_fit_val_loss = np.inf
    patience_left = config.reference_patience
    history: list[dict[str, float]] = []

    for epoch in range(1, config.reference_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for xb_raw, xb_norm, yb in train_loader:
            xb_raw = xb_raw.to(config.device)
            xb_norm = xb_norm.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(xb_norm)
            behavior_loss = mse_loss(out["pred"], yb)
            recon_loss = poisson_reconstruction_loss(out["rates"], xb_raw)
            loss = config.behavior_weight * behavior_loss + config.recon_weight * recon_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(xb_raw)
            train_count += len(xb_raw)

        model.eval()
        fit_val_loss_sum = 0.0
        fit_val_count = 0
        y_true_scaled_parts: list[np.ndarray] = []
        y_pred_scaled_parts: list[np.ndarray] = []
        with torch.no_grad():
            for xb_raw, xb_norm, yb in fit_val_loader:
                xb_raw = xb_raw.to(config.device)
                xb_norm = xb_norm.to(config.device)
                yb = yb.to(config.device)
                out = model(xb_norm)
                behavior_loss = mse_loss(out["pred"], yb)
                recon_loss = poisson_reconstruction_loss(out["rates"], xb_raw)
                total_loss = config.behavior_weight * behavior_loss + config.recon_weight * recon_loss
                fit_val_loss_sum += float(total_loss.item()) * len(xb_raw)
                fit_val_count += len(xb_raw)
                y_true_scaled_parts.append(yb.cpu().numpy())
                y_pred_scaled_parts.append(out["pred"].cpu().numpy())

        fit_val_loss = fit_val_loss_sum / max(fit_val_count, 1)
        scheduler.step(fit_val_loss)
        y_true_fit = y_scaler.inverse_transform(np.concatenate(y_true_scaled_parts, axis=0))
        y_pred_fit = y_scaler.inverse_transform(np.concatenate(y_pred_scaled_parts, axis=0))
        fit_val_metrics = regression_metrics(y_true_fit, y_pred_fit)
        train_loss = train_loss_sum / max(train_count, 1)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "fit_val_loss": float(fit_val_loss),
                "fit_val_rmse": float(fit_val_metrics["rmse"]),
                "fit_val_r2": float(fit_val_metrics["r2"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"Reference epoch {epoch:04d} | train_loss={train_loss:.6f} | "
            f"fit_val_loss={fit_val_loss:.6f} | fit_val_rmse={fit_val_metrics['rmse']:.6f} | "
            f"fit_val_r2={fit_val_metrics['r2']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if fit_val_loss < best_fit_val_loss:
            best_fit_val_loss = fit_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.reference_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Reference-model early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Reference-model training failed to produce a valid checkpoint.")
    model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_out = model(torch.from_numpy(X_train_norm).to(config.device))
        fit_out = model(torch.from_numpy(X_fit_val_norm).to(config.device))
    train_metrics = regression_metrics(Y_train, y_scaler.inverse_transform(train_out["pred"].cpu().numpy()))
    fit_val_metrics = regression_metrics(Y_fit_val, y_scaler.inverse_transform(fit_out["pred"].cpu().numpy()))
    return model, history, train_metrics, fit_val_metrics, input_normalization


def build_day_adaptation_modules(
    reference_model: ReferenceDynamicsModel,
    config: Config,
) -> tuple[ResidualAlignmentNet, nn.Module, nn.Module]:
    align_net = ResidualAlignmentNet(
        input_dim=reference_model.readin.in_features,
        hidden_dim=config.alignment_hidden_dim,
        dropout=(config.reference_alignment_dropout if config.reference_alignment_dropout is not None else config.dropout),
    ).to(config.device)
    day_readin = deepcopy(reference_model.readin).to(config.device)
    day_rate_head = deepcopy(reference_model.rate_head).to(config.device)
    for module in (day_readin, day_rate_head):
        for param in module.parameters():
            param.requires_grad = True
    return align_net, day_readin, day_rate_head


def fit_day_alignment_modules(
    day_counts: np.ndarray,
    reference_model: ReferenceDynamicsModel,
    reference_mean: np.ndarray,
    reference_cov: np.ndarray,
    config: Config,
) -> tuple[ResidualAlignmentNet, nn.Module, nn.Module, dict[str, float], dict[str, object] | None]:
    seq_len = config.seq_len
    end_idx = np.arange(seq_len - 1, len(day_counts), dtype=np.int64)
    X_day = build_count_windows(day_counts, seq_len, end_idx)
    if len(X_day) == 0:
        raise ValueError(f"Not enough bins for seq_len={seq_len}.")
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_day.astype(np.float32, copy=False))),
        batch_size=config.alignment_batch_size,
        shuffle=True,
    )
    align_net, day_readin, day_rate_head = build_day_adaptation_modules(reference_model, config)
    input_normalization = None
    if not config.disable_input_normalization:
        input_normalization = build_input_normalization_stats(
            counts=day_counts,
            bin_size_s=config.bin_size_s,
            sigma_ms=config.input_norm_sigma_ms,
        )
    mean_t, std_t = input_normalization_to_tensors(input_normalization, config.device)

    reference_model.train()
    for param in reference_model.parameters():
        param.requires_grad = False

    ref_mean_t = torch.from_numpy(reference_mean.astype(np.float32, copy=False)).to(config.device)
    ref_cov_t = torch.from_numpy(reference_cov.astype(np.float32, copy=False)).to(config.device)
    optimizer = torch.optim.AdamW(
        chain(align_net.parameters(), day_readin.parameters(), day_rate_head.parameters()),
        lr=config.alignment_lr,
        weight_decay=config.alignment_weight_decay,
    )

    best_state = None
    best_loss = np.inf
    patience_left = config.alignment_patience
    best_metrics: dict[str, float] | None = None

    for epoch in range(1, config.alignment_epochs + 1):
        align_net.train()
        day_readin.train()
        day_rate_head.train()
        total_loss_sum = 0.0
        total_recon_sum = 0.0
        total_kl_sum = 0.0
        total_identity_sum = 0.0
        total_count = 0
        for (xb_raw,) in loader:
            xb_raw = xb_raw.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            xb_norm = apply_input_normalization_tensor(xb_raw, mean_t, std_t)
            aligned = align_net(xb_norm)
            out = reference_model(aligned, readin_layer=day_readin, rate_head=day_rate_head)
            recon_loss = poisson_reconstruction_loss(out["rates"], xb_raw)
            kl_loss = gaussian_kl_to_reference(out["latent"], ref_mean_t, ref_cov_t)
            identity_loss = torch.mean((aligned - xb_norm) ** 2)
            loss = (
                config.recon_weight * recon_loss
                + config.align_kl_weight * kl_loss
                + config.align_identity_weight * identity_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(align_net.parameters()) + list(day_readin.parameters()) + list(day_rate_head.parameters()),
                config.grad_clip,
            )
            optimizer.step()
            total_loss_sum += float(loss.item()) * len(xb_raw)
            total_recon_sum += float(recon_loss.item()) * len(xb_raw)
            total_kl_sum += float(kl_loss.item()) * len(xb_raw)
            total_identity_sum += float(identity_loss.item()) * len(xb_raw)
            total_count += len(xb_raw)

        epoch_loss = total_loss_sum / max(total_count, 1)
        epoch_recon = total_recon_sum / max(total_count, 1)
        epoch_kl = total_kl_sum / max(total_count, 1)
        epoch_identity = total_identity_sum / max(total_count, 1)
        print(
            f"  align epoch={epoch:03d} | loss={epoch_loss:.6f} | "
            f"recon={epoch_recon:.6f} | kl={epoch_kl:.6f} | shift={epoch_identity:.6f}"
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {
                "align_net": {k: v.detach().cpu().clone() for k, v in align_net.state_dict().items()},
                "day_readin": {k: v.detach().cpu().clone() for k, v in day_readin.state_dict().items()},
                "day_rate_head": {k: v.detach().cpu().clone() for k, v in day_rate_head.state_dict().items()},
            }
            best_metrics = {
                "alignment_loss": epoch_loss,
                "reconstruction_loss": epoch_recon,
                "kl_to_reference": epoch_kl,
                "identity_shift": epoch_identity,
            }
            patience_left = config.alignment_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None or best_metrics is None:
        raise RuntimeError("Alignment training failed.")
    align_net.load_state_dict(best_state["align_net"])
    day_readin.load_state_dict(best_state["day_readin"])
    day_rate_head.load_state_dict(best_state["day_rate_head"])
    align_net.eval()
    day_readin.eval()
    day_rate_head.eval()
    return align_net, day_readin, day_rate_head, best_metrics, input_normalization


def train_alignment_network(
    day: DayData,
    reference_model: ReferenceDynamicsModel,
    reference_mean: np.ndarray,
    reference_cov: np.ndarray,
    config: Config,
) -> tuple[ResidualAlignmentNet, nn.Module, nn.Module, dict[str, object] | None, DayAlignmentSummary]:
    seq_len = config.seq_len
    calibration_windows = max(len(day.counts) - seq_len + 1, 0)
    align_net, day_readin, day_rate_head, best_metrics, input_normalization = fit_day_alignment_modules(
        day_counts=day.counts,
        reference_model=reference_model,
        reference_mean=reference_mean,
        reference_cov=reference_cov,
        config=config,
    )
    summary = DayAlignmentSummary(
        day_name=day.day_name,
        matched_unit_count=int(day.unit_mapping.get("matched_unit_count", len(day.unit_cols))),
        missing_unit_count=len(day.unit_mapping.get("missing_unit_cols", [])),
        extra_unit_count=len(day.unit_mapping.get("extra_unit_cols", [])),
        mapping_mode=str(day.unit_mapping.get("mapping_mode", "unknown")),
        calibration_windows=int(calibration_windows),
        alignment_loss=float(best_metrics["alignment_loss"]),
        reconstruction_loss=float(best_metrics["reconstruction_loss"]),
        kl_to_reference=float(best_metrics["kl_to_reference"]),
        identity_shift=float(best_metrics["identity_shift"]),
        input_norm_sigma_ms=float(config.input_norm_sigma_ms),
        input_norm_enabled=not config.disable_input_normalization,
    )
    return align_net, day_readin, day_rate_head, input_normalization, summary


def module_state_dict_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def state_dict_shape_summary(state_dict: dict[str, torch.Tensor] | None) -> dict[str, list[int]] | None:
    if state_dict is None:
        return None
    return {key: list(value.shape) for key, value in state_dict.items()}


def save_prediction_plot(
    output_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_points: int = 1500,
    title: str | None = None,
) -> None:
    n = min(n_points, len(y_true))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(y_true[:n, 0], label="true_vx", linewidth=1.0)
    axes[0].plot(y_pred[:n, 0], label="pred_vx", linewidth=1.0)
    axes[0].set_ylabel("vx")
    axes[0].legend(loc="upper right")
    axes[1].plot(y_true[:n, 1], label="true_vy", linewidth=1.0)
    axes[1].plot(y_pred[:n, 1], label="pred_vy", linewidth=1.0)
    axes[1].set_ylabel("vy")
    axes[1].set_xlabel("Sample")
    axes[1].legend(loc="upper right")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def prepare_latent_datasets(
    days: dict[str, DayData],
    train_days: list[str],
    eval_days: list[str],
    seq_len: int,
    fit_val_ratio: float,
    train_sample_stride_bins: int = 1,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    if int(train_sample_stride_bins) < 1:
        raise ValueError("train_sample_stride_bins must be at least 1.")
    stride = int(train_sample_stride_bins)
    X_parts = {"train": [], "fit_val": [], "eval": []}
    Y_parts = {"train": [], "fit_val": [], "eval": []}
    meta_day = {"train": [], "fit_val": [], "eval": []}
    meta_t = {"train": [], "fit_val": [], "eval": []}

    for day_name in train_days:
        day = days[day_name]
        if day.latent is None:
            raise ValueError(f"Latents are missing for {day_name}.")
        train_end, fit_val_end = split_fit_train_val_indices(len(day.latent), seq_len, fit_val_ratio)
        train_end = train_end[::stride]
        fit_val_end = fit_val_end[::stride]
        X_train, Y_train = build_sequences(day.latent, day.target, seq_len, train_end)
        X_fit_val, Y_fit_val = build_sequences(day.latent, day.target, seq_len, fit_val_end)
        X_parts["train"].append(X_train)
        Y_parts["train"].append(Y_train)
        meta_day["train"].append(np.full(len(train_end), day_name, dtype=object))
        meta_t["train"].append(day.t[train_end])
        X_parts["fit_val"].append(X_fit_val)
        Y_parts["fit_val"].append(Y_fit_val)
        meta_day["fit_val"].append(np.full(len(fit_val_end), day_name, dtype=object))
        meta_t["fit_val"].append(day.t[fit_val_end])

    for day_name in eval_days:
        day = days[day_name]
        if day.latent is None:
            raise ValueError(f"Latents are missing for {day_name}.")
        eval_end = np.arange(seq_len - 1, len(day.latent), dtype=np.int64)
        X_eval, Y_eval = build_sequences(day.latent, day.target, seq_len, eval_end)
        X_parts["eval"].append(X_eval)
        Y_parts["eval"].append(Y_eval)
        meta_day["eval"].append(np.full(len(eval_end), day_name, dtype=object))
        meta_t["eval"].append(day.t[eval_end])

    X = {k: np.concatenate(v, axis=0).astype(np.float32, copy=False) for k, v in X_parts.items() if v}
    Y = {k: np.concatenate(v, axis=0).astype(np.float32, copy=False) for k, v in Y_parts.items() if v}
    day_meta = {k: np.concatenate(v, axis=0) if v else np.array([], dtype=object) for k, v in meta_day.items()}
    t_meta = {
        k: np.concatenate(v, axis=0).astype(np.float64, copy=False) if v else np.array([], dtype=np.float64)
        for k, v in meta_t.items()
    }
    return X, Y, day_meta, t_meta


def write_aligned_latent_cache(
    cache_path: Path,
    config: Config,
    days: dict[str, DayData],
    reference_model: ReferenceDynamicsModel,
    reference_history: list[dict[str, float]],
    reference_fit_train_metrics: dict[str, float],
    reference_fit_val_metrics: dict[str, float],
    reference_mean: np.ndarray,
    reference_cov: np.ndarray,
    base_input_normalization: dict[str, object] | None,
    day_adaptation_state_dicts: dict[str, dict[str, object]],
) -> None:
    """Persist the once-trained NoMAD mapping for later LSTM-only CV searches."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    base_day = days[config.base_day]
    cache_days: dict[str, dict[str, object]] = {}
    for day_name in config.selected_days:
        day = days[day_name]
        if day.latent is None:
            raise ValueError(f"Cannot cache {day_name}: aligned latent is missing.")
        cache_days[day_name] = {
            "latent": day.latent.astype(np.float32, copy=False),
            "target": day.target.astype(np.float32, copy=False),
            "t": day.t.astype(np.float64, copy=False),
            "alignment": json_ready(asdict(day.alignment)) if day.alignment is not None else None,
        }
    payload = {
        "cache_format": "nomad_aligned_latent_v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_config": json_ready(asdict(config)),
        "data_source_identifier": str(config.data_source or Path(config.data_dir).name),
        "data_source_dir": str(config.data_dir.resolve()),
        "base_day": config.base_day,
        "selected_days": list(config.selected_days),
        "unit_cols": list(base_day.unit_cols),
        "target_columns": list(base_day.target_columns),
        "latent_dim": int(config.nomad_latent_dim),
        "reference_model_state_dict": module_state_dict_cpu(reference_model),
        "reference_model_kwargs": {
            "input_dim": int(base_day.counts.shape[1]),
            "readin_dim": int(config.readin_dim),
            "latent_dim": int(config.nomad_latent_dim),
            "dropout": float(config.reference_alignment_dropout if config.reference_alignment_dropout is not None else config.dropout),
            "output_dim": 2,
        },
        "reference_stats": {"mean": reference_mean, "cov": reference_cov, "count": int(len(base_day.latent))},
        "reference_input_normalization": base_input_normalization,
        "reference_fit_train_metrics": reference_fit_train_metrics,
        "reference_fit_val_metrics": reference_fit_val_metrics,
        "reference_history": reference_history,
        "alignment_network_kwargs": {
            "input_dim": int(base_day.counts.shape[1]),
            "hidden_dim": int(config.alignment_hidden_dim),
            "dropout": float(config.reference_alignment_dropout if config.reference_alignment_dropout is not None else config.dropout),
        },
        "alignment_mode": "align_net_plus_observation_adapters",
        "day_adaptation_state_dicts": day_adaptation_state_dicts,
        "days": cache_days,
    }
    torch.save(payload, cache_path)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    metadata = {
        "cache_format": payload["cache_format"],
        "created_at": payload["created_at"],
        "data_source_identifier": payload["data_source_identifier"],
        "base_day": payload["base_day"],
        "selected_days": payload["selected_days"],
        "latent_dim": payload["latent_dim"],
        "source_config": payload["source_config"],
        "day_shapes": {name: list(np.asarray(day["latent"]).shape) for name, day in cache_days.items()},
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def run_pipeline(config: Config) -> None:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    seq_len = config.seq_len
    pairs = discover_day_pairs(config.data_dir, config.selected_days)

    print("Loading selected days:")
    print(f"  data_source={config.data_source or '-'}")
    print(f"  data_dir={config.data_dir}")
    print(f"  train_days={config.train_days}")
    print(f"  eval_days={config.eval_days}")
    print(f"  base_day={config.base_day}")
    print(
        f"  seq_len={seq_len} (window_ms={config.window_ms}, bin_size_s={config.bin_size_s}, "
        f"train_sample_stride_bins={config.train_sample_stride_bins})"
    )

    base_csv, base_dat = pairs[config.base_day]
    base_day = load_day_data(config.base_day, base_csv, base_dat, config.bin_size_s, expected_unit_cols=None)
    base_time_end_s = config.day_time_end_s.get(config.base_day)
    base_day = crop_day_to_time_end(base_day, base_time_end_s)
    days: dict[str, DayData] = {config.base_day: base_day}
    print(
        f"  {config.base_day} | units={len(base_day.unit_cols)} | bins={len(base_day.t)} | "
        f"duration={base_day.t[-1] - base_day.t[0]:.2f}s | mapping=native"
    )
    for day_name in config.selected_days:
        if day_name == config.base_day:
            continue
        csv_path, dat_path = pairs[day_name]
        day_data = load_day_data(day_name, csv_path, dat_path, config.bin_size_s, expected_unit_cols=base_day.unit_cols)
        time_end_s = config.day_time_end_s.get(day_name)
        day_data = crop_day_to_time_end(day_data, time_end_s)
        mapping_mode = day_data.unit_mapping.get("mapping_mode", "unknown")
        matched = int(day_data.unit_mapping.get("matched_unit_count", len(base_day.unit_cols)))
        print(
            f"  {day_name} | units={len(day_data.unit_cols)} | bins={len(day_data.t)} | "
            f"duration={day_data.t[-1] - day_data.t[0]:.2f}s | mapping={mapping_mode} | matched_units={matched}"
            + (f" | cropped_at={time_end_s:g}s" if time_end_s is not None else "")
        )
        days[day_name] = day_data

    print("\nTraining reference dynamics model on the base day:")
    reference_model, reference_history, reference_fit_train_metrics, reference_fit_val_metrics, base_input_normalization = train_reference_model(
        base_day=base_day,
        config=config,
    )

    print("\nExtracting base-day latent dynamics:")
    base_latent, base_rates = encode_full_sequence(
        reference_model,
        base_day.counts,
        config.device,
        config.chunk_len,
        None,
        input_normalization=base_input_normalization,
    )
    base_day.latent = base_latent
    base_recon = regression_metrics(base_day.counts, base_rates)
    reference_mean = np.mean(base_latent.astype(np.float64, copy=False), axis=0)
    centered = base_latent.astype(np.float64, copy=False) - reference_mean
    denom = max(len(base_latent) - 1, 1)
    reference_cov = (centered.T @ centered) / float(denom)
    reference_cov += 1e-4 * np.eye(reference_cov.shape[0], dtype=np.float64)
    base_day.alignment = DayAlignmentSummary(
        day_name=base_day.day_name,
        matched_unit_count=len(base_day.unit_cols),
        missing_unit_count=0,
        extra_unit_count=0,
        mapping_mode="native",
        calibration_windows=max(len(base_day.counts) - seq_len + 1, 0),
        alignment_loss=0.0,
        reconstruction_loss=float(base_recon["mse"]),
        kl_to_reference=0.0,
        identity_shift=0.0,
        input_norm_sigma_ms=float(config.input_norm_sigma_ms),
        input_norm_enabled=not config.disable_input_normalization,
    )
    day_adaptation_state_dicts: dict[str, dict[str, object]] = {
        config.base_day: {
            "is_reference_day": True,
            "align_net_state_dict": None,
            "day_readin_state_dict": module_state_dict_cpu(reference_model.readin),
            "day_rate_head_state_dict": module_state_dict_cpu(reference_model.rate_head),
            "input_normalization": base_input_normalization,
        }
    }

    print("\nFitting NoMAD-style day alignment networks:")
    for day_name in config.selected_days:
        if day_name == config.base_day:
            continue
        align_net, day_readin, day_rate_head, day_input_normalization, alignment_summary = train_alignment_network(
            day=days[day_name],
            reference_model=reference_model,
            reference_mean=reference_mean,
            reference_cov=reference_cov,
            config=config,
        )
        days[day_name].alignment = alignment_summary
        day_latent, _ = encode_full_sequence(
            reference_model,
            days[day_name].counts,
            config.device,
            config.chunk_len,
            align_net,
            input_normalization=day_input_normalization,
            readin_layer=day_readin,
            rate_head=day_rate_head,
        )
        days[day_name].latent = day_latent
        day_adaptation_state_dicts[day_name] = {
            "is_reference_day": False,
            "align_net_state_dict": module_state_dict_cpu(align_net),
            "day_readin_state_dict": module_state_dict_cpu(day_readin),
            "day_rate_head_state_dict": module_state_dict_cpu(day_rate_head),
            "input_normalization": day_input_normalization,
        }
        print(
            f"  {day_name} | latent_shape={day_latent.shape} | "
            f"align_loss={alignment_summary.alignment_loss:.6f} | "
            f"kl={alignment_summary.kl_to_reference:.6f}"
        )

    if config.latent_cache_output is not None:
        write_aligned_latent_cache(
            cache_path=config.latent_cache_output,
            config=config,
            days=days,
            reference_model=reference_model,
            reference_history=reference_history,
            reference_fit_train_metrics=reference_fit_train_metrics,
            reference_fit_val_metrics=reference_fit_val_metrics,
            reference_mean=reference_mean,
            reference_cov=reference_cov,
            base_input_normalization=base_input_normalization,
            day_adaptation_state_dicts=day_adaptation_state_dicts,
        )
        print(f"\nAligned latent cache written to: {config.latent_cache_output.resolve()}")
        if config.build_latent_cache_only:
            print("Decoder training skipped (--build-latent-cache-only)")
            return

    print("\nBuilding latent sequence datasets:")
    X, Y, day_meta, t_meta = prepare_latent_datasets(
        days=days,
        train_days=config.train_days,
        eval_days=config.eval_days,
        seq_len=seq_len,
        fit_val_ratio=config.fit_val_ratio,
        train_sample_stride_bins=config.train_sample_stride_bins,
    )
    for split_name in ["train", "fit_val"]:
        split_counts = {day: int(np.sum(day_meta[split_name] == day)) for day in sorted(set(day_meta[split_name].tolist()))}
        print(f"  {split_name}: X={X[split_name].shape}, Y={Y[split_name].shape}, day_counts={split_counts}")
    if "eval" in X:
        split_counts = {day: int(np.sum(day_meta['eval'] == day)) for day in sorted(set(day_meta['eval'].tolist()))}
        print(f"  eval: X={X['eval'].shape}, Y={Y['eval'].shape}, day_counts={split_counts}")
    else:
        print("  eval: skipped (no eval_days configured)")

    print("\nTraining encoded LSTM decoder on NoMAD-style aligned latents:")
    decoder_model, x_scaler, y_scaler, decoder_history = train_lstm_decoder(
        X_train=X["train"],
        Y_train=Y["train"],
        X_fit_val=X["fit_val"],
        Y_fit_val=Y["fit_val"],
        config=deepcopy(config),
    )

    criterion = nn.MSELoss()
    fit_train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_scaler.transform(X["train"]).astype(np.float32, copy=False)),
            torch.from_numpy(y_scaler.transform(Y["train"]).astype(np.float32, copy=False)),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )
    fit_val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_scaler.transform(X["fit_val"]).astype(np.float32, copy=False)),
            torch.from_numpy(y_scaler.transform(Y["fit_val"]).astype(np.float32, copy=False)),
        ),
        batch_size=config.batch_size,
        shuffle=False,
    )
    fit_train_metrics, _, _ = evaluate_model(decoder_model, fit_train_loader, config.device, y_scaler, criterion=criterion)
    fit_val_metrics, _, _ = evaluate_model(decoder_model, fit_val_loader, config.device, y_scaler, criterion=criterion)

    eval_metrics = None
    y_true_eval = None
    y_pred_eval = None
    if "eval" in X:
        eval_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(x_scaler.transform(X["eval"]).astype(np.float32, copy=False)),
                torch.from_numpy(y_scaler.transform(Y["eval"]).astype(np.float32, copy=False)),
            ),
            batch_size=config.batch_size,
            shuffle=False,
        )
        eval_metrics, y_true_eval, y_pred_eval = evaluate_model(
            decoder_model,
            eval_loader,
            config.device,
            y_scaler,
            criterion=criterion,
        )

    print("\nFinal decoder metrics:")
    print(f"  fit_train: {fit_train_metrics}")
    print(f"  fit_val:   {fit_val_metrics}")
    if eval_metrics is not None:
        print(f"  eval:      {eval_metrics}")
    else:
        print("  eval:      skipped")

    per_day_eval_metrics: dict[str, dict[str, float]] = {}
    if y_true_eval is not None and y_pred_eval is not None:
        for day_name in sorted(set(day_meta["eval"].tolist())):
            mask = day_meta["eval"] == day_name
            per_day_eval_metrics[day_name] = regression_metrics(y_true_eval[mask], y_pred_eval[mask])
        predictions_df = pd.DataFrame(
            {
                "day_name": day_meta["eval"],
                "t": t_meta["eval"],
                "true_vx": y_true_eval[:, 0],
                "true_vy": y_true_eval[:, 1],
                "pred_vx": y_pred_eval[:, 0],
                "pred_vy": y_pred_eval[:, 1],
            }
        )
        predictions_df.to_csv(output_dir / "eval_predictions.csv", index=False)
        save_prediction_plot(output_dir / "eval_predictions.png", y_true_eval, y_pred_eval, title="All eval days")
        for day_name in sorted(set(day_meta["eval"].tolist())):
            mask = day_meta["eval"] == day_name
            save_prediction_plot(output_dir / f"eval_predictions_{day_name}.png", y_true_eval[mask], y_pred_eval[mask], title=f"Eval day {day_name}")

    decoder_kwargs = {
        "decoder_variant": config.decoder_variant,
        "input_dim": int(config.nomad_latent_dim),
        "seq_len": int(seq_len),
        "time_dim": int(config.time_dim),
        "latent_embed_dim": int(config.latent_embed_dim),
        "hidden_dim": int(config.hidden_dim),
        "num_layers": int(config.num_layers),
        "dropout": float(config.dropout),
        "output_dim": 2,
        "use_bias": False,
    }
    checkpoint = {
        "model_family": "nomad_lstm",
        "model_backend": "nomad_style_dynamics",
        "config": json_ready(asdict(config)),
        "data_source_identifier": str(config.data_source or Path(config.data_dir).name),
        "data_source_dir": str(config.data_dir.resolve()),
        "sequence_length": int(seq_len),
        "reference_model_state_dict": reference_model.state_dict(),
        "reference_model_kwargs": {
            "input_dim": int(base_day.counts.shape[1]),
            "readin_dim": int(config.readin_dim),
            "latent_dim": int(config.nomad_latent_dim),
            "dropout": float(config.reference_alignment_dropout if config.reference_alignment_dropout is not None else config.dropout),
            "output_dim": 2,
        },
        "decoder_state_dict": decoder_model.state_dict(),
        "decoder_kwargs": decoder_kwargs,
        "model_state_dict": decoder_model.state_dict(),
        "model_kwargs": decoder_kwargs,
        "x_mean": x_scaler.mean_,
        "x_std": x_scaler.std_,
        "y_mean": y_scaler.mean_,
        "y_std": y_scaler.std_,
        "unit_cols": base_day.unit_cols,
        "target_columns": base_day.target_columns,
        "base_day": config.base_day,
        "train_days": config.train_days,
        "eval_days": config.eval_days,
        "reference_stats": {"mean": reference_mean, "cov": reference_cov, "count": int(len(base_latent))},
        "reference_input_normalization": base_input_normalization,
        "alignment_network_kwargs": {
            "input_dim": int(base_day.counts.shape[1]),
            "hidden_dim": int(config.alignment_hidden_dim),
            "dropout": float(config.reference_alignment_dropout if config.reference_alignment_dropout is not None else config.dropout),
        },
        "alignment_mode": "align_net_plus_observation_adapters",
        "day_alignment": {
            day_name: json_ready(asdict(days[day_name].alignment)) if days[day_name].alignment is not None else None
            for day_name in config.selected_days
        },
        "day_adaptation_state_dicts": day_adaptation_state_dicts,
    }
    checkpoint_path = output_dir / "manifold_lstm_decoder.pt"
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "model_family": "nomad_lstm",
        "model_backend": "nomad_style_dynamics",
        "alignment_mode": "align_net_plus_observation_adapters",
        "config": json_ready(asdict(config)),
        "data_source_identifier": str(config.data_source or Path(config.data_dir).name),
        "data_source_dir": str(config.data_dir.resolve()),
        "sequence_length": int(seq_len),
        "reference_input_normalization": json_ready(base_input_normalization),
        "reference_fit_train_metrics": reference_fit_train_metrics,
        "reference_fit_val_metrics": reference_fit_val_metrics,
        "fit_train_metrics": fit_train_metrics,
        "fit_val_metrics": fit_val_metrics,
        "eval_metrics": eval_metrics,
        "per_day_eval_metrics": per_day_eval_metrics,
        "history": decoder_history,
        "reference_history": reference_history,
        "alignment": {
            day_name: json_ready(asdict(days[day_name].alignment)) if days[day_name].alignment is not None else None
            for day_name in config.selected_days
        },
        "day_adaptation_modules": {
            day_name: {
                "is_reference_day": bool(payload.get("is_reference_day")),
                "has_align_net_state_dict": payload.get("align_net_state_dict") is not None,
                "has_day_readin_state_dict": payload.get("day_readin_state_dict") is not None,
                "has_day_rate_head_state_dict": payload.get("day_rate_head_state_dict") is not None,
                "align_net_state_shapes": state_dict_shape_summary(payload.get("align_net_state_dict")),
                "day_readin_state_shapes": state_dict_shape_summary(payload.get("day_readin_state_dict")),
                "day_rate_head_state_shapes": state_dict_shape_summary(payload.get("day_rate_head_state_dict")),
                "has_input_normalization": payload.get("input_normalization") is not None,
            }
            for day_name, payload in day_adaptation_state_dicts.items()
        },
    }
    summary_payload = json_ready(summary)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"\nArtifacts written to: {output_dir.resolve()}")
    if config.skip_model_export:
        print("Named export folder: skipped (--skip-model-export)")
        print("Named model export: skipped (--skip-model-export)")
        print("Named summary export: skipped (--skip-model-export)")
        return

    exported_model_dir = ensure_dir(DEFAULT_MODEL_DIR)
    temp_export_path = exported_model_dir / f"._tmp_model_export_{int(datetime.now().timestamp() * 1000)}.pt"
    torch.save(checkpoint, temp_export_path)
    modified_at = datetime.fromtimestamp(temp_export_path.stat().st_mtime)
    summary_config = summary_payload.get("config")
    summary_config = summary_config if isinstance(summary_config, dict) else {}
    export_stem = build_model_export_stem(summary_config, modified_at)
    exported_run_dir = resolve_unique_export_dir(exported_model_dir, export_stem)
    copy_directory_contents(output_dir, exported_run_dir)
    exported_checkpoint_path = exported_run_dir / "manifold_lstm_decoder.pt"
    if exported_checkpoint_path.exists():
        exported_checkpoint_path.unlink()
    temp_export_path.replace(exported_checkpoint_path)
    exported_summary_path = exported_run_dir / "summary.json"
    print(f"Named export folder: {exported_run_dir.resolve()}")
    print(f"Named model export: {exported_checkpoint_path.resolve()}")
    print(f"Named summary export: {exported_summary_path.resolve()}")


def main() -> None:
    config = parse_args()
    run_pipeline(config)


if __name__ == "__main__":
    main()
