#!/usr/bin/env python
from __future__ import annotations

"""
Multiday neural manifold alignment + encoded LSTM decoder for the M06 sessions.

Pipeline:
1. Parse sparse spike timestamps from `Online_*.csv`.
2. Parse 2D control voltages from `*_voltage.dat`.
3. Bin spikes per day, fit a factor-analysis neural manifold, and align every day to
   the base-day manifold.
4. Build latent sliding windows and decode voltage with a notebook-inspired encoded LSTM.
"""

import argparse
import json
import math
import random
import re
import shutil
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import FactorAnalysis
from torch.utils.data import DataLoader, TensorDataset


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT_DIR = SCRIPT_DIR / "data"
DEFAULT_TRAIN_DATA_DIR = SCRIPT_DIR / "data" / "train"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "manifold_decoder_outputs"
DEFAULT_MODEL_DIR = SCRIPT_DIR / "model"

# Edit this block when you want to run training directly from the script without
# typing CLI day lists every time. CLI arguments still take priority over these
# values. `selected_days` can stay empty, and the script will automatically use
# the union of `base_day + train_days + eval_days`. By default, the script reads
# every paired CSV + voltage DAT file in `data/train`. The current default split
# uses all available December 2025 paired days for fitting and all available
# January 2026 paired days for held-out eval.
USER_EDITABLE_SETTINGS: dict[str, object] = {
    "data_source": "train",
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
        "2025-12-29",
        "2025-12-30",
    ],
    "eval_days": [
        "2026-01-04",
        "2026-01-05",
        "2026-01-12",
        "2026-01-26",
    ],
    "export_tag": "",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def compact_day_tag(day_name: str | None) -> str:
    normalized = normalize_day_name(day_name)
    if normalized is None:
        return "unknown"
    return normalized[2:].replace("-", "")


def compact_data_source_tag(data_source: object, data_dir: object) -> str:
    raw_source = str(data_source or "").strip()
    if raw_source:
        source_text = raw_source
    else:
        raw_dir = str(data_dir or "").strip()
        source_text = Path(raw_dir).name if raw_dir else "unknownsrc"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", source_text)
    return cleaned or "unknownsrc"


def normalize_subset_suffix_tag(text: object) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "", raw)
    replacements = (
        ("behavior_behavior_", "bh_"),
        ("behavior_", "bh_"),
        ("firing_rate", "fr"),
    )
    for old, new in replacements:
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def split_data_source_prefix_suffix(data_source_tag: str) -> tuple[str, str]:
    match = re.match(r"^(data_kilosort_8070_tmpl\d+(?:_lite)?)(?:_(.+))?$", data_source_tag)
    if not match:
        return data_source_tag, ""
    return match.group(1), normalize_subset_suffix_tag(match.group(2) or "")


def derive_export_tag_extra(export_tag: object, dataset_suffix: str) -> str:
    export_norm = normalize_subset_suffix_tag(export_tag)
    if not export_norm or export_norm.endswith("_default"):
        return ""
    if dataset_suffix:
        legacy_behavior_alias = ""
        if dataset_suffix.startswith("bh_vxvy_"):
            legacy_behavior_alias = "bh_" + dataset_suffix[len("bh_vxvy_") :]
        if export_norm == dataset_suffix or export_norm == f"{dataset_suffix}_default":
            return ""
        if legacy_behavior_alias and (
            export_norm == legacy_behavior_alias or export_norm == f"{legacy_behavior_alias}_default"
        ):
            return ""
        if export_norm.startswith(f"{dataset_suffix}_"):
            return export_norm[len(dataset_suffix) + 1 :]
    for marker in ("_tuned_best", "_best", "_tuned"):
        if export_norm.endswith(marker):
            return marker[1:]
    return export_norm


def build_model_export_stem(
    summary_config: dict[str, object],
    modified_at: datetime,
) -> str:
    base_day = normalize_day_name(summary_config.get("base_day")) or str(summary_config.get("base_day") or "unknown")
    raw_train_days = summary_config.get("train_days")
    train_days = raw_train_days if isinstance(raw_train_days, (list, tuple, set)) else []
    normalized_days = [normalize_day_name(str(day)) or str(day) for day in train_days]
    ordered_days = sorted(normalized_days)
    if not ordered_days:
        train_range = "unknown-unknown"
    else:
        train_range = f"{compact_day_tag(ordered_days[0])}-{compact_day_tag(ordered_days[-1])}"
    data_source_tag = compact_data_source_tag(summary_config.get("data_source"), summary_config.get("data_dir"))
    if data_source_tag and not data_source_tag.startswith("data_"):
        data_source_tag = f"data_{data_source_tag}"
    data_source_prefix, data_source_suffix = split_data_source_prefix_suffix(data_source_tag)
    modified_tag = modified_at.strftime("mod%y%m%d_%H%M")
    export_extra = derive_export_tag_extra(summary_config.get("export_tag"), data_source_suffix)
    stem = f"b{compact_day_tag(base_day)}_tr{train_range}_src{data_source_prefix}_{modified_tag}"
    if data_source_suffix:
        stem += f"_{data_source_suffix}"
    if export_extra:
        stem += f"_{export_extra}"
    return stem


def resolve_unique_export_dir(directory: Path, stem: str) -> Path:
    candidate = directory / stem
    if not candidate.exists():
        return candidate
    version = 2
    while True:
        candidate = directory / f"{stem}_v{version:02d}"
        if not candidate.exists():
            return candidate
        version += 1


def copy_directory_contents(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for item in src_dir.iterdir():
        target = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def as_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def list_parser(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_day_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    match_compact = re.fullmatch(r"(\d{6})(_\d+)?", text)
    if match_compact:
        raw_compact = match_compact.group(1)
        return f"20{raw_compact[:2]}-{raw_compact[2:4]}-{raw_compact[4:6]}{match_compact.group(2) or ''}"
    match_iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})(_\d+)?", text)
    if match_iso:
        return text
    raise ValueError(f"Unsupported day format: {raw}. Use YYYY-MM-DD, YYYY-MM-DD_NN, YYMMDD, or YYMMDD_NN.")


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


def discover_data_source_dirs(data_root: Path | None = None) -> dict[str, Path]:
    root = (data_root or DEFAULT_DATA_ROOT_DIR).resolve()
    mapping: dict[str, Path] = {}
    if not root.exists():
        return mapping
    for child in sorted(root.iterdir()):
        if child.is_dir():
            mapping[child.name] = child.resolve()
    return mapping


def canonical_data_source_name(data_dir: Path, data_root: Path | None = None) -> str | None:
    resolved_dir = data_dir.resolve()
    for name, source_dir in discover_data_source_dirs(data_root).items():
        if source_dir.resolve() == resolved_dir:
            return name
    return None


def resolve_data_dir_from_source(
    explicit_data_dir: Path | None,
    data_source: str | None,
    base_dir: Path,
    default_dir: Path,
) -> Path:
    if explicit_data_dir is not None:
        return explicit_data_dir.resolve()

    raw_source = str(data_source).strip() if data_source is not None else ""
    if not raw_source:
        return default_dir.resolve()

    source_path = Path(raw_source)
    candidate_paths: list[Path] = []
    if source_path.is_absolute():
        candidate_paths.append(source_path)
    else:
        candidate_paths.append((DEFAULT_DATA_ROOT_DIR / raw_source).resolve())
        candidate_paths.append((base_dir / source_path).resolve())

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_dir():
            return candidate

    available_sources = discover_data_source_dirs(DEFAULT_DATA_ROOT_DIR)
    if raw_source in available_sources:
        return available_sources[raw_source]

    available_text = ", ".join(sorted(available_sources.keys())) or "(none)"
    raise FileNotFoundError(
        f"Unknown data source '{raw_source}'. "
        f"Available data sources under {DEFAULT_DATA_ROOT_DIR}: {available_text}"
    )


def parse_day_list_value(value: object) -> list[str] | None:
    if value is None:
        return None
    items: list[str]
    if isinstance(value, str):
        items = list_parser(value)
    elif isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise TypeError("Day list must be a comma-separated string or a list/tuple/set of day strings.")
    normalized = [normalize_day_name(item) for item in items]
    return [item for item in normalized if item is not None] or None


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


def unique_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def window_bins_from_ms(window_ms: float, bin_size_s: float, include_current_bin: bool = True) -> int:
    base_bins = int(round((window_ms * 1e-3) / bin_size_s))
    seq_len = base_bins + (1 if include_current_bin else 0)
    return max(seq_len, 2)


@dataclass
class Config:
    data_source: str | None = None
    data_dir: Path = SCRIPT_DIR / "data"
    output_dir: Path = SCRIPT_DIR / "manifold_decoder_outputs"
    selected_days: list[str] | None = None
    base_day: str = "2025-12-22"
    train_days: list[str] | None = None
    eval_days: list[str] | None = None
    export_tag: str = ""
    skip_model_export: bool = False
    bin_size_s: float = 0.03
    window_ms: float = 780.0
    n_latents: int = 40
    fa_restarts: int = 3
    fa_max_iter: int = 900
    fa_tol: float = 1e-3
    psi_floor: float = 1e-3
    align_n: int = 180
    align_th: float = 5e-3
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
        if not self.base_day:
            raise ValueError("base_day must not be empty")
        if not self.train_days:
            raise ValueError("At least one train_day is required")

        if not self.selected_days:
            self.selected_days = unique_preserve_order([self.base_day, *self.train_days, *self.eval_days])
        else:
            required_days = set([self.base_day, *self.train_days, *self.eval_days])
            missing_days = required_days.difference(self.selected_days)
            if missing_days:
                raise ValueError(
                    "selected_days must include base/train/eval days; missing: "
                    + ", ".join(sorted(missing_days))
                )

        overlap = set(self.train_days).intersection(self.eval_days)
        if overlap:
            raise ValueError(f"train_days and eval_days must be disjoint; overlap: {sorted(overlap)}")
        if self.base_day not in self.train_days:
            raise ValueError("base_day must be included in train_days")

        if self.bin_size_s <= 0:
            raise ValueError("bin_size_s must be positive")
        if self.window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if self.n_latents <= 0:
            raise ValueError("n_latents must be positive")
        if self.align_n < self.n_latents:
            raise ValueError("align_n must be >= n_latents")
        if not (0.0 < self.fit_val_ratio < 0.5):
            raise ValueError("fit_val_ratio must be in (0, 0.5)")
        decoder_variant = str(self.decoder_variant).strip().lower()
        if decoder_variant not in {"encoded", "v2"}:
            raise ValueError("decoder_variant must be either 'encoded' or 'v2'")
        self.decoder_variant = decoder_variant
        if self.time_dim <= 0 or self.latent_embed_dim <= 0:
            raise ValueError("time_dim and latent_embed_dim must be positive")
        if self.hidden_dim <= 0 or self.num_layers <= 0:
            raise ValueError("hidden_dim and num_layers must be positive")
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0 or self.l1_lambda < 0:
            raise ValueError("weight_decay and l1_lambda must be non-negative")
        if self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive")
        if self.patience <= 0 or self.scheduler_patience <= 0:
            raise ValueError("patience and scheduler_patience must be positive")
        if not (0.0 < self.scheduler_factor < 1.0):
            raise ValueError("scheduler_factor must be in (0, 1)")
        if self.min_lr <= 0:
            raise ValueError("min_lr must be positive")


@dataclass
class FAModel:
    mean: np.ndarray
    C: np.ndarray
    psi: np.ndarray
    score: float


@dataclass
class AlignmentResult:
    day_name: str
    transform: np.ndarray
    stable_mask: np.ndarray
    stable_units: list[str]
    loading_rmse_before: float
    loading_rmse_after: float


@dataclass
class DayData:
    day_name: str
    unit_cols: list[str]
    t: np.ndarray
    counts: np.ndarray
    target: np.ndarray
    target_columns: list[str]
    voltage_df: pd.DataFrame
    fa_raw: FAModel | None = None
    fa_aligned: FAModel | None = None
    alignment: AlignmentResult | None = None
    latent: np.ndarray | None = None


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        arr = np.asarray(x, dtype=np.float32)
        self.mean_ = arr.mean(axis=0, keepdims=True)
        self.std_ = arr.std(axis=0, keepdims=True)
        self.std_[self.std_ < 1e-6] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler is not fitted")
        return (np.asarray(x, dtype=np.float32) - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler is not fitted")
        return np.asarray(x, dtype=np.float32) * self.std_ + self.mean_


class EncodedLatentLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        time_dim: int,
        latent_embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int = 2,
        use_bias: bool = False,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.time_encoding = nn.Linear(seq_len, time_dim, bias=use_bias)
        self.latent_encoding = nn.Linear(input_dim, latent_embed_dim, bias=use_bias)
        self.lstm = nn.LSTM(
            input_size=latent_embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        head_hidden = max(hidden_dim // 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = x.transpose(1, 2)
        encoded = self.time_encoding(encoded)
        encoded = encoded.transpose(1, 2)
        encoded = self.latent_encoding(encoded)
        out, _ = self.lstm(encoded)
        return self.head(out[:, -1, :])


class DirectLatentLSTMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int = 2,
        use_bias: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        _ = seq_len
        _ = use_bias
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        head_hidden = max(hidden_dim // 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def build_decoder_model_from_kwargs(kwargs: dict[str, object]) -> nn.Module:
    decoder_variant = str(kwargs.get("decoder_variant", "encoded")).strip().lower()
    if decoder_variant in {"encoded", "v1", "nomad_lstm"}:
        return EncodedLatentLSTMRegressor(
            input_dim=int(kwargs["input_dim"]),
            seq_len=int(kwargs["seq_len"]),
            time_dim=int(kwargs["time_dim"]),
            latent_embed_dim=int(kwargs["latent_embed_dim"]),
            hidden_dim=int(kwargs["hidden_dim"]),
            num_layers=int(kwargs["num_layers"]),
            dropout=float(kwargs["dropout"]),
            output_dim=int(kwargs.get("output_dim", 2)),
            use_bias=bool(kwargs.get("use_bias", False)),
        )
    if decoder_variant in {"v2", "direct", "nomad_lstm_v2"}:
        return DirectLatentLSTMRegressor(
            input_dim=int(kwargs["input_dim"]),
            seq_len=int(kwargs["seq_len"]),
            hidden_dim=int(kwargs["hidden_dim"]),
            num_layers=int(kwargs["num_layers"]),
            dropout=float(kwargs["dropout"]),
            output_dim=int(kwargs.get("output_dim", 2)),
            use_bias=bool(kwargs.get("use_bias", False)),
        )
    raise ValueError(f"Unsupported decoder_variant: {decoder_variant}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Align multiday neural manifolds and train an encoded LSTM decoder."
    )
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
    parser.add_argument("--export-tag", type=str, default=None, help="Optional suffix appended to the exported model folder name.")
    parser.add_argument("--skip-model-export", action="store_true", help="Skip copying this run into the shared model export directory.")
    parser.add_argument("--test-days", dest="eval_days_alias", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bin-size-s", type=float, default=0.03)
    parser.add_argument("--window-ms", type=float, default=780.0)
    parser.add_argument("--n-latents", type=int, default=40)
    parser.add_argument("--fa-restarts", type=int, default=3)
    parser.add_argument("--fa-max-iter", type=int, default=900)
    parser.add_argument("--fa-tol", type=float, default=1e-3)
    parser.add_argument("--psi-floor", type=float, default=1e-3)
    parser.add_argument("--align-n", type=int, default=180)
    parser.add_argument("--align-th", type=float, default=5e-3)
    parser.add_argument(
        "--fit-val-ratio",
        type=float,
        default=0.1,
        help="Within-train-days time-ordered hold-out ratio used only for early stopping.",
    )
    parser.add_argument("--decoder-variant", type=str, default="encoded", choices=["encoded", "v2"])
    parser.add_argument("--val-ratio", dest="fit_val_ratio_alias", type=float, default=None, help=argparse.SUPPRESS)
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
    inline_settings = resolve_inline_settings(SCRIPT_DIR)
    explicit_data_dir = args.data_dir or inline_settings["data_dir"]
    requested_data_source = args.data_source if args.data_source is not None else inline_settings["data_source"]
    data_dir = resolve_data_dir_from_source(
        explicit_data_dir=explicit_data_dir,
        data_source=requested_data_source,
        base_dir=SCRIPT_DIR,
        default_dir=DEFAULT_TRAIN_DATA_DIR,
    )
    data_source_name = canonical_data_source_name(data_dir, DEFAULT_DATA_ROOT_DIR)
    output_dir = args.output_dir or inline_settings["output_dir"] or DEFAULT_OUTPUT_DIR

    default_fit_val_ratio = parser.get_default("fit_val_ratio")
    eval_days_raw = args.eval_days_alias if args.eval_days_alias is not None else args.eval_days
    if args.eval_days_alias is not None and args.eval_days is not None and args.eval_days != args.eval_days_alias:
        parser.error("--eval-days and legacy --test-days disagree; please provide only one.")

    fit_val_ratio = args.fit_val_ratio_alias if args.fit_val_ratio_alias is not None else args.fit_val_ratio
    if (
        args.fit_val_ratio_alias is not None
        and args.fit_val_ratio != default_fit_val_ratio
        and not math.isclose(args.fit_val_ratio, args.fit_val_ratio_alias)
    ):
        parser.error("--fit-val-ratio and legacy --val-ratio disagree; please provide only one.")

    available_pairs = discover_day_pairs(data_dir, None)
    available_days = list(available_pairs.keys())
    if not available_days:
        raise FileNotFoundError(f"No paired spike CSV + voltage DAT files were found in {data_dir}")

    train_days = parse_day_list_value(args.train_days) if args.train_days is not None else inline_settings["train_days"]
    if not train_days:
        train_days = available_days

    eval_days = parse_day_list_value(eval_days_raw) if eval_days_raw is not None else inline_settings["eval_days"]
    if eval_days is None:
        eval_days = []

    base_day = normalize_day_name(args.base_day) if args.base_day is not None else inline_settings["base_day"]
    if base_day is None:
        base_day = train_days[0]

    selected_days = parse_day_list_value(args.selected_days) if args.selected_days is not None else inline_settings["selected_days"]
    if not selected_days:
        selected_days = unique_preserve_order([base_day, *train_days, *eval_days])
    export_tag = args.export_tag if args.export_tag is not None else inline_settings["export_tag"]

    config = Config(
        data_source=data_source_name or str(requested_data_source or "").strip() or None,
        data_dir=data_dir,
        output_dir=output_dir,
        selected_days=selected_days,
        base_day=base_day,
        train_days=train_days,
        eval_days=eval_days,
        export_tag=str(export_tag or "").strip(),
        skip_model_export=args.skip_model_export,
        bin_size_s=args.bin_size_s,
        window_ms=args.window_ms,
        n_latents=args.n_latents,
        fa_restarts=args.fa_restarts,
        fa_max_iter=args.fa_max_iter,
        fa_tol=args.fa_tol,
        psi_floor=args.psi_floor,
        align_n=args.align_n,
        align_th=args.align_th,
        fit_val_ratio=fit_val_ratio,
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


def infer_spike_csv_day_name(csv_name: str) -> str | None:
    """Infer the recording day from known Online CSV naming conventions."""
    tmpl_match = re.search(r"tmpl\d{6}_(20\d{6}|\d{6})(?:_(\d+))?(?:\D|$)", csv_name)
    if tmpl_match:
        raw = tmpl_match.group(1)
        session_suffix = f"_{tmpl_match.group(2)}" if tmpl_match.group(2) else ""
        if len(raw) == 8:
            return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}{session_suffix}"
        return f"20{raw[:2]}-{raw[2:4]}-{raw[4:6]}{session_suffix}"

    suffix_match = re.search(r"(\d{6})(?:_(\d+))?\.csv$", csv_name)
    if suffix_match:
        raw = suffix_match.group(1)
        session_suffix = f"_{suffix_match.group(2)}" if suffix_match.group(2) else ""
        return f"20{raw[:2]}-{raw[2:4]}-{raw[4:6]}{session_suffix}"
    return None


def discover_day_pairs(data_dir: Path, selected_days: Iterable[str] | None) -> dict[str, tuple[Path, Path]]:
    csv_by_day: dict[str, Path] = {}
    dat_by_day: dict[str, Path] = {}

    for csv_path in sorted(data_dir.glob("*.csv")):
        day_name = infer_spike_csv_day_name(csv_path.name)
        if day_name:
            csv_by_day[day_name] = csv_path

    for dat_path in sorted(data_dir.glob("*_voltage*.dat")):
        match = re.search(r"(\d{4}-\d{2}-\d{2})_voltage(?:_(\d+))?\.dat$", dat_path.name)
        if match:
            session_suffix = f"_{match.group(2)}" if match.group(2) else ""
            dat_by_day[f"{match.group(1)}{session_suffix}"] = dat_path

    available_days = sorted(set(csv_by_day).intersection(dat_by_day))
    requested_days = available_days if selected_days is None else list(selected_days)

    pairs: dict[str, tuple[Path, Path]] = {}
    for day_name in requested_days:
        if day_name not in csv_by_day:
            raise FileNotFoundError(f"No spike CSV found for {day_name} in {data_dir}")
        if day_name not in dat_by_day:
            raise FileNotFoundError(f"No voltage DAT found for {day_name} in {data_dir}")
        pairs[day_name] = (csv_by_day[day_name], dat_by_day[day_name])
    return pairs


def read_sparse_spike_times(csv_path: Path) -> tuple[list[str], list[np.ndarray]]:
    df = pd.read_csv(csv_path)
    unit_cols = [str(c) for c in df.columns if str(c).startswith("Unit")]
    if not unit_cols:
        raise ValueError(f"No Unit columns found in {csv_path}")
    unit_times = [df[col].dropna().to_numpy(dtype=np.float64) for col in unit_cols]
    return unit_cols, unit_times


def read_voltage_dat(dat_path: Path) -> pd.DataFrame:
    time_pat = re.compile(
        r"timestamp\s*:\s*([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)"
        r".*?voltage\s*:\s*"
        r"([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)\s+"
        r"([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)"
    )
    point_pat = re.compile(
        r"pointpos\s*:\s*\(\s*([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
        r"([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)\s*\)"
    )
    target_pat = re.compile(
        r"targetpos\s*:\s*\(\s*([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)\s*,\s*"
        r"([+-]?(?:\d+\.?\d*|\d*\.?\d+)(?:[eE][+-]?\d+)?)\s*\)"
    )

    rows: list[dict[str, float]] = []
    with dat_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = time_pat.search(line)
            if not m:
                continue
            row = {
                "t": float(m.group(1)),
                "vx": float(m.group(2)),
                "vy": float(m.group(3)),
                "point_x": np.nan,
                "point_y": np.nan,
                "target_x": np.nan,
                "target_y": np.nan,
            }
            pm = point_pat.search(line)
            if pm:
                row["point_x"] = float(pm.group(1))
                row["point_y"] = float(pm.group(2))
            tm = target_pat.search(line)
            if tm:
                row["target_x"] = float(tm.group(1))
                row["target_y"] = float(tm.group(2))
            rows.append(row)

    if not rows:
        raise ValueError(f"Failed to parse any voltage samples from {dat_path}")

    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    return df.groupby("t", as_index=False).mean(numeric_only=True)


def build_time_grid(voltage_df: pd.DataFrame, bin_size_s: float) -> tuple[np.ndarray, np.ndarray]:
    t_min = float(voltage_df["t"].min())
    t_max = float(voltage_df["t"].max())
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        raise ValueError("Invalid voltage time range")
    edges = np.arange(t_min, t_max + bin_size_s, bin_size_s, dtype=np.float64)
    if edges.size < 2:
        raise ValueError("Too few time bins were created")
    centers = edges[:-1] + 0.5 * bin_size_s
    return edges, centers


def bin_spike_times_to_counts(unit_times: list[np.ndarray], edges: np.ndarray) -> np.ndarray:
    t_start = float(edges[0])
    dt = float(edges[1] - edges[0])
    n_bins = len(edges) - 1
    counts = np.zeros((n_bins, len(unit_times)), dtype=np.float32)

    for unit_idx, spikes in enumerate(unit_times):
        if spikes.size == 0:
            continue
        idx = np.floor((spikes - t_start) / dt).astype(np.int64)
        valid = idx[(idx >= 0) & (idx < n_bins)]
        if valid.size == 0:
            continue
        counts[:, unit_idx] = np.bincount(valid, minlength=n_bins).astype(np.float32, copy=False)

    return counts


def interpolate_columns(voltage_df: pd.DataFrame, t_grid: np.ndarray, cols: list[str]) -> np.ndarray:
    source = voltage_df[["t", *cols]].copy()
    source = source.sort_values("t").groupby("t", as_index=False).mean(numeric_only=True)
    t_src = source["t"].to_numpy(dtype=np.float64)
    out = np.empty((len(t_grid), len(cols)), dtype=np.float32)
    for i, col in enumerate(cols):
        values = source[col].to_numpy(dtype=np.float64)
        interp = np.interp(t_grid, t_src, values)
        interp[(t_grid < t_src[0]) | (t_grid > t_src[-1])] = np.nan
        out[:, i] = interp.astype(np.float32, copy=False)
    return out


def load_day_data(day_name: str, csv_path: Path, dat_path: Path, bin_size_s: float) -> DayData:
    unit_cols, unit_times = read_sparse_spike_times(csv_path)
    voltage_df = read_voltage_dat(dat_path)
    edges, t_centers = build_time_grid(voltage_df, bin_size_s)
    counts = bin_spike_times_to_counts(unit_times, edges)
    target_cols = ["vx", "vy"]
    target = interpolate_columns(voltage_df, t_centers, target_cols)
    valid = np.isfinite(target).all(axis=1)
    t = t_centers[valid]
    counts = counts[valid]
    target = target[valid]
    return DayData(
        day_name=day_name,
        unit_cols=unit_cols,
        t=t.astype(np.float64, copy=False),
        counts=counts.astype(np.float32, copy=False),
        target=target.astype(np.float32, copy=False),
        target_columns=target_cols,
        voltage_df=voltage_df,
    )


def fit_factor_analysis(
    X: np.ndarray,
    n_latents: int,
    n_restarts: int,
    max_iter: int,
    tol: float,
    psi_floor: float,
    seed: int,
) -> FAModel:
    X64 = np.asarray(X, dtype=np.float64)
    best_score = -np.inf
    best_fa: FAModel | None = None

    for restart_idx in range(n_restarts):
        mdl = FactorAnalysis(
            n_components=n_latents,
            max_iter=max_iter,
            tol=tol,
            random_state=seed + restart_idx,
        )
        mdl.fit(X64)
        score = float(mdl.score(X64))
        mean = np.asarray(mdl.mean_, dtype=np.float64)
        C = np.asarray(mdl.components_.T, dtype=np.float64)
        psi = np.asarray(mdl.noise_variance_, dtype=np.float64)
        psi = np.maximum(psi, float(psi_floor))

        fa = FAModel(mean=mean, C=C, psi=psi, score=score)
        if fa.score > best_score:
            best_score = fa.score
            best_fa = fa

    if best_fa is None:
        raise RuntimeError("Factor analysis fitting failed")
    return best_fa


def learn_optimal_orthonormal_transform(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    S = as_float_array(m1).T @ as_float_array(m2)
    U, _, Vt = np.linalg.svd(S, full_matrices=False)
    return U @ Vt


def identify_stable_loading_rows(
    m1: np.ndarray,
    m2: np.ndarray,
    n_stable_rows: int,
    row_norm_threshold: float,
) -> np.ndarray:
    m1 = as_float_array(m1)
    m2 = as_float_array(m2)

    small_1 = np.linalg.norm(m1, axis=1) < row_norm_threshold
    small_2 = np.linalg.norm(m2, axis=1) < row_norm_threshold
    keep_mask = ~(small_1 | small_2)
    clean_rows = np.flatnonzero(keep_mask)

    if clean_rows.size == 0:
        raise ValueError("No loading rows remain after threshold screening")

    n_target = min(int(n_stable_rows), clean_rows.size)
    cur_rows = np.arange(clean_rows.size)
    while cur_rows.size > n_target:
        cur_m1 = m1[clean_rows[cur_rows]]
        cur_m2 = m2[clean_rows[cur_rows]]
        W = learn_optimal_orthonormal_transform(cur_m1, cur_m2)
        row_delta = np.linalg.norm(cur_m1 - cur_m2 @ W.T, axis=1)
        best_sorted = np.argsort(row_delta)[: cur_rows.size - 1]
        cur_rows = np.sort(cur_rows[best_sorted])

    stable_mask = np.zeros(m1.shape[0], dtype=bool)
    stable_mask[clean_rows[cur_rows]] = True
    return stable_mask


def align_factor_model(
    base_fa: FAModel,
    day_fa: FAModel,
    base_unit_cols: list[str],
    day_unit_cols: list[str],
    day_name: str,
    align_n: int,
    align_th: float,
) -> tuple[FAModel, AlignmentResult]:
    base_index = {unit: idx for idx, unit in enumerate(base_unit_cols)}
    day_index = {unit: idx for idx, unit in enumerate(day_unit_cols)}
    common_units = [unit for unit in day_unit_cols if unit in base_index]
    if not common_units:
        raise ValueError(f"No common units remain for manifold alignment on {day_name}")

    base_rows = np.asarray([base_index[unit] for unit in common_units], dtype=np.int64)
    day_rows = np.asarray([day_index[unit] for unit in common_units], dtype=np.int64)
    base_common = base_fa.C[base_rows]
    day_common = day_fa.C[day_rows]

    stable_mask_common = identify_stable_loading_rows(base_common, day_common, align_n, align_th)
    W = learn_optimal_orthonormal_transform(base_common[stable_mask_common], day_common[stable_mask_common])

    before = float(np.sqrt(np.mean((base_common[stable_mask_common] - day_common[stable_mask_common]) ** 2)))
    C_aligned = day_fa.C @ W.T
    after = float(
        np.sqrt(
            np.mean(
                (
                    base_common[stable_mask_common]
                    - C_aligned[day_rows][stable_mask_common]
                ) ** 2
            )
        )
    )

    stable_mask_day = np.zeros(len(day_unit_cols), dtype=bool)
    stable_mask_day[day_rows[stable_mask_common]] = True

    aligned_fa = FAModel(
        mean=day_fa.mean.copy(),
        C=C_aligned,
        psi=day_fa.psi.copy(),
        score=day_fa.score,
    )
    report = AlignmentResult(
        day_name=day_name,
        transform=W,
        stable_mask=stable_mask_day,
        stable_units=[day_unit_cols[i] for i in np.flatnonzero(stable_mask_day)],
        loading_rmse_before=before,
        loading_rmse_after=after,
    )
    return aligned_fa, report


def get_stabilization_matrices(fa: FAModel) -> tuple[np.ndarray, np.ndarray]:
    C = as_float_array(fa.C)
    mean = as_float_array(fa.mean)
    psi_diag = as_float_array(fa.psi)
    cov = C @ C.T + np.diag(psi_diag)
    try:
        beta = np.linalg.solve(cov, C).T
    except np.linalg.LinAlgError:
        beta = C.T @ np.linalg.pinv(cov)
    offset = -beta @ mean
    return beta, offset


def project_latents(X: np.ndarray, fa: FAModel) -> np.ndarray:
    beta, offset = get_stabilization_matrices(fa)
    return (np.asarray(X, dtype=np.float64) @ beta.T + offset).astype(np.float32, copy=False)


def average_aligned_fa_models(fa_models: list[FAModel], psi_floor: float) -> FAModel:
    if not fa_models:
        raise ValueError("At least one aligned FA model is required to build the reference manifold")
    mean = np.mean(np.stack([fa.mean for fa in fa_models], axis=0), axis=0)
    C = np.mean(np.stack([fa.C for fa in fa_models], axis=0), axis=0)
    psi = np.mean(np.stack([fa.psi for fa in fa_models], axis=0), axis=0)
    psi = np.maximum(psi, float(psi_floor))
    score = float(np.mean([fa.score for fa in fa_models]))
    return FAModel(mean=mean, C=C, psi=psi, score=score)


def build_sequences(
    latent: np.ndarray,
    target: np.ndarray,
    seq_len: int,
    end_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    X_seq = np.empty((len(end_indices), seq_len, latent.shape[1]), dtype=np.float32)
    Y = np.empty((len(end_indices), target.shape[1]), dtype=np.float32)
    for i, end_idx in enumerate(end_indices):
        X_seq[i] = latent[end_idx - seq_len + 1 : end_idx + 1]
        Y[i] = target[end_idx]
    return X_seq, Y


def split_fit_train_val_indices(
    n_timepoints: int,
    seq_len: int,
    fit_val_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    end_idx = np.arange(seq_len - 1, n_timepoints, dtype=np.int64)
    n_total = len(end_idx)
    if n_total < 2:
        raise ValueError(
            f"Need at least 2 sequence targets for train/fit_val split, got {n_total}. "
            f"Try reducing window_ms or loading longer recordings."
        )
    n_fit_val = max(1, int(math.floor(n_total * fit_val_ratio)))
    n_fit_val = min(n_fit_val, n_total - 1)
    n_train = n_total - n_fit_val
    return end_idx[:n_train], end_idx[n_train:]


def prepare_datasets(
    days: dict[str, DayData],
    train_days: list[str],
    eval_days: list[str],
    seq_len: int,
    fit_val_ratio: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    X_parts = {"train": [], "fit_val": [], "eval": []}
    Y_parts = {"train": [], "fit_val": [], "eval": []}
    meta_day = {"train": [], "fit_val": [], "eval": []}
    meta_t = {"train": [], "fit_val": [], "eval": []}

    for day_name in train_days:
        day = days[day_name]
        if day.latent is None:
            raise ValueError(f"Latents missing for {day_name}")
        train_end, fit_val_end = split_fit_train_val_indices(len(day.latent), seq_len, fit_val_ratio)
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
            raise ValueError(f"Latents missing for {day_name}")
        eval_end = np.arange(seq_len - 1, len(day.latent), dtype=np.int64)
        X_eval, Y_eval = build_sequences(day.latent, day.target, seq_len, eval_end)
        X_parts["eval"].append(X_eval)
        Y_parts["eval"].append(Y_eval)
        meta_day["eval"].append(np.full(len(eval_end), day_name, dtype=object))
        meta_t["eval"].append(day.t[eval_end])

    X = {k: np.concatenate(v, axis=0).astype(np.float32, copy=False) for k, v in X_parts.items() if v}
    Y = {k: np.concatenate(v, axis=0).astype(np.float32, copy=False) for k, v in Y_parts.items() if v}
    day_meta = {k: np.concatenate(v, axis=0) if v else np.array([], dtype=object) for k, v in meta_day.items()}
    t_meta = {k: np.concatenate(v, axis=0).astype(np.float64, copy=False) if v else np.array([], dtype=np.float64) for k, v in meta_t.items()}
    return X, Y, day_meta, t_meta


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": float(r2)}


def l1_regularization(model: nn.Module, exclude_bias_norm: bool = True) -> torch.Tensor:
    penalties = []
    for name, param in model.named_parameters():
        if param is None or not param.requires_grad:
            continue
        if exclude_bias_norm:
            name_lower = name.lower()
            if name_lower.endswith("bias") or "norm" in name_lower:
                continue
        penalties.append(param.abs().sum())
    if not penalties:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(penalties).sum()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    y_scaler: StandardScaler,
    criterion: nn.Module | None = None,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds_scaled = []
    targets_scaled = []
    scaled_loss_sum = 0.0
    scaled_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            if criterion is not None:
                loss = criterion(pred, yb)
                scaled_loss_sum += float(loss.item()) * len(xb)
                scaled_count += len(xb)
            preds_scaled.append(pred.cpu().numpy())
            targets_scaled.append(yb.cpu().numpy())
    y_pred_scaled = np.concatenate(preds_scaled, axis=0)
    y_true_scaled = np.concatenate(targets_scaled, axis=0)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_true_scaled)
    metrics = regression_metrics(y_true, y_pred)
    if criterion is not None and scaled_count > 0:
        metrics["scaled_loss"] = scaled_loss_sum / scaled_count
    return metrics, y_true, y_pred


def train_lstm_decoder(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_fit_val: np.ndarray,
    Y_fit_val: np.ndarray,
    config: Config,
) -> tuple[nn.Module, StandardScaler, StandardScaler, list[dict[str, float]]]:
    x_scaler = StandardScaler().fit(X_train.reshape(-1, X_train.shape[-1]))
    y_scaler = StandardScaler().fit(Y_train)

    X_train_scaled = x_scaler.transform(X_train)
    X_fit_val_scaled = x_scaler.transform(X_fit_val)
    Y_train_scaled = y_scaler.transform(Y_train)
    Y_fit_val_scaled = y_scaler.transform(Y_fit_val)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_scaled.astype(np.float32, copy=False)),
        torch.from_numpy(Y_train_scaled.astype(np.float32, copy=False)),
    )
    fit_val_ds = TensorDataset(
        torch.from_numpy(X_fit_val_scaled.astype(np.float32, copy=False)),
        torch.from_numpy(Y_fit_val_scaled.astype(np.float32, copy=False)),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    fit_val_loader = DataLoader(fit_val_ds, batch_size=config.batch_size, shuffle=False)

    decoder_kwargs = {
        "decoder_variant": config.decoder_variant,
        "input_dim": int(X_train.shape[-1]),
        "seq_len": int(X_train.shape[1]),
        "time_dim": int(config.time_dim),
        "latent_embed_dim": int(config.latent_embed_dim),
        "hidden_dim": int(config.hidden_dim),
        "num_layers": int(config.num_layers),
        "dropout": float(config.dropout),
        "output_dim": int(Y_train.shape[-1]),
        "use_bias": False,
    }
    model = build_decoder_model_from_kwargs(decoder_kwargs).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=1e-5,
        min_lr=config.min_lr,
    )
    criterion = nn.MSELoss()

    best_state = None
    best_fit_val_loss = np.inf
    patience_left = config.patience
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        for xb, yb in train_loader:
            xb = xb.to(config.device)
            yb = yb.to(config.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            data_loss = criterion(pred, yb)
            reg_loss = config.l1_lambda * l1_regularization(model) if config.l1_lambda > 0 else 0.0
            loss = data_loss + reg_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            running_loss += float(loss.item()) * len(xb)
            running_count += len(xb)

        train_loss = running_loss / max(running_count, 1)
        fit_val_metrics, _, _ = evaluate_model(
            model,
            fit_val_loader,
            config.device,
            y_scaler,
            criterion=criterion,
        )
        fit_val_loss = fit_val_metrics["scaled_loss"]
        scheduler.step(fit_val_loss)
        epoch_info = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"fit_val_{k}": v for k, v in fit_val_metrics.items()},
        }
        history.append(epoch_info)
        print(
            f"Epoch {epoch:04d} | train_loss={train_loss:.6f} | "
            f"fit_val_loss={fit_val_loss:.6f} | fit_val_rmse={fit_val_metrics['rmse']:.6f} | "
            f"fit_val_r2={fit_val_metrics['r2']:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if fit_val_loss < best_fit_val_loss:
            best_fit_val_loss = fit_val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")
    model.load_state_dict(best_state)
    return model, x_scaler, y_scaler, history


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
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("vx")
    axes[1].plot(y_true[:n, 1], label="true_vy", linewidth=1.0)
    axes[1].plot(y_pred[:n, 1], label="pred_vy", linewidth=1.0)
    axes[1].legend(loc="upper right")
    axes[1].set_ylabel("vy")
    axes[1].set_xlabel("Sample")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def json_ready(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    return obj


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
    print(f"  seq_len={seq_len} (window_ms={config.window_ms}, bin_size_s={config.bin_size_s})")
    days: dict[str, DayData] = {}
    reference_unit_cols: list[str] | None = None
    for day_name in config.selected_days:
        csv_path, dat_path = pairs[day_name]
        print(f"  {day_name} | spikes={csv_path.name} | voltage={dat_path.name}")
        day_data = load_day_data(day_name, csv_path, dat_path, config.bin_size_s)
        if reference_unit_cols is None:
            reference_unit_cols = list(day_data.unit_cols)
        elif day_data.unit_cols != reference_unit_cols:
            shared_units = len(set(reference_unit_cols).intersection(day_data.unit_cols))
            print(
                f"    note: unit columns differ from the reference day "
                f"(current={len(day_data.unit_cols)}, reference={len(reference_unit_cols)}, shared={shared_units})"
            )
        print(
            f"    bins={len(day_data.t)} | units={day_data.counts.shape[1]} | "
            f"target_dim={day_data.target.shape[1]} | duration={day_data.t[-1] - day_data.t[0]:.2f}s"
        )
        days[day_name] = day_data

    if config.base_day not in days:
        raise ValueError(f"Base day {config.base_day} is not in selected_days")

    for day_name in unique_preserve_order([*config.train_days, *config.eval_days]):
        if day_name not in days:
            raise ValueError(f"Requested day {day_name} is not loaded")

    print("\nFitting factor-analysis manifolds:")
    for day_name, day_data in days.items():
        day_data.fa_raw = fit_factor_analysis(
            X=day_data.counts,
            n_latents=config.n_latents,
            n_restarts=config.fa_restarts,
            max_iter=config.fa_max_iter,
            tol=config.fa_tol,
            psi_floor=config.psi_floor,
            seed=config.seed,
        )
        print(f"  {day_name} | score={day_data.fa_raw.score:.6f}")

    print("\nAligning manifolds to the base day:")
    base_day = days[config.base_day]
    if base_day.fa_raw is None:
        raise RuntimeError("Base FA model is missing")
    base_day.fa_aligned = base_day.fa_raw
    base_day.alignment = AlignmentResult(
        day_name=base_day.day_name,
        transform=np.eye(config.n_latents, dtype=np.float64),
        stable_mask=np.ones(len(base_day.unit_cols), dtype=bool),
        stable_units=list(base_day.unit_cols),
        loading_rmse_before=0.0,
        loading_rmse_after=0.0,
    )

    for day_name, day_data in days.items():
        if day_name == config.base_day:
            continue
        if day_data.fa_raw is None:
            raise RuntimeError(f"FA model is missing for {day_name}")
        day_data.fa_aligned, day_data.alignment = align_factor_model(
            base_fa=base_day.fa_raw,
            day_fa=day_data.fa_raw,
            base_unit_cols=base_day.unit_cols,
            day_unit_cols=day_data.unit_cols,
            day_name=day_name,
            align_n=config.align_n,
            align_th=config.align_th,
        )
        n_stable = int(day_data.alignment.stable_mask.sum())
        print(
            f"  {day_name} | stable_units={n_stable} | "
            f"loading_rmse_before={day_data.alignment.loading_rmse_before:.6f} | "
            f"after={day_data.alignment.loading_rmse_after:.6f}"
        )

    print("\nProjecting aligned latents:")
    for day_name, day_data in days.items():
        if day_data.fa_aligned is None:
            raise RuntimeError(f"Aligned FA model is missing for {day_name}")
        day_data.latent = project_latents(day_data.counts, day_data.fa_aligned)
        print(f"  {day_name} | latent_shape={day_data.latent.shape}")

    print("\nBuilding sequence datasets:")
    X, Y, day_meta, t_meta = prepare_datasets(
        days=days,
        train_days=config.train_days,
        eval_days=config.eval_days,
        seq_len=seq_len,
        fit_val_ratio=config.fit_val_ratio,
    )
    for split_name in ["train", "fit_val"]:
        if split_name not in X:
            raise ValueError(f"Missing split: {split_name}")
        split_counts = {day: int(np.sum(day_meta[split_name] == day)) for day in sorted(set(day_meta[split_name].tolist()))}
        print(f"  {split_name}: X={X[split_name].shape}, Y={Y[split_name].shape}, day_counts={split_counts}")
    if "eval" in X:
        split_counts = {day: int(np.sum(day_meta["eval"] == day)) for day in sorted(set(day_meta["eval"].tolist()))}
        print(f"  eval: X={X['eval'].shape}, Y={Y['eval'].shape}, day_counts={split_counts}")
    else:
        print("  eval: skipped (no eval_days configured in the current data directory)")

    print("\nTraining encoded LSTM decoder:")
    model, x_scaler, y_scaler, history = train_lstm_decoder(
        X_train=X["train"],
        Y_train=Y["train"],
        X_fit_val=X["fit_val"],
        Y_fit_val=Y["fit_val"],
        config=config,
    )

    criterion = nn.MSELoss()
    eval_metrics = None
    y_true_eval = None
    y_pred_eval = None
    if "eval" in X:
        eval_ds = TensorDataset(
            torch.from_numpy(x_scaler.transform(X["eval"]).astype(np.float32, copy=False)),
            torch.from_numpy(y_scaler.transform(Y["eval"]).astype(np.float32, copy=False)),
        )
        eval_loader = DataLoader(eval_ds, batch_size=config.batch_size, shuffle=False)
        eval_metrics, y_true_eval, y_pred_eval = evaluate_model(
            model,
            eval_loader,
            config.device,
            y_scaler,
            criterion=criterion,
        )

    fit_train_ds = TensorDataset(
        torch.from_numpy(x_scaler.transform(X["train"]).astype(np.float32, copy=False)),
        torch.from_numpy(y_scaler.transform(Y["train"]).astype(np.float32, copy=False)),
    )
    fit_val_ds = TensorDataset(
        torch.from_numpy(x_scaler.transform(X["fit_val"]).astype(np.float32, copy=False)),
        torch.from_numpy(y_scaler.transform(Y["fit_val"]).astype(np.float32, copy=False)),
    )
    fit_train_metrics, _, _ = evaluate_model(
        model,
        DataLoader(fit_train_ds, batch_size=config.batch_size, shuffle=False),
        config.device,
        y_scaler,
        criterion=criterion,
    )
    fit_val_metrics, _, _ = evaluate_model(
        model,
        DataLoader(fit_val_ds, batch_size=config.batch_size, shuffle=False),
        config.device,
        y_scaler,
        criterion=criterion,
    )

    print("\nFinal metrics:")
    print(f"  fit_train: {fit_train_metrics}")
    print(f"  fit_val:   {fit_val_metrics}")
    if eval_metrics is not None:
        print(f"  eval:      {eval_metrics}")
    else:
        print("  eval:      skipped")

    per_day_eval_metrics = {}
    if y_true_eval is not None and y_pred_eval is not None:
        for day_name in sorted(set(day_meta["eval"].tolist())):
            mask = day_meta["eval"] == day_name
            per_day_eval_metrics[day_name] = regression_metrics(y_true_eval[mask], y_pred_eval[mask])

    reference_fa = average_aligned_fa_models(
        [days[day_name].fa_aligned for day_name in config.train_days],
        psi_floor=config.psi_floor,
    )
    reference_update_count = len(config.train_days)

    if y_true_eval is not None and y_pred_eval is not None:
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
            save_prediction_plot(
                output_dir / f"eval_predictions_{day_name}.png",
                y_true_eval[mask],
                y_pred_eval[mask],
                title=f"Eval day {day_name}",
            )

    checkpoint = {
        "config": json_ready(asdict(config)),
        "data_source_identifier": str(config.data_source or Path(config.data_dir).name),
        "data_source_dir": str(config.data_dir.resolve()),
        "sequence_length": seq_len,
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "decoder_variant": config.decoder_variant,
            "input_dim": config.n_latents,
            "seq_len": seq_len,
            "time_dim": config.time_dim,
            "latent_embed_dim": config.latent_embed_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "output_dim": 2,
            "use_bias": False,
        },
        "x_mean": x_scaler.mean_,
        "x_std": x_scaler.std_,
        "y_mean": y_scaler.mean_,
        "y_std": y_scaler.std_,
        "unit_cols": base_day.unit_cols,
        "target_columns": base_day.target_columns,
        "base_day": config.base_day,
        "train_days": config.train_days,
        "eval_days": config.eval_days,
        "base_fa": {
            "mean": base_day.fa_aligned.mean,
            "C": base_day.fa_aligned.C,
            "psi": base_day.fa_aligned.psi,
        },
        "reference_fa": {
            "mean": reference_fa.mean,
            "C": reference_fa.C,
            "psi": reference_fa.psi,
            "score": reference_fa.score,
        },
        "reference_source_days": config.train_days,
        "reference_update_count": reference_update_count,
        "reference_update_mode": "running_average",
        "day_alignment": {
            day_name: {
                "stable_units": days[day_name].alignment.stable_units,
                "stable_mask": days[day_name].alignment.stable_mask,
                "transform": days[day_name].alignment.transform,
                "loading_rmse_before": days[day_name].alignment.loading_rmse_before,
                "loading_rmse_after": days[day_name].alignment.loading_rmse_after,
            }
            for day_name in config.selected_days
        },
    }
    checkpoint_path = output_dir / "manifold_lstm_decoder.pt"
    torch.save(checkpoint, checkpoint_path)

    summary = {
        "config": json_ready(asdict(config)),
        "data_source_identifier": str(config.data_source or Path(config.data_dir).name),
        "data_source_dir": str(config.data_dir.resolve()),
        "sequence_length": seq_len,
        "fit_train_metrics": fit_train_metrics,
        "fit_val_metrics": fit_val_metrics,
        "eval_metrics": eval_metrics,
        "per_day_eval_metrics": per_day_eval_metrics,
        "reference_source_days": config.train_days,
        "reference_update_count": reference_update_count,
        "history": history,
        "alignment": {
            day_name: {
                "stable_unit_count": int(days[day_name].alignment.stable_mask.sum()),
                "loading_rmse_before": days[day_name].alignment.loading_rmse_before,
                "loading_rmse_after": days[day_name].alignment.loading_rmse_after,
            }
            for day_name in config.selected_days
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
