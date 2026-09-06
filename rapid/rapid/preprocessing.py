from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from .rhd_io import RHX_ADC_ZERO, RHX_UV_PER_LSB


@dataclass(frozen=True)
class FilterConfig:
    """The deployed RAPID channel path: mean CAR -> two-sample MAF -> fixed IIR."""

    sample_rate_hz: float = 20_000.0
    maf_samples: int = 2
    threshold_default_uv: float = 50.0
    threshold_mad_multiplier: float = -5.0
    local_peak_search_samples: int = 30


def round_away_from_zero(value: float) -> int:
    """Match the deployed FPGA14 threshold quantization rule exactly."""
    return int(math.floor(value + 0.5)) if value >= 0.0 else int(math.ceil(value - 0.5))


def load_channel_param(path: Path, *, default_threshold_uv: float, mad_multiplier: float) -> tuple[dict[str, int], set[str]]:
    """Read enabled channels and their fixed thresholds in signed ADC counts."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"ChannelParam must be a JSON object: {path}")
    # The established FPGA14 path first quantizes in microvolts, then converts
    # that integer microvolt value to ADC counts.  Collapsing this into one
    # Python round() changes many per-channel thresholds by 1--3 ADC counts.
    defaults = round_away_from_zero(float(default_threshold_uv) / RHX_UV_PER_LSB)
    thresholds: dict[str, int] = {}
    enabled: set[str] = set()
    for channel, value in payload.items():
        item = value if isinstance(value, dict) else {}
        name = str(channel)
        if bool(item.get("Enable", False)):
            enabled.add(name)
        mad = item.get("MadValue")
        # A negative multiplier in the original ChannelParam workflow produces
        # a signed threshold. RAPID detects both polarities, so only abs() is
        # used downstream; signed storage preserves provenance here.
        if mad is not None:
            threshold_uv = round_away_from_zero(float(mad) * float(mad_multiplier))
            thresholds[name] = round_away_from_zero(float(threshold_uv) / RHX_UV_PER_LSB)
        else:
            thresholds[name] = -abs(defaults)
    return thresholds, enabled


@njit
def _wrap_signed(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    value = value & mask
    if value >= sign:
        value -= 1 << bits
    return value


@njit
def filter_car_maf_iir_chunk(
    raw_counts: np.ndarray,
    hp_prev_in1: np.ndarray,
    hp_prev_in2: np.ndarray,
    hp_prev_out1: np.ndarray,
    hp_prev_out2: np.ndarray,
    moving_history: np.ndarray,
    moving_sums: np.ndarray,
    moving_positions: np.ndarray,
    samples_seen: int,
) -> np.ndarray:
    """Count-exact causal FPGA arithmetic with a two-sample MAF.

    The arithmetic and state order mirror the current M06 FPGA14 peak path.
    """
    n_samples, n_channels = raw_counts.shape
    output = np.empty((n_samples, n_channels), dtype=np.int16)
    for sample in range(n_samples):
        total = 0
        for channel in range(n_channels):
            total += int(raw_counts[sample, channel])
        reference = total // n_channels
        for channel in range(n_channels):
            car = _wrap_signed(int(raw_counts[sample, channel]) - reference, 16)
            position = int(moving_positions[channel])
            running_sum = int(moving_sums[channel]) - int(moving_history[channel, position])
            moving_history[channel, position] = car
            running_sum += car
            moving_sums[channel] = running_sum
            moving_positions[channel] = 0 if position == 1 else 1
            # The two-sample MAF has one genuine warm-up sample: the first
            # sample of the *recording*. `sample` is local to an I/O chunk,
            # so using `sample == 0` here would inject an artificial zero at
            # every chunk boundary and then excite the recursive IIR stage.
            maf = 0 if samples_seen + sample == 0 else running_sum >> 1

            x0 = _wrap_signed(int(maf) << 4, 20)
            x1 = _wrap_signed(int(hp_prev_in1[channel]), 20)
            x2 = _wrap_signed(int(hp_prev_in2[channel]), 20)
            y1 = _wrap_signed(int(hp_prev_out1[channel]), 20)
            y2 = _wrap_signed(int(hp_prev_out2[channel]), 20)
            xdiff2 = _wrap_signed(x0 - (x1 << 1) + x2, 20)
            xout = _wrap_signed(xdiff2 - (xdiff2 >> 4), 20)
            ydiff = _wrap_signed(y2 - y1, 20)
            yout = _wrap_signed((y1 << 1) - y2 + (ydiff >> 3) - (y1 >> 7), 20)
            y0 = _wrap_signed(xout + yout, 20)
            hp_prev_in2[channel] = x1
            hp_prev_in1[channel] = x0
            hp_prev_out2[channel] = y1
            hp_prev_out1[channel] = y0
            output[sample, channel] = np.int16(_wrap_signed(y0 >> 4, 16))
    return output


class FilterState:
    def __init__(self, channels: int, config: FilterConfig) -> None:
        if float(config.sample_rate_hz) != 20_000.0:
            raise ValueError("RAPID's fixed FPGA filter is defined at 20 kHz.")
        if int(config.maf_samples) != 2:
            raise ValueError("RAPID's deployed FPGA filter uses a fixed two-sample MAF.")
        self.hp_prev_in1 = np.zeros(channels, dtype=np.int32)
        self.hp_prev_in2 = np.zeros(channels, dtype=np.int32)
        self.hp_prev_out1 = np.zeros(channels, dtype=np.int32)
        self.hp_prev_out2 = np.zeros(channels, dtype=np.int32)
        self.moving_history = np.zeros((channels, 2), dtype=np.int32)
        self.moving_sums = np.zeros(channels, dtype=np.int64)
        self.moving_positions = np.zeros(channels, dtype=np.int32)
        self.samples_seen = 0

    def process(self, raw: np.ndarray) -> np.ndarray:
        output = filter_car_maf_iir_chunk(
            raw,
            self.hp_prev_in1,
            self.hp_prev_in2,
            self.hp_prev_out1,
            self.hp_prev_out2,
            self.moving_history,
            self.moving_sums,
            self.moving_positions,
            self.samples_seen,
        )
        self.samples_seen += int(raw.shape[0])
        return output


def adc_to_uv(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * np.float32(RHX_UV_PER_LSB)


def centered_counts(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.int32) - RHX_ADC_ZERO
