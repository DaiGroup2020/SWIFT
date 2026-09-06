from __future__ import annotations

"""Direct FPGA14 local-peak assignment without an outer residual class.

The reference CSV contains the already-resolved no-boundary intervals.  It is
read directly during unit sorting: an event is assigned to its final outer
unit immediately, rather than first being emitted as RESIDUAL and merged in a
post-processing step.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .artifacts import file_fingerprint


UV_PER_ADC = 0.195


def _round_away_from_zero(value: float) -> int:
    return int(math.floor(value + 0.5)) if value >= 0.0 else int(math.ceil(value - 0.5))


def _parse_bound_uv(value: object) -> float:
    text = str(value).strip().lower()
    if text in {"inf", "+inf", "infinity", "+infinity"}:
        return math.inf
    if text in {"-inf", "-infinity"}:
        return -math.inf
    return float(_round_away_from_zero(float(text) / UV_PER_ADC))


def _format_bound(value: float) -> str | int:
    return "inf" if value == math.inf else "-inf" if value == -math.inf else int(value)


@dataclass(frozen=True)
class LocalPeakUnitInterval:
    channel: str
    polarity: str
    unit_id: str
    unit_index: int
    lower_adc: float
    upper_adc: float


class LocalPeakNoBoundaryMap:
    """Validated direct-assignment view of an FPGA14 no-boundary unit map."""

    def __init__(self, *, source: Path, groups: dict[tuple[str, str], tuple[LocalPeakUnitInterval, ...]]) -> None:
        self.source = source.resolve()
        self.fingerprint = file_fingerprint(self.source)
        self.groups = groups

    @classmethod
    def from_csv(cls, path: Path) -> "LocalPeakNoBoundaryMap":
        path = path.resolve()
        required = {"unit_id", "channel", "polarity", "unit_index", "is_residual", "lower_uv", "upper_uv"}
        raw: dict[tuple[str, str], list[LocalPeakUnitInterval]] = {}
        seen_unit_ids: set[str] = set()
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"No-boundary unit map is missing columns: {sorted(missing)}")
            for row in reader:
                channel = str(row["channel"]).strip()
                polarity = str(row["polarity"]).strip().upper()
                unit_id = str(row["unit_id"]).strip()
                if not channel or polarity not in {"POS", "NEG"} or not unit_id:
                    raise ValueError(f"Invalid no-boundary map row: {row!r}")
                if str(row["is_residual"]).strip().lower() in {"1", "true", "yes", "y"}:
                    raise ValueError(f"No-boundary map must not contain a residual unit: {unit_id}")
                if unit_id in seen_unit_ids:
                    raise ValueError(f"Duplicate unit_id in no-boundary map: {unit_id}")
                seen_unit_ids.add(unit_id)
                interval = LocalPeakUnitInterval(
                    channel=channel,
                    polarity=polarity,
                    unit_id=unit_id,
                    unit_index=int(row["unit_index"]),
                    lower_adc=_parse_bound_uv(row["lower_uv"]),
                    upper_adc=_parse_bound_uv(row["upper_uv"]),
                )
                if interval.lower_adc >= interval.upper_adc:
                    raise ValueError(f"Invalid local-peak interval for {unit_id}")
                raw.setdefault((channel, polarity), []).append(interval)

        groups: dict[tuple[str, str], tuple[LocalPeakUnitInterval, ...]] = {}
        for key, intervals in raw.items():
            ordered = tuple(sorted(intervals, key=lambda item: item.unit_index))
            indexes = [item.unit_index for item in ordered]
            if indexes != list(range(1, len(ordered) + 1)):
                raise ValueError(f"Unit indexes must be consecutive from 1 for {key}: {indexes}")
            if key[1] == "POS":
                if ordered[-1].upper_adc != math.inf:
                    raise ValueError(f"POS no-boundary map must extend to +inf for {key}")
                for left, right in zip(ordered, ordered[1:]):
                    if left.upper_adc != right.lower_adc:
                        raise ValueError(f"Non-contiguous POS intervals for {key}")
            else:
                if ordered[-1].lower_adc != -math.inf:
                    raise ValueError(f"NEG no-boundary map must extend to -inf for {key}")
                for left, right in zip(ordered, ordered[1:]):
                    if left.lower_adc != right.upper_adc:
                        raise ValueError(f"Non-contiguous NEG intervals for {key}")
            groups[key] = ordered
        if not groups:
            raise ValueError(f"No usable unit intervals in {path}")
        return cls(source=path, groups=groups)

    @property
    def catalog(self) -> tuple[LocalPeakUnitInterval, ...]:
        return tuple(interval for key in sorted(self.groups) for interval in self.groups[key])

    def assign(self, *, channel: str, polarity: str, peak_value_adc: int) -> LocalPeakUnitInterval:
        key = (str(channel), str(polarity).upper())
        intervals = self.groups.get(key)
        if intervals is None:
            raise RuntimeError(f"No no-boundary local-peak mapping for {key[0]}/{key[1]}")
        value = int(peak_value_adc)
        if key[1] == "POS":
            for interval in intervals:
                if value >= interval.lower_adc and value < interval.upper_adc:
                    return interval
        else:
            for interval in intervals:
                if value > interval.lower_adc and value <= interval.upper_adc:
                    return interval
        bounds = ", ".join(f"[{_format_bound(item.lower_adc)}, {_format_bound(item.upper_adc)}]" for item in intervals)
        raise RuntimeError(f"Peak {value} ADC is outside the no-boundary local-peak map for {key[0]}/{key[1]}: {bounds}")
