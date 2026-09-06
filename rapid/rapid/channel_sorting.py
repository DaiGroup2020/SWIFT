from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import file_fingerprint, prepare_result_directory, result_manifest, write_json
from .preprocessing import FilterConfig, FilterState, load_channel_param, round_away_from_zero
from .rhd_io import RHD_BLOCK_SIZE, available_channels, open_rhd, raw_block_memmap, raw_counts, timestamps


class FPGAPeakSelector:
    """Algorithm-4 peak/valley detector plus the deployed 30-sample selector."""

    def __init__(self, channels: list[str], thresholds_adc: np.ndarray, window_samples: int, jitter_counts: int = 16) -> None:
        self.channels = channels
        self.thresholds = np.abs(np.asarray(thresholds_adc, dtype=np.int64))
        self.window_samples = int(window_samples)
        self.jitter = int(jitter_counts)
        count = len(channels)
        self.direction = np.ones(count, dtype=bool)
        self.direction_delay = np.ones(count, dtype=bool)
        self.data = np.zeros((3, count), dtype=np.int64)
        self.width = np.zeros(count, dtype=np.int64)
        self.window_events: dict[tuple[int, int], list[dict[str, int | str]]] = {}
        self.previous_x = np.zeros(count, dtype=np.int32)
        self.finalized_window = -1
        self.ticks: dict[int, int] = {}
        self.accepted: list[dict[str, int | str]] = []

    @staticmethod
    def _wrap(value: np.ndarray, bits: int) -> np.ndarray:
        values = np.asarray(value, dtype=np.int64)
        mask = (1 << bits) - 1
        sign = 1 << (bits - 1)
        output = values & mask
        return np.where(output >= sign, output - (1 << bits), output)

    def _step(self, value: np.ndarray, sample: int) -> list[dict[str, int]]:
        old_direction = self.direction.copy()
        old_delay = self.direction_delay.copy()
        data0, data1, data2 = (self._wrap(self.data[index], 16) for index in range(3))
        difference = self._wrap(data0 - data1, 17)
        rise = difference >= self.jitter
        fall = difference <= -self.jitter
        peak = old_delay & (~old_direction)
        valley = (~old_delay) & old_direction
        valid = (peak & (data2 >= self.thresholds)) | (valley & (data2 <= -self.thresholds))
        events = [
            {"column": int(column), "peak_sample": int(sample - 3), "peak_value_adc": int(data2[column]), "polarity": "POS" if bool(old_delay[column]) else "NEG"}
            for column in np.flatnonzero(valid)
        ]
        self.direction = np.where(rise, True, np.where(fall, False, old_direction))
        self.direction_delay = old_direction
        self.data[2] = self.data[1]
        self.data[1] = self.data[0]
        self.data[0] = self._wrap(value, 16)
        return events

    def _finalize(self, last_ready_window: int) -> None:
        for window in range(self.finalized_window + 1, last_ready_window + 1):
            for column in range(len(self.channels)):
                candidates = self.window_events.pop((window, column), [])
                if not candidates:
                    self.previous_x[column] = 0
                    continue
                previous = int(self.previous_x[column])
                if previous > 0:
                    candidates = [item for item in candidates if int(item["peak_sample"]) % self.window_samples > previous]
                if not candidates:
                    self.previous_x[column] = 0
                    continue
                selected = min(candidates, key=lambda item: (-abs(int(item["peak_value_adc"])), int(item["peak_sample"])))
                self.accepted.append(selected)
                self.previous_x[column] = int(selected["peak_sample"]) % self.window_samples
        self.finalized_window = max(self.finalized_window, last_ready_window)

    def process(self, values: np.ndarray, sample_ticks: np.ndarray, *, sample_start: int) -> None:
        for local_index in range(values.shape[0]):
            sample = int(sample_start + local_index)
            self.ticks[sample] = int(sample_ticks[local_index])
            for stale in [key for key in self.ticks if key < sample - self.window_samples - 8]:
                self.ticks.pop(stale, None)
            for event in self._step(values[local_index].astype(np.int64), sample):
                peak_sample = int(event["peak_sample"])
                if peak_sample < 0 or peak_sample not in self.ticks:
                    continue
                column = int(event.pop("column"))
                event.update(
                    {
                        "channel": self.channels[column],
                        "peak_tick": int(self.ticks[peak_sample]),
                        "threshold_tick": int(self.ticks.get(sample, sample_ticks[local_index])),
                        "threshold_sample": sample,
                        "threshold_value_adc": int(event["peak_value_adc"]),
                    }
                )
                self.window_events.setdefault((peak_sample // self.window_samples, column), []).append(event)
            # Algorithm-4 emits a candidate three samples after its peak.  A
            # window must therefore remain open until that final possible
            # candidate has been registered.  This is the deployed FPGA14
            # scheduler: ((sample - latency + 1) // window) - 1.
            self._finalize(((sample - 2) // self.window_samples) - 1)

    def finish(self) -> list[dict[str, int | str]]:
        if self.window_events:
            self._finalize(max(key[0] for key in self.window_events))
        return self.accepted


class LocalPeakDetector:
    """Threshold crossing followed by a maximum-absolute local-peak selection.

    A channel remains independent after the shared preprocessing stage. The
    tail buffer makes detections continuous across RHD read chunks.
    """

    def __init__(self, channels: list[str], thresholds_adc: np.ndarray, search_samples: int) -> None:
        self.channels = channels
        self.thresholds_adc = np.abs(np.asarray(thresholds_adc, dtype=np.int32))
        self.search_samples = int(search_samples)
        self.tail_signal = np.empty((0, len(channels)), dtype=np.int16)
        self.tail_ticks = np.empty(0, dtype=np.int32)
        self.tail_samples = np.empty(0, dtype=np.int64)
        self.next_index = np.zeros(len(channels), dtype=np.int64)

    def process(self, signal: np.ndarray, ticks: np.ndarray, *, sample_start: int = 0, final: bool = False) -> list[dict[str, int | str]]:
        values = np.vstack([self.tail_signal, signal]) if self.tail_signal.size else signal
        all_ticks = np.concatenate([self.tail_ticks, ticks]) if self.tail_ticks.size else ticks
        current_samples = np.arange(int(sample_start), int(sample_start) + signal.shape[0], dtype=np.int64)
        all_samples = np.concatenate([self.tail_samples, current_samples]) if self.tail_samples.size else current_samples
        safe_stop = values.shape[0] if final else max(0, values.shape[0] - self.search_samples)
        rows: list[dict[str, int | str]] = []
        for column, channel in enumerate(self.channels):
            index = int(self.next_index[column])
            threshold = int(self.thresholds_adc[column])
            while index < safe_stop:
                value = int(values[index, column])
                if abs(value) < threshold:
                    index += 1
                    continue
                stop = min(values.shape[0], index + self.search_samples + 1)
                local = values[index:stop, column].astype(np.int32)
                peak_relative = int(np.argmax(np.abs(local)))
                peak_index = index + peak_relative
                peak_value = int(values[peak_index, column])
                rows.append(
                    {
                        "channel": channel,
                        "polarity": "POS" if peak_value >= 0 else "NEG",
                        "threshold_tick": int(all_ticks[index]),
                        "peak_tick": int(all_ticks[peak_index]),
                        "threshold_sample": int(all_samples[index]),
                        "peak_sample": int(all_samples[peak_index]),
                        "threshold_value_adc": value,
                        "peak_value_adc": peak_value,
                    }
                )
                # The configured local-peak interval is also the refractory
                # interval. This is the current channel-base selection rule.
                index = peak_index + self.search_samples
            self.next_index[column] = index
        if final:
            self.tail_signal = np.empty((0, len(self.channels)), dtype=np.int16)
            self.tail_ticks = np.empty(0, dtype=np.int32)
            self.tail_samples = np.empty(0, dtype=np.int64)
            self.next_index.fill(0)
        else:
            keep = min(self.search_samples, values.shape[0])
            self.tail_signal = values[-keep:].copy()
            self.tail_ticks = all_ticks[-keep:].copy()
            self.tail_samples = all_samples[-keep:].copy()
            self.next_index = np.maximum(self.next_index - (values.shape[0] - keep), 0)
        return rows


def _session_id(rhd: Path) -> str:
    return rhd.stem.replace(" ", "_")


def _write_channel_rows(output: Path, rows: list[dict[str, int | str]], sample_rate: float, channels: list[str]) -> None:
    fields = ["channel", "polarity", "threshold_tick", "threshold_sample", "threshold_seconds", "peak_tick", "peak_sample", "peak_seconds", "threshold_value_adc", "peak_value_adc"]
    rows.sort(key=lambda row: (int(row["peak_tick"]), str(row["channel"])))
    with (output / "channel_spikes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            payload = dict(row)
            payload["threshold_seconds"] = format(int(row["threshold_tick"]) / sample_rate, ".17g")
            payload["peak_seconds"] = format(int(row["peak_tick"]) / sample_rate, ".17g")
            writer.writerow(payload)
    for directory in ("unitcsv", "unitcsv_peak_aligned", "unitcsv_peak_aligned_seconds"):
        (output / directory).mkdir(exist_ok=True)
    by_channel: dict[str, list[dict[str, int | str]]] = {channel: [] for channel in channels}
    for row in rows:
        by_channel[str(row["channel"])].append(row)
    for channel, channel_rows in by_channel.items():
        with (output / "unitcsv" / f"{channel}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([[row["threshold_tick"]] for row in channel_rows])
        with (output / "unitcsv_peak_aligned" / f"{channel}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([[row["peak_tick"]] for row in channel_rows])
        with (output / "unitcsv_peak_aligned_seconds" / f"{channel}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([[format(int(row["peak_tick"]) / sample_rate, ".17g")] for row in channel_rows])


def run_channel_sort(
    *, rhd: Path, channel_param: Path, results_root: Path, config: FilterConfig, stream_id: str = "0", stream_name: str | None = None,
    ignore_integrity_checks: bool = False, chunk_blocks: int = 512, write_band_dat: bool = True, overwrite: bool = False,
) -> Path:
    rhd = rhd.resolve()
    channel_param = channel_param.resolve()
    if int(chunk_blocks) < 1 or int(config.local_peak_search_samples) < 1:
        raise ValueError("Chunk blocks and local-peak samples must both be positive.")
    output = results_root.resolve() / "channel" / _session_id(rhd)
    recording = open_rhd(rhd, stream_id=stream_id, stream_name=stream_name, ignore_integrity_checks=ignore_integrity_checks)
    sample_rate = float(recording.get_sampling_frequency())
    if sample_rate != float(config.sample_rate_hz):
        raise ValueError(f"RAPID fixed filter requires {config.sample_rate_hz:g} Hz; RHD reports {sample_rate:g} Hz.")
    available = available_channels(recording)
    thresholds, enabled = load_channel_param(
        channel_param, default_threshold_uv=config.threshold_default_uv, mad_multiplier=config.threshold_mad_multiplier
    )
    channels = [channel for channel in available if channel in enabled]
    if not channels:
        raise RuntimeError("No enabled ChannelParam channels are present in this RHD recording.")
    raw_blocks = raw_block_memmap(recording)
    state = FilterState(len(channels), config)
    detector = FPGAPeakSelector(
        channels,
        np.asarray(
            [
                thresholds.get(channel, -round_away_from_zero(config.threshold_default_uv / 0.195))
                for channel in channels
            ]
        ),
        config.local_peak_search_samples,
    )
    prepare_result_directory(output, overwrite=overwrite, protected_paths=(rhd, channel_param))
    band_handles: dict[str, Any] = {}
    time_handle = None
    if write_band_dat:
        for channel in channels:
            band_handles[channel] = (output / f"band-{channel}.DAT").open("wb")
        time_handle = (output / "time.DAT").open("wb")
    rows: list[dict[str, int | str]] = []
    try:
        sample_start = 0
        for first in range(0, raw_blocks.shape[0], max(1, int(chunk_blocks))):
            block_slice = raw_blocks[first : first + max(1, int(chunk_blocks))]
            counts = raw_counts(block_slice, channels)
            band = state.process(counts)
            ticks = timestamps(block_slice)
            if time_handle is not None:
                ticks.astype("<i4", copy=False).tofile(time_handle)
                for column, channel in enumerate(channels):
                    band[:, column].astype("<i2", copy=False).tofile(band_handles[channel])
            detector.process(band, ticks, sample_start=sample_start)
            sample_start += band.shape[0]
        rows.extend(detector.finish())
    finally:
        if time_handle is not None:
            time_handle.close()
        for handle in band_handles.values():
            handle.close()
    _write_channel_rows(output, rows, sample_rate, channels)
    # Keep the exact threshold input beside each result, so an individual day
    # can be re-audited without consulting another directory.
    (output / "ChannelParam.json").write_bytes(channel_param.read_bytes())
    manifest = result_manifest(
        kind="channel", source_rhd=rhd, output_dir=output, sample_rate_hz=sample_rate,
        parameters={
            "filter": asdict(config),
            "stream_id": stream_id,
            "stream_name": stream_name,
            "chunk_blocks": chunk_blocks,
            "write_band_dat": write_band_dat,
            "channel_param": channel_param.name,
            "channel_param_fingerprint": file_fingerprint(channel_param),
        },
    )
    manifest.update({"enabled_channel_count": len(channels), "detected_event_count": len(rows), "trace_scale_uv_per_lsb": 0.195})
    write_json(output / "manifest.json", manifest)
    return output
