from __future__ import annotations

from pathlib import Path

import numpy as np


RHD_BLOCK_SIZE = 128
RHX_ADC_ZERO = 32768
RHX_UV_PER_LSB = 0.195


def open_rhd(path: Path, *, stream_id: str = "0", stream_name: str | None = None, ignore_integrity_checks: bool = False):
    try:
        import spikeinterface.extractors as se
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("RAPID needs spikeinterface to read Intan .rhd recordings.") from exc
    return se.read_intan(
        str(path),
        stream_id=stream_id,
        stream_name=stream_name,
        use_names_as_ids=True,
        ignore_integrity_checks=ignore_integrity_checks,
    )


def raw_block_memmap(recording) -> np.memmap:
    """Return the Intan block data required for count-exact FPGA replay."""
    neo_reader = getattr(recording, "neo_reader", None)
    raw_data = getattr(neo_reader, "_raw_data", None)
    if raw_data is None or raw_data.dtype.fields is None or "timestamp" not in raw_data.dtype.fields:
        raise RuntimeError(
            "RAPID needs the SpikeInterface Intan RawIO block memmap. "
            "Use spikeinterface.extractors.read_intan on a genuine .rhd file."
        )
    block_size = int(raw_data.dtype["timestamp"].shape[0])
    if block_size != RHD_BLOCK_SIZE:
        raise RuntimeError(f"RAPID expects {RHD_BLOCK_SIZE}-sample Intan blocks, found {block_size}.")
    return raw_data


def available_channels(recording) -> list[str]:
    return [str(item) for item in recording.get_channel_ids()]


def raw_counts(blocks: np.ndarray, channels: list[str]) -> np.ndarray:
    result = np.empty((blocks.shape[0] * RHD_BLOCK_SIZE, len(channels)), dtype=np.int32)
    for index, channel in enumerate(channels):
        result[:, index] = np.asarray(blocks[channel], dtype=np.int32).reshape(-1)
    return result


def timestamps(blocks: np.ndarray) -> np.ndarray:
    return np.asarray(blocks["timestamp"], dtype=np.int32).reshape(-1)


def discover_rhds(root: Path) -> list[Path]:
    return sorted(path.resolve() for path in root.rglob("*.rhd"))
