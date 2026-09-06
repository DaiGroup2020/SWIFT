from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path
from uuid import uuid4

from .artifacts import archive_existing, file_fingerprint, utc_now, write_json


def _is_timestamp_block(buffer: bytes, offset: int) -> bool:
    first, second, third = struct.unpack_from("<iii", buffer, offset)
    if second != first + 1 or third != second + 1:
        return False
    values = struct.unpack_from("<128i", buffer, offset)
    return all(value == first + index for index, value in enumerate(values))


def inspect_rhd_blocks(source: Path, scan_bytes: int = 4 * 1024 * 1024) -> tuple[int, int]:
    """Infer the variable RHD header and fixed data-block size without parsing all data."""
    with source.open("rb") as handle:
        buffer = handle.read(scan_bytes)
    hits: list[int] = []
    for phase in range(4):
        for offset in range(phase, len(buffer) - 512, 4):
            if _is_timestamp_block(buffer, offset):
                hits.append(offset)
    hits.sort()
    if len(hits) < 3:
        raise RuntimeError("Could not find enough Intan timestamp blocks in the source RHD header region.")
    deltas = [right - left for left, right in zip(hits, hits[1:]) if right - left > 512]
    if not deltas:
        raise RuntimeError("Could not infer the RHD data-block byte size.")
    block_bytes, occurrences = Counter(deltas).most_common(1)[0]
    if occurrences < 2:
        raise RuntimeError("RHD data-block spacing was not stable; refusing to create a clip.")
    starts = [offset for offset in hits if offset + 2 * block_bytes < len(buffer) and _is_timestamp_block(buffer, offset + block_bytes) and _is_timestamp_block(buffer, offset + 2 * block_bytes)]
    if not starts:
        raise RuntimeError("Could not confirm three consecutive RHD data blocks.")
    return starts[0], int(block_bytes)


def clip_rhd(*, source: Path, destination: Path, duration_seconds: float, sample_rate_hz: float = 20_000.0, overwrite: bool = False) -> dict[str, int | float | str]:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination or (destination.exists() and source.samefile(destination)):
        raise ValueError("RHD clip destination must differ from its source.")
    if float(sample_rate_hz) <= 0:
        raise ValueError("Sample rate must be positive.")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing RHD clip: {destination}")
    metadata_path = destination.with_suffix(destination.suffix + ".manifest.json")
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing RHD clip metadata: {metadata_path}")
    header_bytes, block_bytes = inspect_rhd_blocks(source)
    block_count = int(float(duration_seconds) * float(sample_rate_hz) // 128)
    if block_count < 1:
        raise ValueError("Duration is shorter than one 128-sample RHD block.")
    payload_bytes = block_count * block_bytes
    if header_bytes + payload_bytes > source.stat().st_size:
        raise ValueError("Requested duration exceeds the available RHD recording.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{uuid4().hex}.tmp")
    with source.open("rb") as reader, temporary.open("wb") as writer:
        remaining = header_bytes + payload_bytes
        while remaining:
            chunk = reader.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("Unexpected end of source while clipping RHD.")
            writer.write(chunk)
            remaining -= len(chunk)
    for previous in (destination, metadata_path):
        if previous.exists():
            archive_existing(previous)
    temporary.replace(destination)
    metadata: dict[str, int | float | str] = {
        "system": "RAPID",
        "artifact_type": "rhd_clip",
        "created_at": utc_now(),
        "source_rhd": source.name,
        "source_fingerprint": file_fingerprint(source),
        "destination_rhd": destination.name,
        "header_bytes": header_bytes,
        "data_block_bytes": block_bytes,
        "block_count": block_count,
        "sample_rate_hz": float(sample_rate_hz),
        "duration_seconds": block_count * 128 / float(sample_rate_hz),
        "output_bytes": destination.stat().st_size,
        "destination_fingerprint": file_fingerprint(destination),
    }
    write_json(metadata_path, metadata)
    return metadata
