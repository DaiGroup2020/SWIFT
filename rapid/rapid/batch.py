from __future__ import annotations

from pathlib import Path

from .artifacts import ensure_output_separate
from .channel_sorting import _session_id, run_channel_sort
from .preprocessing import FilterConfig
from .rhd_io import discover_rhds
from .unit_sorting import run_unit_sort


def run_all_days(
    *, input_root: Path, channel_param: Path, reference_model: Path, results_root: Path, config: FilterConfig,
    stream_id: str, stream_name: str | None, ignore_integrity_checks: bool, chunk_blocks: int, write_band_dat: bool, overwrite: bool,
) -> list[tuple[Path, Path]]:
    rhds = discover_rhds(input_root.resolve())
    if not rhds:
        raise RuntimeError(f"No .rhd files found below {input_root}")
    sessions: dict[str, Path] = {}
    for rhd in rhds:
        session = _session_id(rhd).casefold()
        if session in sessions:
            raise ValueError(f"RHD names map to the same result directory: {sessions[session]} and {rhd}. Rename one input first.")
        sessions[session] = rhd
        for kind in ("channel", "unit"):
            ensure_output_separate(results_root / kind / _session_id(rhd), tuple(rhds) + (channel_param, reference_model))
    completed: list[tuple[Path, Path]] = []
    for index, rhd in enumerate(rhds, start=1):
        print(f"[{index}/{len(rhds)}] RAPID channel: {rhd}", flush=True)
        channel = run_channel_sort(
            rhd=rhd, channel_param=channel_param, results_root=results_root, config=config, stream_id=stream_id, stream_name=stream_name,
            ignore_integrity_checks=ignore_integrity_checks, chunk_blocks=chunk_blocks, write_band_dat=write_band_dat, overwrite=overwrite,
        )
        print(f"[{index}/{len(rhds)}] RAPID unit: {rhd}", flush=True)
        unit = run_unit_sort(channel_result=channel, reference_model=reference_model, results_root=results_root, overwrite=overwrite)
        completed.append((channel, unit))
    return completed
