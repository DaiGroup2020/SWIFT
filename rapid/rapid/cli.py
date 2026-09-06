#!/usr/bin/env python
"""Single entry point for the standalone RAPID pipeline."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

ROOT = Path.cwd()

def _filter_config(args: argparse.Namespace):
    from .preprocessing import FilterConfig

    return FilterConfig(
        sample_rate_hz=float(args.sample_rate_hz),
        maf_samples=int(args.maf_samples),
        threshold_default_uv=float(args.default_threshold_uv),
        threshold_mad_multiplier=float(args.threshold_mad_multiplier),
        local_peak_search_samples=int(args.local_peak_samples),
    )


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--results-root", type=Path, default=ROOT / "results", help="Contains only the channel and unit result trees.")
    parser.add_argument("--sample-rate-hz", type=float, default=20_000.0)
    parser.add_argument("--maf-samples", type=int, default=2)
    parser.add_argument("--default-threshold-uv", type=float, default=50.0)
    parser.add_argument("--threshold-mad-multiplier", type=float, default=-5.0, help="Multiply ChannelParam MadValue; use -1 to use a stored signed threshold directly.")
    parser.add_argument("--local-peak-samples", type=int, default=30)
    parser.add_argument("--stream-id", default="0")
    parser.add_argument("--stream-name", default=None)
    parser.add_argument("--ignore-integrity-checks", action="store_true")
    parser.add_argument("--chunk-blocks", type=int, default=512)
    parser.add_argument("--no-band-dat", action="store_true", help="Do not save filtered traces. Not suitable for a reference-PCA run.")
    parser.add_argument("--overwrite", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rapid", description="RAPID standalone channel and unit spike sorting")
    commands = parser.add_subparsers(dest="command", required=True)
    channel = commands.add_parser("channel", help="Run CAR+MAF+fixed-IIR channel sorting for one RHD.")
    channel.add_argument("--rhd", type=Path, required=True)
    channel.add_argument("--channel-param", type=Path, required=True)
    _common(channel)
    reference = commands.add_parser("reference-pca", help="Generate a no-outer-boundary PCA unit model from a baseline channel result.")
    reference.add_argument("--channel-result", type=Path, required=True)
    reference.add_argument("--model", type=Path, default=ROOT / "models" / "reference_pca_unit_model.json")
    reference.add_argument("--groups", type=int, default=2, help="Local-peak groups per channel/polarity.")
    reference.add_argument("--pre-samples", type=int, default=15)
    reference.add_argument("--post-samples", type=int, default=30)
    reference.add_argument("--max-events-per-group", type=int, default=10_000)
    reference.add_argument("--overwrite", action="store_true", help="Archive the exact existing model artifacts before replacing them.")
    unit = commands.add_parser("unit", help="Assign one channel result to the baseline PCA model; no outer boundary is applied.")
    unit.add_argument("--channel-result", type=Path, required=True)
    unit.add_argument(
        "--model", type=Path, required=True,
        help="FPGA14 local-peak no-boundary CSV (recommended) or legacy PCA JSON model.",
    )
    unit.add_argument("--results-root", type=Path, default=ROOT / "results")
    unit.add_argument("--overwrite", action="store_true")
    batch = commands.add_parser("all-days", help="Run channel then no-boundary unit sorting for every RHD below an input root.")
    batch.add_argument("--input-root", type=Path, required=True)
    batch.add_argument("--channel-param", type=Path, required=True)
    batch.add_argument(
        "--model", type=Path, required=True,
        help="FPGA14 local-peak no-boundary CSV (recommended) or legacy PCA JSON model.",
    )
    _common(batch)
    clip = commands.add_parser("clip-rhd", help="Create a valid short RHD by copying its header and complete Intan blocks.")
    clip.add_argument("--source", type=Path, required=True)
    clip.add_argument("--output", type=Path, required=True)
    clip.add_argument("--duration-s", type=float, default=60.0)
    clip.add_argument("--sample-rate-hz", type=float, default=20_000.0)
    clip.add_argument("--overwrite", action="store_true")
    commands.add_parser("check-env", help="Verify the standalone RAPID Python requirements.")
    return parser


def check_environment() -> int:
    required = ("numpy", "numba", "sklearn", "spikeinterface", "neo")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing RAPID dependencies: " + ", ".join(missing))
        return 1
    print("RAPID environment check passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check-env":
        return check_environment()
    if args.command == "clip-rhd":
        from .rhd_clip import clip_rhd

        metadata = clip_rhd(source=args.source, destination=args.output, duration_seconds=args.duration_s, sample_rate_hz=args.sample_rate_hz, overwrite=args.overwrite)
        print(metadata["destination_rhd"])
        return 0
    if args.command == "channel":
        from .channel_sorting import run_channel_sort

        output = run_channel_sort(
            rhd=args.rhd, channel_param=args.channel_param, results_root=args.results_root, config=_filter_config(args),
            stream_id=args.stream_id, stream_name=args.stream_name, ignore_integrity_checks=args.ignore_integrity_checks,
            chunk_blocks=args.chunk_blocks, write_band_dat=not args.no_band_dat, overwrite=args.overwrite,
        )
        print(output)
        return 0
    if args.command == "reference-pca":
        from .reference_pca import build_reference_pca

        output = build_reference_pca(
            channel_result=args.channel_result, output_model=args.model, groups=args.groups, pre_samples=args.pre_samples,
            post_samples=args.post_samples, max_events_per_group=args.max_events_per_group,
            overwrite=args.overwrite,
        )
        print(output)
        return 0
    if args.command == "unit":
        from .unit_sorting import run_unit_sort

        output = run_unit_sort(channel_result=args.channel_result, reference_model=args.model, results_root=args.results_root, overwrite=args.overwrite)
        print(output)
        return 0
    from .batch import run_all_days

    completed = run_all_days(
        input_root=args.input_root, channel_param=args.channel_param, reference_model=args.model, results_root=args.results_root,
        config=_filter_config(args), stream_id=args.stream_id, stream_name=args.stream_name,
        ignore_integrity_checks=args.ignore_integrity_checks, chunk_blocks=args.chunk_blocks, write_band_dat=not args.no_band_dat, overwrite=args.overwrite,
    )
    print(f"Completed {len(completed)} RAPID days.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
