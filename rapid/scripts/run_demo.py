"""Safely replay the bundled 60-second RAPID demonstration recording."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rapid.channel_sorting import run_channel_sort
from rapid.artifacts import ensure_output_separate
from rapid.preprocessing import FilterConfig
from rapid.unit_sorting import run_unit_sort


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rhd", type=Path, default=ROOT / "data" / "M06_20260129_L_260129_161328_60s.rhd")
    parser.add_argument("--channel-param", type=Path, default=ROOT / "data" / "ChannelParam_20260129.json")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "fpga14_local_peak_no_boundary_tmpl0129.csv")
    parser.add_argument("--results-root", type=Path, default=ROOT / "outputs", help="Untracked destination for generated results.")
    parser.add_argument("--chunk-blocks", type=int, default=256)
    parser.add_argument("--write-band-dat", action="store_true", help="Also save filtered traces; this increases output size substantially.")
    parser.add_argument("--overwrite", action="store_true", help="Archive each existing demo result directory beside its replacement.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for label, path in (("RHD", args.rhd), ("ChannelParam", args.channel_param), ("reference map", args.model)):
        if not path.is_file():
            raise FileNotFoundError(f"RAPID demo {label} is missing: {path}")
    for kind in ("channel", "unit"):
        ensure_output_separate(args.results_root / kind / args.rhd.stem.replace(" ", "_"), (args.rhd, args.channel_param, args.model))
    channel = run_channel_sort(
        rhd=args.rhd,
        channel_param=args.channel_param,
        results_root=args.results_root,
        config=FilterConfig(),
        chunk_blocks=args.chunk_blocks,
        write_band_dat=args.write_band_dat,
        overwrite=args.overwrite,
    )
    unit = run_unit_sort(channel_result=channel, reference_model=args.model, results_root=args.results_root, overwrite=args.overwrite)
    print("RAPID demo complete")
    print(f"channel: {channel}")
    print(f"unit:    {unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
