"""Verify bundled assets and the complete 60-second demo (standard library only)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = "M06_20260129_L_260129_161328_60s"
ASSETS = {
    f"data/{STEM}.rhd": "89b7a7b796dcfcec18c999f416cf9179abfbb60820cf88121e66fa367d3efd8c",
    "data/ChannelParam_20260129.json": "383252e222ac47eb70b176b18b8f39003a85e754f121fae9a9b0381956b69b90",
    "models/fpga14_local_peak_no_boundary_tmpl0129.csv": "dd7b69bdc08efe9abe9f7a2b48dd511c0b6133c8da6b6429525582ca5a16d984",
}
OUTPUT_HASHES = {
    "channel": "c3235272b0a88100daedc2b95113380a87262c3e90682e282cd3de565560374e",
    "unit": "6c2b2046e414384557afe6094d368a6836026db2a6840c12ee54f6e8a0cee71a",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        first = stream.read(1024 * 1024)
        require(not first.startswith(b"version https://git-lfs.github.com/spec/v1"),
                f"Git LFS pointer, not data: {path}. Run git lfs pull.")
        digest.update(first)
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(results_root: Path | None) -> dict:
    for name, expected in ASSETS.items():
        require(sha256(ROOT / name) == expected, f"Asset checksum mismatch: {name}")
    report = {"assets": "PASS", "asset_count": len(ASSETS)}
    if results_root is None:
        return report
    for kind, file_count, nonempty in (("channel", 191, 191), ("unit", 591, 575)):
        directory = results_root / kind / STEM
        require(sha256(directory / f"{kind}_spikes.csv") == OUTPUT_HASHES[kind],
                f"Unexpected {kind} event CSV: use the complete bundled demo.")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        count_key = "detected_event_count" if kind == "channel" else "assigned_event_count"
        require(manifest[count_key] == 439597, f"Incorrect {kind} event count")
        require(manifest["source_fingerprint"] == ASSETS[f"data/{STEM}.rhd"],
                f"Incorrect {kind} input fingerprint")
        if kind == "channel":
            require(manifest["enabled_channel_count"] == 191, "Incorrect channel count")
            require(sha256(directory / "ChannelParam.json") == ASSETS["data/ChannelParam_20260129.json"],
                    "Copied channel parameters differ")
            if manifest["parameters"]["write_band_dat"]:
                traces = list(directory.glob("band-*.DAT"))
                require(len(traces) == 191 and all(p.stat().st_size == 2400000 for p in traces),
                        "Filtered trace count or length differs")
                require((directory / "time.DAT").stat().st_size == 4800000, "Time trace length differs")
        else:
            require(manifest["catalog_unit_count"] == 591, "Incorrect catalogue count")
            require(manifest["nonempty_assigned_unit_count"] == 575, "Incorrect nonempty unit count")
            require(sha256(directory / "local_peak_no_boundary_map.csv") ==
                    ASSETS["models/fpga14_local_peak_no_boundary_tmpl0129.csv"], "Copied unit map differs")
        for export in ("unitcsv", "unitcsv_peak_aligned", "unitcsv_peak_aligned_seconds"):
            files = list((directory / export).glob("*.csv"))
            require(len(files) == file_count, f"Incorrect file count: {kind}/{export}")
            require(sum(p.stat().st_size > 0 for p in files) == nonempty,
                    f"Incorrect nonempty file count: {kind}/{export}")
            rows = 0
            for path in files:
                with path.open(encoding="utf-8") as stream:
                    rows += sum(bool(line.strip()) for line in stream)
            require(rows == 439597, f"Incomplete events: {kind}/{export}")
        report[kind] = {"events": 439597, "files_per_export": file_count, "status": "PASS"}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT / "outputs" / "demo-60s")
    parser.add_argument("--assets-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(None if args.assets_only else args.results_root), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Demo verification FAILED: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
