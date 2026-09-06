from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .artifacts import file_fingerprint, prepare_result_directory, result_manifest, write_json
from .local_peak_no_boundary import LocalPeakNoBoundaryMap


def _assign(thresholds: list[float], amplitude: float) -> int:
    for index, threshold in enumerate(thresholds):
        if amplitude < float(threshold):
            return index
    return len(thresholds)


def _run_pca_unit_sort(*, channel_result: Path, reference_model: Path, results_root: Path, overwrite: bool = False) -> Path:
    """Assign all channel events to reference-PCA units without an outer boundary."""
    channel_result = channel_result.resolve()
    reference_model = reference_model.resolve()
    channel_manifest = json.loads((channel_result / "manifest.json").read_text(encoding="utf-8"))
    model = json.loads(reference_model.read_text(encoding="utf-8"))
    if float(model["sample_rate_hz"]) != float(channel_manifest["sample_rate_hz"]):
        raise ValueError("Reference PCA model and channel result have different sample rates.")
    output = results_root.resolve() / "unit" / channel_result.name
    with (channel_result / "channel_spikes.csv").open(newline="", encoding="utf-8-sig") as handle:
        events = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for event in events:
        channel = event["channel"]
        polarity = event["polarity"]
        key = model.get("channels", {}).get(channel, {}).get(polarity)
        if key is None:
            raise RuntimeError(f"Reference PCA model lacks {channel}/{polarity}; no silent fallback is permitted.")
        index = _assign(key["thresholds_adc"], float(event["peak_value_adc"]))
        unit = key["units"][index]
        rows.append({**event, "unit_id": unit["unit_id"], "unit_index": unit["index"], "no_outer_boundary": True})
    rows.sort(key=lambda row: (int(row["peak_tick"]), row["unit_id"]))
    prepare_result_directory(output, overwrite=overwrite, protected_paths=(channel_result, reference_model))
    fields = ["channel", "polarity", "unit_id", "unit_index", "threshold_tick", "threshold_sample", "threshold_seconds", "peak_tick", "peak_sample", "peak_seconds", "threshold_value_adc", "peak_value_adc", "no_outer_boundary"]
    with (output / "unit_spikes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for directory in ("unitcsv", "unitcsv_peak_aligned", "unitcsv_peak_aligned_seconds"):
        (output / directory).mkdir(exist_ok=True)
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_unit.setdefault(row["unit_id"], []).append(row)
    # Keep empty per-unit files as an explicit zero-event result for a day.
    # This makes the unit catalog stable across dates without inventing spikes.
    for channel_model in model.get("channels", {}).values():
        for polarity_model in channel_model.values():
            for unit in polarity_model.get("units", []):
                by_unit.setdefault(str(unit["unit_id"]), [])
    for unit_id, unit_rows in by_unit.items():
        with (output / "unitcsv" / f"{unit_id}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([[row["threshold_tick"]] for row in unit_rows])
        with (output / "unitcsv_peak_aligned" / f"{unit_id}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([[row["peak_tick"]] for row in unit_rows])
        with (output / "unitcsv_peak_aligned_seconds" / f"{unit_id}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([[row["peak_seconds"]] for row in unit_rows])
    manifest = result_manifest(
        kind="unit", source_rhd=None, source_name=str(channel_manifest["source_rhd"]),
        source_fingerprint=str(channel_manifest["source_fingerprint"]), output_dir=output,
        sample_rate_hz=float(channel_manifest["sample_rate_hz"]),
        parameters={"reference_pca_model": reference_model.name, "reference_pca_model_fingerprint": file_fingerprint(reference_model), "no_outer_boundary": True},
        parent=channel_result / "manifest.json",
    )
    manifest.update({"assigned_event_count": len(rows), "assigned_unit_count": len(by_unit), "reference_pca_model": reference_model.name})
    write_json(output / "manifest.json", manifest)
    return output


def run_local_peak_no_boundary_sort(*, channel_result: Path, reference_map: Path, results_root: Path, overwrite: bool = False) -> Path:
    """Assign each event directly with FPGA14 local-peak no-boundary intervals.

    This deliberately never creates a residual event or performs a later merge.
    The final POS outer unit includes the +inf tail; the final NEG outer unit
    includes the -inf tail.
    """
    channel_result = channel_result.resolve()
    reference_map = reference_map.resolve()
    channel_manifest = json.loads((channel_result / "manifest.json").read_text(encoding="utf-8"))
    mapping = LocalPeakNoBoundaryMap.from_csv(reference_map)
    output = results_root.resolve() / "unit" / channel_result.name

    with (channel_result / "channel_spikes.csv").open(newline="", encoding="utf-8-sig") as handle:
        events = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for event in events:
        interval = mapping.assign(
            channel=event["channel"], polarity=event["polarity"], peak_value_adc=int(event["peak_value_adc"])
        )
        rows.append(
            {
                **event,
                "unit_id": interval.unit_id,
                "unit_index": interval.unit_index,
                "no_outer_boundary": True,
                "assignment_mode": "fpga14_local_peak_no_boundary_direct",
                "bin_lower_adc": interval.lower_adc,
                "bin_upper_adc": interval.upper_adc,
            }
        )
    rows.sort(key=lambda row: (int(row["peak_tick"]), row["unit_id"]))
    prepare_result_directory(output, overwrite=overwrite, protected_paths=(channel_result, reference_map))
    fields = [
        "channel", "polarity", "unit_id", "unit_index", "threshold_tick", "threshold_sample", "threshold_seconds",
        "peak_tick", "peak_sample", "peak_seconds", "threshold_value_adc", "peak_value_adc", "no_outer_boundary",
        "assignment_mode", "bin_lower_adc", "bin_upper_adc",
    ]
    with (output / "unit_spikes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for directory in ("unitcsv", "unitcsv_peak_aligned", "unitcsv_peak_aligned_seconds"):
        (output / directory).mkdir(exist_ok=True)
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_unit.setdefault(str(row["unit_id"]), []).append(row)
    for interval in mapping.catalog:
        by_unit.setdefault(interval.unit_id, [])
    for unit_id, unit_rows in by_unit.items():
        with (output / "unitcsv" / f"{unit_id}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([[row["threshold_tick"]] for row in unit_rows])
        with (output / "unitcsv_peak_aligned" / f"{unit_id}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([[row["peak_tick"]] for row in unit_rows])
        with (output / "unitcsv_peak_aligned_seconds" / f"{unit_id}.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerows([[row["peak_seconds"]] for row in unit_rows])

    shutil.copy2(reference_map, output / "local_peak_no_boundary_map.csv")
    manifest = result_manifest(
        kind="unit", source_rhd=None, source_name=str(channel_manifest["source_rhd"]),
        source_fingerprint=str(channel_manifest["source_fingerprint"]), output_dir=output,
        sample_rate_hz=float(channel_manifest["sample_rate_hz"]),
        parameters={
            "reference_local_peak_no_boundary_map": reference_map.name,
            "reference_local_peak_no_boundary_map_fingerprint": mapping.fingerprint,
            "assignment_mode": "fpga14_local_peak_no_boundary_direct",
            "no_outer_boundary": True,
        },
        parent=channel_result / "manifest.json",
    )
    manifest.update(
        {
            "assigned_event_count": len(rows),
            "assigned_unit_count": len(by_unit),
            "catalog_unit_count": len(mapping.catalog),
            "nonempty_assigned_unit_count": sum(bool(unit_rows) for unit_rows in by_unit.values()),
            "reference_local_peak_no_boundary_map": reference_map.name,
        }
    )
    write_json(output / "manifest.json", manifest)
    return output


def run_unit_sort(*, channel_result: Path, reference_model: Path, results_root: Path, overwrite: bool = False) -> Path:
    """Dispatch to direct FPGA14 no-boundary CSV assignment or legacy PCA JSON."""
    if reference_model.suffix.lower() == ".csv":
        return run_local_peak_no_boundary_sort(
            channel_result=channel_result, reference_map=reference_model, results_root=results_root, overwrite=overwrite
        )
    return _run_pca_unit_sort(
        channel_result=channel_result, reference_model=reference_model, results_root=results_root, overwrite=overwrite
    )
