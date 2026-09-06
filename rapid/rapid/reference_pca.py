from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .artifacts import archive_existing, ensure_output_separate, file_fingerprint, utc_now, write_json


def _read_channel_events(channel_result: Path) -> list[dict[str, str]]:
    path = channel_result / "channel_spikes.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RAPID channel result: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _metadata(channel_result: Path) -> dict[str, Any]:
    path = channel_result / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RAPID channel manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _snippets(channel_result: Path, channel: str, ticks: np.ndarray, before: int, after: int) -> tuple[np.ndarray, np.ndarray]:
    trace_path = channel_result / f"band-{channel}.DAT"
    if not trace_path.is_file():
        raise FileNotFoundError(
            f"Reference PCA requires filtered traces. Missing {trace_path}; rerun channel sorting without --no-band-dat."
        )
    trace = np.memmap(trace_path, dtype="<i2", mode="r")
    valid = (ticks >= before) & (ticks + after < trace.size)
    selected = ticks[valid]
    data = np.empty((selected.size, before + after + 1), dtype=np.float32)
    for index, tick in enumerate(selected):
        data[index] = trace[int(tick) - before : int(tick) + after + 1]
    return data, selected


def _thresholds_from_pca(scores: np.ndarray, amplitudes: np.ndarray, groups: int) -> tuple[list[float], np.ndarray]:
    if amplitudes.size == 0:
        return [], np.empty(0, dtype=np.int32)
    group_count = min(max(1, int(groups)), int(amplitudes.size), int(np.unique(scores, axis=0).shape[0]))
    if group_count == 1:
        return [], np.zeros(amplitudes.size, dtype=np.int32)
    # Clustering is performed in PCA space using the full waveform. The
    # deployed unit assignment remains a deterministic 0-ms/local-peak rule:
    # PCA labels are sorted by median local amplitude before midpoint splits
    # are derived for the real-time-compatible unit model.
    feature_count = min(3, scores.shape[1])
    labels = KMeans(n_clusters=group_count, random_state=42, n_init=20).fit_predict(scores[:, :feature_count])
    # Degenerate waveforms can yield fewer distinct clusters than requested.
    order = sorted(np.unique(labels), key=lambda label: float(np.median(amplitudes[labels == label])))
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[int(label)] for label in labels], dtype=np.int32)
    centers = [float(np.median(amplitudes[labels == label])) for label in range(len(order))]
    return [float((left + right) / 2.0) for left, right in zip(centers, centers[1:])], labels


def build_reference_pca(
    *, channel_result: Path, output_model: Path, groups: int = 2, pre_samples: int = 15, post_samples: int = 30, max_events_per_group: int = 10_000, overwrite: bool = False
) -> Path:
    """Create a reference-day unit model without any outer boundary.

    Each channel and polarity receives a PCA audit and local-peak thresholds.
    Every later event maps to one interval spanning (-inf, +inf), so there is
    intentionally no residual/out-of-boundary class.
    """
    channel_result = channel_result.resolve()
    output_model = output_model.resolve()
    if int(groups) < 1 or int(max_events_per_group) < 1 or int(pre_samples) < 0 or int(post_samples) < 0:
        raise ValueError("Groups and maximum events must be positive; snippet extents must be non-negative.")
    output_paths = (output_model, output_model.with_name(output_model.stem + "_audit.csv"), output_model.with_name(output_model.stem + "_templates.npz"))
    for path in output_paths:
        ensure_output_separate(path, (channel_result,))
        if path.exists() and not overwrite:
            raise FileExistsError(f"Reference PCA output already exists: {path}. Use --overwrite to archive and replace it.")
        if path.exists() and not path.is_file():
            raise ValueError(f"Reference PCA output is not a file: {path}")
    manifest = _metadata(channel_result)
    sample_rate = float(manifest["sample_rate_hz"])
    events = _read_channel_events(channel_result)
    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in events:
        by_key.setdefault((row["channel"], row["polarity"]), []).append(row)
    model_channels: dict[str, dict[str, Any]] = {}
    audit_rows: list[dict[str, Any]] = []
    template_arrays: dict[str, np.ndarray] = {}
    for (channel, polarity), rows in sorted(by_key.items()):
        if len(rows) > int(max_events_per_group):
            rng = np.random.default_rng(42)
            rows = [rows[index] for index in np.sort(rng.choice(len(rows), size=int(max_events_per_group), replace=False))]
        samples = np.asarray([int(row["peak_sample"]) for row in rows], dtype=np.int64)
        snippets, valid_samples = _snippets(channel_result, channel, samples, int(pre_samples), int(post_samples))
        if snippets.size == 0:
            continue
        center = int(pre_samples)
        amplitudes = snippets[:, center].astype(np.float32)
        n_components = min(3, snippets.shape[0], snippets.shape[1])
        if snippets.shape[0] == 1 or not np.any(np.var(snippets, axis=0)):
            scores = np.zeros((snippets.shape[0], n_components), dtype=np.float32)
            explained_variance_ratio = [0.0] * n_components
        else:
            pca = PCA(n_components=n_components, random_state=42)
            scores = pca.fit_transform(snippets)
            explained_variance_ratio = [float(value) for value in pca.explained_variance_ratio_]
        thresholds, labels = _thresholds_from_pca(scores, amplitudes, groups)
        unit_count = len(thresholds) + 1
        units: list[dict[str, Any]] = []
        for index in range(unit_count):
            mask = labels == index
            template = np.median(snippets[mask], axis=0).astype(np.float32)
            unit_id = f"{channel}-{polarity}-U{index + 1:02d}"
            template_arrays[unit_id] = template
            units.append({"unit_id": unit_id, "index": index + 1, "event_count": int(mask.sum()), "median_peak_adc": float(np.median(amplitudes[mask]))})
        model_channels.setdefault(channel, {})[polarity] = {
            "thresholds_adc": thresholds,
            "units": units,
            "no_outer_boundary": True,
            "pca_components": int(n_components),
            "pca_explained_variance_ratio": explained_variance_ratio,
            "reference_event_samples": [int(value) for value in valid_samples],
        }
        audit_rows.append({"channel": channel, "polarity": polarity, "reference_events": int(snippets.shape[0]), "pca_components": int(n_components), "unit_count": unit_count, "thresholds_adc": ";".join(f"{value:.6g}" for value in thresholds), "score_std_pc1": float(np.std(scores[:, 0]))})
    if not model_channels:
        raise RuntimeError("Reference PCA found no usable channel events.")
    payload = {
        "schema_version": 1,
        "system": "RAPID",
        "artifact_type": "reference_pca_unit_model_no_boundary",
        "created_at": utc_now(),
        "reference_channel_result": channel_result.name,
        "reference_channel_manifest_fingerprint": file_fingerprint(channel_result / "manifest.json"),
        "sample_rate_hz": sample_rate,
        "filter": manifest["parameters"]["filter"],
        "pca": {"groups_requested": int(groups), "pre_samples": int(pre_samples), "post_samples": int(post_samples), "max_events_per_group": int(max_events_per_group)},
        "assignment_rule": "channel and polarity, then local-peak interval; all intervals extend to infinity (no outer boundary)",
        "channels": model_channels,
    }
    for path in output_paths:
        if path.exists():
            archive_existing(path)
    write_json(output_model, payload)
    with (output_model.with_name(output_model.stem + "_audit.csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    np.savez_compressed(output_model.with_name(output_model.stem + "_templates.npz"), **template_arrays)
    return output_model
