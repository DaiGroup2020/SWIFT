from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Stable content fingerprint, deliberately independent of modification time."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def ensure_output_separate(path: Path, protected_paths: tuple[Path, ...]) -> None:
    """Reject output aliases that contain or replace any required input."""
    output = path.resolve()
    for protected in protected_paths:
        source = protected.resolve()
        if source == output or output in source.parents or (source.is_dir() and source in output.parents):
            raise ValueError(f"RAPID output overlaps a required input: {output} and {source}")


def archive_existing(path: Path) -> Path:
    """Preserve an exact previous artifact beside its replacement."""
    backup = path.with_name(f"{path.name}.backup-{uuid4().hex}")
    path.rename(backup)
    return backup


def prepare_result_directory(path: Path, *, overwrite: bool, protected_paths: tuple[Path, ...] = ()) -> Path:
    """Create a clean result directory; explicit overwrites keep a backup."""
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError(f"Refusing to replace a linked RAPID output directory: {path}")
    path = path.resolve()
    if path == Path(path.anchor) or path == Path.home().resolve() or path == Path.cwd().resolve():
        raise ValueError(f"Refusing a broad RAPID output directory: {path}")
    ensure_output_separate(path, protected_paths)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"RAPID output already exists: {path}. Use --overwrite to replace it.")
        if not path.is_dir():
            raise NotADirectoryError(f"RAPID output path is not a directory: {path}")
        archive_existing(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def result_manifest(
    *, kind: str, source_rhd: Path | None, output_dir: Path, sample_rate_hz: float, parameters: dict[str, Any],
    parent: Path | None = None, source_name: str | None = None, source_fingerprint: str | None = None,
) -> dict[str, Any]:
    if kind not in {"channel", "unit"}:
        raise ValueError(f"RAPID result kind must be channel or unit, received {kind!r}")
    if source_rhd is not None:
        source_rhd = source_rhd.resolve()
        source_name = source_rhd.name
        source_fingerprint = file_fingerprint(source_rhd)
    if not source_name or not source_fingerprint:
        raise ValueError("RAPID result manifests require a source file name and content fingerprint.")
    return {
        "schema_version": 1,
        "system": "RAPID",
        "result_name": kind,
        "created_at": utc_now(),
        "source_rhd": source_name,
        "source_fingerprint": source_fingerprint,
        "output_dir": output_dir.name,
        "sample_rate_hz": float(sample_rate_hz),
        "parameters": parameters,
        "parent_artifact": None if parent is None else parent.name,
    }
