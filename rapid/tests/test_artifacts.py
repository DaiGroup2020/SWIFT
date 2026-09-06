from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rapid.artifacts import prepare_result_directory, result_manifest
from rapid.rhd_clip import clip_rhd


def _temporary_directory() -> tempfile.TemporaryDirectory:
    """Allow a constrained runner to supply a writable test-only directory."""
    root = os.environ.get("RAPID_TEST_TMP")
    return tempfile.TemporaryDirectory(dir=root)


class ArtifactTests(unittest.TestCase):
    def test_overwrite_creates_a_clean_session_directory(self) -> None:
        with _temporary_directory() as temporary:
            output = Path(temporary) / "results" / "channel" / "fixture"
            output.mkdir(parents=True)
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            prepare_result_directory(output, overwrite=True)
            self.assertEqual(list(output.iterdir()), [])
            backups = list(output.parent.glob("fixture.backup-*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / "stale.txt").read_text(encoding="utf-8"), "stale")
            with self.assertRaises(FileExistsError):
                prepare_result_directory(output, overwrite=False)

    def test_overwrite_rejects_directory_containing_an_input(self) -> None:
        with _temporary_directory() as temporary:
            output = Path(temporary) / "fixture"
            output.mkdir()
            source = output / "source.rhd"
            source.write_bytes(b"recording")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                prepare_result_directory(output, overwrite=True, protected_paths=(source,))
            self.assertEqual(source.read_bytes(), b"recording")

    def test_clip_refuses_to_overwrite_source(self) -> None:
        with _temporary_directory() as temporary:
            source = Path(temporary) / "source.rhd"
            source.write_bytes(b"recording")
            with self.assertRaisesRegex(ValueError, "differ"):
                clip_rhd(source=source, destination=source, duration_seconds=1, overwrite=True)
            self.assertEqual(source.read_bytes(), b"recording")

    def test_manifest_uses_portable_identifiers(self) -> None:
        with _temporary_directory() as temporary:
            root = Path(temporary)
            source = root / "input.rhd"
            source.write_bytes(b"fixture")
            manifest = result_manifest(
                kind="channel", source_rhd=source, output_dir=root / "output", sample_rate_hz=20_000.0, parameters={}
            )
            self.assertEqual(manifest["source_rhd"], "input.rhd")
            self.assertEqual(manifest["output_dir"], "output")
            self.assertNotIn(str(root), str(manifest))

    def test_rhd_clip_writes_portable_metadata(self) -> None:
        with _temporary_directory() as temporary:
            root = Path(temporary)
            source = root / "source.rhd"
            header = b"header" * 4
            blocks = []
            for start in (0, 128, 256):
                timestamps = np.asarray(range(start, start + 128), dtype="<i4").tobytes()
                blocks.append(timestamps + b"\0" * 8)
            source.write_bytes(header + b"".join(blocks))
            destination = root / "clip.rhd"
            metadata = clip_rhd(source=source, destination=destination, duration_seconds=1, sample_rate_hz=128)
            self.assertEqual(metadata["source_rhd"], "source.rhd")
            self.assertEqual(metadata["destination_rhd"], "clip.rhd")
            self.assertNotIn(str(root), str(metadata))


if __name__ == "__main__":
    unittest.main()
