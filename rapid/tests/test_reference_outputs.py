"""Standard-library checks for the compact reference-output verifier."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("rapid_reference_verifier", ROOT / "scripts" / "verify_demo.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class ReferenceOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        hashes = {}
        for kind in ("channel", "unit"):
            data = b"synthetic verifier fixture\n"
            (self.directory / f"{kind}_spikes.csv").write_bytes(data)
            hashes[kind] = hashlib.sha256(data).hexdigest()
            manifest = {
                "source_fingerprint": verifier.ASSETS[f"data/{verifier.STEM}.rhd"],
                "detected_event_count": 439597, "assigned_event_count": 439597,
                "enabled_channel_count": 191, "catalog_unit_count": 591,
                "nonempty_assigned_unit_count": 575,
            }
            (self.directory / f"{kind}_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        assets = patch.object(verifier, "verify", return_value={"assets": "PASS", "asset_count": 3})
        assets.start()
        self.addCleanup(assets.stop)
        expected = patch.dict(verifier.OUTPUT_HASHES, hashes, clear=True)
        expected.start()
        self.addCleanup(expected.stop)

    def test_compact_scope_does_not_require_per_unit_exports(self) -> None:
        report = verifier.verify_reference(self.directory)
        self.assertEqual(report["scope"], "compact_reference_only")
        self.assertEqual(report["channel"]["status"], "PASS")
        self.assertEqual(report["unit"]["status"], "PASS")

    def test_modified_reference_csv_is_rejected(self) -> None:
        (self.directory / "channel_spikes.csv").write_bytes(b"modified\n")
        with self.assertRaisesRegex(ValueError, "reference CSV"):
            verifier.verify_reference(self.directory)

    def test_incorrect_manifest_count_is_rejected(self) -> None:
        path = self.directory / "unit_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["assigned_event_count"] = 1
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference event count"):
            verifier.verify_reference(self.directory)


if __name__ == "__main__":
    unittest.main()
