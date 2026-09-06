from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rapid.batch import run_all_days
from rapid.preprocessing import FilterConfig
from rapid.reference_pca import build_reference_pca
from rapid.unit_sorting import run_unit_sort


class ReferencePcaTests(unittest.TestCase):
    def _fixture(self, root: Path, samples: tuple[int, ...] = (3, 9)) -> Path:
        channel = root / "channel" / "fixture"
        channel.mkdir(parents=True)
        (channel / "manifest.json").write_text(json.dumps({
            "sample_rate_hz": 20000, "parameters": {"filter": {}},
            "source_rhd": "not-distributed.rhd", "source_fingerprint": "fixture-fingerprint",
        }), encoding="utf-8")
        trace = np.zeros(16, dtype="<i2")
        trace[list(samples)] = 100
        trace.tofile(channel / "band-A-000.DAT")
        with (channel / "channel_spikes.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["channel", "polarity", "threshold_tick", "threshold_sample", "threshold_seconds", "peak_tick", "peak_sample", "peak_seconds", "threshold_value_adc", "peak_value_adc"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for sample in samples:
                writer.writerow({"channel": "A-000", "polarity": "POS", "threshold_tick": sample, "threshold_sample": sample, "threshold_seconds": sample / 20000, "peak_tick": sample, "peak_sample": sample, "peak_seconds": sample / 20000, "threshold_value_adc": 100, "peak_value_adc": 100})
        return channel

    def test_degenerate_pca_and_portable_unit_roundtrip(self) -> None:
        for samples in ((3,), (3, 9)):
            with self.subTest(samples=samples), tempfile.TemporaryDirectory(dir=os.environ.get("RAPID_TEST_TMP")) as temporary:
                root = Path(temporary)
                channel = self._fixture(root, samples)
                model = build_reference_pca(channel_result=channel, output_model=root / "model.json", pre_samples=1, post_samples=1)
                text = model.read_text(encoding="utf-8")
                self.assertNotIn("NaN", text)
                payload = json.loads(text)
                self.assertEqual(len(payload["channels"]["A-000"]["POS"]["units"]), 1)
                output = run_unit_sort(channel_result=channel, reference_model=model, results_root=root / "results")
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["assigned_event_count"], len(samples))
                self.assertEqual(manifest["source_rhd"], "not-distributed.rhd")

    def test_pca_checks_all_output_files_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory(dir=os.environ.get("RAPID_TEST_TMP")) as temporary:
            root = Path(temporary)
            channel = self._fixture(root)
            audit = root / "model_audit.csv"
            audit.write_text("existing audit", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                build_reference_pca(channel_result=channel, output_model=root / "model.json")
            self.assertFalse((root / "model.json").exists())
            self.assertEqual(audit.read_text(encoding="utf-8"), "existing audit")
            build_reference_pca(channel_result=channel, output_model=root / "model.json", pre_samples=1, post_samples=1, overwrite=True)
            self.assertEqual(len(list(root.glob("model_audit.csv.backup-*"))), 1)

    def test_unit_rejects_replacing_its_input_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=os.environ.get("RAPID_TEST_TMP")) as temporary:
            root = Path(temporary)
            channel = self._fixture(root)
            model = build_reference_pca(channel_result=channel, output_model=root / "model.json", pre_samples=1, post_samples=1)
            input_dir = root / "results" / "unit" / "fixture"
            input_dir.parent.mkdir(parents=True)
            channel.rename(input_dir)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                run_unit_sort(channel_result=input_dir, reference_model=model, results_root=root / "results", overwrite=True)
            self.assertTrue((input_dir / "channel_spikes.csv").is_file())

    def test_batch_rejects_colliding_session_names_before_processing(self) -> None:
        with tempfile.TemporaryDirectory(dir=os.environ.get("RAPID_TEST_TMP")) as temporary:
            root = Path(temporary)
            for folder in ("first", "second"):
                (root / folder).mkdir()
                (root / folder / "same.rhd").write_bytes(b"fixture")
            with self.assertRaisesRegex(ValueError, "same result directory"):
                run_all_days(input_root=root, channel_param=root / "param.json", reference_model=root / "model.json", results_root=root / "results", config=FilterConfig(), stream_id="0", stream_name=None, ignore_integrity_checks=False, chunk_blocks=1, write_band_dat=False, overwrite=True)
            self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
