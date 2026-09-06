"""Data-free regression checks for prediction and evaluation boundaries."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_signal_decoder.manifold_decoder import interpolate_columns
from neural_signal_decoder.predict import (
    DEFAULT_OUTPUT_DIR,
    build_prediction_windows,
    load_checkpoint,
    load_spike_csv_aligned,
    require_materialized_file,
)
from neural_signal_decoder.predict_nomad import (
    evaluate_prediction_segments,
    make_output_paths,
    post_calibration_window_mask,
    prepare_output_paths,
)


class PredictionContractTests(unittest.TestCase):
    def test_decoder_windows_are_causal(self) -> None:
        latent = np.arange(12, dtype=np.float32).reshape(6, 2)
        windows, endpoints = build_prediction_windows(latent, 3)
        np.testing.assert_array_equal(endpoints, [2, 3, 4, 5])
        np.testing.assert_array_equal(windows[1], latent[1:4])

    def test_post_calibration_excludes_cross_boundary_windows(self) -> None:
        times = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
        actual = post_calibration_window_mask(
            times, sequence_length=3, bin_size_s=0.5, calibration_time_end=1.0
        )
        np.testing.assert_array_equal(actual, [False, False, False, True, True])

    def test_segment_metrics_use_only_valid_future_ground_truth(self) -> None:
        times = np.arange(6, dtype=float)
        target = np.column_stack([times, times])
        prediction = target.copy()
        prediction[:4] += 10
        target[-1] = np.nan
        evaluation = evaluate_prediction_segments(
            times, prediction, target,
            sequence_length=2, bin_size_s=1.0,
            same_session_calibration=True, calibration_time_end=2.0,
        )
        self.assertGreater(evaluation["metrics"]["mse"], 0)
        post = evaluation["post_calibration"]
        self.assertEqual(post["prediction_count"], 2)
        self.assertEqual(post["valid_ground_truth_count"], 1)
        self.assertEqual(post["metrics"]["mse"], 0)

    def test_spike_only_evaluation_has_no_metric(self) -> None:
        evaluation = evaluate_prediction_segments(
            np.arange(5, dtype=float), np.zeros((5, 2)), None,
            sequence_length=2, bin_size_s=1.0,
            same_session_calibration=True, calibration_time_end=1.0,
        )
        self.assertIsNone(evaluation["metrics"])
        self.assertIsNone(evaluation["post_calibration"]["metrics"])
        self.assertEqual(evaluation["valid_ground_truth_count"], 0)

    def test_direct_mode_does_not_claim_post_calibration_score(self) -> None:
        evaluation = evaluate_prediction_segments(
            np.arange(5, dtype=float), np.zeros((5, 2)), np.ones((5, 2)),
            sequence_length=2, bin_size_s=1.0,
            same_session_calibration=False, calibration_time_end=1.0,
        )
        self.assertIsNone(evaluation["post_calibration"])

    def test_ground_truth_is_not_extrapolated(self) -> None:
        voltage = pd.DataFrame({"t": [1.0, 2.0], "vx": [2.0, 4.0], "vy": [4.0, 8.0]})
        values = interpolate_columns(voltage, np.array([0.5, 1.5, 2.5]), ["vx", "vy"])
        self.assertTrue(np.isnan(values[[0, 2]]).all())
        np.testing.assert_array_equal(values[1], [3.0, 6.0])

    def test_existing_outputs_require_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_output_paths(Path(temporary), "2026-02-02")
            paths["predictions_csv"].write_text("existing result", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                prepare_output_paths(paths, overwrite=False)
            prepare_output_paths(paths, overwrite=True)
            self.assertEqual(paths["predictions_csv"].read_text(encoding="utf-8"), "existing result")

    def test_lfs_pointer_produces_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pointer = Path(temporary) / "checkpoint.pt"
            pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                require_materialized_file(pointer)

    def test_named_unit_alignment_preserves_checkpoint_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv = Path(temporary) / "spikes.csv"
            csv.write_text("Unit2,Unit1,Unit3\n0.2,0.1,0.3\n", encoding="utf-8")
            times, mapping = load_spike_csv_aligned(csv, ["Unit1", "Unit2", "Unit4"], allow_missing_units=True)
            np.testing.assert_array_equal(times[0], [0.1])
            np.testing.assert_array_equal(times[1], [0.2])
            self.assertEqual(times[2].size, 0)
            self.assertEqual(mapping["missing_unit_cols"], ["Unit4"])
            self.assertEqual(mapping["extra_unit_cols"], ["Unit3"])

    def test_missing_unit_identity_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv = Path(temporary) / "spikes.csv"
            csv.write_text("Unit1\n0.1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing 1 checkpoint Unit"):
                load_spike_csv_aligned(csv, ["Unit1", "Unit2"])

    def test_positional_unit_mapping_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv = Path(temporary) / "spikes.csv"
            csv.write_text("Unit101,Unit102\n0.1,0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no checkpoint Unit identities match"):
                load_spike_csv_aligned(csv, ["Unit1", "Unit2"])
            with self.assertRaisesRegex(ValueError, "no checkpoint Unit identities match"):
                load_spike_csv_aligned(csv, ["Unit1", "Unit2"], allow_missing_units=True)
            times, mapping = load_spike_csv_aligned(csv, ["Unit1", "Unit2"], allow_positional_mapping=True)
            np.testing.assert_array_equal(times[0], [0.1])
            self.assertEqual(mapping["mapping_mode"], "positional_fallback")

    def test_invalid_spike_timestamps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv = Path(temporary) / "spikes.csv"
            csv.write_text("Unit1\ninf\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite, non-negative"):
                load_spike_csv_aligned(csv, ["Unit1"])

    def test_checkpoint_with_explicit_decoder_state_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            torch.save({"decoder_state_dict": {"weight": torch.ones(1)}}, checkpoint)
            loaded = load_checkpoint(checkpoint, "cpu")
            self.assertIn("decoder_state_dict", loaded)

    def test_default_outputs_are_outside_installed_package(self) -> None:
        self.assertFalse(DEFAULT_OUTPUT_DIR.is_absolute())
        self.assertEqual(DEFAULT_OUTPUT_DIR.parts, ("outputs", "prediction"))


if __name__ == "__main__":
    unittest.main()
