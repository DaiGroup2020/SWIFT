from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rapid.preprocessing import FilterConfig, FilterState, round_away_from_zero


class PreprocessingTests(unittest.TestCase):
    def test_chunking_is_bitwise_invariant(self) -> None:
        raw = np.asarray(
            [
                [32768, 32774, 32759], [32770, 32760, 32780], [32764, 32780, 32758],
                [32775, 32762, 32769], [32758, 32775, 32763], [32772, 32761, 32781],
                [32766, 32776, 32757], [32777, 32759, 32771], [32761, 32782, 32764],
            ],
            dtype=np.int32,
        )
        whole_state = FilterState(3, FilterConfig())
        whole = whole_state.process(raw)
        split_state = FilterState(3, FilterConfig())
        split = np.vstack([split_state.process(raw[:2]), split_state.process(raw[2:5]), split_state.process(raw[5:])])
        np.testing.assert_array_equal(whole, split)
        self.assertEqual(whole_state.samples_seen, raw.shape[0])
        self.assertEqual(split_state.samples_seen, raw.shape[0])

    def test_external_nonuniform_chunk_regression(self) -> None:
        """Regression supplied with the synchronized external RAPID source."""
        rng = np.random.default_rng(20260906)
        channels = 191
        samples = 65_537
        raw = rng.integers(30_000, 36_000, size=(samples, channels), dtype=np.int32)
        one_pass_state = FilterState(channels, FilterConfig())
        one_pass = one_pass_state.process(raw)

        chunked_state = FilterState(channels, FilterConfig())
        pieces: list[np.ndarray] = []
        offset = 0
        for size in (1, 127, 128, 32_768, 8_192, 4_096, 16_384):
            if offset >= samples:
                break
            stop = min(samples, offset + size)
            pieces.append(chunked_state.process(raw[offset:stop]))
            offset = stop
        if offset < samples:
            pieces.append(chunked_state.process(raw[offset:]))
        chunked = np.concatenate(pieces, axis=0)

        np.testing.assert_array_equal(one_pass, chunked)
        self.assertEqual(one_pass_state.samples_seen, samples)
        self.assertEqual(chunked_state.samples_seen, samples)

    def test_empty_chunk_does_not_advance_filter_state(self) -> None:
        state = FilterState(3, FilterConfig())
        output = state.process(np.empty((0, 3), dtype=np.int32))
        self.assertEqual(output.shape, (0, 3))
        self.assertEqual(state.samples_seen, 0)

    def test_common_signal_is_removed(self) -> None:
        raw = np.tile(np.arange(32760, 32770, dtype=np.int32)[:, None], (1, 3))
        result = FilterState(3, FilterConfig()).process(raw)
        np.testing.assert_array_equal(result, np.zeros_like(result))

    def test_rounding_is_away_from_zero(self) -> None:
        self.assertEqual(round_away_from_zero(1.5), 2)
        self.assertEqual(round_away_from_zero(-1.5), -2)
        self.assertEqual(round_away_from_zero(0.49), 0)


if __name__ == "__main__":
    unittest.main()
