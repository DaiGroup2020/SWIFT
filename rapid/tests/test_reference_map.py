from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rapid.local_peak_no_boundary import LocalPeakNoBoundaryMap


MAP = ROOT / "models" / "fpga14_local_peak_no_boundary_tmpl0129.csv"


class ReferenceMapTests(unittest.TestCase):
    def test_release_map_is_direct_and_path_sanitised(self) -> None:
        mapping = LocalPeakNoBoundaryMap.from_csv(MAP)
        self.assertEqual(len(mapping.catalog), 591)
        self.assertEqual(len(mapping.groups), 382)
        self.assertEqual(len({item.channel for item in mapping.catalog}), 191)
        with MAP.open(newline="", encoding="utf-8") as handle:
            self.assertEqual(
                csv.DictReader(handle).fieldnames,
                ["unit_id", "channel", "polarity", "unit_index", "is_residual", "lower_uv", "upper_uv"],
            )

    def test_outer_intervals_assign_directly(self) -> None:
        mapping = LocalPeakNoBoundaryMap.from_csv(MAP)
        for (channel, polarity), intervals in mapping.groups.items():
            if polarity == "POS":
                assigned = mapping.assign(channel=channel, polarity=polarity, peak_value_adc=1_000_000)
            else:
                assigned = mapping.assign(channel=channel, polarity=polarity, peak_value_adc=-1_000_000)
            self.assertEqual(assigned.unit_id, intervals[-1].unit_id)


if __name__ == "__main__":
    unittest.main()
