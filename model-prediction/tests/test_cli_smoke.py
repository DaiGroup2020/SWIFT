"""Fast, data-free checks for the documented command-line interfaces."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CommandLineSmokeTests(unittest.TestCase):
    def run_help(self, relative_script: str) -> None:
        completed = subprocess.run(
            [sys.executable, relative_script, "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("usage:", completed.stdout.lower())

    def test_demo_command_is_exposed(self) -> None:
        self.run_help("scripts/run_demo.py")

    def test_prediction_command_is_exposed(self) -> None:
        self.run_help("scripts/predict_future_day.py")


if __name__ == "__main__":
    unittest.main()
