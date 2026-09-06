from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CliSmokeTests(unittest.TestCase):
    def test_source_cli_help(self) -> None:
        completed = subprocess.run([sys.executable, str(ROOT / "rapid.py"), "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("channel", completed.stdout)
        self.assertIn("unit", completed.stdout)

    def test_demo_cli_help(self) -> None:
        completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_demo.py"), "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("results-root", completed.stdout)

    def test_package_cli_includes_reference_overwrite(self) -> None:
        completed = subprocess.run([sys.executable, "-m", "rapid", "reference-pca", "--help"], cwd=ROOT, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--overwrite", completed.stdout)


if __name__ == "__main__":
    unittest.main()
