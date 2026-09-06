"""PyCharm-friendly entry point for the bundled prediction demo.

Run this file directly after selecting the project interpreter. Results are
written to outputs/demo_prediction/.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "run_demo.py"),
        run_name="__main__",
    )
