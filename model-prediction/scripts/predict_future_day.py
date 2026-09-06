#!/usr/bin/env python
"""Command-line entry point for future-session spike-only deployment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_signal_decoder.predict_nomad import main


if __name__ == "__main__":
    main()
