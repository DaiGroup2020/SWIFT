"""Backward-compatible wrapper for the safe bundled RAPID demo."""
from scripts.run_demo import main


if __name__ == "__main__":
    raise SystemExit(main())
