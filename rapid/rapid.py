#!/usr/bin/env python
"""Source-checkout compatibility wrapper for the installed RAPID command."""
from rapid.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
