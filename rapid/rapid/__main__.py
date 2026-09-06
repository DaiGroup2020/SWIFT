"""Run the packaged command line interface with ``python -m rapid``."""
from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
