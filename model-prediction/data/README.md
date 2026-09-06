# Demo data

`demo/` contains the bundled 240-second spike CSV and matching voltage log.
The files are tracked with Git LFS; run `git lfs pull` after cloning.

Run `python scripts/run_demo.py --device cpu` from the `model-prediction/` project directory. Spikes
are used for calibration and prediction; voltage is used only for scoring and
the reference traces. Add `--skip-evaluation` to omit voltage loading.

See [demo/README.md](demo/README.md) for file details and
[DATA_FORMAT.md](../docs/DATA_FORMAT.md) for the supported input schema.
