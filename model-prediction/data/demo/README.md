# Prediction demo data

`Online_demo_20260202.csv` contains spike timestamps for 213 identified units
from the 240-second demo. `M06-2026-02-02_voltage.dat` contains the matching
target values used only for scoring and plotting reference traces. Both files
share the same clock and use seconds.

Run `python scripts/run_demo.py --device cpu` from the `model-prediction/` project directory. It
calibrates on the first 200 seconds using spikes only, then writes the whole
replay to `outputs/demo_prediction/`. The 7,950 output rows include the
calibration period; a separate post-calibration metric excludes decoder
windows that overlap that period.

The `top64` name identifies the upstream channel selection. The released
decoder expects all 213 Unit identities in this CSV. Run with
`--skip-evaluation` to omit the voltage file, and use `--overwrite` or another
output directory when repeating the example.
