# Neural Signal Prediction

Run a pretrained neural decoder on the bundled 240-second example. The demo
uses spike timestamps to calibrate the model and predicts two-dimensional
velocity. It saves the predictions, an evaluation summary, and a figure.

## System requirements

- Tested on Windows 11 Pro, 64-bit (build 22631), Python 3.10.11, and an Intel
  Core Ultra 9 285K. macOS and Linux have not been validated for this release.
- Validated scientific dependencies: PyTorch 2.8.0+cpu, NumPy 2.1.3, pandas
  2.3.3, Matplotlib 3.10.8, and scikit-learn 1.7.2. All exact dependency
  versions are in [requirements-lock.txt](requirements-lock.txt).
- No non-standard hardware, FPGA board, acquisition hardware, or dedicated
  GPU is required for this recorded-data demo. Git LFS retrieves the assets.

## Install

Use Python 3.10 and Git LFS. From the SWIFT root directory, enter this project
with `cd model-prediction` first. Run these commands from the
`model-prediction/` project directory in Windows PowerShell:

```powershell
git lfs install
git lfs pull
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation .
```

The locked environment includes CPU PyTorch so the example needs no dedicated
GPU. For GPU use, install a compatible CUDA-enabled PyTorch build and select
`--device cuda` when running the demo.

Typical installation on the tested desktop took approximately 2-3 minutes
(140.58 seconds), excluding package and data downloads. The CPU PyTorch
wheel is approximately 619 MB; download speed can substantially increase
setup time.

## Run the example

Run from the `model-prediction/` project directory:

```powershell
python scripts/run_demo.py --device cpu
```

The command uses the bundled checkpoint, spike CSV, and matching voltage
log.

Expected runtime is approximately 74 seconds on the tested desktop using the
locked CPU environment. This is an offline software replay, not a guarantee
of online or hardware latency.

Results are saved in `outputs/demo_prediction/`:

| File | Contents |
|---|---|
| `predictions_2026-02-02.csv` | Timestamps, predicted velocity, and reference velocity |
| `prediction_summary_2026-02-02.json` | Input checksums, calibration details, and metrics |
| `predictions_2026-02-02.png` | Predicted and reference velocity traces, with joint vx/vy R² |

The demo calibrates using spikes from the first 200 seconds. Its 7,950 decoded
rows cover the whole replay, including that period. The summary reports both
whole-replay metrics and a separate post-calibration score that excludes
decoder windows overlapping calibration.

The figure reports joint vx/vy R² over all valid samples in the plotted time
interval, calculated before display downsampling. For the default full replay,
this includes calibration and is approximately 0.8884. The separate
post-calibration R² is recorded in the JSON summary, not substituted for the
full-replay figure score. No R² is shown when evaluation is skipped.

Precomputed CSV/JSON/PNG examples are included in
[outputs/reference](outputs/reference/README.md). Normal runs use
`outputs/demo_prediction/`, leaving these reference files unchanged.

## Run again or choose another output folder

```powershell
python scripts/run_demo.py --device cpu --overwrite
python scripts/run_demo.py --device cpu --output-dir outputs/my_run
```

Add `--skip-evaluation` to predict without reading the voltage log. This writes
predicted velocity without reference columns or evaluation scores.

## Use your own recording and further details

For another compatible recording, see the command-line options with
`python scripts/predict_future_day.py --help` and read the
[input format](docs/DATA_FORMAT.md). The checkpoint expects 213 named units;
`top64` describes its upstream channel selection, not its input-column count.

The [functional description](docs/REPRODUCIBILITY.md) explains calibration,
evaluation, and configuration. Model details are in the
[model card](models/top64_local_peak/MODEL_CARD.md).

## Citation and license

This independent DaiGroup PyTorch implementation is informed by the latent
dynamics alignment method of [Karpowicz et al. (2025)](https://doi.org/10.1038/s41467-025-59652-y).
See [method attribution and provenance](docs/PROVENANCE.md) for the citation,
implementation differences, and source and model records.

The DaiGroup-owned software, demo data, pretrained weights, and reference
outputs are licensed under GNU GPL version 3 only (`GPL-3.0-only`). See [LICENSE](LICENSE),
[NOTICE](NOTICE), and [CITATION.cff](CITATION.cff). This license does not grant
rights to third-party software or patents. Covered assets are listed in the
[asset license table](../ASSET_LICENSES.md).
