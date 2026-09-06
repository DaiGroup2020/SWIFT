# Neural Signal Prediction

Run a pretrained neural decoder on the bundled 240-second example. The demo
uses spike timestamps to calibrate the model and predicts two-dimensional
velocity. It saves the predictions, an evaluation summary, and a figure.

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

## Run the example

Run from the `model-prediction/` project directory:

```powershell
python scripts/run_demo.py --device cpu
python scripts/verify_demo.py --output-dir outputs/demo_prediction
```

The first command uses the bundled checkpoint, spike CSV, and matching voltage
log. The second checks the generated results against the demo reference.

Results are saved in `outputs/demo_prediction/`:

| File | Contents |
|---|---|
| `predictions_2026-02-02.csv` | Timestamps, predicted velocity, and reference velocity |
| `prediction_summary_2026-02-02.json` | Input checksums, calibration details, and metrics |
| `predictions_2026-02-02.png` | Predicted and reference velocity traces |

The demo calibrates using spikes from the first 200 seconds. Its 7,950 decoded
rows cover the whole replay, including that period. The summary reports both
whole-replay metrics and a separate post-calibration score that excludes
decoder windows overlapping calibration.

## Run again or choose another output folder

```powershell
python scripts/run_demo.py --device cpu --overwrite
python scripts/run_demo.py --device cpu --output-dir outputs/my_run
```

Add `--skip-evaluation` to predict without reading the voltage log. This writes
predicted velocity without reference columns or evaluation scores.

## Tests and details

```powershell
python -m unittest discover -s tests -v
```

For another compatible recording, see the command-line options with
`python scripts/predict_future_day.py --help` and read the
[input format](docs/DATA_FORMAT.md). The checkpoint expects 213 named units;
`top64` describes its upstream channel selection, not its input-column count.

The [reproducibility guide](docs/REPRODUCIBILITY.md) explains calibration,
evaluation, and configuration. Model details are in the
[model card](models/top64_local_peak/MODEL_CARD.md), and validation records are
in [deployment validation](docs/DEPLOYMENT_VALIDATION.md).

## Citation and license

See [CITATION.cff](CITATION.cff) for citation metadata and [LICENSE](LICENSE)
for the software license.
