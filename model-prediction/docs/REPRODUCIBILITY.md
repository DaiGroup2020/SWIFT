# Reproduce the prediction demo

Run commands on this page from the `model-prediction/` project directory. From
the SWIFT root directory, enter it with `cd model-prediction` first.

## Install and run

Follow the Python 3.10 installation commands in [README.md](../README.md).
`requirements-lock.txt` fixes the dependency versions, including stable
PyTorch 2.8.0 with CPU support. CPU is the shared verification environment;
the program also supports a compatible CUDA-enabled PyTorch installation.

From the `model-prediction/` project directory, run:

```powershell
python -m unittest discover -s tests -v
python scripts/run_demo.py --device cpu
python scripts/verify_demo.py --output-dir outputs/demo_prediction
```

Existing session outputs require `--overwrite`, or choose a fresh directory
with `--output-dir`. Generated files are described in the README. The input
checkpoint, spike CSV, and voltage log are Git LFS assets; run `git lfs pull`
if they have not been downloaded.

## What the demo computes

The model bins 213 identified spike trains into 30 ms intervals. It fits a
session-specific alignment using only spikes in the first 200 seconds, then
replays the sequence through the saved latent-dynamics model and decoder.
The decoder uses 51 bins per window, spanning 1.5 seconds between first and
last bin centers. Voltage values are loaded only after predictions are made.

The default output contains 7,950 predictions over the whole replay. Keep
these two score definitions separate:

| Summary field | Evaluated predictions |
|---|---|
| `evaluation.metrics` | All decoded endpoints with valid voltage targets, including the calibration period |
| `evaluation.post_calibration.metrics` | Only windows with their first bin strictly after the calibration end |

For the default calibration end of 200 seconds, the second score requires
`endpoint_time - 1.5 > 200`. It therefore excludes windows that cross the
calibration boundary, not only endpoints within calibration. The recurrent
model preserves its causal history from earlier bins. Each score includes
its valid-ground-truth count in the summary; the post-calibration section also
records its selected prediction count and first/last endpoint times.

`--time-start 200` restricts output endpoints but can still include decoder
windows crossing 200 seconds. The separate post-calibration summary continues
to apply the stricter rule. `--skip-evaluation` skips all target loading and
writes predictions without reference columns or scores.

## Configuration and another compatible recording

Use `python scripts/predict_future_day.py --help` or the installed
`neural-signal-predict --help` command. Inputs must follow
[DATA_FORMAT.md](DATA_FORMAT.md). Missing Unit identities are rejected unless
an explicit mapping option is requested.

Settings can also be supplied with `--config docs/prediction.json`, for example:

```json
{
  "model_path": "../models/top64_local_peak/manifold_lstm_decoder.pt",
  "spike_csv": "../data/demo/Online_demo_20260202.csv",
  "voltage_dat": "../data/demo/M06-2026-02-02_voltage.dat",
  "output_dir": "../outputs/configured_prediction",
  "day_name": "2026-02-02",
  "calibration_time_start": 0,
  "calibration_time_end": 200
}
```

The example assumes the JSON file is saved in `docs/`; paths inside it are
relative to that file. Explicit command-line values take precedence. With no
custom configuration, the default calibration is 0–200 seconds and the default
installed-command destination is `outputs/prediction/` in the current folder.

Generated summaries store input filenames and SHA-256 hashes, output filenames,
mapping details, and evaluation settings. Historical path strings embedded in
the frozen checkpoint are provenance metadata and are not used to find inputs
or choose output destinations.

## Validation record and documentation references

See [DEPLOYMENT_VALIDATION.md](DEPLOYMENT_VALIDATION.md) for recorded results.
For comparisons, use the same input files, locked environment, calibration,
device, and evaluation interval. GPU/library changes can alter numerical
results slightly.

The concise installation/demo/output layout follows examples such as
[Kilosort4](https://github.com/MouseLand/Kilosort) (Nature Methods) and
[speechBCI](https://github.com/fwillett/speechBCI) (Nature).
[Nature's code publication guidance](https://www.nature.com/documents/GuidelinesCodePublication.pdf)
informs the environment and run records; it does not prescribe a single
README template.
