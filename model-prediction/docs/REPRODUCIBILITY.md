# Prediction demonstration and functional description

Run commands on this page from the `model-prediction/` project directory. From
the SWIFT root directory, enter it with `cd model-prediction` first.

## Install and run

Follow the Python 3.10 installation commands in [README.md](../README.md).
`requirements-lock.txt` fixes the dependency versions, including stable
PyTorch 2.8.0 with CPU support. CPU is the documented demonstration environment;
the program also supports a compatible CUDA-enabled PyTorch installation.

From the `model-prediction/` project directory, run:

```powershell
python scripts/run_demo.py --device cpu
```

Existing session outputs require `--overwrite`, or choose a fresh directory
with `--output-dir`. Generated files are described in the README. The input
checkpoint, spike CSV, and voltage log are Git LFS assets; run `git lfs pull`
if they have not been downloaded.

## Functional description and pseudocode

This section describes the prediction code's operations, model, calibration
objective, and decoding procedure.

### Inputs and fixed model

The input is a sparse CSV of identified spike timestamps. Columns are matched
to the checkpoint's 213 named units and ordered accordingly; missing names
are rejected unless an explicit override is requested. The checkpoint contains
the reference dynamics, decoder, saved scalers, and reference latent mean and
covariance. Its weights are not retrained or overwritten by the demo.

The reference model has a linear read-in, layer normalization, a unidirectional
GRU, latent normalization, a positive firing-rate reconstruction head, and a
behavior head used during reference-model training. The released latent
dimension is 176. The separate encoded decoder applies learned time and latent
projections to a latent window, followed by two LSTM layers (hidden size 144),
normalization, and a two-dimensional regression head. Its released dropout
setting is 0.15. See the [model card](../models/top64_local_peak/MODEL_CARD.md).

### Spike-only session calibration

The first 200 seconds of spikes define the default calibration interval.
Per-unit normalization statistics are estimated from those spikes; the
Gaussian smoothing setting is used to estimate these statistics, not to load
future voltage targets. Each model input is normalized using the fitted
statistics. The reference dynamics and saved behavior decoder remain fixed.

A residual alignment network and copies of the observation read-in and rate
head are adapted with AdamW. The minimized objective is a weighted sum of:

1. Poisson reconstruction loss on raw spike counts, implemented as the mean
   of `rate - count * log(rate)` after clamping positive rates.
2. Multivariate Gaussian KL divergence from the current latent mean/covariance
   to the saved reference statistics, with covariance regularization.
3. An identity penalty, the mean squared change from normalized input to
   aligned input.

No voltage or behavior labels enter this calibration loss. The random seed is
42, and the epoch limit, learning rate, weights, dropout, gradient clipping,
and early-stopping settings are read from checkpoint metadata. The best
calibration state is used only for that run. `--no-daily-calibration` bypasses
session adaptation and uses the saved shared dynamics directly.

### Continuous decoding and evaluation

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

### Demo pseudocode

```text
load checkpoint, reference model, encoded decoder, scalers and reference statistics
validate the spike CSV and match its unit names to checkpoint order
bin spike timestamps into 30-ms count vectors
select count vectors in the spike-only calibration interval
freeze the saved reference model and decoder
fit session alignment, read-in and rate-head copies using spike-only losses
encode the full count sequence, preserving GRU state across processing chunks
scale latents and build 51-bin windows ending at each prediction time
run the fixed encoded LSTM decoder and undo output scaling
apply any requested output-time limits and save velocity predictions
if evaluation is enabled:
    load matching voltage targets only after prediction
    interpolate within target coverage; do not extrapolate missing targets
    score the whole replay and the non-overlapping post-calibration windows separately
save the prediction CSV, input hashes, settings, summary and plot
```

The full session is an offline replay. In particular, the fitted calibration
is also applied when replaying its initial 200-second interval; whole-replay
scores must not be described as a prospective online test.

### Source locations and optional training

- [predict_aligned_decoder.py](../src/neural_signal_decoder/predict_aligned_decoder.py):
  checkpoint loading, calibration, prediction, output and evaluation boundaries.
- [train_aligned_decoder.py](../src/neural_signal_decoder/train_aligned_decoder.py):
  model definitions, spike-count windows, normalization, reconstruction/KL losses,
  session adaptation and continuous latent encoding.
- [manifold_decoder.py](../src/neural_signal_decoder/manifold_decoder.py):
  the encoded LSTM, scaling and shared regression utilities.

The optional training module first fits reference dynamics using spike
reconstruction and supervised behavior loss, adapts other sessions using
spikes, then trains an encoded LSTM on aligned latents and paired targets.
It requires compatible paired training recordings. The supplied demonstration
loads self-trained weights; it does not retrain the checkpoint or reproduce
all historical multi-session results. Method inspiration and implementation
differences are recorded in [PROVENANCE.md](PROVENANCE.md).

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

## Example results

The [reference outputs](../outputs/reference/README.md) contain the prediction
CSV, evaluation summary, and figure for the bundled recording.
For comparisons, use the same input files, locked environment, calibration,
device, and evaluation interval. GPU/library changes can alter numerical
results slightly.
