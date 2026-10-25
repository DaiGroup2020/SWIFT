# Computational methods

## Intended replay

RAPID replays a fixed 20 kHz channel-based processing path for Intan RHD recordings. The bundled example detects events and assigns them using the supplied channel configuration and interval map. The measured CPU replay time is recorded separately in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Channel path

For each RHD block, RAPID selects channel identifiers that are both present in the recording and enabled in `ChannelParam_20260129.json`. It then applies the following stages in order:

1. Mean common-average reference across selected channels at each sample.
2. A causal, two-sample moving average.
3. Signed-width fixed-point high-pass IIR arithmetic.
4. Peak/valley detection with a three-sample latency and a per-channel amplitude threshold.
5. One maximum-absolute candidate per 30-sample selection window.

Thresholds use the channel `MadValue` multiplied by `-5` by default. RAPID rounds away from zero first in microvolts and then again after conversion to ADC counts. Missing `MadValue` values use the configured 50 microvolt default.

The moving-average, IIR, and detector state are preserved across RHD chunks. Therefore splitting a recording with `--chunk-blocks` changes memory use but not the calculated channel-event sequence.

### Fixed-point arithmetic and peak scheduling

The common average is an integer channel sum divided by the number of enabled
channels. Subtraction wraps to signed 16-bit counts. The two-sample moving
average uses an arithmetic right shift; only the first sample of the recording
is a warm-up zero. For the IIR, `wrap20` denotes signed 20-bit modular wrapping
and `>>` is an arithmetic right shift:

```text
x0 = wrap20(moving_average << 4)
d  = wrap20(x0 - 2*x1 + x2)
u  = wrap20(d - (d >> 4))
v  = wrap20(2*y1 - y2 + (wrap20(y2 - y1) >> 3) - (y1 >> 7))
y0 = wrap20(u + v)
filtered_sample = wrap16(y0 >> 4)
shift x0/x1 and y0/y1 into the persistent state
```

The deployed selector tracks rising/falling direction with a 16-count jitter
threshold. A detected turning point is registered three samples after its
peak. For each channel, the 30-sample windows retain the maximum absolute
candidate, breaking ties by the earliest peak sample, subject to the selector's
previous-window phase rule. Window finalization waits for the three-sample
detector delay, and the final pending windows are flushed at end of input.
These are software arithmetic/scheduler rules, not measured hardware latency.

## Direct unit assignment

Each detected channel event has a channel, polarity, and peak amplitude in ADC counts. The fixed CSV map defines contiguous intervals for each channel/polarity pair. Positive intervals include their lower bound and exclude their upper bound; negative intervals exclude their lower bound and include their upper bound. The outer positive interval extends to `+inf` and the outer negative interval extends to `-inf`.

Consequently, every map-covered event is assigned immediately to one final `unit_id`. RAPID creates neither a residual class nor a later merge operation.

## Demo pseudocode and source locations

```text
validate the RHD header, 20-kHz sample rate and channel configuration
select channels that are present in the recording and enabled by the configuration
quantize each channel threshold and initialize persistent filter/selector state
for each RHD block:
    read signed ADC counts and their recording-clock timestamps
    subtract the integer common average across enabled channels
    update the two-sample moving average and fixed-point IIR
    update peak/valley candidates and finalize ready selection windows
flush the remaining selection windows at end of recording
write detected events, manifests and per-channel timestamp exports
for each event:
    find its channel/polarity interval in the fixed reference map
    assign the corresponding unit_id
write assignments, manifests and per-unit timestamp exports
```

The implementation is in [preprocessing.py](../rapid/preprocessing.py)
(quantization and filter state), [channel_sorting.py](../rapid/channel_sorting.py)
(`FPGAPeakSelector` and channel exports), [local_peak_no_boundary.py](../rapid/local_peak_no_boundary.py)
(fixed-map loading), and [unit_sorting.py](../rapid/unit_sorting.py)
(assignment and unit exports). The entry point is
[run_demo.py](../scripts/run_demo.py); [verify_demo.py](../scripts/verify_demo.py)
checks the expected example outputs.

The bundled replay produces 439,597 detected events and the same number of
unit assignments, with 591 catalogue units and 575 nonempty units. Main CSVs
and manifests are included in [outputs/reference](../outputs/reference/README.md).
The reference timestamps preserve the recording clock, including any negative
pre-trigger values. Source fingerprints and settings are stored in manifests.

This repository provides the host-computer software replay only. It does not
contain FPGA HDL, vendor build projects, bitstreams, or a hardware-validation
claim. The channel and prediction demonstrations are independent.

## Fixed map and optional PCA command

The demonstration loads the fixed reference map directly; it requires no fitting step. The optional `reference-pca` command builds a separate exploratory PCA-based model from channel events and filtered waveform snippets. It is not part of the demonstration and does not recreate the supplied fixed CSV map.

To use this optional command, first save filtered traces with `--write-band-dat` when running the demo. Then supply the resulting channel directory:

```powershell
python -m rapid reference-pca --channel-result outputs/demo-60s/channel/M06_20260129_L_260129_161328_60s --model outputs/reference-pca/model.json
```

This writes a JSON model, an `_audit.csv` file, and a `_templates.npz` file. PCA and clustering use a fixed seed. A single waveform or identical snippets are represented as one group with finite, zero explained-variance ratios. Existing artifacts require `--overwrite`, which preserves them as sibling backup files.
