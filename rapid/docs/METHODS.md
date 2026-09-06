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

## Direct unit assignment

Each detected channel event has a channel, polarity, and peak amplitude in ADC counts. The fixed CSV map defines contiguous intervals for each channel/polarity pair. Positive intervals include their lower bound and exclude their upper bound; negative intervals exclude their lower bound and include their upper bound. The outer positive interval extends to `+inf` and the outer negative interval extends to `-inf`.

Consequently, every map-covered event is assigned immediately to one final `unit_id`. RAPID creates neither a residual class nor a later merge operation.

## Fixed map and optional PCA command

The demonstration loads the fixed reference map directly; it requires no fitting step. The optional `reference-pca` command builds a separate exploratory PCA-based model from channel events and filtered waveform snippets. It is not part of the demonstration and does not recreate the supplied fixed CSV map.

To use this optional command, first save filtered traces with `--write-band-dat` when running the demo. Then supply the resulting channel directory:

```powershell
python -m rapid reference-pca --channel-result outputs/demo-60s/channel/M06_20260129_L_260129_161328_60s --model outputs/reference-pca/model.json
```

This writes a JSON model, an `_audit.csv` file, and a `_templates.npz` file. PCA and clustering use a fixed seed. A single waveform or identical snippets are represented as one group with finite, zero explained-variance ratios. Existing artifacts require `--overwrite`, which preserves them as sibling backup files.
