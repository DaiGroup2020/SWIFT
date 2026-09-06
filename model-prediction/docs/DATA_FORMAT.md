# Prediction input format

Prediction requires a sparse spike CSV. A matching voltage log is optional and
is used only after prediction to calculate metrics and draw reference traces.

## Sparse spike CSV

Each column starting with `Unit` represents one identified unit. Entries are
finite, non-negative spike timestamps in seconds. Blank cells represent padding
for units with fewer events; they are not zero-time spikes. For example:

```csv
Unit226,Unit227,Unit228
0.014,0.008,0.021
0.042,0.055,
,0.088,0.119
```

This is a schema illustration. The released checkpoint requires all 213 names
in its `unit_cols` metadata, which match the bundled CSV. The loader restores
checkpoint column order by name and ignores extra Unit columns. `top64` refers
to the upstream selection of 64 channels; it does not mean 64 input units.

Missing expected names cause an error by default. For a deliberate experiment,
`--allow-missing-units` fills missing units with empty spike trains, provided at
least one expected name matches. `--allow-positional-unit-mapping` permits
first-N column mapping only when no names match and enough columns exist. Use
that option only when the correspondence has been independently established.
The JSON summary records the mapping used.

An eight-digit date in the filename, such as `Online_session_20260202.csv`,
allows automatic day-name inference. Otherwise supply `--day-name YYYY-MM-DD`.

## Optional voltage log

Supply a log explicitly with `--voltage-dat recording.dat`. If that option is
omitted, the command searches for matching files such as
`M06-2026-02-02_voltage.dat` beside the spike CSV. Use `--skip-evaluation` to
disable both loading and automatic discovery.

Each usable text line contains a timestamp and two target values:

```text
timestamp: 0.000000 voltage: 0.1200000 -0.0500000 pointpos: (0.0, 0.0) targetpos: (0.0, 0.0)
```

Only `timestamp` and the two values after `voltage:` are required. The position
fields are optional. Targets and spikes must share the same time origin and
seconds scale. Targets are interpolated to prediction times; times outside
the recorded target range are excluded from metrics rather than extrapolated.

## Calibration and output intervals

`--calibration-time-start` and `--calibration-time-end` select the spike-only
alignment interval; defaults are 0 and 200 seconds. `--time-start` and
`--time-end` optionally restrict the decoded endpoints written to the CSV.
Without those output limits, the demo writes a whole-session replay.

The summary's `evaluation.metrics` covers every decoded endpoint with valid
ground truth, including calibration when present. Its
`evaluation.post_calibration` separately scores windows whose first bin is
strictly after calibration ends. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md)
for the exact boundary rule.
