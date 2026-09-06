# Demo validation record

The runnable example uses the released checkpoint, the 240-second
2026-02-02 spike CSV, and its matching voltage log. Calibration uses the first
200 seconds of spikes only. All 213 checkpoint Unit identities are present.

## Current verification commands

Install the Python 3.10 locked environment described in the README, then run:

```powershell
python -m unittest discover -s tests -v
python scripts/run_demo.py --device cpu --output-dir outputs/demo_prediction
python scripts/verify_demo.py --output-dir outputs/demo_prediction
```

Use a new output directory or `--overwrite` for a repeat run. The 16 data-free
tests cover command availability, input identities, checkpoint loading,
output protection, and evaluation boundaries. The full-demo verification
checks the actual generated artifacts separately.

The whole-replay score includes the calibration period. The separate
`evaluation.post_calibration` score excludes all decoder windows whose first
bin is at or before 200 seconds; it preserves causal recurrent history.

## Validated stable environment — 2026-09-07

A new isolated environment was created without access to system packages:
Windows 11 Pro (build 22631), Python 3.10.11, Intel Core Ultra 9 285K,
and PyTorch 2.8.0+cpu. All other exact versions are in `requirements-lock.txt`.
Package installation, `pip check`, and all 16 regression tests passed.

The full README demo completed in **74.00 seconds**, generating the CSV,
summary, and PNG. All 213 input identities matched. The independent verifier
passed input hashes, all 7,950 finite prediction/target rows, the time grid,
summary consistency, whole-replay metrics, and post-calibration metrics.

| Evaluation interval | Rows | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Whole replay, endpoints 1.515–239.985 s | 7,950 | 0.398225541 | 0.306468904 | 0.888419564 |
| Non-overlapping post-calibration windows, endpoints 201.525–239.985 s | 1,283 | 0.445492875 | 0.344329089 | 0.828259196 |

The verifier recalculates metrics from the exported CSV. Absolute/relative
tolerances are `2e-6` for CSV-versus-summary checks and `1e-4` for reference
metrics, allowing small differences between CPU math libraries. Exact hashes
below document this run but are not required across platforms:

| Artifact | SHA-256 |
|---|---|
| `predictions_2026-02-02.csv` | `8B113826EC5B24DA0B362695414468BB7033DE2E60D3FE26F6506A2F9D40CD79` |
| `predictions_2026-02-02.png` | `F68646AA42FAE99B7CCA4EE8367DA323DBD1556845EBE3DE7C3A1DECD5DC264D` |

The non-editably installed `neural-signal-predict` command was also tested
outside the source checkout. A 10-second direct, spike-only smoke run produced
283 predictions with no voltage input, reference columns, or evaluation score.

Base dependency preparation took 112.35 seconds. Installing the verified
PyTorch wheel and this project took a further 28.23 seconds, excluding network
transfer. The CPU wheel is approximately 619 MB; download time depends on the
connection, and the validation download required a retry. Runtime measurements
are examples from this computer, not performance guarantees.

Normal GitHub Actions runs perform the data-free tests. The optional manual
`full_demo` job additionally retrieves LFS data, runs the complete demo, and
verifies outputs. The workflow was checked locally; it has not yet run on GitHub.

## Comparison with the earlier local environment

The earlier local check used the previously installed nightly PyTorch
environment, not the new stable PyTorch 2.8.0 CPU lock. Two runs produced
7,950 predictions over the whole replay with these metrics:

| RMSE | MAE | R2 |
|---:|---:|---:|
| 0.3982238945 | 0.3064672351 | 0.8884204930 |

These metrics include decoded endpoints within the calibration interval and
must not be labeled as post-calibration scores. Historical artifact hashes:

| Artifact | SHA-256 |
|---|---|
| `predictions_2026-02-02.csv` | `FBB977083078F531C85D963616DF4D0E83557586ACD0AD89C27471D1074FBE4D` |
| `predictions_2026-02-02.png` | `E1DF777A22930F960EB69B9FC7F8DFC58A1E6CB9CF6734C8A2F088E27A81A91D` |

These hashes document that earlier environment. Use the current demo reference
and verification command for the locked environment rather than treating the
historical image hash as a cross-platform requirement.

The updated code was also rerun in that existing environment on 2026-09-07.
It completed in 60.49 seconds and passed the independent demo verifier. The
small differences from the stable environment are within its tolerance.

## Earlier model-development results

The original development notes recorded a 2026-02-02 offline score of
RMSE 0.465512, MAE 0.343437, R2 0.840551 and a separate deployment run of
RMSE 0.419830, MAE 0.326186, R2 0.870491. These used different alignment or
recording settings. They are retained only as historical context and are not
the expected results of the bundled demo.
