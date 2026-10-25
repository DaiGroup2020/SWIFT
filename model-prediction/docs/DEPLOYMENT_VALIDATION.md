# Prediction demonstration validation

The example uses the released checkpoint, the 240-second spike CSV, and the
matching voltage log. Calibration uses only the first 200 seconds of spikes.
The detailed functionality and pseudocode are in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md#functional-description-and-pseudocode).

## Reference environment and timing

The full demo was validated in an isolated Python 3.10.11 environment on
Windows 11 Pro x64 (build 22631), using an Intel Core Ultra 9 285K and the
unchanged [requirements-lock.txt](../requirements-lock.txt), including
PyTorch 2.8.0+cpu, NumPy 2.1.3, pandas 2.3.3, Matplotlib 3.10.8, and
scikit-learn 1.7.2. Package build/installation, `pip check`, and the installed
command entry point passed; LICENSE and NOTICE were included in the package.

Base dependency preparation took 112.35 seconds; installing the verified
PyTorch wheel and project took another 28.23 seconds, excluding network
transfer. Total measured preparation was 140.58 seconds. The CPU wheel is
approximately 619 MB. A complete demo took 74.00 seconds on the tested
desktop. Network time, processor and numerical libraries affect these values.

## Generate and check the complete demo

From `model-prediction/` after installation:

```powershell
python -m unittest discover -s tests -v
python scripts/run_demo.py --device cpu --output-dir outputs/demo_prediction
python scripts/verify_demo.py --output-dir outputs/demo_prediction
```

Use a new output directory or `--overwrite` for another run.

| Evaluation interval | Rows | RMSE | MAE | R2 |
| --- | ---: | ---: | ---: | ---: |
| Whole replay, endpoints 1.515-239.985 s | 7,950 | 0.39822554 | 0.30646890 | 0.88841956 |
| Non-overlapping post-calibration windows, endpoints 201.525-239.985 s | 1,283 | 0.44549288 | 0.34432911 | 0.82825918 |

All 213 unit identities match. The whole-replay score includes the calibration
period and must not be described as a prospective post-calibration result.
The separate future-window metric excludes decoder windows overlapping the
calibration interval but retains continuous recurrent history.

The verifier checks three input hashes, every prediction row and timestamp,
summary/CSV consistency, expected metrics and the PNG signature. It uses
tolerance `2e-6` for CSV-versus-summary metrics and `1e-4` for the reference
scores; rendering and floating-point differences need not be byte-identical.

## Included reference results

The [reference directory](../outputs/reference/README.md) contains the full
prediction CSV, summary JSON and PNG copied unchanged from the verified run.
Check these included files with the standard library only:

```powershell
python scripts/verify_demo.py --output-dir outputs/reference
```

The complete generated demo and the committed reference files both passed
independent verification on 2026-10-25. The frozen checkpoint remains
unchanged (SHA-256
`4f87529ee8250dab814387f77e72e2d349d70771f47cf31491a048836b2c3565`).

## Current checks and scope

All 21 current data-free tests passed on 2026-10-25, including renamed-module
help, current/legacy checkpoint tags, exact equivalence of decoder aliases,
unit mapping, output protection and evaluation boundaries. This latest
test-only rerun used the existing Python 3.10.11 desktop environment with the
same NumPy/pandas/Matplotlib/scikit-learn versions above and PyTorch
2.11.0.dev20251215+cu128. The published full reference results instead use the
stable CPU lock above. No numerical model algorithm changed in this
documentation/reference-output update.

The installed CLI also completed a short direct, spike-only check from
outside the checkout without voltage targets. The training parser accepted
`--latent-dim` with an explicit data directory; this was not a full retraining
or independent audit of historical training.

No new test environment was created for the reference checks. Dedicated demo
validation environments have been removed. The code and bundled assets use
GPL-3.0-only; see [NOTICE](../NOTICE) and [asset licenses](../../ASSET_LICENSES.md).
