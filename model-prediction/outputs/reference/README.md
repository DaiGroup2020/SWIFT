# Verified prediction reference outputs

These files were produced on 2026-10-25 with Python 3.10.11 and the locked
CPU environment. They passed
`scripts/verify_demo.py` and were copied unchanged for release 0.2.1.

| File | Expected content |
| --- | --- |
| `predictions_2026-02-02.csv` | 7,950 timestamps, predictions, and matching voltage targets |
| `prediction_summary_2026-02-02.json` | Input hashes, calibration details, and evaluation scores |
| `predictions_2026-02-02.png` | Predicted and reference two-dimensional velocity traces |

The first 200 seconds of spikes are used for calibration. The complete replay
has RMSE approximately 0.39822554 and R2 approximately 0.88841956. The separate
1,283-point, non-overlapping post-calibration segment has RMSE approximately
0.44549288 and R2 approximately 0.82825918. It retains causal recurrent
history, as explained in [the protocol](../../docs/REPRODUCIBILITY.md).

From `model-prediction/`, check these included files without installing
scientific dependencies:

```powershell
python scripts/verify_demo.py --output-dir outputs/reference
```

To regenerate them, install the project as described in the README and run
the normal demo into a separate directory:

```powershell
python scripts/run_demo.py --device cpu --output-dir outputs/demo_prediction
python scripts/verify_demo.py --output-dir outputs/demo_prediction
```

The verifier compares numerical results within the documented tolerances;
PNG bytes are not required to match across rendering/library versions.
These results are licensed under GPL-3.0-only; see [NOTICE](../../NOTICE).
