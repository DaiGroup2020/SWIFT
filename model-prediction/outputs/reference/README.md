# Prediction example outputs

These files were produced on 2026-10-25 with Python 3.10.11 and the locked
CPU environment and are included unchanged for release 0.2.1.

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

To regenerate them, install the project as described in the README and run
the normal demo into a separate directory from `model-prediction/`:

```powershell
python scripts/run_demo.py --device cpu --output-dir outputs/demo_prediction
```

Small numerical differences can occur across devices and library versions;
PNG bytes can also differ across rendering versions.
These results are licensed under GPL-3.0-only; see [NOTICE](../../NOTICE).
