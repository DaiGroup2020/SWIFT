# Local-peak top64 model

## Checkpoint

| Property | Value |
|---|---|
| File | `manifold_lstm_decoder.pt` |
| Family | NoMAD-style latent dynamics, spike-only alignment, encoded LSTM decoder |
| Input | 213 identified spike units; names stored in checkpoint `unit_cols` |
| Output | Two-dimensional velocity (`vx`, `vy`) |
| Bin size | 30 ms |
| Decoder sequence | 51 bins; 1,500 ms from first to last bin center |
| Latent dimension | 176 |
| Decoder | Two LSTM layers, hidden size 144, dropout 0.15 |
| Reference day | 2026-01-27 |

`top64` denotes upstream selection of 64 channels. Multiple units from those
channels make up the 213 input columns; the demo does not reduce them to 64
Unit columns. Input identities are matched by name and missing names cause an
error unless an explicit mapping option is requested.

## Bundled demo protocol

The first 200 seconds of spikes fit a fresh session-alignment module. The
saved reference dynamics and decoder then produce the whole-session replay.
Matching voltage is used only for evaluation after predictions are computed.

The summary distinguishes whole-replay metrics from post-calibration metrics.
The latter retain only windows whose first bin is later than the calibration
end; recurrent history remains continuous. See
[REPRODUCIBILITY.md](../../docs/REPRODUCIBILITY.md) for the exact selection rule.

## Historical training record

The accompanying `summary.json` records the original training configuration
and results. That record used every tenth decoder window for training/fit
validation, the first 300 seconds of 2026-01-29 in training, reference/alignment
dropout 0.25, and decoder dropout 0.15.

Its reported aggregate over 13 future sessions was RMSE 0.620843, MAE 0.482758,
and R2 0.764512. These are historical model-training results, not scores
recomputed by the bundled 240-second demo. Current demo validation belongs in
[DEPLOYMENT_VALIDATION.md](../../docs/DEPLOYMENT_VALIDATION.md).
