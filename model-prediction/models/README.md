# Released model

`top64_local_peak/manifold_lstm_decoder.pt` is the pretrained checkpoint used
by the bundled demo. It is stored with Git LFS and expects 213 identified Unit
columns. See the [model card](top64_local_peak/MODEL_CARD.md) for its dimensions
and evaluation protocol.

`summary.json` records the historical model-training run. Those historical
multi-session scores are separate from the results produced by this demo.
Historical local path strings inside the frozen checkpoint are unused
provenance metadata; prediction uses the input and output paths supplied at
runtime.
