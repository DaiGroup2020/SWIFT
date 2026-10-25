# Prediction source review

## Scope and result

Review date: 2026-10-25. The local prediction release was compared with the
[official NoMAD source at commit aa3c8faadf053d0b22160458854c076dcb13eb7e](https://github.com/snel-repo/nomad/tree/aa3c8faadf053d0b22160458854c076dcb13eb7e).
All 42 Python files listed by that commit's recursive tree were fetched for
read-only comparison; upstream code was neither executed nor added to this
package. Local coverage was 11 Python files: five package modules, three
scripts, the demo wrapper, and two test modules.

- No identical contiguous blocks of eight or more nonempty, stripped lines
  were found after removing comment-only lines.
- No identical function ASTs containing at least 50 nodes were found. This
  comparison ignores formatting, but does not normalize renamed identifiers
  or remove docstrings.
- Absolute imports were inspected across those local files. None import
  `nomad`, `lfads_tf2`, or `tensorflow`; neither the direct dependency list nor
  the tested lock file lists those packages.
- The runtime path was inspected: it constructs the local PyTorch
  `ReferenceDynamicsModel`, loads its state from the supplied checkpoint,
  fits the local session adapters, and runs the local LSTM decoder. No call
  to the official NoMAD runtime was identified in that path.

This screening found no direct reuse evidence within its stated scope. It
cannot prove independent development or exclude shorter fragments,
translations, rewritten logic, other versions, or other third-party sources.
It is separate from the authors' origin declaration in
[PROVENANCE.md](PROVENANCE.md), and is not a legal or patent clearance.

## Local implementation and dependencies

| Component | Implementation used by this release |
| --- | --- |
| Reference dynamics | Local `ReferenceDynamicsModel`: linear read-in, layer normalization, GRU, and rate/behavior heads in `train_aligned_decoder.py`. |
| Session adaptation | Local residual alignment network and observation adapters; Poisson reconstruction, Gaussian latent-distribution KL, and identity losses. |
| Prediction | Local checkpoint loading and calibration in `predict_aligned_decoder.py`, with decoder construction in `manifold_decoder.py`. |
| Numeric operations and data | NumPy and pandas. |
| Neural-network computation | PyTorch. |
| Exploratory manifold utilities | scikit-learn FactorAnalysis in the local shared module. |
| Figures | Matplotlib. |

Dependency names here identify runtime libraries, not bundled third-party
source code or a completed dependency-license audit. Exact tested package
versions are in [requirements-lock.txt](../requirements-lock.txt).

Before version 0.2.0, local filenames and checkpoint metadata included
historical identifiers such as `nomad_lstm` and `nomad_style_dynamics`.
Version 0.2.0 uses `train_aligned_decoder.py` and `predict_aligned_decoder.py`
for the public modules and neutral identifiers for new exports. The released
checkpoint and its historical training summary are unchanged; legacy family
and decoder tags remain accepted for compatibility. These identifiers and
their renaming do not establish code origin or change third-party rights.

## Weight and data checks

The authors confirm that they trained the released checkpoint themselves.
The bundled `summary.json` records its local training configuration and
historical results. `scripts/verify_demo.py` checks hashes for the released
checkpoint and two demo inputs; matching hashes establish file identity, not
the historical training process or distribution permission. Training logs,
input provenance, and the applicable asset permissions should be retained as
described in [PROVENANCE.md](PROVENANCE.md).

The source comparison above did not change the model architecture or
checkpoint. The subsequent 0.2.0 release renamed the modules and selected
GPL-3.0-or-later for DaiGroup-owned software; this is not a new source-origin
or patent-clearance determination.
