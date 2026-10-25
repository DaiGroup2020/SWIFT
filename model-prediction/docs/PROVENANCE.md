# Method attribution and provenance

## Independent implementation

The project authors confirm that DaiGroup developed this prediction code
independently with reference to the NoMAD paper, without copying, translating,
or adapting the official NoMAD source code. This statement records the authors'
declaration of origin. The authors further confirm that they did not download
the official NoMAD source code during development and trained the released
weights themselves. A limited source comparison is recorded in
[SOURCE_REVIEW.md](SOURCE_REVIEW.md); it is not a legal review.
This package is not an official release of NoMAD.

## Method reference

Karpowicz, B. M. et al. Stabilizing brain-computer interfaces through alignment
of latent dynamics. *Nature Communications* **16**, 4662 (2025).
https://doi.org/10.1038/s41467-025-59652-y

The paper motivates the use of a fixed reference latent-dynamics model and
spike-only adaptation of a session-specific mapping for stable decoding.
The method reference is also recorded in `references` in this project's
[CITATION.cff](../CITATION.cff). The accompanying SWIFT manuscript remains the
preferred citation for this release.

## Implementation in this release

The published NoMAD method uses an LFADS-based architecture. This package
implements its own model in PyTorch rather than loading the official
TensorFlow/LFADS implementation:

- `ReferenceDynamicsModel` uses a linear read-in, layer normalization, a GRU,
  and rate and behavior readout heads.
- `ResidualAlignmentNet` and session-specific observation adapters fit the
  spike-only alignment while the reference dynamics remain fixed.
- An encoded LSTM decodes the aligned latent sequence into two-dimensional
  velocity. The bundled example uses 213 identified units, 30-ms bins, and
  51-bin decoding windows.
- The bundled replay calibrates on the first 200 seconds of spikes. Voltage
  targets are used for evaluation after prediction; the summary separates
  whole-replay and post-calibration metrics.

These details describe this implementation, not a claim of numerical or
architectural equivalence to the original NoMAD software. See the
[model card](../models/top64_local_peak/MODEL_CARD.md) and
[reproducibility guide](REPRODUCIBILITY.md) for the released configuration.

## Source and asset records

| Item | Available source information | Record to retain or complete |
| --- | --- | --- |
| Prediction source code | Authors' independent-development declaration above; software contributors in `CITATION.cff`; limited source and import review in `SOURCE_REVIEW.md` | Preserve original development files, dated revisions, and a list of third-party code, if any. |
| Method inspiration | NoMAD paper and DOI above | Cite the paper in publications and describe the actual implementation. |
| Bundled checkpoint | Authors confirm that they trained the released weights themselves; configuration and historical training results in the model card and `summary.json`; checksum verified by `scripts/verify_demo.py` | Retain training scripts, data provenance, training logs, and the applicable permission to distribute weights. The authors' declaration is distinct from independently auditing the historical training run. |
| Demo spike CSV and voltage log | File descriptions in `data/demo/README.md`; input checksums in `scripts/verify_demo.py` | Retain recording provenance and the applicable permission to distribute the data. |
| Python dependencies | Direct dependencies in `pyproject.toml`; exact tested versions in `requirements-lock.txt` | Keep dependency license notices as required when redistributing dependencies. |

The table identifies existing records and remaining release checks. It does not
assert that missing historical records or asset permissions have been supplied.

## Licensing status

On 2026-10-25 the project authors selected GNU GPL version 3 or, at the user's
option, any later version (`GPL-3.0-or-later`) for the DaiGroup-owned software;
see [LICENSE](../LICENSE) and [NOTICE](../NOTICE). This selection does not claim
a license from the NoMAD rights holders or a completed patent review. Academic
citation, renaming, and a statement of independent implementation do not
themselves grant third-party rights. Dependency licenses and separate asset
permissions remain applicable.
