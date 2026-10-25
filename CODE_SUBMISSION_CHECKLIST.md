# Code and Software Submission Checklist: repository map

This map describes the current release, 0.2.1. Corresponding authors are Teng
Gao and Xiaochuan Dai, as recorded in [CITATION.cff](CITATION.cff).

| Checklist item | Repository evidence |
| --- | --- |
| Source code and version | RAPID `rapid/rapid/`, prediction `model-prediction/src/`; version 0.2.1 in package metadata and CITATION.cff |
| Small real demonstration dataset | [RAPID data](rapid/data/README.md), [prediction data](model-prediction/data/demo/README.md); large inputs and weights use Git LFS |
| Dependencies, operating system and tested versions | Each project README and `requirements-lock.txt` |
| Non-standard hardware | Each README explicitly states that the bundled host-software demo needs no FPGA board or dedicated GPU |
| Installation instructions and typical install time | [RAPID README](rapid/README.md), [prediction README](model-prediction/README.md); download time is separate |
| Demo instructions, expected output and expected runtime | The same READMEs, verifier commands and recorded validation results |
| Precomputed example results | [RAPID reference outputs](rapid/outputs/reference/README.md), [prediction reference outputs](model-prediction/outputs/reference/README.md) |
| Use with compatible user data | [RAPID input/output schema](rapid/docs/INPUT_OUTPUT_SCHEMA.md), [prediction input format](model-prediction/docs/DATA_FORMAT.md), README examples and CLI help |
| Software and bundled-asset license | GNU GPL version 3 only (`GPL-3.0-only`); [LICENSE](LICENSE), project NOTICE files and [asset license table](ASSET_LICENSES.md) |
| Public code location | https://github.com/DaiGroup2020/SWIFT |
| Detailed functionality and pseudocode | [RAPID computational methods](rapid/docs/METHODS.md), [prediction functional description](model-prediction/docs/REPRODUCIBILITY.md#functional-description-and-pseudocode) |
| Optional manuscript-wide quantitative reproduction | Not claimed: these are two independent recorded-data demonstrations, not full reproduction of all manuscript results |

## Wording for the form

License:

> GNU General Public License, version 3 only (GPL-3.0-only), for DaiGroup-owned
> SWIFT source code and bundled demonstration data, reference maps, pretrained
> weights and reference outputs. Third-party dependencies retain their own licenses.

For "Elsewhere (specify)" use the repository documentation locations:

- RAPID: https://github.com/DaiGroup2020/SWIFT/blob/main/rapid/docs/METHODS.md
- Prediction: https://github.com/DaiGroup2020/SWIFT/blob/main/model-prediction/docs/REPRODUCIBILITY.md#functional-description-and-pseudocode

These links identify documentation in GitHub; they do not state that the same
description appears in the manuscript. If the editor requires a manuscript
methods description, that remains a separate editorial matter. This map is
not an assertion of acceptance or a substitute for the journal's checklist.

The checklist asks for expected output; it does not prescribe a directory
named `output` or require all intermediate files to be uploaded. This release
additionally includes verified reference files under each project's `outputs/`
directory. Normal runs generate a separate result tree without overwriting
those references. See [Nature's guidance](https://www.nature.com/documents/GuidelinesCodePublication.pdf).
