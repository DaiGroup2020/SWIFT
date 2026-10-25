# RAPID demonstration validation

This page records validation of the currently distributed software replay and
its included reference outputs. Detailed functionality and pseudocode are in
[METHODS.md](METHODS.md).

## Reference environment and installation

The full 60-second demo was validated on Windows 11 Pro x64 (build 22631),
Python 3.10.11, and an Intel Core Ultra 9 285K. The isolated environment used
the versions in [requirements-lock.txt](../requirements-lock.txt): NumPy 2.0.2,
Numba 0.60.0, scikit-learn 1.6.1, SpikeInterface 0.103.2, and Neo 0.14.3.
Package installation, dependency checks, and installed command help passed.

Dependency installation took approximately 129.18 seconds, excluding input
downloads. A complete software replay with `--chunk-blocks 64` took 40.06
seconds on the tested desktop. These timings are examples, not hardware
throughput or performance guarantees. No FPGA board, GPU or acquisition
hardware is required; FPGA HDL, build projects and bitstreams are not included.

## Generate the complete outputs

From `rapid/` after installation:

```powershell
python -m unittest discover -s tests -v
python scripts/run_demo.py --results-root outputs/demo-60s --chunk-blocks 64
python scripts/verify_demo.py --results-root outputs/demo-60s
```

The complete demo produces:

| Check | Expected result |
| --- | --- |
| Detected channel events | 439,597 |
| Assigned unit events | 439,597 |
| Enabled channels / files per channel export | 191 |
| Catalogue units / files per unit export | 591 |
| Nonempty assigned units | 575 |
| Source RHD SHA-256 | `89b7a7b796dcfcec18c999f416cf9179abfbb60820cf88121e66fa367d3efd8c` |
| Channel CSV SHA-256 | `c3235272b0a88100daedc2b95113380a87262c3e90682e282cd3de565560374e` |
| Unit CSV SHA-256 | `6c2b2046e414384557afe6094d368a6836026db2a6840c12ee54f6e8a0cee71a` |

The independent verifier checks input fingerprints, main event tables,
manifests and all three per-channel/per-unit timestamp exports. Filtered
traces are not written by default; `--write-band-dat` adds them.

## Included compact reference

The main CSVs and manifests are committed in
[outputs/reference](../outputs/reference/README.md). Fetch their large CSVs
using Git LFS and check them independently:

```powershell
python scripts/verify_demo.py --reference-only --results-root outputs/reference
```

This mode checks assets and the exact main CSV/manifest values. It explicitly
does not claim to check omitted per-channel/per-unit exports. A full raw
result tree and this compact reference both passed their respective verifiers
on 2026-10-25.

## Current checks and scope

All 22 current data-free tests passed on 2026-10-25, including the three compact
reference checks, filter chunk continuity, quantization, map assignment, input
protection, degenerate PCA, and command help. This latest test-only rerun used
the existing Python 3.10.11 desktop environment (NumPy 2.1.3, Numba 0.65.1 and
scikit-learn 1.7.2); the recorded full reference run used the separate locked
environment above. No numerical processing algorithm changed in this
documentation/reference-output update.

No new test environment was created for the reference checks. Dedicated demo
validation environments have been removed. Generated full outputs remain
outside the repository; normal user runs are ignored by Git.

This is a software demonstration, not FPGA validation or full reproduction
of all manuscript results. The current code and bundled assets use
GPL-3.0-only; see [NOTICE](../NOTICE) and [asset licenses](../../ASSET_LICENSES.md).
