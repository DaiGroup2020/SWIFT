# Demonstration reproducibility

Run commands on this page from the `rapid/` project directory. From the SWIFT
root directory, enter it with `cd rapid` first.

## Tested environment and installation

The demonstration was checked on 2026-09-07 in a new Windows 11 Pro environment (build 22631) with Python 3.10.11 and an Intel Core Ultra 9 285K CPU. Use the versions in `requirements-lock.txt`; the direct scientific dependencies are NumPy 2.0.2, Numba 0.60.0, scikit-learn 1.6.1, SpikeInterface 0.103.2, and Neo 0.14.3.

Follow the installation commands in the [README](../README.md), then run:

```powershell
python -m pip check
python -m rapid check-env
python -m unittest discover -s tests -v
```

All 19 tests passed. They cover filter continuity across unequal chunks, direct interval assignment, portable manifests, recoverable output replacement, input protection, PCA edge cases, duplicate session names, and command-line entry points. These tests use synthetic fixtures and the small included reference map; they do not read the demonstration RHD.

Dependency installation took 129.18 seconds in the tested environment. Download speed, package caches, and hardware affect installation time.

## Full demonstration

Run these commands from the `rapid/` project directory:

```powershell
python scripts/run_demo.py --results-root outputs/demo-60s
python scripts/verify_demo.py --results-root outputs/demo-60s
```

The demo reads the included RHD, channel configuration, and reference map. It generates channel events and direct unit assignments under the requested output directory. Add `--write-band-dat` to also save the filtered waveform traces.

Existing result directories are rejected by default. With `--overwrite`, each old result directory is renamed to a sibling `.backup-<UUID>` directory before replacement. Inputs must remain separate from outputs. The optional PCA and RHD clipping commands likewise preserve existing artifacts when explicit replacement is requested.

## Independently installed command

The package also supplies `rapid` and `python -m rapid`. Their output paths default to the current working directory. The source-checkout command `python rapid.py` remains available.

The 2026-09-07 validation used the installed package from outside the checkout and ran the equivalent `all-days` command against the bundled data, with `--chunk-blocks 256` and filtered-trace output enabled. It completed the full 60 seconds in 43.8 seconds on the test CPU, including channel detection and unit assignment. This is a measured example, not a runtime guarantee.

For example, from the `rapid/` project directory:

```powershell
python -m rapid all-days --input-root data --channel-param data/ChannelParam_20260129.json --model models/fpga14_local_peak_no_boundary_tmpl0129.csv --results-root outputs/all-days --chunk-blocks 256
```

This command saves filtered traces by default; add `--no-band-dat` to omit them. Each recording must have a distinct filename stem.

## Reference results

The full replay produced the following results, confirmed from the generated manifests and CSV files:

| Result | Expected value |
| --- | ---: |
| Recording duration | 60 seconds |
| Enabled channels | 191 |
| Channel events | 439,597 |
| Unit assignments | 439,597 |
| Units in the catalogue | 591 |
| Nonempty units | 575 |

The event CSV SHA-256 digests are:

```text
channel_spikes.csv
C3235272B0A88100DAEDC2B95113380A87262C3E90682E282CD3DE565560374E

unit_spikes.csv
6C2B2046E414384557AFE6094D368A6836026DB2A6840C12EE54F6E8A0CEE71A
```

These hashes also match the previous two full replays on 2026-09-06. Manifest creation timestamps vary between runs and are not expected to be byte-identical.

The README demo entry point was also run for all 60 seconds with `--chunk-blocks 64` and without filtered-trace output. It completed in 40.06 seconds and produced the same two CSV hashes. The verification script passed for both runs: input checksums, main CSV hashes, manifest counts, all six per-channel/per-unit export folders, and optional trace sizes. Neither run used data or modules from the original development folder.

Normal GitHub Actions runs perform the data-free tests. A manual `full_demo` run additionally downloads the LFS assets, executes the complete example, and verifies outputs. The workflow has been checked locally but has not yet been executed on GitHub.

## Optional chunk-continuity check

An earlier 0.9984-second clipped replay produced 7,279 channel events with both `--chunk-blocks 1` and `--chunk-blocks 64`. The channel CSV files were byte-identical, with SHA-256 `625A17EED45B8997CCCAD639B9AD4E7BE44D34EA7BBDD18FBD9183210B32D108`.

The corresponding unit assignment contained 7,279 events across 493 nonempty units, with CSV SHA-256 `03969B4E5F0B974DDDF008CD6E5468BA8915D72090242D21CDBE5DCB2E947F90`. The test suite also checks the synchronized 191-channel, 65,537-sample nonuniform filter regression.

## Documentation references

The short installation/demo/output layout follows examples such as [Kilosort4](https://github.com/MouseLand/Kilosort) (Nature Methods) and [speechBCI](https://github.com/fwillett/speechBCI) (Nature). Detailed descriptions are kept outside the README. [Nature's code publication guidance](https://www.nature.com/documents/GuidelinesCodePublication.pdf) informs the environment and run records; it does not prescribe a single README template.
