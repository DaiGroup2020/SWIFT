# Changelog

## Unreleased

- Synchronized the external filter repair using an explicit cumulative `samples_seen` counter and added its large nonuniform-chunk regression test.
- Added installed command-line entry points and 19 regression tests, including degenerate PCA and input/output protection.
- Changed explicit output replacement to preserve recoverable backup directories.
- Added a tested dependency snapshot and a complete-demo asset/output verifier; simplified the README to installation and usage.

## 0.1.0 — 2026-09-06

- Prepared a clean research-code release from the standalone RAPID source.
- Added a Git-LFS-managed 60-second RHD demonstration input, fixed reference map, package metadata, tests, CI, and reproducibility documentation.
- Removed internal source paths from release model metadata and generated manifests.
- Corrected filter-state continuity across RHD chunks and made `--overwrite` replace the exact target result directory.
