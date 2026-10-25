# Changelog

## 0.2.0 — 2026-10-25

- Selected GPL-3.0-or-later for DaiGroup-owned software and synchronized license metadata.
- Renamed the prediction and training modules to describe latent-dynamics alignment; updated imports and the installed command entry point.
- Retained method attribution and unchanged released checkpoint bytes; old family tags remain accepted for compatibility, while new exports use neutral tags.
- Added regression tests for current and legacy checkpoint families and decoder aliases, plus renamed-module command checks.

- Enforced checkpoint Unit identities by default and added explicit overrides for alternate mapping.
- Added LFS/input validation, output replacement protection, spike-only evaluation skipping, and configuration precedence fixes.
- Added separately labeled post-calibration scores while preserving whole-demo predictions and metrics.
- Added 16 regression tests, a dependency snapshot, and a complete-demo verifier; simplified the README to installation and usage.

## 0.1.0

- Initial prediction-only research release.
- Released top64 local-peak checkpoint and compact validation demo.
- Added public command-line smoke tests, continuous integration, and
  reproducibility documentation.
