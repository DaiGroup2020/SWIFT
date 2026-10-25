# License scope for release 0.2.1

Copyright (C) 2026 DaiGroup.

The project owner has selected the GNU General Public License, version 3 only
(`GPL-3.0-only`), for the DaiGroup-owned source, documentation, and bundled
demonstration assets listed below. The full terms are in [LICENSE](LICENSE).
This grant applies to the rights DaiGroup holds in these materials; it does
not create third-party software, data, or patent rights.

| Material | Location | License |
| --- | --- | --- |
| RAPID source and documentation | `rapid/rapid/`, `rapid/scripts/`, `rapid/docs/` | GPL-3.0-only |
| Prediction source and documentation | `model-prediction/src/`, `model-prediction/scripts/`, `model-prediction/docs/` | GPL-3.0-only |
| RAPID demo recording | `rapid/data/M06_20260129_L_260129_161328_60s.rhd` | GPL-3.0-only |
| RAPID channel parameters | `rapid/data/ChannelParam_20260129.json` | GPL-3.0-only |
| RAPID fixed reference map | `rapid/models/fpga14_local_peak_no_boundary_tmpl0129.csv` | GPL-3.0-only |
| Prediction demo spike timestamps and voltage targets | `model-prediction/data/demo/` | GPL-3.0-only |
| Self-trained decoder weights and historical training metadata | `model-prediction/models/top64_local_peak/` | GPL-3.0-only |
| Reference outputs | `rapid/outputs/reference/`, `model-prediction/outputs/reference/` | GPL-3.0-only |

Third-party dependencies are installed separately, not relicensed by this
table, and remain subject to their respective licenses. The prediction
[provenance record](model-prediction/docs/PROVENANCE.md) distinguishes method
inspiration, implementation origin, and historical training records.
