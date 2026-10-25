# Verified RAPID reference outputs

These files were produced by the full bundled 60-second software replay on
2026-10-25 with Python 3.10.11, the locked dependencies, and
`--chunk-blocks 64`, without filtered-trace output. The full
result tree passed `scripts/verify_demo.py` before these files were copied
unchanged into this compact reference directory for release 0.2.1.

| File | Expected content |
| --- | --- |
| `channel_spikes.csv` | All 439,597 detected channel events |
| `unit_spikes.csv` | All 439,597 unit assignments; 591 catalogue units, 575 nonempty |
| `channel_manifest.json` | Input fingerprint, channel count, and processing settings |
| `unit_manifest.json` | Input/map fingerprints and assignment counts |

The CSVs use Git LFS. Fetch them with `git lfs pull`, then run from `rapid/`:

```powershell
python scripts/verify_demo.py --reference-only --results-root outputs/reference
```

This checks the source assets, exact main CSV hashes and manifest fields.
It does not check the per-channel/per-unit export files, which are omitted
here to avoid distributing thousands of redundant files. Generate and verify
the complete result tree separately:

```powershell
python scripts/run_demo.py --results-root outputs/demo-60s --chunk-blocks 64
python scripts/verify_demo.py --results-root outputs/demo-60s
```

CSV SHA-256 values:

- Channel: `c3235272b0a88100daedc2b95113380a87262c3e90682e282cd3de565560374e`
- Unit: `6c2b2046e414384557afe6094d368a6836026db2a6840c12ee54f6e8a0cee71a`

Recorded timestamps are retained from the RHD clock and can include negative
pre-trigger values. Normal demo runs do not write to this reference directory.
These results are licensed under GPL-3.0-only; see [NOTICE](../../NOTICE).
