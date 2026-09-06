# Input and output schema

## Inputs

### Intan RHD

RAPID accepts a genuine Intan RHD file with 128-sample data blocks and a 20 kHz sampling frequency. The reader uses the Intan extractor supplied by SpikeInterface. The bundled demonstration file is the supported reference input.

### Channel parameters

`ChannelParam_20260129.json` is a JSON object keyed by channel identifier. Each entry may contain:

```json
{"Enable": true, "MadValue": 9.2}
```

Only enabled channels present in the RHD are processed. `MadValue` determines the fixed threshold unless a channel has no value.

### No-boundary map

The direct-assignment CSV requires these columns:

```text
unit_id, channel, polarity, unit_index, is_residual, lower_uv, upper_uv
```

`is_residual` must be false for every row. Unit indexes must be consecutive in each channel/polarity group, and intervals must be contiguous with the required infinite outer bound.

## Outputs

Each demo run creates these result classes below the selected output root:

```text
channel/<rhd-stem>/
  channel_spikes.csv
  ChannelParam.json
  manifest.json
  unitcsv*/

unit/<rhd-stem>/
  unit_spikes.csv
  local_peak_no_boundary_map.csv
  manifest.json
  unitcsv*/
```

`channel_spikes.csv` records threshold and peak ticks, sample indexes, seconds, polarity, and ADC values. `unit_spikes.csv` retains those fields and adds the final unit assignment and bin bounds. Per-channel and per-unit CSVs retain explicit empty files when the corresponding channel or unit has no events.

The demo omits filtered traces by default; add `--write-band-dat` to save `band-<channel>.DAT` and `time.DAT`. The lower-level `channel` and `all-days` commands save those files by default and offer `--no-band-dat` to omit them. Filtered channel traces are little-endian signed 16-bit ADC counts, with a scale of 0.195 microvolts per count. `time.DAT` contains little-endian signed 32-bit timestamp ticks.

Manifests contain file names, SHA-256 input fingerprints, settings, and event counts. Their creation timestamps vary between runs. The [reproducibility guide](REPRODUCIBILITY.md) records the expected demonstration CSV hashes and counts.

Existing session directories are rejected unless `--overwrite` is supplied. With that flag, previous directories are renamed to sibling `<session>.backup-<UUID>` folders before new results are written. Outputs that overlap required inputs are rejected. The `all-days` command also rejects recordings whose names would map to the same output directory.
