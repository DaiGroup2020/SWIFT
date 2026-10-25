# RAPID

This package provides a Python software replay of RAPID's FPGA14 signal-processing workflow on recorded 20 kHz Intan RHD data. It includes a 60-second recording, channel parameters, and a fixed reference map for detecting channel events and assigning them to units.

The bundled demonstration runs on the host computer using Python, NumPy, and Numba. It does not execute on or program an FPGA board, or invoke an RTL simulator. FPGA HDL, hardware build projects, and bitstreams are not included in this package.

## System requirements

- Tested on Windows 11 Pro, 64-bit (build 22631), with Python 3.10.11 and an Intel Core Ultra 9 285K on 2026-09-07. macOS and Linux have not been validated for this release.
- Scientific dependencies: NumPy 2.0.2, Numba 0.60.0, scikit-learn 1.6.1, SpikeInterface 0.103.2, and Neo 0.14.3. All Python dependencies and exact versions are in [requirements-lock.txt](requirements-lock.txt).
- No FPGA board, GPU, acquisition hardware, or vendor FPGA toolchain is required for this offline demonstration.
- Git and Git LFS are needed to download the bundled recording (approximately 629 MB).

## Installation

Use Python 3.10. From the SWIFT root directory, enter this project with `cd rapid` first. Run the commands below from the `rapid/` project directory. The RHD recording is stored with Git LFS; the following commands retrieve it and install the demonstration environment.

```powershell
git lfs install
git lfs pull

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps --no-build-isolation .
```

Dependency installation took approximately 2 minutes on the tested desktop (129.18 seconds). Allow additional time to download the recording and packages; network speed and package caches affect the total setup time.

## Run the demonstration

From the `rapid/` project directory:

```powershell
python -m rapid check-env
python scripts/run_demo.py --results-root outputs/demo-60s --chunk-blocks 64
python scripts/verify_demo.py --results-root outputs/demo-60s
```

The included recording produces 439,597 channel events and 439,597 unit assignments. The verification command checks the demonstration results against the recorded reference.

Expected runtime is approximately 40 seconds on the tested desktop: this 60-second replay took 40.06 seconds with `--chunk-blocks 64` and no filtered-trace output. This is software replay time, not FPGA throughput or latency. The included checks validate the software outputs, not equivalence to a physical FPGA implementation.

## Results

Results are written under `outputs/demo-60s/`:

```text
channel/M06_20260129_L_260129_161328_60s/
  channel_spikes.csv
  manifest.json
  unitcsv*/
unit/M06_20260129_L_260129_161328_60s/
  unit_spikes.csv
  manifest.json
  unitcsv*/
```

The two CSV files contain the detected events and their unit assignments. The per-channel and per-unit folders contain event times; manifests record the inputs and settings.

Add `--write-band-dat` to the demo command to save filtered traces. To repeat a run in the same location, add `--overwrite`; existing result directories are retained beside the new results as `.backup-<UUID>` folders.

## Use your own recording

Supply a 20 kHz Intan RHD recording with 128-sample data blocks, its channel-parameter JSON, and a compatible no-boundary reference-map CSV. Replace the example paths below with your files:

```powershell
python scripts/run_demo.py --rhd "path/to/recording.rhd" --channel-param "path/to/ChannelParam.json" --model "path/to/reference_map.csv" --results-root outputs/my-recording --chunk-blocks 64
```

Channel identifiers, enabled channels, thresholds, and reference-map intervals must match your recording. The bundled parameters and map are specific to the example; do not assume they apply to other recordings. See the [input/output schema](docs/INPUT_OUTPUT_SCHEMA.md) for file formats. `verify_demo.py` checks only the bundled example and its expected results, not arbitrary recordings.

## License and further information

The DaiGroup-owned software is licensed under GNU GPL version 3 or, at your option, any later version (`GPL-3.0-or-later`); see [LICENSE](LICENSE) and [NOTICE](NOTICE). Third-party software retains its own licensing terms. See [CITATION.cff](CITATION.cff) for citation metadata, the [demonstration validation record](docs/REPRODUCIBILITY.md) for measured results, and [method details](docs/METHODS.md) for the processing steps.
