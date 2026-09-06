# RAPID

RAPID detects channel events and assigns them to units in 20 kHz Intan RHD recordings. This repository includes the code, a 60-second demonstration recording, channel parameters, and the fixed reference map needed to run the demonstration.

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

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead. The locked environment was tested on Windows with Python 3.10.11.

## Run the demonstration

From the `rapid/` project directory:

```powershell
python -m rapid check-env
python scripts/run_demo.py --results-root outputs/demo-60s
python scripts/verify_demo.py --results-root outputs/demo-60s
```

The included recording produces 439,597 channel events and 439,597 unit assignments. The verification command checks the demonstration results against the recorded reference.

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

See the [demonstration validation record](docs/REPRODUCIBILITY.md), [method details](docs/METHODS.md), and [input/output schema](docs/INPUT_OUTPUT_SCHEMA.md) for further information.
