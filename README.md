# SWIFT

This repository contains two independent demonstrations:

| Project | Demonstration |
| --- | --- |
| [RAPID](rapid/README.md) | Detect channel events and assign units from the included 60-second Intan RHD recording. |
| [Neural Signal Prediction](model-prediction/README.md) | Run the included pretrained decoder on a 240-second example and save predictions, evaluation results, and a figure. |

## Download

Install Git and Git LFS, then run:

```powershell
git lfs install
git clone https://github.com/DaiGroup2020/SWIFT.git
cd SWIFT
git lfs pull
```

The recording, prediction data, and checkpoint use Git LFS. Their full contents are needed to run the demonstrations.

## Install and run a project

From the SWIFT root directory, enter `rapid/` with `cd rapid`, or enter `model-prediction/` with `cd model-prediction`. Follow the selected project's README to create its environment, install its dependencies, run the demonstration, and verify the outputs.

Each project uses its own virtual environment and dependency lock file. Use a separate terminal session for each project. The project READMEs and documentation contain the tested environment, expected results, citation metadata, and license.

## Automated checks

The [RAPID workflow](.github/workflows/rapid.yml) and [prediction workflow](.github/workflows/model-prediction.yml) run their respective data-free tests. Manually enabling `full_demo` also downloads the LFS assets, runs the selected complete demonstration, and verifies its outputs.
