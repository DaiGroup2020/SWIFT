# Demonstration data and code

This repository provides the RAPID source code, a fixed reference map, channel configuration, and the 60-second demonstration RHD needed for the included example. The project owner has confirmed that the RHD may be included in the public GitHub repository.

The RHD is managed through Git LFS. After cloning, run `git lfs pull` to retrieve its complete contents. File details and the input digest are in [data/README.md](../data/README.md); the fixed map is described in [models/README.md](../models/README.md).

Generated results are written locally under the selected output directory. The [README](../README.md) gives the complete installation, run, and verification commands. The [reproducibility guide](REPRODUCIBILITY.md) records the tested environment and expected results.

Complete the author and repository identifiers in `CITATION.cff` before publication.
