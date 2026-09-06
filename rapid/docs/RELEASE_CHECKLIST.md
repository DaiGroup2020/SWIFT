# Release checklist

- [ ] Replace `Manuscript authors` in `LICENSE`, `CITATION.cff`, and `pyproject.toml`.
- [ ] Add the final GitHub repository URL and available manuscript/archival identifiers to `CITATION.cff`.
- [ ] Run `git lfs install` and verify that the demonstration RHD is managed by LFS.
- [ ] From a fresh checkout, follow the README installation instructions and run both the demo and its verification command.
- [ ] Review `git status` so generated results, backups, environment folders, and private local paths are excluded.
- [ ] Push the source and LFS objects, then confirm that a new clone retrieves the complete demonstration recording.
- [ ] Create a version tag and archive the release with a DOI when needed.

The bundled demonstration RHD is included for public distribution as confirmed by the project owner. The current local validation record is in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
