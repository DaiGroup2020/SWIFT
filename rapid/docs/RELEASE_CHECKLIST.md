# Release checklist

- [x] Set the software copyright holder and package author to `DaiGroup` and the software license to `GPL-3.0-or-later`; retain LICENSE and NOTICE, and record named software contributors and manuscript authors in `CITATION.cff`.
- [x] Add the GitHub repository URL, manuscript title, author order, and correspondence details to `CITATION.cff`.
- [ ] Add final publication details and manuscript/archival identifiers to `CITATION.cff` when available.
- [ ] Run `git lfs install` and verify that the demonstration RHD is managed by LFS.
- [ ] From a fresh checkout, follow the README installation instructions and run both the demo and its verification command.
- [ ] Review `git status` so generated results, backups, environment folders, and private local paths are excluded.
- [ ] Push the source and LFS objects, then confirm that a new clone retrieves the complete demonstration recording.
- [ ] Create a version tag and archive the release with a DOI when needed.

The bundled demonstration RHD is included for public distribution as confirmed by the project owner. The current local validation record is in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
