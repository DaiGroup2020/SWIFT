# Release checklist

Use this checklist before publishing a tagged GitHub release or creating an
external archival deposit.

1. The software copyright holder and package author are DaiGroup. The selected
   software license is GPL-3.0-or-later; retain LICENSE and NOTICE. CITATION.cff
   records named software contributors, the manuscript authors and title,
   correspondence details, and the repository URL. Add final publication
   details and DOI when available.
2. Confirm that the compact demo data and checkpoint are authorized for public
   redistribution and that their license and access conditions are documented.
3. Install the locked Python 3.10 environment, run all tests, run the bundled
   demonstration, and check its outputs with scripts/verify_demo.py.
4. Confirm that git lfs ls-files lists the checkpoint and compact demo data,
   and that the corresponding LFS objects have been pushed.
5. Confirm that generated outputs, credentials, local IDE settings, and models
   other than the released checkpoint are excluded by .gitignore.
6. Record the release tag, dependency versions, input provenance, expected
   runtime, and hardware in the manuscript's Code and Data Availability
   statements.
7. Archive the immutable release in an approved DOI-minting repository if the
   accompanying manuscript requires one.
8. Follow [PROVENANCE.md](PROVENANCE.md): preserve the original development
   history, retain the NoMAD method citation, and complete the checkpoint,
   data, and third-party source records before finalizing release licensing.
   The authors have confirmed independent development without downloading
   official NoMAD source and self-trained weights. The limited source and
   dependency review is recorded in [SOURCE_REVIEW.md](SOURCE_REVIEW.md).
