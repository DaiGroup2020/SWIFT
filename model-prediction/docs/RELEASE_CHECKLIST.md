# Release checklist

Use this checklist before publishing a tagged GitHub release or creating an
external archival deposit.

1. Replace the generic author entry in CITATION.cff and add the final
   repository URL, manuscript citation, and DOI when available.
2. Confirm that the compact demo data and checkpoint are authorized for public
   redistribution and that their license and access conditions are documented.
3. Install the locked Python 3.10 environment, run the 16 tests, run the bundled
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
