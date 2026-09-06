# Contributing

Thank you for helping improve this research release.

1. Do not add raw recordings, participant-sensitive material, secrets, or
   generated outputs artifacts to version control.
2. Keep each change focused and update the documentation when a command-line
   interface, input contract, model card, or result format changes.
3. Run python -m unittest discover -s tests -v before opening a pull request.
4. For a reproducible bug report, include the command, software versions,
   platform, and the smallest authorized input that demonstrates the issue.

Use Git LFS for the released checkpoint and compact demo assets. Do not replace
them with files from a different recording without updating the model card,
data documentation, and validation record.
