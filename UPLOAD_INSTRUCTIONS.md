# Upload and replacement instructions

This package is a **complete replacement of the files present in the uploaded `CIRRUS_Plus_Plus-main.zip`**. It is not a snapshot of every historical file on GitHub.

1. Keep a backup of your existing GitHub repository (or use git history).
2. If you want the repository to match this package **exactly**, remove obsolete tracked files first. Uploading files in the GitHub web interface does **not** automatically delete old files.
3. Upload the entire **contents of this folder**, including `.gitignore`, `docs/` and `paper_figures/`, to the repository root. Do not upload an extra enclosing folder.
4. Review the changed files before committing. Keep unrelated research artifacts if they are needed; do not delete them blindly.
5. Test installation and `streamlit run app.py` in a fresh Python environment, verify both paper demos, and check https://cirrus.fids.ur.de/ separately: updating GitHub does not automatically mean the hosted app has deployed the same commit.
6. A comprehensive PDF user guide will be produced as a separate next step; do not advertise it as available before it exists.

**Known concern:** The provided source code contains an eye-tracking validity selector, but this control was not visible in the previously inspected hosted-app screenshots. The hosted deployment and repository may differ. The two paper demos avoid relying on that selector.
