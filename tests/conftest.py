"""Shared pytest environment.

Workspace detection resolves repo_root via `git rev-parse --show-toplevel`.
On machines where the user home itself is a git repository, tmp_path-based
tests would resolve their workspace to the home directory and run gencode's
workspace-level git/snapshot operations against the whole home tree. Ceiling
git discovery at the OS temp directory keeps tmp_path workspaces self-contained.
"""

import os
import tempfile

os.environ["GIT_CEILING_DIRECTORIES"] = tempfile.gettempdir()
