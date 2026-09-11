"""Portable command helpers used by deterministic evaluators."""

import re
import os
import shutil
import subprocess
import sys


_PYTHON3_TOKEN = re.compile(r"(?<![\w.-])python3(?:\.\d+)?(?=\s|$)", re.IGNORECASE)


def portable_verifier_command(command):
    """Use the active interpreter when a benchmark says ``python3``."""
    text = str(command or "")
    if os.name != "nt" and shutil.which("python3"):
        return text
    executable = subprocess.list2cmdline([sys.executable])
    return _PYTHON3_TOKEN.sub(lambda _match: executable, text)
