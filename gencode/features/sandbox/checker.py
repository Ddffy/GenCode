"""Sandbox backend availability checks."""


class SandboxChecker:
    def __init__(self, which):
        self.which = which

    def backend_path(self, backend):
        backend = "bubblewrap" if backend == "auto" else backend
        if backend in {"none", "off"}:
            return ""
        if backend == "bubblewrap":
            return self.which("bwrap") or ""
        return ""
