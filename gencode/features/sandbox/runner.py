"""Optional shell sandbox runner."""

import os
import subprocess
from pathlib import Path
from shutil import which as default_which

from .checker import SandboxChecker
from .command_matcher import command_is_excluded
from .config import SandboxConfig


class SandboxRunner:
    def __init__(self, config=None, *, which=None, run=None, emit_event=None):
        self.config = config or SandboxConfig()
        self.which = which or default_which
        self.run_process = run
        self.emit_event = emit_event or (lambda event, payload: None)

    def run(self, command, *, cwd, env, timeout):
        config = self.config
        if config.mode == "off" or (
            config.mode != "required"
            and command_is_excluded(command, config.excluded_commands)
        ):
            return self._plain(command, cwd=cwd, env=env, timeout=timeout)

        backend_path = SandboxChecker(self.which).backend_path(config.backend)
        if not backend_path:
            self.emit_event(
                "sandbox_unavailable",
                {
                    "mode": config.mode,
                    "backend": config.backend,
                    "command": str(command or "")[:200],
                },
            )
            if config.mode == "required":
                raise RuntimeError("sandbox required but unavailable")
            return self._plain(command, cwd=cwd, env=env, timeout=timeout)

        argv = self._bubblewrap_argv(backend_path, command, Path(cwd), config)
        run_process = self.run_process or subprocess.run
        return run_process(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

    def _plain(self, command, *, cwd, env, timeout):
        run_process = self.run_process or subprocess.run
        kwargs = {
            "cwd": cwd,
            "shell": True,
            "capture_output": True,
            "text": True,
            "timeout": timeout,
            "env": env,
        }
        if os.name == "nt":
            kwargs["executable"] = os.environ.get("COMSPEC") or r"C:\Windows\System32\cmd.exe"
        return run_process(command, **kwargs)

    def _bubblewrap_argv(self, backend_path, command, cwd, config):
        argv = [
            backend_path,
            "--die-with-parent",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/lib64",
            "/lib64",
        ]
        bind_mode = "--bind" if config.workspace_write else "--ro-bind"
        argv.extend([bind_mode, str(cwd), str(cwd)])
        for path in config.extra_readonly_paths:
            argv.extend(["--ro-bind", path, path])
        for path in (*config.deny_read, *config.deny_write):
            argv.extend(["--tmpfs", path])
        argv.extend(["--chdir", str(cwd), "--", "/bin/sh", "-lc", str(command)])
        return argv
