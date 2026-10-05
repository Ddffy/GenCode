import asyncio
import os
import subprocess
import textwrap


async def tool_run_shell_async(agent, args):
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    runner = getattr(agent, "sandbox_runner", None)
    if runner is None:
        result = await asyncio.create_subprocess_shell(
            command,
            cwd=agent.root,
            executable=(
                os.environ.get("COMSPEC") or r"C:\Windows\System32\cmd.exe"
                if os.name == "nt"
                else None
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=agent.shell_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
            result.terminate()
            try:
                await asyncio.wait_for(result.wait(), 1)
            except asyncio.TimeoutError:
                result.kill()
                await result.wait()
            if isinstance(exc, asyncio.TimeoutError):
                raise TimeoutError(f"shell command exceeded {timeout}s") from exc
            raise
        completed = subprocess.CompletedProcess(
            command,
            result.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )
    else:
        completed = await runner.run_async(
            command, cwd=agent.root, env=agent.shell_env(), timeout=timeout
        )
    return textwrap.dedent(
        f"""\
        exit_code: {completed.returncode}
        stdout:
        {completed.stdout.strip() or "(empty)"}
        stderr:
        {completed.stderr.strip() or "(empty)"}
        """
    ).strip()
