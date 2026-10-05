"""Shared pytest environment.

Workspace detection resolves repo_root via `git rev-parse --show-toplevel`.
On machines where the user home itself is a git repository, tmp_path-based
tests would resolve their workspace to the home directory and run gencode's
workspace-level git/snapshot operations against the whole home tree. Ceiling
git discovery at the OS temp directory keeps tmp_path workspaces self-contained.
"""

import asyncio
import os
import tempfile


async def _collect_events(stream, include_text):
    events = []
    async for event in stream:
        if include_text or event.get("type") != "text_delta":
            events.append(event)
    return events


def collect_events(stream, include_text=False):
    return asyncio.run(_collect_events(stream, include_text))


def collect_stream_events(stream):
    return collect_events(stream, include_text=True)


def run_tool(agent, name, args):
    return asyncio.run(agent.run_tool(name, args))

os.environ["GIT_CEILING_DIRECTORIES"] = tempfile.gettempdir()
