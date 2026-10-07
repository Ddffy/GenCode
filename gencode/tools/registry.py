"""工具定义、参数校验与执行。"""

import asyncio
import shutil
import subprocess
from functools import partial

from pydantic import ValidationError

from ..core.workspace import IGNORED_PATH_NAMES
from . import knowledge as knowledge_tools
from . import media as media_tools
from . import repomap as repomap_tools
from .ask_user import (
    ASK_USER_TOOL_EXAMPLES,
    ASK_USER_TOOL_SPECS,
    tool_ask_user,
)
from .base import RegisteredTool, ToolCapability, ToolEffect
from .plan import (
    PLAN_TOOL_EXAMPLES,
    PLAN_TOOL_SPECS,
    tool_enter_plan_mode,
    tool_exit_plan_mode,
)
from .schemas import (
    AskUserArgs,
    EnterPlanModeArgs,
    ExitPlanModeArgs,
    InspectImageArgs,
    ListFilesArgs,
    PatchFileArgs,
    ReadFileArgs,
    RepoMapArgs,
    RunShellArgs,
    SearchArgs,
    TodoAddArgs,
    TodoListArgs,
    TodoUpdateArgs,
    WriteFileArgs,
    first_error_message,
)
from .shell import tool_run_shell_async
from .todos import (
    TODO_TOOL_EXAMPLES,
    TODO_TOOL_SPECS,
    tool_todo_add,
    tool_todo_list,
    tool_todo_update,
)

_TOOL_SCHEMAS = {
    "list_files": ListFilesArgs,
    "read_file": ReadFileArgs,
    "search": SearchArgs,
    "inspect_image": InspectImageArgs,
    "run_shell": RunShellArgs,
    "write_file": WriteFileArgs,
    "patch_file": PatchFileArgs,
    "todo_add": TodoAddArgs,
    "todo_update": TodoUpdateArgs,
    "todo_list": TodoListArgs,
    "enter_plan_mode": EnterPlanModeArgs,
    "exit_plan_mode": ExitPlanModeArgs,
    "ask_user": AskUserArgs,
    "repo_map": RepoMapArgs,
    **knowledge_tools.TOOL_SCHEMAS,
    **knowledge_tools.TOOL_SCHEMAS,
}
BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "capability": ToolCapability(ToolEffect.READ, True),
        "description": "List files in the workspace.",
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
        "capability": ToolCapability(ToolEffect.READ, True),
        "description": "Read a UTF-8 file by line range.",
    },
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
        "capability": ToolCapability(ToolEffect.READ, True),
        "description": "Search the workspace with rg or a simple fallback.",
    },
    "run_shell": {
        "schema": {"command": "str", "timeout": "int=20"},
        "risky": True,
        "description": "Run a shell command in the repo root.",
    },
    "write_file": {
        "schema": {"path": "str", "content": "str"},
        "risky": True,
        "description": "Write a text file.",
    },
    "patch_file": {
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
        "description": "Replace one exact text block in a file.",
    },
    **knowledge_tools.TOOL_SPECS,
    **media_tools.MEDIA_TOOL_SPECS,
    **TODO_TOOL_SPECS,
    **PLAN_TOOL_SPECS,
    **ASK_USER_TOOL_SPECS,
    **repomap_tools.REPO_MAP_TOOL_SPECS,
}
TOOL_EXAMPLES = {
    "list_files": '<tool>{"name":"list_files","args":{"path":"."}}</tool>',
    "read_file": '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":80}}</tool>',
    "search": '<tool>{"name":"search","args":{"pattern":"binary_search","path":"."}}</tool>',
    "run_shell": '<tool>{"name":"run_shell","args":{"command":"uv run --with pytest python -m pytest -q","timeout":20}}</tool>',
    "write_file": '<tool name="write_file" path="binary_search.py"><content>def binary_search(nums, target):\n    return -1\n</content></tool>',
    "patch_file": '<tool name="patch_file" path="binary_search.py"><old_text>return -1</old_text><new_text>return mid</new_text></tool>',
    **knowledge_tools.TOOL_EXAMPLES,
    **media_tools.MEDIA_TOOL_EXAMPLES,
    **TODO_TOOL_EXAMPLES,
    **PLAN_TOOL_EXAMPLES,
    **ASK_USER_TOOL_EXAMPLES,
    **repomap_tools.REPO_MAP_TOOL_EXAMPLES,
}
def build_tool_registry(agent):
    # 工具不是动态发现的，而是显式注册的。
    # 这样模型看到的是一个有边界、可审计的动作集合。
    tools = {
        name: RegisteredTool(
            name=name,
            schema=spec["schema"],
            description=spec["description"],
            risky=bool(spec["risky"]),
            runner=partial(_async_tool_runner, name, agent),
            capability=spec.get("capability", ToolCapability()),
        )
        for name, spec in BASE_TOOL_SPECS.items()
    }
    return tools


async def _async_tool_runner(name, agent, args):
    if name == "run_shell":
        return await tool_run_shell_async(agent, args)
    if name == "ask_user":
        return await agent.ask_user_async(
            str(args["question"]), choices=args.get("choices", []) or []
        )
    if name == "inspect_image":
        return await media_tools.tool_inspect_image_async(agent, args)
    if name in {"todo_add", "todo_update", "todo_list", "enter_plan_mode", "exit_plan_mode"}:
        return _TOOL_RUNNERS[name](agent, args)
    task = asyncio.create_task(asyncio.to_thread(_TOOL_RUNNERS[name], agent, args))
    tool = agent.tools.get(name)
    cancellable_read = bool(
        tool is not None
        and tool.read_only
        and tool.capability.effect is ToolEffect.READ
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if cancellable_read:
            task.add_done_callback(_consume_task_exception)
        else:
            await task
        raise


def _consume_task_exception(task):
    if not task.cancelled():
        task.exception()


def tool_example(name):
    return TOOL_EXAMPLES.get(name, "")
def build_native_tool_definitions(agent):
    """Derive native JSON Schemas from the same Pydantic validation models."""
    definitions = []
    for name, tool in agent.available_tools().items():
        schema_cls = _TOOL_SCHEMAS.get(name)
        if schema_cls is not None and hasattr(schema_cls, "model_json_schema"):
            parameters = schema_cls.model_json_schema()
        else:  # pragma: no cover - every built-in tool currently has a model
            parameters = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }
        definitions.append(
            {
                "name": name,
                "description": tool.description,
                "parameters": parameters,
            }
        )
    return definitions


def validate_tool(agent, name, args):
    args = args or {}

    schema_cls = _TOOL_SCHEMAS.get(name)
    if schema_cls is not None:
        try:
            schema_cls.model_validate(args)
        except ValidationError as exc:
            raise ValueError(first_error_message(exc)) from exc

    # Workspace-aware checks that require the agent (path safety, file state).
    if name == "list_files":
        path = agent.path(args.get("path", "."))
        if not path.is_dir():
            raise ValueError("path is not a directory")

    elif name == "read_file":
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")

    elif name == "search":
        agent.path(args.get("path", "."))
    elif name in knowledge_tools.TOOL_SCHEMAS:
        knowledge_tools.validate(agent, name, args)

    elif name in media_tools.MEDIA_TOOL_NAMES:
        media_tools.validate_media_runtime(agent, name, args)

    elif name == "write_file":
        path = agent.path(args["path"])
        if path.exists() and path.is_dir():
            raise ValueError("path is a directory")

    elif name == "patch_file":
        # patch_file 故意做得很严格：old_text 必须精确命中且只能出现一次，
        # 这样修改行为才是确定的，失败原因也更容易解释。
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")
        text = path.read_text(encoding="utf-8")
        count = text.count(str(args.get("old_text", "")))
        if count != 1:
            raise ValueError(f"old_text must occur exactly once, found {count}")



def tool_list_files(agent, args):
    path = agent.path(args.get("path", "."))
    if not path.is_dir():
        raise ValueError("path is not a directory")
    entries = _visible_entries(path)
    lines = []
    for entry in entries[:200]:
        kind = "[D]" if entry.is_dir() else "[F]"
        lines.append(f"{kind} {entry.relative_to(agent.root).as_posix()}")
        if entry.is_dir():
            for child in _visible_entries(entry)[:12]:
                child_kind = "[D]" if child.is_dir() else "[F]"
                lines.append(f"  {child_kind} {child.relative_to(agent.root).as_posix()}")
    return "\n".join(lines) or "(empty)"


def _visible_entries(path):
    return [
        item
        for item in sorted(
            path.iterdir(), key=lambda item: (item.is_file(), item.name.lower())
        )
        if item.name not in IGNORED_PATH_NAMES
    ]


def tool_read_file(agent, args):
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    start = int(args.get("start", 1))
    default_end = 80 if getattr(agent, "fast_read_only_qa", False) else 200
    end = int(args.get("end", default_end))
    if start < 1 or end < start:
        raise ValueError("invalid line range")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    body = "\n".join(
        f"{number:>4}: {line}"
        for number, line in enumerate(lines[start - 1 : end], start=start)
    )
    return f"# {path.relative_to(agent.root).as_posix()}\n{body}"


def tool_search(agent, args):
    pattern = str(args.get("pattern", "")).strip()
    if not pattern:
        raise ValueError("pattern must not be empty")
    path = agent.path(args.get("path", "."))

    if shutil.which("rg"):
        # 优先用 rg，因为搜索会非常频繁，搜索延迟会直接影响 agent 控制循环。
        result = subprocess.run(
            ["rg", "-n", "--smart-case", "--max-count", "200", pattern, str(path)],
            cwd=agent.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.stdout.strip() or result.stderr.strip() or "(no matches)"

    matches = []
    files = (
        [path]
        if path.is_file()
        else [
            item
            for item in path.rglob("*")
            if item.is_file()
            and not any(
                part in IGNORED_PATH_NAMES
                for part in item.relative_to(agent.root).parts
            )
        ]
    )
    for file_path in files:
        for number, line in enumerate(
            file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if pattern.lower() in line.lower():
                matches.append(f"{file_path.relative_to(agent.root).as_posix()}:{number}:{line}")
                if len(matches) >= 200:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def tool_write_file(agent, args):
    path = agent.path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"wrote {path.relative_to(agent.root).as_posix()} ({len(content)} chars)"


def tool_patch_file(agent, args):
    path = agent.path(args["path"])
    if not path.is_file():
        raise ValueError("path is not a file")
    old_text = str(args.get("old_text", ""))
    if not old_text:
        raise ValueError("old_text must not be empty")
    if "new_text" not in args:
        raise ValueError("missing new_text")
    text = path.read_text(encoding="utf-8")
    count = text.count(old_text)
    if count != 1:
        raise ValueError(f"old_text must occur exactly once, found {count}")
    path.write_text(text.replace(old_text, str(args["new_text"]), 1), encoding="utf-8")
    return f"patched {path.relative_to(agent.root).as_posix()}"


_TOOL_RUNNERS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "search": tool_search,
    "knowledge_read": knowledge_tools.read_wiki,
    "knowledge_propose": knowledge_tools.run,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
    "todo_add": tool_todo_add,
    "todo_update": tool_todo_update,
    "todo_list": tool_todo_list,
    "enter_plan_mode": tool_enter_plan_mode,
    "exit_plan_mode": tool_exit_plan_mode,
    "ask_user": tool_ask_user,
    **media_tools.MEDIA_TOOL_RUNNERS,
    **repomap_tools.REPO_MAP_TOOL_RUNNERS,
}
