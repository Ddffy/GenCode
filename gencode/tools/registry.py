"""工具定义、参数校验与执行。"""

import os
import shutil
import subprocess
import textwrap
from functools import partial

from pydantic import ValidationError

from ..core.workspace import IGNORED_PATH_NAMES
from . import media as media_tools
from . import repomap as repomap_tools
from .base import RegisteredTool
from .agents import (
    AGENT_TOOL_EXAMPLES,
    AGENT_TOOL_NAMES,
    AGENT_TOOL_SPECS,
    tool_agent,
    tool_send_message,
    tool_task_stop,
    validate_agent_runtime,
)
from .ask_user import (
    ASK_USER_TOOL_EXAMPLES,
    ASK_USER_TOOL_SPECS,
    tool_ask_user,
)
from .plan import (
    PLAN_TOOL_EXAMPLES,
    PLAN_TOOL_SPECS,
    tool_enter_plan_mode,
    tool_exit_plan_mode,
)
from .todos import (
    TODO_TOOL_EXAMPLES,
    TODO_TOOL_SPECS,
    tool_todo_add,
    tool_todo_list,
    tool_todo_update,
)
from .schemas import (
    AgentArgs,
    AskUserArgs,
    EnterPlanModeArgs,
    ExitPlanModeArgs,
    InspectImageArgs,
    ListFilesArgs,
    KnowledgeReadArgs,
    PatchFileArgs,
    ReadFileArgs,
    RunShellArgs,
    RepoMapArgs,
    SearchArgs,
    SendMessageArgs,
    TaskStopArgs,
    TodoAddArgs,
    TodoListArgs,
    TodoUpdateArgs,
    WriteFileArgs,
    first_error_message,
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
    "agent": AgentArgs,
    "send_message": SendMessageArgs,
    "task_stop": TaskStopArgs,
    "enter_plan_mode": EnterPlanModeArgs,
    "exit_plan_mode": ExitPlanModeArgs,
    "ask_user": AskUserArgs,
    "repo_map": RepoMapArgs,
    "knowledge_read": KnowledgeReadArgs,
}
BASE_TOOL_SPECS = {
    "list_files": {
        "schema": {"path": "str='.'"},
        "risky": False,
        "description": "List files in the workspace.",
    },
    "read_file": {
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
        "description": "Read a UTF-8 file by line range.",
    },
    "search": {
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
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
    "knowledge_read": {
        "schema": {"id": "str", "kind": "str='wiki'", "section": "str=''", "max_chars": "int=12000"},
        "risky": False,
        "description": "Read an approved Wiki page or Markdown section on demand.",
    },
    **media_tools.MEDIA_TOOL_SPECS,
    **TODO_TOOL_SPECS,
    **AGENT_TOOL_SPECS,
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
    "knowledge_read": '<tool>{"name":"knowledge_read","args":{"id":"runtime-overview","section":"Execution"}}</tool>',
    **media_tools.MEDIA_TOOL_EXAMPLES,
    **TODO_TOOL_EXAMPLES,
    **AGENT_TOOL_EXAMPLES,
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
            runner=partial(_TOOL_RUNNERS[name], agent),
        )
        for name, spec in BASE_TOOL_SPECS.items()
    }
    return tools
def tool_example(name):
    return TOOL_EXAMPLES.get(name, "")
def build_native_tool_definitions(agent):
    """Build provider-neutral JSON Schemas for native tool calling.

    The prompt-facing ``RegisteredTool.schema`` is intentionally compact and
    human readable.  Native providers need a real JSON Schema, so derive it
    from the same Pydantic models used by ``validate_tool`` instead of keeping
    a second, drift-prone schema registry.
    """
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
    elif name == "knowledge_read":
        if not hasattr(agent, "knowledge_store"):
            raise ValueError("typed knowledge is unavailable")
        if hasattr(agent, "feature_enabled") and not agent.feature_enabled("typed_knowledge"):
            raise ValueError("typed knowledge is disabled")
        if str(args.get("kind", "wiki")).lower() != "wiki":
            raise ValueError("knowledge_read only reads kind=wiki")
        if not agent.knowledge_store.get(args["id"], kind="wiki", include_inactive=True):
            raise ValueError("unknown wiki")

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

    elif name in AGENT_TOOL_NAMES:
        validate_agent_runtime(agent, name, args)


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
    end = int(args.get("end", 200))
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


def tool_knowledge_read(agent, args):
    return agent.knowledge_store.read_wiki(
        args["id"],
        section=args.get("section", ""),
        max_chars=args.get("max_chars", 12_000),
    )
def tool_run_shell(agent, args):
    command = str(args.get("command", "")).strip()
    if not command:
        raise ValueError("command must not be empty")
    timeout = int(args.get("timeout", 20))
    if timeout < 1 or timeout > 120:
        raise ValueError("timeout must be in [1, 120]")
    runner = getattr(agent, "sandbox_runner", None)
    if runner is None:
        kwargs = {
            "cwd": agent.root,
            "shell": True,
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "timeout": timeout,
            "env": agent.shell_env(),
            "check": False,
        }
        if os.name == "nt":
            kwargs["executable"] = os.environ.get("COMSPEC") or r"C:\Windows\System32\cmd.exe"
        result = subprocess.run(command, **kwargs)
    else:
        result = runner.run(
            command,
            cwd=agent.root,
            env=agent.shell_env(),
            timeout=timeout,
        )
    return textwrap.dedent(
        f"""\
        exit_code: {result.returncode}
        stdout:
        {result.stdout.strip() or "(empty)"}
        stderr:
        {result.stderr.strip() or "(empty)"}
        """
    ).strip()


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
    "knowledge_read": tool_knowledge_read,
    "run_shell": tool_run_shell,
    "write_file": tool_write_file,
    "patch_file": tool_patch_file,
    "todo_add": tool_todo_add,
    "todo_update": tool_todo_update,
    "todo_list": tool_todo_list,
    "agent": tool_agent,
    "send_message": tool_send_message,
    "task_stop": tool_task_stop,
    "enter_plan_mode": tool_enter_plan_mode,
    "exit_plan_mode": tool_exit_plan_mode,
    "ask_user": tool_ask_user,
    **media_tools.MEDIA_TOOL_RUNNERS,
    **repomap_tools.REPO_MAP_TOOL_RUNNERS,
}
