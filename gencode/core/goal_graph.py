"""Validated, serializable DAG contracts used by Goal runs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath

MAX_GOAL_NODES = 16
MAX_SCOPE_PATHS = 16
NODE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,47}$")


class GoalPlanError(ValueError):
    pass


def validate_plan(value):
    if not isinstance(value, dict):
        raise GoalPlanError("plan must be a JSON object")
    raw_nodes = value.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes or len(raw_nodes) > MAX_GOAL_NODES:
        raise GoalPlanError(f"plan nodes must contain 1..{MAX_GOAL_NODES} items")
    nodes = []
    ids = set()
    for raw in raw_nodes:
        if not isinstance(raw, dict):
            raise GoalPlanError("each node must be an object")
        node_id = str(raw.get("id", "")).strip()
        if not NODE_ID_RE.fullmatch(node_id) or node_id in ids:
            raise GoalPlanError(f"invalid or duplicate node id: {node_id!r}")
        ids.add(node_id)
        title = _required_text(raw, "title", 160)
        prompt = _required_text(raw, "prompt", 6000)
        dependencies = _string_list(raw.get("depends_on", []), "depends_on", 32)
        if len(set(dependencies)) != len(dependencies) or node_id in dependencies:
            raise GoalPlanError(f"invalid dependencies for node {node_id}")
        scope = _string_list(raw.get("write_scope", []), "write_scope", MAX_SCOPE_PATHS)
        if not scope:
            raise GoalPlanError(f"node {node_id} needs a non-empty write_scope")
        for path in scope:
            _validate_scope_path(path, node_id)
        acceptance = _string_list(raw.get("acceptance", []), "acceptance", 12)
        if not acceptance:
            raise GoalPlanError(f"node {node_id} needs an acceptance criterion")
        nodes.append(
            {
                "id": node_id,
                "title": title,
                "prompt": prompt,
                "depends_on": dependencies,
                "write_scope": scope,
                "acceptance": acceptance,
            }
        )
    for node in nodes:
        missing = sorted(set(node["depends_on"]) - ids)
        if missing:
            raise GoalPlanError(f"node {node['id']} has unknown dependencies: {missing}")
    topological_order(nodes)
    verifiers = value.get("verifiers")
    if not isinstance(verifiers, list) or not verifiers:
        raise GoalPlanError("plan must provide at least one deterministic verifier")
    normalized_verifiers = [_normalize_verifier(item) for item in verifiers]
    return {"nodes": nodes, "verifiers": normalized_verifiers}


def topological_order(nodes):
    by_id = {str(node["id"]): node for node in nodes}
    pending = {node_id: set(node.get("depends_on", [])) for node_id, node in by_id.items()}
    order = []
    while pending:
        ready = [node_id for node_id, dependencies in pending.items() if not dependencies]
        if not ready:
            raise GoalPlanError("plan contains a dependency cycle")
        for node_id in ready:
            order.append(node_id)
            pending.pop(node_id)
            for dependencies in pending.values():
                dependencies.discard(node_id)
    return order


def node_fingerprint(node):
    contract = {
        key: node.get(key)
        for key in ("title", "prompt", "depends_on", "write_scope", "acceptance")
    }
    return hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _required_text(raw, key, limit):
    value = str(raw.get(key, "")).strip()
    if not value or len(value) > limit:
        raise GoalPlanError(f"{key} must contain 1..{limit} characters")
    return value


def _string_list(value, name, limit):
    if not isinstance(value, list) or len(value) > limit:
        raise GoalPlanError(f"{name} must be a list with at most {limit} items")
    items = [str(item).strip() for item in value]
    if any(not item or len(item) > 500 for item in items):
        raise GoalPlanError(f"{name} contains an empty or oversized item")
    return items


def _validate_scope_path(value, node_id):
    path = PurePosixPath(value.replace("\\", "/"))
    if (
        value.strip() in {"", ".", "/"}
        or re.match(r"^[A-Za-z]:", value)
        or path.is_absolute()
        or any(part in {"..", ".git", ".gencode", ".pico"} for part in path.parts)
        or any("*" in part or "?" in part for part in path.parts)
    ):
        raise GoalPlanError(f"unsafe write_scope path for {node_id}: {value!r}")


def _validate_verifier(value):
    if not isinstance(value, list) or not value:
        raise GoalPlanError("each verifier must be an argv list")
    argv = [str(item) for item in value]
    if any(not item or "\x00" in item or len(item) > 300 for item in argv):
        raise GoalPlanError("verifier argv contains an invalid item")
    executable = PurePosixPath(argv[0].replace("\\", "/")).name.lower()
    allowed = {
        "pytest": lambda args: True,
        "ruff": lambda args: bool(args) and args[0] in {"check", "format"},
        "mypy": lambda args: True,
        "pyright": lambda args: True,
        "npm": lambda args: args in (["test"], ["run", "test:unit"]),
        "pnpm": lambda args: args in (["test"], ["run", "test"]),
        "yarn": lambda args: args == ["test"],
        "cargo": lambda args: bool(args) and args[0] == "test",
        "go": lambda args: bool(args) and args[0] == "test",
        "dotnet": lambda args: bool(args) and args[0] == "test",
    }
    args = argv[1:]
    if executable in {"python", "python3", "py"}:
        if len(args) < 2 or args[0] != "-m" or args[1] not in {"pytest", "unittest"}:
            raise GoalPlanError("Python verifier must use `python -m pytest|unittest`")
    elif executable not in allowed or not allowed[executable](args):
        raise GoalPlanError(f"unsupported verifier executable: {executable}")
    if any(token in {";", "&&", "||", "|", ">", "<"} for token in argv):
        raise GoalPlanError("verifiers must not contain shell operators")
    return argv


def _normalize_verifier(value):
    if isinstance(value, dict):
        if set(value) - {"id", "node_id", "argv"} or not isinstance(value.get("argv"), list):
            raise GoalPlanError("verifier objects may contain only id, node_id, and argv")
        value = value["argv"]
    return _validate_verifier(value)
