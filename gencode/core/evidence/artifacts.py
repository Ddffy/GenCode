"""Runtime artifact graph and verifier suggestion helpers."""

import json
import re
from pathlib import Path

DEPENDENCY_FILES = {"package.json", "pyproject.toml", "requirements.txt", "uv.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock"}
ROUTE_RE = re.compile(r"""["'](/api/[^"'\s)]*)["']""")


def build_artifact_graph(root, changed_paths):
    root = Path(root)
    paths = sorted(dict.fromkeys(str(path) for path in changed_paths if str(path).strip()))
    graph = {
        "changed_paths": paths,
        "categories": {name: [] for name in ("backend", "frontend", "docs", "tests", "dependencies", "other")},
        "route_refs": [],
        "api_refs": [],
    }
    for relative in paths:
        graph["categories"][_category(relative)].append(relative)
        text = _read_text(root / relative)
        if text:
            _collect_refs(graph, text)
    graph["route_refs"] = sorted(set(graph["route_refs"]))
    graph["api_refs"] = sorted(set(graph["api_refs"]))
    return graph


def build_verifier_suggestions(root, graph):
    root = Path(root)
    suggestions = []
    package_json = root / "package.json"
    if package_json.exists():
        scripts = _package_scripts(package_json)
        if "test" in scripts:
            suggestions.append({"command": "npm test", "reason": "package.json defines a test script"})
        if "build" in scripts:
            suggestions.append({"command": "npm run build", "reason": "package.json defines a build script"})
    if _has_python_tests(root):
        suggestions.append({"command": "uv run python -m pytest -q", "reason": "Python tests are present"})
    elif any(path.endswith(".py") for path in graph.get("changed_paths", [])):
        suggestions.append({"command": "uv run python -m compileall .", "reason": "Python files changed and no tests were found"})
    return suggestions[:8]


def _category(path):
    normalized = path.replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1]
    suffix = Path(name).suffix.lower()
    if name in DEPENDENCY_FILES:
        return "dependencies"
    if normalized.startswith("tests/") or name.startswith("test_") or name.endswith("_test.py"):
        return "tests"
    if suffix in {".md", ".rst", ".txt"} or normalized.startswith("docs/"):
        return "docs"
    if suffix in {".js", ".jsx", ".ts", ".tsx", ".css", ".html", ".vue", ".svelte"}:
        return "frontend"
    if suffix in {".py", ".rb", ".go", ".rs", ".java", ".kt", ".php"}:
        return "backend"
    return "other"


def _collect_refs(graph, text):
    for line in text.splitlines()[:500]:
        refs = ROUTE_RE.findall(line)
        if not refs:
            continue
        if "fetch(" in line or "axios" in line:
            graph["api_refs"].extend(refs)
        if "@" in line or "route" in line.lower() or ".get(" in line or ".post(" in line:
            graph["route_refs"].extend(refs)


def _read_text(path):
    try:
        if not path.is_file() or path.stat().st_size > 200_000:
            return ""
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _package_scripts(path):
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")).get("scripts", {}) or {})
    except (OSError, json.JSONDecodeError):
        return {}


def _has_python_tests(root):
    tests_dir = root / "tests"
    return tests_dir.is_dir() and any(path.suffix == ".py" for path in tests_dir.rglob("*.py"))
