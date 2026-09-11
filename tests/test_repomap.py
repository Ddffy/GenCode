"""Tests for the tree-sitter repo map pipeline."""

from typing import ClassVar

import pytest

from gencode.core.context_manager import ContextManager
from gencode.features.repomap import RepoMapBuilder
from gencode.features.repomap import graph as graphlib
from gencode.features.repomap import rank as ranklib
from gencode.features.repomap import render as renderlib
from gencode.features.repomap import tags as tagslib
from gencode.tools.repomap import tool_repo_map


def _write_repo(tmp_path):
    (tmp_path / "storage.py").write_text(
        "class Storage:\n"
        "    def load(self, key):\n"
        "        return key\n",
        encoding="utf-8",
    )
    (tmp_path / "service.py").write_text(
        "from storage import Storage\n"
        "\n"
        "class Service:\n"
        "    def run(self):\n"
        "        box = Storage()\n"
        "        return box.load('k')\n",
        encoding="utf-8",
    )
    (tmp_path / "unrelated.py").write_text(
        "def totally_different():\n    return 1\n",
        encoding="utf-8",
    )


@pytest.fixture
def repo(tmp_path):
    _write_repo(tmp_path)
    return tmp_path


requires_pack = pytest.mark.skipif(
    not tagslib.pack_available(), reason="tree-sitter-language-pack not installed"
)


@requires_pack
def test_extract_python_tags(repo):
    file_tags = graphlib.load_tags(repo)["storage.py"]
    names = {name for name, _line in file_tags.definitions}
    assert "Storage" in names
    assert "load" in names


@requires_pack
def test_reference_graph_builds_weighted_edges(repo):
    tags_by_file = graphlib.load_tags(repo)
    _defines, edges = graphlib.build_reference_graph(tags_by_file)
    # service.py 引用 storage.py 定义的 Storage/load -> 至少一条有向边。
    assert edges.get(("service.py", "storage.py"), 0) > 0


@requires_pack
def test_query_personalization_promotes_relevant_files(repo):
    builder = RepoMapBuilder(repo, cache_dir=repo / "cache")
    text, meta = builder.build(query="Storage load", budget_chars=4000)
    assert meta["enabled"] is True
    lines = text.splitlines()[1:]
    paths = [line.split(" (")[0].lstrip("- ") for line in lines]
    # 命中查询符号的文件必须排在完全无关文件前面。
    assert paths.index("service.py") < paths.index("unrelated.py")
    assert paths.index("storage.py") < paths.index("unrelated.py")


@requires_pack
def test_map_is_deterministic(repo):
    builder = RepoMapBuilder(repo, cache_dir=repo / "cache")
    first = builder.build(query="Storage", budget_chars=2000)[0]
    second = builder.build(query="Storage", budget_chars=2000)[0]
    assert first == second


@requires_pack
def test_cache_invalidates_on_file_change(repo):
    builder = RepoMapBuilder(repo, cache_dir=repo / "cache")
    builder.build(query="", budget_chars=2000)
    (repo / "newmod.py").write_text("def fresh_symbol_here():\n    return 2\n", encoding="utf-8")
    text, _meta = builder.build(query="fresh_symbol_here", budget_chars=4000)
    assert "newmod.py" in text


@requires_pack
def test_render_respects_budget(repo):
    tags_by_file = graphlib.load_tags(repo)
    ranked = sorted(tags_by_file)
    text = renderlib.render_repo_map(ranked, tags_by_file, 80)
    assert len(text) <= 80


def test_pagerank_without_seeds_ranks_referenced_target_higher():
    scores = ranklib.personalized_pagerank(
        ["a.py", "b.py"],
        {("a.py", "b.py"): 1.0},
        seeds=None,
    )
    assert set(scores) == {"a.py", "b.py"}
    assert scores["b.py"] > scores["a.py"]


def test_pagerank_empty_graph():
    assert ranklib.personalized_pagerank([], {}, seeds=None) == {}


@requires_pack
def test_builder_degrades_without_any_parser(repo, monkeypatch):
    monkeypatch.setattr(tagslib, "get_parser", None)
    monkeypatch.setattr(tagslib, "_FALLBACK_LANGUAGE_MODULES", {})
    monkeypatch.setattr(tagslib, "_resolved_languages", {})
    monkeypatch.setattr(tagslib, "_unresolvable_languages", set())
    monkeypatch.setattr(tagslib, "_pack_available_cache", None)
    text, meta = RepoMapBuilder(repo, cache_dir=repo / "cache").build(
        query="Storage", budget_chars=2000
    )
    assert text == ""
    assert meta["enabled"] is False


@requires_pack
def test_tool_repo_map_returns_text():
    class _Agent:
        def build_repo_map(self, query="", budget_chars=6000, recent_paths=None):
            return "Repository map:\n- a.py (A)", {"enabled": True}

    text = tool_repo_map(_Agent(), {"query": "Storage load", "limit_chars": 4000})
    assert "a.py" in text


def test_tool_repo_map_reports_unavailable_reason():
    class _Agent:
        def build_repo_map(self, query="", budget_chars=6000, recent_paths=None):
            return "", {"enabled": False, "reason": "no_supported_files"}

    text = tool_repo_map(_Agent(), {"query": "", "limit_chars": 4000})
    assert "no_supported_files" in text


class _FakeTool:
    schema: ClassVar[dict] = {}
    risky: ClassVar[bool] = False
    description: ClassVar[str] = "stub"


class _StubAgent:
    """Minimal agent surface for ContextManager integration checks."""

    def __init__(self, map_text):
        self.prefix = "PREFIX"
        self.skills = {}
        self.session = {"history": []}
        self.max_new_tokens = 8192
        self._map_text = map_text

    def feature_enabled(self, name):
        return name in {"memory", "relevant_memory", "context_reduction"}

    def memory_text(self):
        return "Memory:\n- task: x"

    def available_tools(self):
        return {"read_file": _FakeTool()}

    def build_repo_map_section(self, user_message):
        return self._map_text


def _context_prompt(map_text, user_message):
    manager = ContextManager(_StubAgent(map_text), total_budget=60000)
    prompt, metadata = manager.build(user_message)
    return prompt, metadata


def test_context_manager_includes_nonempty_repo_map():
    prompt, metadata = _context_prompt("Repository map:\n- a.py (A)", "fix the bug")
    assert "Repository map:" in prompt
    assert metadata["sections"]["repo_map"]["raw_chars"] > 0


def test_context_manager_skips_empty_repo_map():
    prompt, metadata = _context_prompt("", "fix the bug")
    assert "Repository map:" not in prompt
    assert metadata["sections"]["repo_map"]["raw_chars"] == 0
