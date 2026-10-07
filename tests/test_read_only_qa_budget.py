import asyncio

import pytest

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.core import engine_helpers
from gencode.core.engine import (
    _is_read_only_qa_request,
    _read_only_qa_final_only,
    _read_only_qa_prompt,
    _set_fast_read_only_qa,
    _turn_tool_step_budget,
)
from gencode.core.task_state import TaskState
from gencode.testing import ScriptedModelClient
from gencode.tools.registry import tool_read_file


def test_repository_explanation_gets_a_smaller_tool_budget():
    assert _turn_tool_step_budget("介绍 GenCode 多 Agent 是怎么做的", 50) == 28
    assert _turn_tool_step_budget("修复多 Agent worker 的 bug", 50) == 50
    assert _is_read_only_qa_request("介绍 GenCode 多 Agent 是怎么做的") is True
    assert _is_read_only_qa_request(
        "请解释 GenCode 多 Agent，不要修改代码"
    ) is True
    assert _is_read_only_qa_request("修改代码并修复多 Agent worker") is False
    assert _read_only_qa_final_only(
        read_only_qa=True, tool_steps=12, native_mode=True
    ) is True
    assert _read_only_qa_final_only(
        read_only_qa=True, tool_steps=12, native_mode=False
    ) is False


def test_goal_worker_tool_budget_ignores_read_only_qa_heuristic():
    prompt = "Goal Worker requested command describes an implementation task"
    assert _turn_tool_step_budget(prompt, 48) == 28
    assert _turn_tool_step_budget(prompt, 48, allow_read_only_qa=False) == 48


def test_read_only_qa_prompt_requests_evidence_without_progress_narration():
    prompt = _read_only_qa_prompt("介绍 GenCode 多 Agent")

    assert prompt.startswith("介绍 GenCode 多 Agent")
    assert "不使用 Shell" in prompt
    assert "文件/函数出处" in prompt
    assert "必须显式传 start/end" in prompt
    assert "单次最多 80 行" in prompt


def test_fast_read_only_qa_exposes_only_readonly_tools_and_restores_profile(tmp_path):
    agent = GenCode(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )
    original_profile = agent.active_tool_profile.name

    _set_fast_read_only_qa(agent, True)

    assert agent.active_tool_profile.name == "readonly"
    assert "read_file" in agent.available_tools()
    assert "search" in agent.available_tools()
    assert "repo_map" in agent.available_tools()
    assert "run_shell" not in agent.available_tools()
    assert "write_file" not in agent.available_tools()
    assert "agent" not in agent.available_tools()

    _set_fast_read_only_qa(agent, False)

    assert agent.active_tool_profile.name == original_profile
    assert "run_shell" in agent.available_tools()


def test_fast_read_only_qa_defaults_file_reads_to_eighty_lines(tmp_path):
    agent = GenCode(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )
    source = tmp_path / "large.txt"
    source.write_text("\n".join(f"line {index}" for index in range(1, 121)), encoding="utf-8")
    agent.fast_read_only_qa = True

    result = tool_read_file(agent, {"path": "large.txt"})

    assert "  80: line 80" in result
    assert "  81: line 81" not in result


@pytest.mark.asyncio
async def test_read_only_qa_uses_sparse_retrieval_without_cold_repo_map(tmp_path):
    from types import SimpleNamespace

    agent = GenCode(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )
    agent.current_task_state = TaskState.create(
        task_id="task_test", user_request="介绍项目多 Agent"
    )
    agent.fast_read_only_qa = True
    agent.build_repo_map = lambda **_kwargs: (
        "local repo map",
        {"enabled": True, "files_scanned": 10, "chars": 15},
    )
    route_args = {}
    sync_args = {}
    retrieve_args = {}
    service = agent.knowledge_store.retrieval

    def route(**kwargs):
        route_args.update(kwargs)
        return {
            "strategy": "mini_repo_map_plus_hybrid",
            "channels": ("sparse", "dense"),
        }

    def sync_code(root, **kwargs):
        sync_args.update({"root": root, **kwargs})
        return {"changed": False, "dense_complete": False}

    service.start_background_code_sync = lambda *_args, **_kwargs: True

    def retrieve(query, **kwargs):
        retrieve_args.update({"query": query, **kwargs})
        return SimpleNamespace(
            to_dict=lambda: {"strategy": "hybrid_rrf"},
            insufficient_evidence=False,
            evidence_text="retrieved code evidence",
        )

    service.route = route
    service.sync_code = sync_code
    service.retrieve = retrieve

    section = agent.build_retrieval_section("介绍项目多 Agent")

    assert not section.startswith("local repo map")
    assert "Large-repository retrieval:\nretrieved code evidence" in section
    assert agent.last_repo_map_metadata["reason"] == "deferred_for_fast_read_only_qa"
    assert route_args["force_hybrid"] is True
    assert sync_args["root"] == agent.root
    assert retrieve_args["source_types"] == ("code",)
    assert retrieve_args["channels"] == ("sparse",)
    assert retrieve_args["rerank"] is False
    assert agent.last_code_retrieval["strategy"] == "hybrid_rrf"


def test_fast_read_only_qa_includes_typed_knowledge_retrieval(tmp_path):
    agent = GenCode(
        model_client=ScriptedModelClient([]),
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )
    agent.fast_read_only_qa = True
    agent.build_repo_map_section = lambda _message: ""

    _prompt, metadata = agent.context_manager.build("Explain the worker runtime")

    assert agent.last_knowledge_retrieval is not None
    assert metadata["knowledge"]["enabled"] is True
    assert metadata["knowledge"]["selected_wiki_ids"] == []


@pytest.mark.asyncio
async def test_step_limit_summary_repeats_request_and_requires_final_protocol(monkeypatch):
    captured = {}

    class Runtime:
        model_client = object()
        max_new_tokens = 8192

        async def _build_prompt_and_metadata_async(self, request):
            captured["request"] = request
            return "assembled prompt", {}

        @staticmethod
        def parse(text):
            from gencode.core.model_output import parse

            return parse(text)

        @staticmethod
        def emit_trace(_task_state, _event, payload):
            captured["trace"] = payload

    async def complete(_client, prompt, max_new_tokens, *, tools=None, messages=None):
        captured["completion"] = (prompt, max_new_tokens, tools, messages)
        return type("Result", (), {"text": "<final>完整回答</final>"})()

    monkeypatch.setattr(engine_helpers, "complete_model_async", complete)
    result = await engine_helpers.request_step_limit_summary(
        type("Engine", (), {"runtime": Runtime()})(),
        type("Task", (), {"run_id": "run_test"})(),
        "请说明多 Agent 的调度方式",
    )

    assert result == "完整回答"
    assert "请说明多 Agent 的调度方式" in captured["completion"][0]
    assert "<final>...</final>" in captured["completion"][0]
    assert captured["completion"][1:3] == (700, [])


@pytest.mark.asyncio
async def test_native_step_limit_summary_returns_plain_text_without_final_tag(monkeypatch):
    captured = {}

    class NativeClient:
        supports_native_tool_calling = True

        async def stream_messages(self):
            pass

    class Runtime:
        model_client = NativeClient()
        max_new_tokens = 8192
        context_manager = type(
            "Context", (), {"last_rendered_sections": {"prefix": "system context"}}
        )()
        fast_read_only_qa = True

        def __init__(self):
            self.session = {
                "history": [
                    {"role": role, "content": f"evidence_{index}"}
                    for index in range(20)
                    for role in ("user", "assistant")
                ]
            }

        async def _build_prompt_and_metadata_async(self, request):
            captured["request"] = request
            return "assembled prompt", {}

        @staticmethod
        def parse(_text):
            return pytest.fail("native plain-text final should bypass text protocol parsing")

        @staticmethod
        def emit_trace(_task_state, _event, payload):
            captured["trace"] = payload

    async def complete(_client, prompt, max_new_tokens, *, tools=None, messages=None):
        captured["completion"] = (prompt, max_new_tokens, tools, messages)
        return type("Result", (), {"text": "这里是普通文本终答", "tool_calls": ()})()

    monkeypatch.setattr(engine_helpers, "complete_model_async", complete)
    result = await engine_helpers.request_step_limit_summary(
        type("Engine", (), {"runtime": Runtime()})(),
        type("Task", (), {"run_id": "run_native"})(),
        "解释多 Agent",
    )

    assert result == "这里是普通文本终答"
    assert "return ordinary answer text; do not use XML tags" in captured["completion"][0]
    assert "<final>...</final>" not in captured["completion"][0]
    assert captured["completion"][1:3] == (700, None)
    assert any(
        message.get("content") == "evidence_0"
        for message in captured["completion"][3]
    )
    assert captured["completion"][3][-1] == {
        "role": "user",
        "content": captured["completion"][3][-1]["content"],
    }
    assert captured["trace"]["kind"] == "native_final"


@pytest.mark.asyncio
async def test_step_limit_summary_times_out_without_rebuilding_context(monkeypatch):
    captured = {}

    class Runtime:
        model_client = object()
        max_new_tokens = 8192
        prefix = "cached prefix"
        context_manager = type(
            "Context",
            (),
            {"last_rendered_sections": {"prefix": "cached prefix", "history": "old evidence"}},
        )()

        async def _build_prompt_and_metadata_async(_request):
            pytest.fail("summary must reuse rendered context, not rebuild retrieval")

        @staticmethod
        def parse(_text):
            return ("error", None)

        @staticmethod
        def emit_trace(_task_state, event, payload):
            captured["trace"] = (event, payload)

    async def complete(*_args, **_kwargs):
        await asyncio.sleep(0.1)

    monkeypatch.setattr(engine_helpers, "STEP_LIMIT_SUMMARY_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(engine_helpers, "complete_model_async", complete)
    result = await engine_helpers.request_step_limit_summary(
        type("Engine", (), {"runtime": Runtime()})(),
        type("Task", (), {"run_id": "run_timeout"})(),
        "Explain the worker runtime",
    )

    assert result is None
    assert captured["trace"][0] == "step_limit_summary_failed"


def test_native_final_only_request_keeps_structured_messages_without_tools(monkeypatch):
    captured = {}

    class NativeClient:
        supports_native_tool_calling = True

        async def stream_messages(self, *_args, **_kwargs):
            yield None

    agent = type(
        "Runtime",
        (),
        {
            "model_client": NativeClient(),
            "context_manager": type("Context", (), {"last_rendered_sections": {}})(),
        },
    )()

    def stream_model(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "stream"

    monkeypatch.setattr(engine_helpers, "stream_model", stream_model)
    result = engine_helpers.complete_runtime_model(
        agent, "prompt", "question", 100, tools=[]
    )

    assert result == "stream"
    assert captured["kwargs"]["messages"] == [{"role": "user", "content": "prompt"}]
    assert captured["kwargs"]["tools"] == []
