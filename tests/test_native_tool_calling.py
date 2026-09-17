import json
from unittest.mock import patch

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.providers.base import ModelResult
from gencode.providers import AnthropicCompatibleModelClient, OpenAICompatibleModelClient
from gencode.testing import NativeScriptedModelClient


TOOL = {
    "name": "read_file",
    "description": "Read a file.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
}


class _Response:
    headers = {"Content-Type": "application/json"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_openai_native_tool_call_is_normalized_and_schema_is_sent():
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    }
                ]
            }
        )

    client = OpenAICompatibleModelClient(
        model="gpt-test",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=0.0,
        timeout=30,
    )
    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete_result("inspect the readme", 64, tools=[TOOL])

    assert captured["body"]["tools"][0]["type"] == "function"
    assert captured["body"]["tools"][0]["parameters"] == TOOL["parameters"]
    assert result.tool_calls == (
        {"id": "call_1", "name": "read_file", "args": {"path": "README.md"}},
    )
    assert result.metadata["native_tool_calling"] is True


def test_anthropic_native_tool_use_is_normalized_and_schema_is_sent():
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response(
            {
                "stop_reason": "tool_use",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    }
                ],
            }
        )

    client = AnthropicCompatibleModelClient(
        model="claude-test",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=0.0,
        timeout=30,
    )
    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete_result("inspect the readme", 64, tools=[TOOL])

    assert captured["body"]["tools"][0]["input_schema"] == TOOL["parameters"]
    assert captured["body"]["tool_choice"] == {"type": "auto"}
    assert result.tool_calls == (
        {"id": "toolu_1", "name": "read_file", "args": {"path": "README.md"}},
    )
    assert result.stop_reason == "tool_use"


def test_openai_native_messages_replay_function_call_output():
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _Response({"output_text": "done"})

    client = OpenAICompatibleModelClient(
        model="gpt-test",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=0.0,
        timeout=30,
    )
    messages = [
        {"role": "system", "content": "You are gencode."},
        {"role": "user", "content": "Read README."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "read_file", "content": "hello"},
    ]
    with patch("urllib.request.urlopen", fake_urlopen):
        result = client.complete_messages(messages, 64, tools=[TOOL])

    assert result.text == "done"
    assert captured["body"]["input"][-2]["type"] == "function_call"
    assert captured["body"]["input"][-1] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "hello",
    }
    assert result.metadata["native_message_mode"] is True


def test_native_engine_uses_structured_message_entry_point(tmp_path):
    (tmp_path / "README.md").write_text("native messages\n", encoding="utf-8")
    client = NativeScriptedModelClient(
        [
            ModelResult(
                text="",
                metadata={"native_tool_calling": True},
                tool_calls=({"id": "call-1", "name": "read_file", "args": {"path": "README.md"}},),
                stop_reason="tool_use",
            ),
            ModelResult(text="done", metadata={"native_tool_calling": True}),
        ]
    )
    agent = GenCode(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )

    assert agent.ask("Read README") == "done"
    assert len(client.messages) == 2
    replay = client.messages[1]
    assistant = next(message for message in replay if message.get("role") == "assistant")
    tool = next(message for message in replay if message.get("role") == "tool")
    assert assistant["tool_calls"][0]["id"] == "call-1"
    assert tool["tool_call_id"] == "call-1"


def test_engine_executes_native_call_and_accepts_plain_final_text(tmp_path):
    (tmp_path / "README.md").write_text("native path\n", encoding="utf-8")

    class FakeNativeClient:
        supports_native_tool_calling = True

        def __init__(self):
            self.calls = 0

        def complete_result(self, prompt, max_new_tokens, tools=None, **kwargs):
            del prompt, max_new_tokens, kwargs
            self.calls += 1
            assert tools
            if self.calls == 1:
                return ModelResult(
                    text="I will inspect the README.",
                    metadata={
                        "native_tool_calling": True,
                        "native_tool_call_count": 1,
                    },
                    tool_calls=(
                        {
                            "id": "call_1",
                            "name": "read_file",
                            "args": {"path": "README.md"},
                        },
                    ),
                    stop_reason="tool_use",
                )
            return ModelResult(
                text="README inspected successfully.",
                metadata={"native_tool_calling": True},
            )

    client = FakeNativeClient()
    agent = GenCode(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".gencode" / "sessions"),
        approval_policy="auto",
    )

    assert agent.ask("Inspect README.md") == "README inspected successfully."
    assert client.calls == 2
    assert any(
        item.get("role") == "tool" and item.get("name") == "read_file"
        for item in agent.session["history"]
    )


def _multi_call_history(prepend):
    """History whose 16-item tail can land inside a multi-call group."""
    items = [{"role": "user", "content": f"历史消息 {index}"} for index in range(prepend)]
    index = 0
    for width in (2, 1, 2, 1, 2, 1, 2, 1):
        items.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": f"call_{index + offset}", "name": "read_file",
                     "args": {"path": f"f{index + offset}.py"}}
                    for offset in range(width)
                ],
            }
        )
        items.extend(
            {
                "role": "tool",
                "tool_call_id": f"call_{index + offset}",
                "name": "read_file",
                "content": f"内容 {index + offset}",
            }
            for offset in range(width)
        )
        index += width
    return items


def test_native_history_window_never_splits_a_tool_call_group():
    """回归：窗口盲切会留下孤儿 tool_result，Anthropic 直接拒绝该请求。

    报错原文是 "each tool_result block must have a corresponding tool_use block"，
    位置固定在 messages[0].content[0]——也就是被切掉调用方的那条结果。
    """
    from gencode.core.native_messages import _native_history_tail
    from gencode.providers.clients import _anthropic_messages_to_payload

    for prepend in range(14):
        canonical, _count = _native_history_tail(_multi_call_history(prepend))
        _system, converted = _anthropic_messages_to_payload(
            [
                {"role": "system", "content": "system"},
                *canonical,
                {"role": "user", "content": "现在的问题"},
            ]
        )
        use_ids = {
            block.get("id")
            for message in converted
            if message["role"] == "assistant"
            for block in (message["content"] if isinstance(message["content"], list) else [])
            if isinstance(block, dict)
        }
        result_ids = [
            block["tool_use_id"]
            for message in converted
            for block in (message["content"] if isinstance(message["content"], list) else [])
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        assert result_ids, "窗口内应当仍然保留工具结果"
        assert not [rid for rid in result_ids if rid not in use_ids], (
            f"prepend={prepend} 时出现孤儿 tool_result"
        )


def test_native_history_drops_tool_results_without_their_call():
    from gencode.core.native_messages import _native_history_tail

    history = [
        {"role": "user", "content": "做点事"},
        {"role": "tool", "tool_call_id": "call_ghost", "name": "read_file",
         "content": "没有配对调用的结果"},
    ]

    messages, _count = _native_history_tail(history)

    assert [message["role"] for message in messages] == ["user"], "孤儿结果必须被丢弃"


def _pairing(converted):
    """(orphan tool_use ids, orphan tool_result ids) in an Anthropic payload."""
    uses, results = [], []
    for message in converted:
        blocks = message["content"] if isinstance(message["content"], list) else []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                uses.append(block["id"])
            elif block.get("type") == "tool_result":
                results.append(block["tool_use_id"])
    return [i for i in uses if i not in results], [i for i in results if i not in uses]


def test_native_history_drops_tool_use_without_a_result():
    """回归：步数预算或中断截断一批调用时，历史里会留下没有结果的 tool_use。

    服务端报 "tool_use ids were found without tool_result blocks immediately
    after"，而且是**下一次请求**才报——所以它会让会话永久不可用，不只是当轮失败。
    这里断言已经损坏的历史仍然能被发送出去。
    """
    from gencode.core.native_messages import _native_history_tail
    from gencode.providers.clients import _anthropic_messages_to_payload

    damaged = [
        {"role": "user", "content": "做事"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_a", "name": "read_file", "args": {"path": "a.py"}},
                {"id": "call_b", "name": "read_file", "args": {"path": "b.py"}},
                {"id": "call_c", "name": "read_file", "args": {"path": "c.py"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_a", "name": "read_file", "content": "A"},
        {"role": "tool", "tool_call_id": "call_b", "name": "read_file", "content": "B"},
        # call_c 的结果从未写入——正是截断留下的形状
    ]

    canonical, _count = _native_history_tail(damaged)
    _system, converted = _anthropic_messages_to_payload(
        [{"role": "system", "content": "s"}, *canonical, {"role": "user", "content": "继续"}]
    )

    orphan_uses, orphan_results = _pairing(converted)
    assert not orphan_uses, "没有结果的 tool_use 必须被丢弃"
    assert not orphan_results
    use_ids = [
        block["id"]
        for message in converted
        for block in (message["content"] if isinstance(message["content"], list) else [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    assert "call_c" not in use_ids
    assert "call_a" in use_ids and "call_b" in use_ids, "有结果的调用必须保留"


def test_skipped_tool_call_records_a_result_for_the_orphan():
    """产生源修复：预算用尽时不是 break，而是给被跳过的调用补一条结果。"""
    from gencode.core.engine_helpers import record_skipped_tool_call

    class _TaskState:
        run_id = "run-1"

    class _Agent:
        def __init__(self):
            self.recorded = []

        def record(self, item):
            self.recorded.append(item)

        def emit_trace(self, *args, **kwargs):
            pass

    class _Engine:
        def __init__(self):
            self.runtime = _Agent()

    engine = _Engine()
    events = list(
        record_skipped_tool_call(
            engine,
            _TaskState(),
            {"name": "read_file", "args": {"path": "c.py"}, "id": "call_c"},
            reason="step_budget_exhausted",
        )
    )

    item = engine.runtime.recorded[0]
    assert item["role"] == "tool"
    assert item["tool_call_id"] == "call_c"
    assert item["tool_error_code"] == "step_budget_exhausted"
    assert item["workspace_changed"] is False
    assert "skipped" in item["content"]
    assert events[0]["type"] == "tool_result"
