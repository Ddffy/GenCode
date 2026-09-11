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
