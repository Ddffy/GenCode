import json
import time
import urllib.request

import pytest

import gencode.cli as gencode_cli
from gencode.core.runtime import GenCode
from gencode.core.model_router import ModelClientRouter
from gencode.core.session_store import SessionStore
from gencode.core.workspace import WorkspaceContext
from gencode.providers.clients import AnthropicCompatibleModelClient, OpenAICompatibleModelClient
from gencode.testing import ScriptedModelClient


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeResponse:
    def __init__(self, payload, content_type="application/json"):
        self.payload = payload
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RecordingVisionClient(ScriptedModelClient):
    def __init__(self, outputs):
        super().__init__(outputs)


def build_agent(tmp_path, model_client=None, model_client_router=None):
    (tmp_path / "README.md").write_text("demo\n", encoding="utf-8")
    workspace = WorkspaceContext.build(tmp_path)
    store = SessionStore(tmp_path / ".gencode" / "sessions")
    return GenCode(
        model_client=model_client or RecordingVisionClient(["A small red chart."]),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        max_steps=4,
        model_client_router=model_client_router,
    )


def write_png(tmp_path, name="chart.png"):
    path = tmp_path / name
    path.write_bytes(PNG_BYTES)
    return path


def test_openai_client_sends_image_blocks_in_responses_payload(monkeypatch):
    from gencode.core.content_blocks import ImageBlock, ModelInput

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"output_text": "<final>ok</final>"})

    client = OpenAICompatibleModelClient(
        model="vision-model",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
    )
    image = ImageBlock(
        path="chart.png",
        mime_type="image/png",
        data=PNG_BYTES,
        sha256="abc123",
    )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = client.complete(ModelInput(text="Describe it.", images=[image]), 64)

    assert result == "<final>ok</final>"
    content = captured["body"]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Describe it."}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")
    assert "abc123" not in content[1]["image_url"]
    assert client.last_completion_metadata["image_input_count"] == 1


def test_anthropic_client_sends_image_blocks_in_messages_payload(monkeypatch):
    from gencode.core.content_blocks import ImageBlock, ModelInput

    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"content": [{"type": "text", "text": "<final>ok</final>"}]})

    client = AnthropicCompatibleModelClient(
        model="claude-vision",
        base_url="https://example.test/v1",
        api_key="sk-test",
        temperature=None,
        timeout=30,
    )
    image = ImageBlock(
        path="chart.png",
        mime_type="image/png",
        data=PNG_BYTES,
        sha256="abc123",
    )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = client.complete(ModelInput(text="Describe it.", images=[image]), 64)

    assert result == "<final>ok</final>"
    content = captured["body"]["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[0]["source"]["data"]
    assert content[1] == {"type": "text", "text": "Describe it."}
    assert client.last_completion_metadata["image_input_count"] == 1


def test_deepseek_profile_defaults_to_openai_vision_provider(tmp_path, monkeypatch):
    from gencode.config import resolve_provider_config

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    config = resolve_provider_config("deepseek", start=tmp_path)

    assert config.name == "deepseek"
    assert config.model == "deepseek-v4-flash"
    assert config.supports_vision is False
    assert config.vision_provider == "openai"


def test_vision_provider_uses_vision_specific_env_overrides(tmp_path, monkeypatch):
    from gencode.config import resolve_vision_provider_config

    monkeypatch.setenv("GENCODE_OPENAI_API_KEY", "sk-text")
    monkeypatch.setenv("GENCODE_OPENAI_MODEL", "text-model")
    monkeypatch.setenv("GENCODE_OPENAI_API_BASE", "https://text.example/v1")
    monkeypatch.setenv("GENCODE_VISION_API_KEY", "sk-vision")
    monkeypatch.setenv("GENCODE_VISION_MODEL", "vision-model")
    monkeypatch.setenv("GENCODE_VISION_API_BASE", "https://vision.example/v1")

    config = resolve_vision_provider_config("openai", start=tmp_path)

    assert config.name == "openai"
    assert config.protocol == "openai"
    assert config.api_key == "sk-vision"
    assert config.model == "vision-model"
    assert config.base_url == "https://vision.example/v1"


def test_build_agent_uses_separate_vision_provider_for_deepseek(tmp_path, monkeypatch):
    args = gencode_cli.build_arg_parser().parse_args(
        ["--cwd", str(tmp_path), "--provider", "deepseek"]
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_MODEL", "vision-model")
    monkeypatch.setenv("OPENAI_API_BASE", "https://vision.example/v1")

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(gencode_cli, "AnthropicCompatibleModelClient", lambda **kwargs: ("anthropic", kwargs))
        patcher.setattr(gencode_cli, "OpenAICompatibleModelClient", lambda **kwargs: ("openai", kwargs))
        agent = gencode_cli.build_agent(args)
        vision_client = agent.model_client_router.vision_client()

    assert agent.model_client[0] == "anthropic"
    assert agent.model_client[1]["model"] == "deepseek-v4-flash"
    assert not hasattr(agent, "vision_model_client")
    assert vision_client[0] == "openai"
    assert vision_client[1]["model"] == "vision-model"
    assert vision_client[1]["base_url"] == "https://vision.example/v1"


def test_build_agent_uses_vision_specific_client_overrides(tmp_path, monkeypatch):
    args = gencode_cli.build_arg_parser().parse_args(
        ["--cwd", str(tmp_path), "--provider", "deepseek", "--vision-timeout", "45"]
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-text")
    monkeypatch.setenv("OPENAI_MODEL", "text-model")
    monkeypatch.setenv("OPENAI_API_BASE", "https://text.example/v1")
    monkeypatch.setenv("GENCODE_VISION_API_KEY", "sk-vision")
    monkeypatch.setenv("GENCODE_VISION_MODEL", "vision-model")
    monkeypatch.setenv("GENCODE_VISION_API_BASE", "https://vision.example/v1")

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(gencode_cli, "AnthropicCompatibleModelClient", lambda **kwargs: ("anthropic", kwargs))
        patcher.setattr(gencode_cli, "OpenAICompatibleModelClient", lambda **kwargs: ("openai", kwargs))
        agent = gencode_cli.build_agent(args)
        vision_client = agent.model_client_router.vision_client()

    assert agent.model_client[0] == "anthropic"
    assert agent.model_client[1]["model"] == "deepseek-v4-flash"
    assert vision_client[0] == "openai"
    assert vision_client[1]["api_key"] == "sk-vision"
    assert vision_client[1]["model"] == "vision-model"
    assert vision_client[1]["base_url"] == "https://vision.example/v1"
    assert vision_client[1]["timeout"] == 45


def test_inspect_image_uses_separate_vision_model_when_configured(tmp_path):
    from gencode.core.content_blocks import ModelInput
    from gencode.core.task_state import TaskState

    write_png(tmp_path)
    main_client = RecordingVisionClient(["unused main model output"])
    vision_client = RecordingVisionClient(["vision provider summary"])
    router = ModelClientRouter(main_client=main_client, vision_client=vision_client)
    agent = build_agent(tmp_path, model_client=main_client, model_client_router=router)
    task_state = TaskState.create(run_id="run_direct", task_id="task_direct", user_request="inspect")
    agent.current_task_state = task_state
    agent.current_run_dir = agent.run_store.start_run(task_state)

    result = agent.run_tool(
        "inspect_image",
        {"path": "chart.png", "question": "What is shown?", "profile": "general"},
    )

    assert "vision provider summary" in result
    assert main_client.prompts == []
    assert isinstance(vision_client.prompts[0], ModelInput)
    assert vision_client.prompts[0].images[0].path == "chart.png"


def test_inspect_image_keeps_medium_summary_inline(tmp_path):
    from gencode.core.task_state import TaskState

    write_png(tmp_path)
    summary = "vision detail\n" * 180
    client = RecordingVisionClient([summary])
    agent = build_agent(tmp_path, model_client=client)
    task_state = TaskState.create(run_id="run_direct", task_id="task_direct", user_request="inspect")
    agent.current_task_state = task_state
    agent.current_run_dir = agent.run_store.start_run(task_state)

    result = agent.run_tool("inspect_image", {"path": "chart.png", "question": "What is shown?"})

    assert summary.strip() in result
    assert "full output saved:" not in result
    assert agent._last_tool_result_metadata["full_output_artifact"] == ""


def test_image_inspection_prompt_preserves_complete_ocr_extraction():
    from gencode.core.vision import image_inspection_prompt

    prompt = image_inspection_prompt(
        "rows.png",
        "Extract all data rows from this screenshot.",
        "general",
        "",
    )

    assert "complete requested extraction" in prompt
    assert "Do not omit rows" in prompt
    assert "Return concise" not in prompt


def test_vision_model_call_has_total_timeout(monkeypatch):
    from gencode.core.content_blocks import ModelInput
    from gencode.core.vision import complete_model_with_timeout

    class SlowClient:
        timeout = 0.01

    def slow_complete(*_args, **_kwargs):
        time.sleep(0.2)

    monkeypatch.setattr("gencode.core.vision.complete_model", slow_complete)

    with pytest.raises(TimeoutError, match="vision provider request exceeded"):
        complete_model_with_timeout(SlowClient(), ModelInput(text="describe"), 64)


def test_load_workspace_image_rejects_path_escape_and_records_safe_metadata(tmp_path):
    from gencode.core.media import load_workspace_image

    write_png(tmp_path)
    outside = tmp_path.parent / "outside.png"
    outside.write_bytes(PNG_BYTES)
    agent = build_agent(tmp_path)

    loaded = load_workspace_image(agent, "chart.png")
    assert loaded.block.mime_type == "image/png"
    assert loaded.metadata["path"] == "chart.png"
    assert loaded.metadata["sha256"] == loaded.block.sha256
    assert "base64" not in json.dumps(loaded.metadata)

    with pytest.raises(ValueError, match="path escapes workspace"):
        load_workspace_image(agent, "../outside.png")


def test_load_workspace_image_rejects_fake_image_extension(tmp_path):
    from gencode.core.media import load_workspace_image

    (tmp_path / "fake.png").write_text("not really a png\n", encoding="utf-8")
    agent = build_agent(tmp_path)

    with pytest.raises(ValueError, match="unsupported image type"):
        load_workspace_image(agent, "fake.png")


def test_run_store_writes_binary_artifact(tmp_path):
    from gencode.core.task_state import TaskState

    agent = build_agent(tmp_path)
    task_state = TaskState.create(run_id="run_test", task_id="task_test", user_request="inspect")
    agent.current_run_dir = agent.run_store.start_run(task_state)

    path = agent.run_store.write_binary_artifact(task_state, "image", PNG_BYTES, ".png")
    assert path.read_bytes() == PNG_BYTES
    assert agent.run_store.artifact_ref(task_state, path).endswith(".png")


def test_inspect_image_tool_calls_model_with_model_input_and_records_media_refs(tmp_path):
    from gencode.core.content_blocks import ModelInput
    from gencode.core.task_state import TaskState

    write_png(tmp_path)
    client = RecordingVisionClient(["The image contains a one-pixel chart."])
    agent = build_agent(tmp_path, model_client=client)
    task_state = TaskState.create(run_id="run_direct", task_id="task_direct", user_request="inspect")
    agent.current_task_state = task_state
    agent.current_run_dir = agent.run_store.start_run(task_state)

    result = agent.run_tool(
        "inspect_image",
        {"path": "chart.png", "question": "What is shown?", "profile": "general"},
    )

    assert "The image contains a one-pixel chart." in result
    assert isinstance(client.prompts[0], ModelInput)
    assert client.prompts[0].images[0].mime_type == "image/png"
    media_refs = agent._last_tool_result_metadata["media_refs"]
    assert media_refs[0]["path"] == "chart.png"
    assert media_refs[0]["artifact_ref"].endswith(".png")
    assert "base64" not in json.dumps(agent._last_tool_result_metadata)


def test_inspect_image_tool_trace_and_history_do_not_store_base64(tmp_path):
    write_png(tmp_path)
    client = RecordingVisionClient(
        [
            '<tool>{"name":"inspect_image","args":{"path":"chart.png","question":"Describe it"}}</tool>',
            "one-pixel image",
            "<final>Image inspected.</final>",
        ]
    )
    agent = build_agent(tmp_path, model_client=client)

    events = list(agent.engine.run_turn("inspect chart.png"))

    assert any(event["type"] == "final" for event in events)
    trace_text = (agent.current_run_dir / "trace.jsonl").read_text(encoding="utf-8")
    assert "media_refs" in trace_text
    assert "base64" not in trace_text

    prompt, metadata = agent.context_manager.build("continue")
    assert "[image]" in prompt
    assert "chart.png" in prompt
    assert "base64" not in prompt
    assert metadata["history"]["summarized_tool_count"] >= 0


def test_same_image_inspection_is_budgeted_per_turn(tmp_path):
    write_png(tmp_path)
    client = RecordingVisionClient(
        [
            '<tool>{"name":"inspect_image","args":{"path":"chart.png","question":"first pass"}}</tool>',
            "first observations",
            '<tool>{"name":"inspect_image","args":{"path":"chart.png","question":"second pass"}}</tool>',
            "second observations",
            '<tool>{"name":"inspect_image","args":{"path":"chart.png","question":"third pass"}}</tool>',
            "<final>done</final>",
        ]
    )
    agent = build_agent(tmp_path, model_client=client)

    events = list(agent.engine.run_turn("inspect chart.png repeatedly"))

    assert any(event["type"] == "final" for event in events)
    tool_items = [item for item in agent.session["history"] if item.get("name") == "inspect_image"]
    assert len(tool_items) == 3
    assert "first observations" in tool_items[0]["content"]
    assert "second observations" in tool_items[1]["content"]
    assert tool_items[2]["content"].startswith("error: repeated identical tool call")
