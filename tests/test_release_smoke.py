"""Release smoke tests.

This module verifies the two end-to-end user journeys that must work before
public release:

1. **basic edit flow** — read a file, propose a patch, apply it
2. **dream consolidation** — auto-dream produces non-empty topic files

These tests use ScriptedModelClient (deterministic) by default so CI always runs.
Set GENCODE_LIVE_SMOKE=1 with a provider configured to run them against a real model.
"""
import os
import textwrap

import pytest

from gencode import GenCode, SessionStore, WorkspaceContext
from gencode.testing import ScriptedModelClient


def _build_workspace(tmp_path):
    (tmp_path / "README.md").write_text(
        "# demo\n\nQuick Start coming soon.\n", encoding="utf-8"
    )
    (tmp_path / "TODO").write_text("- [ ] write docs\n", encoding="utf-8")
    return WorkspaceContext.build(tmp_path)


def _build_agent(tmp_path, outputs):
    workspace = _build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".gencode" / "sessions")
    return GenCode(
        model_client=ScriptedModelClient(outputs),
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        auto_dream=False,
    )


def test_read_then_edit_completes_in_one_turn(tmp_path):
    """模型读 README → 改一行 → 完成。验证读取-编辑闭环。"""
    agent = _build_agent(
        tmp_path,
        [
            '<tool>{"name":"read_file","args":{"path":"README.md","start":1,"end":20}}</tool>',
            '<tool>{"name":"patch_file","args":{"path":"README.md","old_text":"Quick Start coming soon.","new_text":"Quick Start: gencode --help"}}</tool>',
            "<final>已把 Quick Start 段从占位文本改成实际的 gencode --help 提示。</final>",
        ],
    )

    final = agent.engine.ask("把 README 里的 Quick Start 段从占位文本改成实际命令")

    assert "Quick Start" in final
    updated = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "gencode --help" in updated
    assert "Quick Start coming soon" not in updated


def test_search_then_summarize_succeeds(tmp_path):
    """模型 search 找 TODO 标记 → 汇报。验证只读探索流程。"""
    agent = _build_agent(
        tmp_path,
        [
            '<tool>{"name":"search","args":{"pattern":"TODO","path":"."}}</tool>',
            "<final>workspace 里有 1 条 TODO：在 TODO 文件里。</final>",
        ],
    )

    final = agent.engine.ask("workspace 里有多少条 TODO？")

    assert "TODO" in final
    history = agent.session["history"]
    tool_calls = [item for item in history if item["role"] == "tool"]
    assert tool_calls and tool_calls[0]["name"] == "search"


def test_step_limit_default_can_handle_realistic_workflows(tmp_path):
    """默认 max_steps 应该够 read+search+edit+verify 这种流程，且能扛全栈
    脚手架（17+ 文件写入）。"""
    outputs = []
    for i in range(7):
        outputs.append(
            f'<tool>{{"name":"read_file","args":{{"path":"README.md","start":1,"end":{i+5}}}}}</tool>'
        )
    outputs.append("<final>看完了。</final>")

    agent = _build_agent(tmp_path, outputs)
    assert agent.max_steps >= 50

    final = agent.engine.ask("仔细读一遍 README，确认结构 OK")

    assert "看完了" in final
    tool_calls = [item for item in agent.session["history"] if item["role"] == "tool"]
    assert len(tool_calls) == 7


def test_empty_response_does_not_silently_stop(tmp_path):
    """empty_response 错误必须用户可见，不再'Stopped after model error' 静默。"""
    from gencode.providers.errors import ProviderError

    err = ProviderError(
        "empty",
        provider="anthropic",
        model="test",
        base_url="http://test",
        code="empty_response",
        retryable=False,
    )
    # 两次空响应：第一次会被 should_retry_model_error 重试一次，第二次报失败
    agent = _build_agent(tmp_path, [err, err])

    final = agent.engine.ask("hello")

    assert "空响应" in final or "empty_response" in final
    # 不能是历史的静默"Stopped after model error"格式
    assert not final.startswith("Stopped after")


def _has_live_provider():
    keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
    return any(os.environ.get(k) for k in keys) and os.environ.get("GENCODE_LIVE_SMOKE") == "1"


@pytest.mark.skipif(
    not _has_live_provider(),
    reason="set GENCODE_LIVE_SMOKE=1 and a provider API key to run against a real model",
)
def test_dream_produces_non_empty_topics_with_live_provider(tmp_path):
    """End-to-end: 真实 provider 跑一次 dream，topics/ 必须产出非空文件。"""
    from gencode.config import resolve_provider_config
    from gencode.providers import (
        AnthropicCompatibleModelClient,
        OpenAICompatibleModelClient,
    )

    workspace = _build_workspace(tmp_path)
    store = SessionStore(tmp_path / ".gencode" / "sessions")

    config = resolve_provider_config(None, start=str(tmp_path))
    client_cls = (
        OpenAICompatibleModelClient if config.protocol == "openai" else AnthropicCompatibleModelClient
    )
    model = client_cls(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=0.0,
        timeout=180,
    )

    log_path = tmp_path / ".gencode" / "memory" / "logs" / "2026" / "05" / "2026-05-13.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        textwrap.dedent(
            """\
            # 2026-05-13 daily log

            - 测试项目使用 pytest 而非 unittest
            - 部署前必须运行 `make test` 和 `make lint`
            - 切勿提交真实的 API key
            """
        ),
        encoding="utf-8",
    )

    agent = GenCode(
        model_client=model,
        workspace=workspace,
        session_store=store,
        approval_policy="auto",
        auto_dream=False,
    )

    agent.run_dream()

    topics_dir = tmp_path / ".gencode" / "memory" / "topics"
    written = [p for p in topics_dir.glob("*.md") if p.read_text(encoding="utf-8").strip()]
    assert written, "dream 必须至少产出一个非空 topic 文件"

    index = (tmp_path / ".gencode" / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "topics/" in index
