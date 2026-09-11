"""Model-error cleanup, trace, checkpoint, and user messaging path."""

from ..providers.errors import ProviderError
from .turn_transitions import emit_terminal_transition
from .workspace import clip, now


def finish_model_error(engine, task_state, user_message, prompt_metadata, exc, duration_ms, run_duration_ms):
    agent = engine.runtime
    error_metadata = _error_metadata(exc)
    prompt_metadata.update(error_metadata)
    agent.last_completion_metadata = error_metadata
    agent.last_prompt_metadata = prompt_metadata
    error = dict(error_metadata.get("provider_error", {}))
    code = str(error.get("code") or "model_error")
    if code == "empty_response":
        final = (
            "模型返回空响应。可能原因：max_new_tokens 太小（当前 "
            f"{agent.max_new_tokens}）、provider 临时异常、或 prompt 超出窗口。"
            "建议：加大 --max-new-tokens、检查 provider 状态、或减少历史长度后重试。"
        )
    elif code in {"prompt_too_long", "context_length_exceeded"}:
        final = (
            f"Prompt 超出模型上下文窗口（{code}）。建议：/compact 压缩历史、"
            "或 /clear 开新 session。"
        )
    else:
        final = f"模型错误：{code}（{clip(str(error.get('body_excerpt', '')), 200)}）"
    task_state.stop_model_error(final)
    agent.record({"role": "assistant", "content": final, "created_at": now()})
    agent.emit_trace(
        task_state,
        "model_error",
        {"duration_ms": duration_ms, "completion_metadata": error_metadata, "error": error},
    )
    agent.session_event_bus.emit(
        "model_error",
        {"run_id": task_state.run_id, "code": code, "retryable": bool(error.get("retryable")), "duration_ms": duration_ms},
    )
    agent.session_event_bus.emit(
        "assistant_message",
        {"run_id": task_state.run_id, "kind": "model_error", "content": final},
    )
    checkpoint = agent.create_checkpoint(task_state, user_message, trigger="model_error")
    agent.run_store.write_task_state(task_state)
    emit_terminal_transition(
        agent, task_state, reason=task_state.stop_reason,
        stop_reason=task_state.stop_reason,
    )
    agent.emit_trace(
        task_state,
        "checkpoint_created",
        {"checkpoint_id": checkpoint["checkpoint_id"], "trigger": "model_error"},
    )
    agent.emit_trace(
        task_state,
        "run_finished",
        {
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "final_answer": final,
            "run_duration_ms": run_duration_ms,
        },
    )
    agent.session_event_bus.emit(
        "turn_finished",
        {
            "run_id": task_state.run_id,
            "status": task_state.status,
            "stop_reason": task_state.stop_reason,
            "duration_ms": run_duration_ms,
        },
    )
    agent.run_store.write_report(task_state, agent.redact_artifact(agent.build_report(task_state)))
    agent.current_turn_id = ""
    agent.current_run_id = ""
    yield {"type": "stop", "run_id": task_state.run_id, "content": final}
    yield {
        "type": "turn_finished",
        "run_id": task_state.run_id,
        "status": task_state.status,
        "stop_reason": task_state.stop_reason,
    }


def _error_metadata(exc):
    if isinstance(exc, ProviderError):
        return exc.to_metadata()
    return {
        "provider_error": {
            "code": "model_client_error",
            "retryable": False,
            "attempts": 1,
            "retry_count": 0,
            "cause_type": type(exc).__name__,
            "body_excerpt": clip(str(exc), 500),
        }
    }
