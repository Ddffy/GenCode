"""Provider-facing result types."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelResult:
    text: str
    metadata: dict = field(default_factory=dict)
    # Provider-native function/tool calls.  Each item is normalized to
    # {id, name, args}; the runtime does not need to know whether it came from
    # OpenAI Responses, Chat Completions, or Anthropic Messages.
    tool_calls: tuple = field(default_factory=tuple)
    stop_reason: str = ""


def complete_model(model_client, prompt, max_new_tokens, tools=None, messages=None, **kwargs):
    """Call a model while keeping the legacy prompt contract intact.

    Native-capable clients may expose ``complete_messages``.  The engine uses
    that method only when it has assembled structured history; lightweight test
    clients and legacy providers continue to receive the original string
    prompt.  This makes the protocol migration opt-in instead of changing the
    benchmark contract in one step.
    """
    call_kwargs = dict(kwargs)
    if tools:
        call_kwargs["tools"] = tools
    if messages is not None and hasattr(model_client, "complete_messages"):
        return model_client.complete_messages(
            messages,
            max_new_tokens,
            **call_kwargs,
        )
    if hasattr(model_client, "complete_result"):
        return model_client.complete_result(prompt, max_new_tokens, **call_kwargs)
    text = model_client.complete(prompt, max_new_tokens, **call_kwargs)
    metadata = dict(getattr(model_client, "last_completion_metadata", {}) or {})
    return ModelResult(text=str(text), metadata=metadata)
