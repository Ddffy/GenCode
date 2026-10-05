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


@dataclass(frozen=True)
class ModelStreamEvent:
    type: str
    text: str = ""
    result: ModelResult | None = None


async def stream_model(model_client, prompt, max_new_tokens, tools=None, messages=None, **kwargs):
    call_kwargs = dict(kwargs)
    if tools:
        call_kwargs["tools"] = tools
    if messages is not None:
        if not hasattr(model_client, "stream_messages"):
            raise TypeError(
                "model_client must implement async stream_messages() when structured messages are used"
            )
        stream = model_client.stream_messages(messages, max_new_tokens, **call_kwargs)
    elif hasattr(model_client, "stream_result"):
        stream = model_client.stream_result(prompt, max_new_tokens, **call_kwargs)
    else:
        raise TypeError(
            "model_client must implement async stream_result() or stream_messages()"
        )
    async for event in stream:
        yield event


async def complete_model_async(model_client, prompt, max_new_tokens, tools=None, messages=None, **kwargs):
    result = None
    async for event in stream_model(
        model_client, prompt, max_new_tokens, tools=tools, messages=messages, **kwargs
    ):
        if event.type == "completed":
            result = event.result
    if result is None:
        raise RuntimeError("model stream ended without a completion")
    return result


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
