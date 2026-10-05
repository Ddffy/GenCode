"""Testing helpers for deterministic GenCode runtime checks."""

from .providers.base import ModelResult, ModelStreamEvent


class ScriptedModelClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.prompts = []
        self.supports_prompt_cache = False
        self.last_completion_metadata = {}

    def complete(self, prompt, max_new_tokens, **kwargs):
        self.prompts.append(prompt)
        if not getattr(self, "last_completion_metadata", None):
            self.last_completion_metadata = {}
        if not self.outputs:
            raise RuntimeError("scripted model ran out of outputs")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        return output

    def complete_result(self, prompt, max_new_tokens, **kwargs):
        return ModelResult(
            text=self.complete(prompt, max_new_tokens, **kwargs),
            metadata=dict(self.last_completion_metadata),
        )

    async def stream_result(self, prompt, max_new_tokens, **kwargs):
        result = self.complete_result(prompt, max_new_tokens, **kwargs)
        for offset in range(0, len(result.text), 128):
            yield ModelStreamEvent("text_delta", result.text[offset : offset + 128])
        yield ModelStreamEvent("completed", result=result)


class NativeScriptedModelClient:
    """Deterministic structured-message client for native Tool Calling tests.

    It deliberately does not implement the legacy ``complete`` method.  This
    makes tests fail loudly if the Engine accidentally falls back to the text
    protocol while exercising the native path, while the existing
    ``ScriptedModelClient`` keeps all legacy benchmark scripts unchanged.
    """

    supports_native_tool_calling = True
    supports_prompt_cache = False

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.messages = []
        self.tools_seen = []
        self.last_completion_metadata = {}

    def complete_messages(self, messages, max_new_tokens, tools=None, **kwargs):
        del max_new_tokens, kwargs
        self.messages.append([dict(message) for message in messages])
        self.tools_seen.append(list(tools or []))
        if not self.outputs:
            raise RuntimeError("native scripted model ran out of outputs")
        output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        if isinstance(output, ModelResult):
            self.last_completion_metadata = dict(output.metadata or {})
            return output
        return ModelResult(
            text=str(output),
            metadata={"native_tool_calling": True},
        )

    async def stream_messages(self, messages, max_new_tokens, tools=None, **kwargs):
        result = self.complete_messages(messages, max_new_tokens, tools, **kwargs)
        for offset in range(0, len(result.text), 128):
            yield ModelStreamEvent("text_delta", result.text[offset : offset + 128])
        yield ModelStreamEvent("completed", result=result)
