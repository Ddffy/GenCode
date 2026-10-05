"""Tool abstraction shared by the runtime and prompt builder."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum


class ToolEffect(str, Enum):
    """Declared side-effect class used by runtime scheduling policy."""

    UNKNOWN = "unknown"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    EXTERNAL = "external"


@dataclass(frozen=True)
class ToolCapability:
    """Fail-closed execution properties; not exposed as model arguments."""

    effect: ToolEffect = ToolEffect.UNKNOWN
    concurrency_safe: bool = False


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    schema: dict
    description: str
    risky: bool
    runner: Callable[[dict], Awaitable[str]]
    capability: ToolCapability = ToolCapability()

    @property
    def read_only(self):
        return not self.risky

    async def execute(self, args):
        result = await self.runner(args)
        if isinstance(result, ToolResult):
            return result
        return ToolResult(content=str(result))

    def __getitem__(self, key):
        if key == "run":
            return self.runner
        return getattr(self, key)
