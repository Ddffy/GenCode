"""Slash command registry and parsers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SlashCommand:
    name: str
    usage: str
    description: str
    aliases: tuple[str, ...] = field(default_factory=tuple)


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("help", "/help", "Show commands.", ("h",)),
    SlashCommand("clear", "/clear", "Create a new empty session."),
    SlashCommand("compact", "/compact", "Compact older session history."),
    SlashCommand("context", "/context", "Show prompt context usage."),
    SlashCommand("dream", "/dream", "Extract Skill/Wiki/Spec candidates from prior conversations."),
    SlashCommand("goal", "/goal <objective>", "Run a persistent, bounded DAG Goal with isolated Workers."),
    SlashCommand("history", "/history", "List saved sessions."),
    SlashCommand("knowledge", "/knowledge [approve|reject] [kind:]id", "List or review typed durable knowledge."),
    SlashCommand("memory", "/memory", "Alias for listing typed Skill/Wiki/Spec knowledge."),
    SlashCommand("mode", "/mode", "Show runtime mode."),
    SlashCommand("model", "/model [name]", "Show or switch the current model."),
    SlashCommand("plan", "/plan <topic>", "Enter plan mode."),
    SlashCommand("plan-exit", "/plan-exit", "Exit plan mode."),
    SlashCommand("remember", "/remember <text>", "Ask Dream to classify text into an inactive typed-knowledge candidate."),
    SlashCommand("reset", "/reset", "Reset current session memory and history."),
    SlashCommand("resume", "/resume <id|index|latest>", "Resume a saved session."),
    SlashCommand("session", "/session", "Show session status."),
    SlashCommand("skills", "/skills", "List available GenCode skills.", ("sk",)),
    SlashCommand("skill", "/skill <name> [args]", "Load and run a GenCode skill."),
    SlashCommand("spec", "/spec [use|clear] [id]", "List or bind durable task specifications."),
    SlashCommand("usage", "/usage", "Show model/provider usage metadata."),
    SlashCommand("undo", "/undo", "Reset the latest safe GenCode Git commit."),
    SlashCommand("working-memory", "/working-memory", "Show working memory."),
    SlashCommand("exit", "/exit", "Exit GenCode.", ("quit",)),
)


def command_help_text() -> str:
    lines = ["Commands:"]
    for command in SLASH_COMMANDS:
        lines.append(f"{command.usage:<32} {command.description}")
    return "\n".join(lines)


def resolve_command(name: str) -> SlashCommand | None:
    normalized = str(name or "").strip().lstrip("/").lower()
    if not normalized:
        return None
    for command in SLASH_COMMANDS:
        if normalized == command.name or normalized in command.aliases:
            return command
    return None


def suggest_commands(text: str, limit: int = 8) -> list[SlashCommand]:
    raw = str(text or "")
    if not raw.startswith("/"):
        return []
    body = raw[1:]
    if " " in body:
        return []
    token = body.lower()
    matches = []
    for command in SLASH_COMMANDS:
        names = (command.name, *command.aliases)
        if not token or any(name.startswith(token) for name in names):
            matches.append(command)
    return matches[:limit]
