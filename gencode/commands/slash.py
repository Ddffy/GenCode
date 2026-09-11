"""Slash command registry and parsers."""

from __future__ import annotations

import shlex
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
    SlashCommand("dream", "/dream", "Consolidate durable memory."),
    SlashCommand("history", "/history", "List saved sessions."),
    SlashCommand("memory", "/memory", "Show durable memory index."),
    SlashCommand("mode", "/mode", "Show runtime mode."),
    SlashCommand("model", "/model [name]", "Show or switch the current model."),
    SlashCommand("plan", "/plan <topic>", "Enter plan mode."),
    SlashCommand("plan-exit", "/plan-exit", "Exit plan mode."),
    SlashCommand("remember", "/remember <text>", "Save a durable memory note."),
    SlashCommand("reset", "/reset", "Reset current session memory and history."),
    SlashCommand("resume", "/resume <id|index|latest>", "Resume a saved session."),
    SlashCommand("session", "/session", "Show session status."),
    SlashCommand("skills", "/skills", "List available GenCode skills.", ("sk",)),
    SlashCommand("skill", "/skill <name> [args]", "Load and run a GenCode skill."),
    SlashCommand("agents", "/agents", "Show subagent worker status.", ("agent",)),
    SlashCommand(
        "subagent",
        "/subagent explore <task>",
        "Launch a bounded local child run: Explore or scoped worker.",
        ("sub",),
    ),
    SlashCommand("usage", "/usage", "Show model/provider usage metadata."),
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


def parse_subagent_args(args: str) -> tuple[dict | None, str]:
    usage = "Usage: /subagent explore <task> or /subagent worker --scope <path[,path]> <task>"
    try:
        tokens = shlex.split(str(args or ""))
    except ValueError as exc:
        return None, f"{usage}. {exc}"
    if not tokens:
        return None, usage

    subagent_type = "Explore"
    if tokens[0].lower() in {"explore", "worker"}:
        subagent_type = "worker" if tokens.pop(0).lower() == "worker" else "Explore"

    write_scope: list[str] = []
    task_parts: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--scope":
            index += 1
            if index >= len(tokens):
                return None, usage
            write_scope.extend(_split_scope(tokens[index]))
        elif token.startswith("--scope="):
            write_scope.extend(_split_scope(token.split("=", 1)[1]))
        else:
            task_parts.append(token)
        index += 1

    prompt = " ".join(task_parts).strip()
    if not prompt:
        return None, usage
    if subagent_type == "worker" and not write_scope:
        return None, usage
    return {
        "description": prompt[:80],
        "prompt": prompt,
        "subagent_type": subagent_type,
        "write_scope": write_scope,
    }, ""


def _split_scope(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]
