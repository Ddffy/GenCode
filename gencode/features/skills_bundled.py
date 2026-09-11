"""Built-in GenCode skills."""

from __future__ import annotations

from .skills import Skill


def bundled_skills():
    return [
        Skill(
            name="simplify",
            description="Review changed code for reuse, quality, and efficiency, then fix issues found",
            when_to_use="After making code changes, to clean up and improve the code",
            argument_hint="focus",
            source="builtin",
            prompt_fn=_with_optional_section(
                "# Simplify: Code Review and Cleanup",
                [
                    "Run git diff, inspect changed files, remove duplication, simplify overly complex logic, and fix issues directly.",
                    "After changes, run relevant tests or linters and report exact verification.",
                ],
                "Additional Focus",
            ),
        ),
        Skill(
            name="review",
            description="Review code changes and report issues without making fixes",
            when_to_use="Before committing or merging code changes",
            argument_hint="focus",
            source="builtin",
            prompt_fn=_with_optional_section(
                "# Code Review",
                [
                    "Inspect git status and diff first.",
                    "Report correctness, security, performance, readability, and missing-test findings by severity.",
                    "Do not modify files.",
                ],
                "Additional Focus",
            ),
        ),
        Skill(
            name="commit",
            description="Create a focused git commit from the current staged changes",
            when_to_use="When ready to commit a coherent change",
            argument_hint="message",
            source="builtin",
            prompt_fn=_with_optional_section(
                "# Git Commit",
                [
                    "Inspect git status and staged diff.",
                    "Stage only coherent task changes if needed, then create a concise conventional commit.",
                    "Do not include unrelated files.",
                ],
                "User Instructions",
            ),
        ),
        Skill(
            name="test",
            description="Run the project's test suite and analyze results",
            when_to_use="To verify code changes with the relevant test path",
            argument_hint="filter",
            source="builtin",
            prompt_fn=_with_optional_section(
                "# Run Tests",
                [
                    "Identify the relevant test command from project files.",
                    "Run the smallest meaningful verification first, then broaden if needed.",
                    "If tests fail, diagnose root cause before changing code.",
                ],
                "Specific Instructions",
            ),
        ),
    ]


def _with_optional_section(title, paragraphs, section_title):
    def render(arguments=""):
        lines = [title, "", *paragraphs]
        if arguments:
            lines.extend(["", f"## {section_title}", "", str(arguments)])
        return "\n".join(lines)

    return render
