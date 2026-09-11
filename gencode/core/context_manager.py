"""Prompt assembly and context budget control.

ContextManager decides how much prefix, memory, relevant notes, transcript
history, and current user input reach the model for one turn. It reports
budget evidence but does not mutate session history or compact the conversation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..features import memory as memorylib, skills as skillslib
from .context_report import ContextReportBuilder, RELEVANT_MEMORY_LIMIT
from .context_sections import (
    CURRENT_REQUEST_SECTION,
    MIN_SECTION_BUDGETS,
    REDUCTION_ORDER,
    SECTION_ORDER,
    compute_budget_chars,
    compute_section_budgets,
)
from .turn_history import TurnHistoryBuilder, tail_clip

DEFAULT_TOTAL_BUDGET = 60000
DEFAULT_SECTION_FLOORS = MIN_SECTION_BUDGETS
DEFAULT_REDUCTION_ORDER = REDUCTION_ORDER


@dataclass(frozen=True)
class _PromptPressure:
    ratio: float
    tier: str
    source: str = "char_estimate"

@dataclass
class SectionRender:
    raw: str
    budget: int
    rendered: str
    details: dict | None = None

    @property
    def raw_chars(self):
        return len(self.raw)

    @property
    def rendered_chars(self):
        return len(self.rendered)


class ContextManager:
    def __init__(
        self,
        agent,
        total_budget=None,
        context_window=None,
        section_budgets=None,
        section_floors=None,
        reduction_order=None,
    ):
        self.agent = agent
        if total_budget is not None:
            self.total_budget = int(total_budget)
        elif context_window:
            self.total_budget = compute_budget_chars(int(context_window))
        else:
            self.total_budget = DEFAULT_TOTAL_BUDGET
        self.section_budgets = compute_section_budgets(self.total_budget)
        if section_budgets:
            self.section_budgets.update({str(key): int(value) for key, value in section_budgets.items()})
        self._section_floor_overrides = {str(key): int(value) for key, value in (section_floors or {}).items()}
        self.section_floors = self._compute_section_floors()
        self.reduction_order = tuple(reduction_order or REDUCTION_ORDER)
        self.history_builder = TurnHistoryBuilder(agent)
        # The legacy API returns one rendered prompt.  Native providers also
        # need the rendered section boundaries so Runtime can preserve a
        # structured message history without rebuilding context a second time.
        self.last_rendered_sections = {}

    def _remember_rendered_sections(self, rendered):
        self.last_rendered_sections = {
            str(section): str(value.rendered)
            for section, value in rendered.items()
        }

    def build(self, user_message):
        """按预算组装一轮完整 prompt。

        为什么存在：
        仅靠用户这一轮输入，模型并不知道当前仓库状态、会话里已经读过什么、
        哪些旧信息还值得继续参考。这个函数负责把“稳定基线 + 工作记忆 +
        相关笔记 + 历史 + 当前请求”拼成真正发给模型的 prompt。

        输入 / 输出：
        - 输入：`user_message`，也就是用户当前这一轮的新请求。
        - 输出：`(prompt, metadata)`。
          `prompt` 是最终发送给模型的文本；
          `metadata` 记录了每个 section 的原始长度、裁剪后的长度、是否触发了
          预算收缩等信息，后续会进入 trace/report，便于解释这轮 prompt
          是怎么被拼出来的。

        在 agent 链路里的位置：
        它位于 `GenCode.ask()` 的每轮模型调用之前，是“真正发请求给模型”
        的最后一道组装工序。`WorkspaceContext` 提供稳定前缀，`LayeredMemory`
        提供工作记忆，这个函数则把它们和当前请求合成一份可控大小的 prompt。
        """
        user_message = str(user_message)
        self.section_floors = self._compute_section_floors()
        memory_enabled = True
        relevant_memory_enabled = True
        context_reduction_enabled = True
        if hasattr(self.agent, "feature_enabled"):
            memory_enabled = self.agent.feature_enabled("memory")
            relevant_memory_enabled = self.agent.feature_enabled("relevant_memory")
            context_reduction_enabled = self.agent.feature_enabled("context_reduction")
        memory_text = "Memory:\n- disabled" if not memory_enabled else str(self.agent.memory_text())
        section_texts = {
            "prefix": str(getattr(self.agent, "prefix", "")),
            "memory": memory_text,
            "skills": skillslib.render_prompt_section(getattr(self.agent, "skills", {})),
            "history": "",
            CURRENT_REQUEST_SECTION: f"Current user request:\n{user_message}",
        }
        # repo map 与 relevant_memory 同构：吃当前请求做个性化排序，
        # 未启用/不支持时保持空串，组装时会跳过。
        if hasattr(self.agent, "build_repo_map_section"):
            section_texts["repo_map"] = str(self.agent.build_repo_map_section(user_message) or "")
        else:
            section_texts["repo_map"] = ""
        if hasattr(self.agent, "todo_ledger"):
            section_texts["memory"] += "\n\n" + self.agent.todo_ledger.render_prompt()
        checkpoint_text = ""
        if hasattr(self.agent, "render_checkpoint_text"):
            checkpoint_text = str(self.agent.render_checkpoint_text() or "").strip()
        if checkpoint_text:
            section_texts["memory"] += "\n\n" + checkpoint_text
        if memory_enabled and hasattr(self.agent, "memory_dir"):
            section_texts["memory"] += "\n\n" + memorylib.build_memory_system_section(self.agent.memory_dir)
        selected_notes = []
        if memory_enabled and relevant_memory_enabled and hasattr(self.agent, "memory") and hasattr(self.agent.memory, "retrieval_candidates"):
            selected_notes = self.agent.memory.retrieval_candidates(user_message, limit=RELEVANT_MEMORY_LIMIT)

        if not context_reduction_enabled:
            rendered = self._render_sections_without_reduction(section_texts, selected_notes=selected_notes)
            prompt = self._assemble_prompt(rendered)
            self._remember_rendered_sections(rendered)
            metadata = self._metadata(
                prompt=prompt,
                rendered=rendered,
                budgets={section: render.budget for section, render in rendered.items() if section != CURRENT_REQUEST_SECTION},
                reduction_log=[],
                selected_notes=selected_notes,
                user_message=user_message,
                section_texts=section_texts,
            )
            return prompt, metadata

        budgets = dict(self.section_budgets)
        rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes)
        prompt = self._assemble_prompt(rendered)
        pressure = self._prompt_pressure(len(prompt))
        if pressure.tier != "tier0_observe":
            budgets = self._pressure_adjusted_budgets(budgets, pressure)
            rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes, pressure=pressure)
            prompt = self._assemble_prompt(rendered)
        reduction_log = []

        while len(prompt) > self.total_budget:
            overflow = len(prompt) - self.total_budget
            reduced = False
            for section in self.reduction_order:
                floor = int(self.section_floors.get(section, 0))
                current_budget = int(budgets.get(section, 0))
                if current_budget <= floor:
                    continue
                new_budget = max(floor, current_budget - overflow)
                if new_budget >= current_budget:
                    continue
                reduction_log.append(
                    {
                        "section": section,
                        "before_chars": current_budget,
                        "after_chars": new_budget,
                        "overflow_chars": overflow,
                    }
                )
                budgets[section] = new_budget
                rendered = self._render_sections(section_texts, budgets, selected_notes=selected_notes, pressure=pressure)
                prompt = self._assemble_prompt(rendered)
                reduced = True
                break
            if not reduced:
                break

        metadata = self._metadata(
            prompt=prompt,
            rendered=rendered,
            budgets=budgets,
            reduction_log=reduction_log,
            selected_notes=selected_notes,
            user_message=user_message,
            section_texts=section_texts,
            pressure=pressure,
        )
        self._remember_rendered_sections(rendered)
        return prompt, metadata

    def _render_sections_without_reduction(self, section_texts, selected_notes=None):
        selected_notes = selected_notes or []
        relevant_lines = ["Relevant memory:"]
        if selected_notes:
            relevant_lines.extend(f"- {note['text']}" for note in selected_notes)
        else:
            relevant_lines.append("- none")
        relevant_raw = "\n".join(relevant_lines)
        history = list(getattr(self.agent, "session", {}).get("history", []))
        history_raw = self.history_builder.raw_text(history)
        repo_map_raw = str(section_texts.get("repo_map", ""))
        return {
            "prefix": SectionRender(raw=section_texts["prefix"], budget=len(section_texts["prefix"]), rendered=section_texts["prefix"], details={}),
            "memory": SectionRender(raw=section_texts["memory"], budget=len(section_texts["memory"]), rendered=section_texts["memory"], details={}),
            "skills": SectionRender(raw=section_texts["skills"], budget=len(section_texts["skills"]), rendered=section_texts["skills"], details={}),
            "repo_map": SectionRender(raw=repo_map_raw, budget=len(repo_map_raw), rendered=repo_map_raw, details={}),
            "relevant_memory": SectionRender(
                raw=relevant_raw,
                budget=len(relevant_raw),
                rendered=relevant_raw,
                details={
                    "selected_notes": [note["text"] for note in selected_notes],
                    "rendered_notes": [note["text"] for note in selected_notes],
                    "selected_count": len(selected_notes),
                    "rendered_count": len(selected_notes),
                    "note_budget": 0,
                },
            ),
            "history": SectionRender(raw=history_raw, budget=len(history_raw), rendered=history_raw, details={"rendered_entries": []}),
            CURRENT_REQUEST_SECTION: SectionRender(
                raw=section_texts[CURRENT_REQUEST_SECTION],
                budget=0,
                rendered=section_texts[CURRENT_REQUEST_SECTION],
                details={},
            ),
        }

    def _compute_section_floors(self):
        floors = dict(MIN_SECTION_BUDGETS)
        for section, budget in self.section_budgets.items():
            if section not in floors:
                floors[section] = max(20, int(budget) // 4)
        floors.update(self._section_floor_overrides)
        return floors

    def _render_sections(self, section_texts, budgets, selected_notes=None, pressure=None):
        rendered = {}
        for section in SECTION_ORDER:
            budget = budgets.get(section)
            if section == CURRENT_REQUEST_SECTION:
                raw = section_texts[section]
                rendered[section] = SectionRender(raw=raw, budget=0, rendered=raw, details={})
            elif section == "relevant_memory":
                rendered[section] = self._render_relevant_memory(selected_notes or [], int(budget or 0))
            elif section == "history":
                rendered[section] = self._render_history_section(int(budget or 0), pressure=pressure)
            else:
                raw = section_texts[section]
                rendered_text = tail_clip(raw, int(budget)) if budget is not None else raw
                rendered[section] = SectionRender(raw=raw, budget=int(budget) if budget is not None else 0, rendered=rendered_text, details={})
        return rendered

    def _prompt_pressure(self, prompt_chars):
        ratio = int(prompt_chars) / max(1, self.total_budget)
        if ratio >= 0.95:
            tier = "tier3_summary"
        elif ratio >= 0.80:
            tier = "tier2_prune"
        elif ratio >= 0.60:
            tier = "tier1_snip"
        else:
            tier = "tier0_observe"
        return _PromptPressure(ratio=round(ratio, 4), tier=tier)

    def _pressure_adjusted_budgets(self, budgets, pressure):
        adjusted = dict(budgets)
        tier = str(getattr(pressure, "tier", ""))
        if tier in {"tier1_snip", "tier2_prune"}:
            adjusted["relevant_memory"] = max(80, int(adjusted.get("relevant_memory", 0) * 0.7))
        if tier == "tier2_prune":
            adjusted["skills"] = int(adjusted.get("skills", 0) * 0.5)
        return adjusted

    def _render_relevant_memory(self, selected_notes, budget):
        header = "Relevant memory:"
        note_texts = [str(note.get("text", "")) for note in selected_notes if str(note.get("text", "")).strip()]
        raw_lines = [header] + [f"- {text}" for text in note_texts]
        raw = "\n".join(raw_lines) if note_texts else "\n".join([header, "- none"])
        if not note_texts:
            rendered = raw
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "selected_notes": [],
                    "rendered_notes": [],
                    "selected_count": 0,
                    "rendered_count": 0,
                    "note_budget": 0,
                },
            )

        per_note_budget = self._per_note_budget(budget, len(note_texts), header)
        rendered_notes = []
        while True:
            # 让每条 note 平分这一段的预算，避免一条超长笔记把其他笔记都挤掉。
            rendered_notes = [tail_clip(text, per_note_budget) for text in note_texts]
            rendered = "\n".join([header] + [f"- {text}" for text in rendered_notes])
            if len(rendered) <= budget or per_note_budget <= 1:
                break
            per_note_budget -= 1

        if len(rendered) > budget and budget > 0:
            rendered = tail_clip(raw, budget)
            rendered_notes = [rendered]

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details={
                "selected_notes": note_texts,
                "rendered_notes": rendered_notes,
                "selected_count": len(note_texts),
                "rendered_count": len(rendered_notes),
                "note_budget": per_note_budget,
            },
        )

    def _per_note_budget(self, budget, note_count, header):
        if note_count <= 0:
            return 0
        overhead = len(header) + 3 * note_count
        usable = max(0, budget - overhead)
        return max(1, usable // note_count)

    def _render_history_section(self, budget, pressure=None):
        history = list(getattr(self.agent, "session", {}).get("history", []))
        raw = self.history_builder.raw_text(history)
        if not history:
            rendered = "Transcript:\n- empty"
            return SectionRender(
                raw=raw,
                budget=budget,
                rendered=rendered,
                details={
                    "rendered_entries": [],
                    "older_entries_count": 0,
                    "collapsed_duplicate_reads": 0,
                    "reused_file_summary_count": 0,
                    "summarized_tool_count": 0,
                    "rendered_turns": 0,
                },
            )

        rendered, history_details = self.history_builder.render_section(budget, pressure=pressure)

        return SectionRender(
            raw=raw,
            budget=budget,
            rendered=rendered,
            details=history_details,
        )

    def _assemble_prompt(self, rendered):
        # 顺序是刻意设计的：稳定规则放前面，最新请求放最后。
        # repo_map 为空（未启用/不支持）时整段跳过，不给 prompt 留空洞。
        return "\n\n".join(
            rendered[section].rendered
            for section in SECTION_ORDER
            if section != "repo_map" or rendered[section].rendered
        ).strip()

    def _metadata(self, prompt, rendered, budgets, reduction_log, selected_notes, user_message, section_texts, pressure=None):
        metadata = ContextReportBuilder(
            self.agent,
            total_budget=self.total_budget,
            reduction_order=self.reduction_order,
        ).build(
            prompt=prompt,
            rendered=rendered,
            budgets=budgets,
            reduction_log=reduction_log,
            selected_notes=selected_notes,
            user_message=user_message,
            section_texts=section_texts,
        )
        if pressure:
            metadata["pressure"] = {
                "ratio": pressure.ratio,
                "tier": pressure.tier,
                "source": pressure.source,
            }
        else:
            metadata["pressure"] = {
                "ratio": 0.0,
                "tier": "tier0_observe",
                "source": "no_reduction",
            }
        return metadata
