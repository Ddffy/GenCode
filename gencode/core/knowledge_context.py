"""Prompt assembly adapter for typed durable knowledge."""


def assemble_typed_knowledge(
    agent, user_message, section_texts, selected_notes, *, enabled=True
):
    """Inject each knowledge kind through its own context boundary."""
    # ContextManager reads this marker to protect mandatory Spec text from
    # optional-memory trimming.  Reset it every turn so a cleared binding
    # cannot leak into the next prompt.
    agent._protected_spec_text = ""
    if not enabled or not hasattr(agent, "knowledge_store"):
        agent.last_knowledge_retrieval = None
        return selected_notes

    task_state = getattr(agent, "current_task_state", None)
    spec_ids = (
        list(getattr(task_state, "active_spec_ids", []))
        if task_state is not None
        else list(agent.active_spec_ids())
    )
    knowledge = agent.knowledge_store.retrieve(user_message, spec_ids=spec_ids)
    agent.last_knowledge_retrieval = knowledge
    section_texts["memory"] += "\n\n" + agent.knowledge_store.prompt_contract()

    specs = agent.knowledge_store.render_specs(knowledge["specs"])
    agent._protected_spec_text = specs
    if specs:
        # This section is tail-clipped, so mandatory constraints stay after
        # optional durable-memory prose when prompt pressure rises.
        section_texts["memory"] += "\n\n" + specs
    skills = agent.knowledge_store.render_skills(knowledge["skills"])
    if skills:
        section_texts["skills"] += "\n\n" + skills
    return [
        *selected_notes,
        *agent.knowledge_store.wiki_as_notes(knowledge["wiki"]),
    ]
