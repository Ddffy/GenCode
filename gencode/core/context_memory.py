"""Memory-section rendering helpers for ContextManager."""

from .turn_history import tail_clip


def render_memory_section(agent, raw, budget):
    """Trim optional memory while retaining bound Spec constraints.

    A bound Spec is marked by ``assemble_typed_knowledge``.  If pressure leaves
    less room than the mandatory block, keep it intact and expose the visible
    over-budget condition instead of silently dropping a constraint.
    """
    raw = str(raw or "")
    budget = int(budget)
    protected = str(getattr(agent, "_protected_spec_text", "") or "")
    if not protected or protected not in raw:
        rendered = tail_clip(raw, budget)
        return rendered, {"protected_spec": False}
    prefix, _, _suffix = raw.rpartition(protected)
    if budget <= len(protected):
        rendered = protected
    else:
        available = max(0, budget - len(protected) - 2)
        optional = tail_clip(prefix.rstrip(), available)
        rendered = (optional + "\n\n" if optional else "") + protected
    return rendered, {
        "protected_spec": True,
        "protected_spec_chars": len(protected),
    }
