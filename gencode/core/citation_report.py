"""Citation validation for retrieved evidence and wiki notes.

Kept out of the runtime knowledge mixin so the mixin stays a thin assembly hook
and the validation policy lives in one bounded module.
"""

from ..features.retrieval.assembler import validate_answer_citations


def build_citation_report(final_answer, *, code_retrieval=None, knowledge_retrieval=None):
    """Validate an answer against every citation scheme retrieval rendered.

    Two schemes can reach the prompt and they are numbered independently, so the
    verdicts are reported per source rather than merged: ``[E1]`` from evidence
    assembly and ``wiki:<id>#<section>`` from a knowledge note are different
    claims even when the visible number matches.

    Runs on every turn regardless of the typed-knowledge flag, because an answer
    can cite retrieved evidence even when knowledge maintenance is disabled.
    """
    code = dict(code_retrieval or {})
    knowledge = dict(knowledge_retrieval or {})
    citation_map = dict((code.get("metrics") or {}).get("citations") or {})
    return validate_answer_citations(
        final_answer,
        citation_map=citation_map,
        wiki_records=list(knowledge.get("wiki", []) or []),
    )
