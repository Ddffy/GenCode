"""Tool boundary for proposing inactive Skill/Wiki/Spec knowledge."""

from .base import ToolCapability, ToolEffect
from .schemas import KnowledgeProposeArgs, KnowledgeReadArgs

TOOL_SCHEMAS = {"knowledge_read": KnowledgeReadArgs, "knowledge_propose": KnowledgeProposeArgs}
TOOL_SPECS = {
    "knowledge_read": {
        "schema": {"id": "str", "kind": "str='wiki'", "section": "str=''", "max_chars": "int=12000"},
        "risky": False,
        "description": "Read an approved Wiki page or Markdown section on demand.",
    },
    "knowledge_propose": {
        "schema": {
            "kind": "skill|wiki|spec", "id": "str=''", "title": "str",
            "description": "str=''", "summary": "str=''", "body": "str",
            "tags": "list[str]=[]", "source_paths": "list[str]=[]",
            "source_sessions": "list[str]=[]", "when_to_use": "str=''",
            "paths": "list[str]=[]", "allowed_tools": "list[str]=[]",
            "constraints": "list[str]=[]", "invariants": "list[str]=[]",
            "acceptance": "list[str]=[]",
        },
        "risky": True,
        "capability": ToolCapability(ToolEffect.WRITE, False),
        "description": (
            "Create an inactive Skill/Wiki/Spec candidate for review. Never activates "
            "knowledge; use only for explicit remember requests or Dream extraction."
        ),
    }
}
TOOL_EXAMPLES = {
    "knowledge_read": '<tool>{"name":"knowledge_read","args":{"id":"runtime-overview","section":"Execution"}}</tool>',
    "knowledge_propose": (
        '<tool>{"name":"knowledge_propose","args":{"kind":"wiki",'
        '"id":"workspace-conventions","title":"Workspace conventions",'
        '"description":"Stable repository guidance","body":"...",'
        '"source_sessions":["session-id"]}}</tool>'
    )
}


def validate(agent, name, args):
    if not hasattr(agent, "knowledge_store"):
        raise ValueError("typed knowledge is unavailable")
    if hasattr(agent, "feature_enabled") and not agent.feature_enabled("typed_knowledge"):
        raise ValueError("typed knowledge is disabled")
    if name == "knowledge_read":
        if str(args.get("kind", "wiki")).lower() != "wiki":
            raise ValueError("knowledge_read only reads kind=wiki")
        if not agent.knowledge_store.get(args["id"], kind="wiki", include_inactive=True):
            raise ValueError("unknown wiki")
        return
    permitted = set(
        getattr(agent, "knowledge_source_sessions", {str(agent.session.get("id", ""))})
    )
    requested = {str(item) for item in args.get("source_sessions", [])}
    if not requested.issubset(permitted):
        raise ValueError("source_sessions must refer to supplied conversation evidence")


def read_wiki(agent, args):
    return agent.knowledge_store.read_wiki(
        args["id"], section=args.get("section", ""), max_chars=args.get("max_chars", 12_000)
    )


def run(agent, args):
    return agent.propose_knowledge_candidate(args)
