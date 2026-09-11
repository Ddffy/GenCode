"""Read-only repo_map tool: query-personalized repository symbol map."""

REPO_MAP_TOOL_SPECS = {
    "repo_map": {
        "schema": {"query": "str=''", "limit_chars": "int=6000"},
        "risky": False,
        "description": (
            "Show a dependency-ranked map of repository files and symbols, "
            "ordered by relevance to an optional query."
        ),
    },
}

REPO_MAP_TOOL_EXAMPLES = {
    "repo_map": '<tool>{"name":"repo_map","args":{"query":"permission checker","limit_chars":6000}}</tool>',
}


def tool_repo_map(agent, args):
    query = str(args.get("query", "") or "")
    limit_chars = int(args.get("limit_chars", 6000) or 6000)
    recent = []
    if hasattr(agent, "memory"):
        try:
            recent = agent.memory.to_dict()["working"]["recent_files"]
        except (KeyError, AttributeError, TypeError):
            recent = []
    text, meta = agent.build_repo_map(
        query=query,
        budget_chars=limit_chars,
        recent_paths=recent,
    )
    if not text:
        reason = str(meta.get("reason", "unavailable"))
        return f"Repository map unavailable: {reason}"
    return text


REPO_MAP_TOOL_RUNNERS = {"repo_map": tool_repo_map}
