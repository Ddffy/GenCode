"""Safe proposal boundary for Skill/Wiki/Spec candidates."""


def propose_candidate(agent, args):
    """Persist a structured proposal as inactive knowledge with verified provenance."""
    if not agent.feature_enabled("typed_knowledge"):
        raise ValueError("typed knowledge is disabled")

    allowed_sources = set(getattr(agent, "knowledge_source_sessions", set()))
    source_sessions = list(dict.fromkeys(str(item) for item in args.get("source_sessions", [])))
    if not source_sessions:
        # Provenance is runtime-owned. If the model omits the optional field,
        # attach the bounded evidence set that Dream actually saw.
        source_sessions = sorted(item for item in allowed_sources if item)
    if not set(source_sessions).issubset(allowed_sources):
        raise ValueError("source_sessions must refer to supplied conversation evidence")

    kind = str(args.get("kind", "")).strip().lower()
    metadata = {
        "when_to_use": str(args.get("when_to_use", "")).strip(),
        "paths": list(args.get("paths", [])),
        "allowed_tools": list(args.get("allowed_tools", [])),
        "user_invocable": False,
    }
    record = agent.knowledge_store.upsert(
        kind,
        args.get("id", ""),
        title=args.get("title", ""),
        description=args.get("description", ""),
        summary=args.get("summary", ""),
        body=args.get("body", ""),
        tags=args.get("tags", []),
        source_paths=args.get("source_paths", []),
        status="candidate",
        trusted=False,
        metadata=metadata,
        constraints=args.get("constraints", []),
        invariants=args.get("invariants", []),
        acceptance=args.get("acceptance", []),
        provenance={
            "source": str(getattr(agent, "knowledge_proposal_source", "assistant_tool")),
            "source_sessions": source_sessions,
            "session_id": str(agent.session.get("id", "")),
            "run_id": str(getattr(agent, "current_run_id", "")),
        },
    )
    bucket = "quarantined" if record.get("status") == "quarantined" else "candidates"
    audit = dict(agent.last_knowledge_maintenance or {})
    audit.setdefault("candidates", [])
    audit.setdefault("quarantined", [])
    audit.setdefault("errors", [])
    audit.setdefault("auto_dream", {})
    audit[bucket].append(agent.knowledge_store._trace_record(record))
    agent.last_knowledge_maintenance = audit
    agent.session_event_bus.emit(
        "knowledge_candidate_proposed",
        {
            "kind": kind,
            "id": record["id"],
            "status": record["status"],
            "source": str(getattr(agent, "knowledge_proposal_source", "assistant_tool")),
            "source_sessions": source_sessions,
            "run_id": str(getattr(agent, "current_run_id", "")),
        },
    )
    return (
        f"Created inactive {kind} candidate:{record['id']} "
        f"(status={record['status']}); review it with "
        f"/knowledge approve {kind}:{record['id']}."
    )
