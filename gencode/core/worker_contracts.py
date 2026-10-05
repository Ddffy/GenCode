import json


def clean_type(value):
    subagent_type = str(value or "worker").strip()
    if subagent_type not in {"worker", "Explore"}:
        raise ValueError("subagent_type must be worker or Explore")
    return subagent_type


def clean_scope(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise TypeError("write_scope must be a list of workspace paths")
    return [str(item).strip() for item in value if str(item).strip()]


def dumps_payload(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
