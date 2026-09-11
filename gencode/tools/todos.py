"""Todo ledger tool definitions."""

TODO_TOOL_SPECS = {
    "todo_add": {
        "schema": {"content": "str", "status": "str='pending'", "priority": "str='normal'", "note": "str=''"},
        "risky": False,
        "description": "Add an item to the session task ledger.",
    },
    "todo_update": {
        "schema": {"todo_id": "str", "status": "str?", "content": "str?", "priority": "str?", "note": "str?"},
        "risky": False,
        "description": "Update an item in the session task ledger.",
    },
    "todo_list": {"schema": {}, "risky": False, "description": "List the session task ledger."},
}

TODO_TOOL_EXAMPLES = {
    "todo_add": '<tool>{"name":"todo_add","args":{"content":"Implement parser","priority":"high"}}</tool>',
    "todo_update": '<tool>{"name":"todo_update","args":{"todo_id":"todo_1","status":"done"}}</tool>',
    "todo_list": '<tool>{"name":"todo_list","args":{}}</tool>',
}



def tool_todo_add(agent, args):
    item = agent.todo_ledger.add(
        args["content"],
        status=args.get("status", "pending"),
        priority=args.get("priority", "normal"),
        note=args.get("note", ""),
    )
    return f"added {item['id']} [{item['status']}] {item['priority']} - {item['content']}"


def tool_todo_update(agent, args):
    item = agent.todo_ledger.update(
        args["todo_id"],
        status=args.get("status"),
        content=args.get("content"),
        priority=args.get("priority"),
        note=args.get("note"),
    )
    return f"updated {item['id']} [{item['status']}] {item['priority']} - {item['content']}"


def tool_todo_list(agent, args):
    return agent.todo_ledger.render_list()
