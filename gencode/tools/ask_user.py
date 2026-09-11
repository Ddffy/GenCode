"""User clarification tool definitions."""

ASK_USER_TOOL_SPECS = {
    "ask_user": {
        "schema": {"question": "str", "choices": "list[str]=[]"},
        "risky": False,
        "description": "Ask the interactive user a real blocking clarification question.",
    },
}

ASK_USER_TOOL_EXAMPLES = {
    "ask_user": '<tool>{"name":"ask_user","args":{"question":"Which target should I deploy?","choices":["staging","production"]}}</tool>',
}



def tool_ask_user(agent, args):
    return agent.ask_user(str(args["question"]), choices=args.get("choices", []) or [])
