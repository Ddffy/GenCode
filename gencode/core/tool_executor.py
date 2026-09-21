"""Tool-call validation, authorization, execution, and evidence recording."""

from .tool_execution import execute_prepared_tool, prepare_tool_call


def run_tool(agent, name, args):
    """Run one tool through the serial compatibility path."""
    prepared = prepare_tool_call(agent, name, args)
    return execute_prepared_tool(agent, prepared).content
