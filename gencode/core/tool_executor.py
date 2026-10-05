"""Tool-call validation, authorization, execution, and evidence recording."""

from .tool_execution import execute_prepared_tool, prepare_tool_call


async def run_tool(agent, name, args):
    prepared = await prepare_tool_call(agent, name, args)
    return (await execute_prepared_tool(agent, prepared)).content
