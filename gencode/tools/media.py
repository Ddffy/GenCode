"""Media inspection tools."""

from ..core import vision

MEDIA_TOOL_NAMES = {"inspect_image"}

MEDIA_TOOL_SPECS = {
    "inspect_image": {
        "schema": {
            "path": "str",
            "question": "str",
            "profile": "str='general'",
            "output_schema": "str=''",
        },
        "risky": False,
        "description": "Inspect a workspace image with a vision-capable model call.",
    },
}

MEDIA_TOOL_EXAMPLES = {
    "inspect_image": '<tool>{"name":"inspect_image","args":{"path":"chart.png","question":"Describe the chart","profile":"chart"}}</tool>',
}


def validate_media_runtime(agent, name, args):
    if name == "inspect_image":
        path = agent.path(args["path"])
        if not path.is_file():
            raise ValueError("path is not a file")


def tool_inspect_image(agent, args):
    return vision.inspect_image_with_model(
        agent,
        args["path"],
        args["question"],
        profile=args.get("profile", "general"),
        output_schema=args.get("output_schema", ""),
    )


MEDIA_TOOL_RUNNERS = {
    "inspect_image": tool_inspect_image,
}
