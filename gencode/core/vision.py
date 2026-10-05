"""Vision inspection helper for image-aware tools."""

from __future__ import annotations

import asyncio

from ..providers.base import stream_model
from .content_blocks import ModelInput
from .media import load_workspace_image


async def complete_model_with_timeout(model_client, model_input, max_new_tokens):
    timeout = float(getattr(model_client, "timeout", 60))

    async def collect_result():
        result = None
        async for event in stream_model(model_client, model_input, max_new_tokens):
            if event.type == "completed":
                result = event.result
        if result is None:
            raise RuntimeError("vision provider stream ended without a completion")
        return result

    try:
        return await asyncio.wait_for(collect_result(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"vision provider request exceeded {timeout}s") from exc


async def inspect_image_with_model(agent, path, question, profile="general", output_schema=""):
    loaded = load_workspace_image(agent, path)
    prompt = image_inspection_prompt(
        loaded.metadata["path"], question, profile, output_schema
    )
    model_input = ModelInput(text=prompt, images=[loaded.block])
    model_client = agent.model_client_router.client_for_input(model_input)
    result = None
    async for event in stream_model(
        model_client, model_input, agent.max_new_tokens
    ):
        if event.type == "completed":
            result = event.result
    if result is None:
        raise RuntimeError("vision provider stream ended without a completion")
    media_ref = dict(loaded.metadata)
    task_state = getattr(agent, "current_task_state", None)
    if task_state is not None:
        artifact = agent.run_store.write_binary_artifact(
            task_state,
            "image",
            loaded.block.data,
            image_suffix(loaded.block.mime_type),
        )
        media_ref["artifact_ref"] = agent.run_store.artifact_ref(task_state, artifact)
    else:
        media_ref["artifact_ref"] = ""
    agent._pending_tool_result_metadata = {
        "media_refs": [media_ref],
        "vision_completion_metadata": dict(result.metadata),
    }
    return (
        f"image inspected: {loaded.metadata['path']}\n"
        f"profile: {profile or 'general'}\n"
        f"summary:\n{result.text}"
    )


def image_inspection_prompt(path, question, profile, output_schema):
    question_text = str(question or "")
    lines = [
        "Inspect this workspace image for a coding-agent task.",
        f"Image path: {path}",
        f"Inspection profile: {profile or 'general'}",
        f"Question: {question_text}",
    ]
    schema = str(output_schema or "").strip()
    if schema:
        lines.append(f"Return shape: {schema}")
    if _needs_complete_visual_extraction(question_text, profile, schema):
        lines.append("Return the complete requested extraction. Do not omit rows, cells, or trailing items.")
    else:
        lines.append("Return concise, task-useful observations.")
    lines.append("Do not mention base64.")
    return "\n".join(lines)


def _needs_complete_visual_extraction(question, profile, output_schema):
    text = " ".join(str(value or "").lower() for value in (question, profile, output_schema))
    keywords = ("ocr", "extract", "all rows", "every visible row", "list rows", "table", "csv")
    return any(keyword in text for keyword in keywords)


def image_suffix(mime_type):
    return {
        "image/gif": ".gif",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(str(mime_type), ".img")
