"""Workspace-safe media loading for model input."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .content_blocks import ImageBlock

IMAGE_MIME_BY_SUFFIX = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
MAX_IMAGE_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class LoadedImage:
    block: ImageBlock
    metadata: dict


def load_workspace_image(agent, raw_path):
    path = agent.path(raw_path)
    if not path.is_file():
        raise ValueError("path is not a file")
    data = path.read_bytes()
    if not data:
        raise ValueError("image file is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"image file is too large ({len(data)} bytes)")
    mime_type = detect_image_mime(path.name, data)
    optimized = maybe_downsample_image(data, mime_type)
    sha256 = hashlib.sha256(optimized).hexdigest()
    relative = path.relative_to(agent.root).as_posix()
    block = ImageBlock(
        path=relative,
        mime_type=mime_type,
        data=optimized,
        sha256=sha256,
    )
    return LoadedImage(
        block=block,
        metadata={
            "path": relative,
            "mime_type": mime_type,
            "sha256": sha256,
            "bytes": len(optimized),
        },
    )


def detect_image_mime(filename, data):
    suffix = "." + str(filename).rsplit(".", 1)[-1].lower() if "." in str(filename) else ""
    expected = IMAGE_MIME_BY_SUFFIX.get(suffix)
    actual = _mime_from_magic(data)
    if not actual or actual not in set(IMAGE_MIME_BY_SUFFIX.values()):
        raise ValueError("unsupported image type")
    if expected and expected != actual:
        raise ValueError("image extension does not match file content")
    return actual


def maybe_downsample_image(data, mime_type):
    try:
        from PIL import Image
        from io import BytesIO
    except Exception:
        return data

    if len(data) <= MAX_IMAGE_BYTES // 2:
        return data
    try:
        with Image.open(BytesIO(data)) as image:
            image.thumbnail((2048, 2048))
            out = BytesIO()
            fmt = _pillow_format(mime_type)
            image.save(out, format=fmt, optimize=True)
            optimized = out.getvalue()
            return optimized if optimized else data
    except Exception:
        return data


def _mime_from_magic(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return ""


def _pillow_format(mime_type):
    return {
        "image/gif": "GIF",
        "image/jpeg": "JPEG",
        "image/png": "PNG",
        "image/webp": "WEBP",
    }[mime_type]
