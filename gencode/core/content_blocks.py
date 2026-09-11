"""Typed model input blocks.

GenCode keeps conversation history as text, but provider adapters can accept
structured model input when a turn needs media. Image bytes live only at the
adapter boundary; trace and history should store metadata references instead.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageBlock:
    path: str
    mime_type: str
    data: bytes
    sha256: str
    detail: str = "auto"

    @property
    def byte_count(self):
        return len(self.data)

    def base64_data(self):
        return base64.b64encode(self.data).decode("ascii")

    def data_url(self):
        return f"data:{self.mime_type};base64,{self.base64_data()}"


@dataclass(frozen=True)
class ModelInput:
    text: str
    images: tuple[ImageBlock, ...] = field(default_factory=tuple)

    def __init__(self, text: str, images=None):
        object.__setattr__(self, "text", str(text))
        object.__setattr__(self, "images", tuple(images or ()))

    @property
    def image_count(self):
        return len(self.images)


def ensure_model_input(value):
    if isinstance(value, ModelInput):
        return value
    return ModelInput(text=str(value))
