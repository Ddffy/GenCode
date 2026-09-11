"""Provider runtime assembly for CLI and other frontends."""

import os
from dataclasses import dataclass
from typing import Callable

from ..config import (
    ENV_VISION_TIMEOUT,
    default_max_tokens_for_provider,
    resolve_provider_config,
    resolve_vision_provider_config,
)
from ..core.model_router import ModelClientRouter
from .clients import AnthropicCompatibleModelClient, OpenAICompatibleModelClient


@dataclass(frozen=True)
class ProviderClientClasses:
    openai: type = OpenAICompatibleModelClient
    anthropic: type = AnthropicCompatibleModelClient


@dataclass(frozen=True)
class ProviderRuntime:
    provider_config: object
    model_client: object
    model_client_router: ModelClientRouter
    model_client_factory: Callable[[], object]
    max_new_tokens: int


def build_provider_runtime(args, client_classes=None):
    client_classes = client_classes or ProviderClientClasses()
    provider_config = _resolve_main_provider_config(args)
    model_client = build_model_client(
        args, config=provider_config, client_classes=client_classes
    )
    return ProviderRuntime(
        provider_config=provider_config,
        model_client=model_client,
        model_client_router=_build_model_client_router(
            args, provider_config, model_client, client_classes
        ),
        model_client_factory=lambda: build_model_client(
            args, client_classes=client_classes
        ),
        max_new_tokens=(
            args.max_new_tokens
            if getattr(args, "max_new_tokens", None) is not None
            else default_max_tokens_for_provider(provider_config.name)
        ),
    )


def build_model_client(
    args,
    *,
    provider=None,
    config=None,
    use_cli_overrides=True,
    timeout=None,
    client_classes=None,
):
    client_classes = client_classes or ProviderClientClasses()
    resolved_config = config or _resolve_main_provider_config(
        args, provider=provider, use_cli_overrides=use_cli_overrides
    )
    return model_client_from_config(
        resolved_config, args, timeout=timeout, client_classes=client_classes
    )


def model_client_from_config(config, args, *, timeout=None, client_classes=None):
    client_classes = client_classes or ProviderClientClasses()
    timeout = getattr(args, "openai_timeout", 300) if timeout is None else timeout
    if config.protocol == "openai":
        return client_classes.openai(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=args.temperature,
            timeout=timeout,
        )
    if config.protocol == "anthropic":
        return client_classes.anthropic(
            model=config.model,
            base_url=config.base_url,
            api_key=config.api_key,
            temperature=args.temperature,
            timeout=timeout,
        )

    raise ValueError(f"unknown provider protocol: {config.protocol}")


def _resolve_main_provider_config(args, provider=None, use_cli_overrides=True):
    return resolve_provider_config(
        provider if provider is not None else getattr(args, "provider", None),
        start=getattr(args, "cwd", "."),
        config_path=getattr(args, "config", None),
        model=getattr(args, "model", None) if use_cli_overrides else None,
        base_url=getattr(args, "base_url", None) if use_cli_overrides else None,
        api_key=getattr(args, "api_key", None) if use_cli_overrides else None,
        vision_provider=(
            getattr(args, "vision_provider", None) if use_cli_overrides else None
        ),
    )


def _build_vision_model_client(args, provider, client_classes):
    config = resolve_vision_provider_config(
        provider,
        start=getattr(args, "cwd", "."),
        config_path=getattr(args, "config", None),
        model=getattr(args, "vision_model", None),
        base_url=getattr(args, "vision_base_url", None),
        api_key=getattr(args, "vision_api_key", None),
    )
    return model_client_from_config(
        config,
        args,
        timeout=_vision_timeout(args),
        client_classes=client_classes,
    )


def _vision_timeout(args):
    value = getattr(args, "vision_timeout", None)
    if value is None:
        value = os.environ.get(ENV_VISION_TIMEOUT)
    return int(value) if value else getattr(args, "openai_timeout", 300)


def _build_model_client_router(args, provider_config, model_client, client_classes):
    if provider_config.supports_vision:
        return ModelClientRouter(main_client=model_client, vision_client=model_client)
    if not provider_config.vision_provider:
        return ModelClientRouter(main_client=model_client)

    def vision_client_factory():
        return _build_vision_model_client(
            args, provider_config.vision_provider, client_classes
        )

    return ModelClientRouter(
        main_client=model_client,
        vision_client_factory=vision_client_factory,
    )
