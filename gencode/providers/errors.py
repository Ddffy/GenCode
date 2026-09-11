"""Structured provider failure types."""

from urllib.parse import urlsplit, urlunsplit


class ProviderError(RuntimeError):
    def __init__(
        self,
        message,
        *,
        provider="",
        model="",
        base_url="",
        code="provider_error",
        http_status=None,
        retryable=False,
        attempts=1,
        retry_count=0,
        body_excerpt="",
        cause_type="",
    ):
        super().__init__(message)
        self.provider = str(provider or "")
        self.model = str(model or "")
        self.base_url = sanitize_url(base_url)
        self.code = str(code or "provider_error")
        self.http_status = http_status
        self.retryable = bool(retryable)
        self.attempts = int(attempts or 1)
        self.retry_count = int(retry_count or 0)
        self.body_excerpt = _clip(body_excerpt, 500)
        self.cause_type = str(cause_type or "")

    def to_metadata(self):
        payload = {
            "provider_error": {
                "code": self.code,
                "retryable": self.retryable,
                "attempts": self.attempts,
                "retry_count": self.retry_count,
            }
        }
        error = payload["provider_error"]
        if self.provider:
            error["provider"] = self.provider
        if self.model:
            error["model"] = self.model
        if self.base_url:
            error["base_url"] = self.base_url
        if self.http_status is not None:
            error["http_status"] = int(self.http_status)
        if self.body_excerpt:
            error["body_excerpt"] = self.body_excerpt
        if self.cause_type:
            error["cause_type"] = self.cause_type
        return payload


def _clip(value, limit):
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def sanitize_url(value):
    text = str(value or "")
    if not text:
        return ""
    if "://" not in text and "@" in text:
        return text.rsplit("@", 1)[1].split("?", 1)[0].split("#", 1)[0]
    try:
        parsed = urlsplit(text)
    except ValueError:
        safe = text.split("?", 1)[0].split("#", 1)[0]
        scheme, sep, rest = safe.partition("://")
        if "@" in rest:
            rest = rest.rsplit("@", 1)[1]
        return f"{scheme}{sep}{rest}" if sep else rest
    hostname = parsed.hostname or ""
    if not hostname:
        return urlunsplit((parsed.scheme, "", parsed.path, "", ""))
    netloc = hostname
    if ":" in hostname and not hostname.startswith("["):
        netloc = f"[{hostname}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
