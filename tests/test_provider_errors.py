from gencode.providers.errors import sanitize_url


def test_sanitize_url_drops_credentials_query_and_fragment_from_malformed_url():
    sanitized = sanitize_url("http://user:secret@[::1/v1?api_key=x#frag")

    assert "user" not in sanitized
    assert "secret" not in sanitized
    assert "api_key" not in sanitized
    assert "#" not in sanitized
    assert sanitized.startswith("http://")


def test_sanitize_url_drops_credentials_from_scheme_less_url():
    sanitized = sanitize_url("user:secret@example.com/v1?api_key=x#frag")

    assert sanitized == "example.com/v1"
