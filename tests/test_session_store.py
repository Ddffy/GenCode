import os

import pytest

import gencode.core.runtime.persistence.session_store as session_store_module
from gencode.core.runtime.persistence.session_store import SessionStore


@pytest.mark.skipif(os.name != "nt", reason="Windows-only replace sharing semantics")
def test_session_save_retries_transient_windows_replace_denial(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    replace = session_store_module.os.replace
    calls = []

    def transient_denial(source, destination):
        calls.append((source, destination))
        if len(calls) < 3:
            error = PermissionError("transient sharing violation")
            error.winerror = 32
            raise error
        replace(source, destination)

    monkeypatch.setattr(session_store_module.os, "replace", transient_denial)
    store.save({"id": "session-1", "revision": 3})

    assert len(calls) == 3
    assert store.load("session-1") == {"id": "session-1", "revision": 3}
    assert list((tmp_path / "sessions").glob("*.tmp")) == []


@pytest.mark.skipif(os.name != "nt", reason="Windows-only replace sharing semantics")
def test_session_save_stops_after_bounded_replace_retries(tmp_path, monkeypatch):
    store = SessionStore(tmp_path / "sessions")
    calls = []

    def persistent_denial(source, destination):
        calls.append((source, destination))
        error = PermissionError("persistent sharing violation")
        error.winerror = 32
        raise error

    monkeypatch.setattr(session_store_module.os, "replace", persistent_denial)

    with pytest.raises(PermissionError, match="persistent sharing violation"):
        store.save({"id": "session-1", "revision": 3})

    assert len(calls) == 4
    assert list((tmp_path / "sessions").glob("*.tmp")) == []
