import pytest

from gencode.features.sandbox.config import SandboxConfig, resolve_sandbox_config


def test_sandbox_config_defaults_to_off():
    config = resolve_sandbox_config({})

    assert config == SandboxConfig()
    assert config.enabled is False


def test_sandbox_config_accepts_required_bubblewrap_mode():
    config = resolve_sandbox_config(
        {
            "sandbox": {
                "mode": "required",
                "backend": "bubblewrap",
                "workspace_write": False,
                "excluded_commands": ["git *"],
                "filesystem": {
                    "extra_readonly_paths": ["/usr/bin"],
                    "deny_read": ["/tmp/private"],
                    "deny_write": ["/"],
                },
            }
        }
    )

    assert config.mode == "required"
    assert config.backend == "bubblewrap"
    assert config.workspace_write is False
    assert config.excluded_commands == ("git *",)
    assert config.extra_readonly_paths == ("/usr/bin",)
    assert config.deny_read == ("/tmp/private",)
    assert config.deny_write == ("/",)


def test_sandbox_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="sandbox.mode"):
        resolve_sandbox_config({"sandbox": {"mode": "strict"}})
