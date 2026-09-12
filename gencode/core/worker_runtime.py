"""Child runtime construction for worker tasks."""

from .workspace import WorkspaceContext


def build_child_runtime(parent, subagent_type, write_scope):
    from .runtime import GenCode

    child = GenCode(
        model_client=new_model_client(parent),
        workspace=WorkspaceContext.build(parent.root, repo_root_override=parent.root),
        session_store=parent.session_store,
        run_store=parent.run_store,
        approval_policy="never" if subagent_type == "Explore" else "auto",
        max_steps=parent.max_steps,
        max_new_tokens=parent.max_new_tokens,
        depth=parent.depth + 1,
        max_depth=parent.max_depth,
        read_only=subagent_type == "Explore"
        or (subagent_type == "worker" and not write_scope),
        secret_env_names=parent.secret_env_names,
        shell_env_allowlist=parent.shell_env_allowlist,
        feature_flags=parent.feature_flags,
        write_scope=write_scope,
        model_client_factory=getattr(parent, "model_client_factory", None),
        sandbox_config=getattr(parent, "sandbox_config", None),
        ask_user_callback=getattr(parent, "ask_user_callback", None),
        git_auto_commit=getattr(getattr(parent, "git", None), "auto_commit", True),
        git_auto_undo=getattr(getattr(parent, "git", None), "auto_undo", True),
    )
    child.set_tool_profile("readonly" if subagent_type == "Explore" else "worker")
    child.refresh_prefix(force=True)
    return child


def new_model_client(parent):
    factory = getattr(parent, "model_client_factory", None)
    if factory is not None:
        return factory()
    return parent.model_client
