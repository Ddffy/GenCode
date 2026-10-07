"""Child runtime construction for Goal Worker attempts."""


def build_child_runtime(parent, role, write_scope, *, workspace=None, read_only=None):
    from .runtime import GenCode

    child_workspace = workspace or parent.workspace
    child = GenCode(
        model_client=new_model_client(parent),
        # The child starts in the exact same repository.  Reuse the parent's
        # immutable snapshot here; prepare_turn refreshes it before the first
        # model call. This keeps Goal attempt setup bounded.
        workspace=child_workspace,
        session_store=parent.session_store,
        run_store=parent.run_store,
        approval_policy="never" if role == "Explore" else "auto",
        max_steps=parent.max_steps,
        max_new_tokens=parent.max_new_tokens,
        depth=parent.depth + 1,
        max_depth=parent.max_depth,
        read_only=(
            bool(read_only)
            if read_only is not None
            else role == "Explore"
            or (role == "worker" and not write_scope)
        ),
        secret_env_names=parent.secret_env_names,
        shell_env_allowlist=parent.shell_env_allowlist,
        feature_flags=parent.feature_flags,
        write_scope=write_scope,
        model_client_factory=getattr(parent, "model_client_factory", None),
        sandbox_config=getattr(parent, "sandbox_config", None),
        ask_user_callback=getattr(parent, "ask_user_callback", None),
        git_auto_commit=getattr(getattr(parent, "git", None), "auto_commit", True),
        git_auto_undo=getattr(getattr(parent, "git", None), "auto_undo", True),
        stable_workspace_prefix=True,
    )
    child.disable_fast_read_only_qa = role == "worker"
    child.goal_worker = role == "worker"
    profile_name = "readonly" if role == "Explore" else "worker"
    previous_signature = child.tool_signature()
    child.set_tool_profile(profile_name)
    child.session["knowledge"]["active_specs"] = list(parent.active_spec_ids())
    # The constructor has already resolved the workspace.  Re-scanning Git and
    # repository metadata here made background spawn synchronous and slow.
    # Only rebuild the prefix when changing profiles actually changes tools.
    if child.tool_signature() != previous_signature:
        child._apply_prefix_state(child.build_prefix())
    child._reuse_workspace_snapshot_once = True
    return child


def new_model_client(parent):
    factory = getattr(parent, "model_client_factory", None)
    if factory is not None:
        return factory()
    return parent.model_client
