from src.profile import getUserName


def render_user(user):
    return f"User: {getUserName(user)}"
