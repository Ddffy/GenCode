from src.profile import getUserName
from src.report import render_user


def greeting(user):
    return f"Hello, {getUserName(user)}"


def report(user):
    return render_user(user)
