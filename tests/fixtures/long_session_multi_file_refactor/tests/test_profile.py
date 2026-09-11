from src.main import greeting, report
from src.profile import getUserName


def test_greeting_uses_user_name():
    assert greeting({"name": " ada "}) == "Hello, Ada"
    assert getUserName({"name": "grace"}) == "Grace"
    assert report({"name": "alan"}) == "User: Alan"
