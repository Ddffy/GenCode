import requests


def build_session():
    session = requests.Session()
    session.trust_env = True
    return session
