import requests


def fetch_text(url):
    response = requests.get(url, timeout=None)
    return response.text
