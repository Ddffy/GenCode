import requests


def fetch_json(url):
    response = requests.get(url, verify=False)
    return response.json()
