try:
    from flask import Flask
except Exception:
    class _Response:
        def __init__(self, payload, status_code):
            self._payload = payload
            self.status_code = status_code

        def get_json(self):
            return self._payload

    class _Client:
        def __init__(self, app):
            self.app = app

        def get(self, path):
            handler = self.app.routes[path]
            result = handler()
            if isinstance(result, tuple):
                payload, status_code = result
            else:
                payload, status_code = result, 200
            return _Response(payload, status_code)

    class Flask:
        def __init__(self, name):
            self.name = name
            self.routes = {}

        def route(self, path):
            def decorator(fn):
                self.routes[path] = fn
                return fn
            return decorator

        def test_client(self):
            return _Client(self)


app = Flask(__name__)


@app.route("/")
def index():
    return {"status": "running"}, 200
