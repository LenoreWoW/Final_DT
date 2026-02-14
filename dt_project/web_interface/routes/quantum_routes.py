"""Route stub: quantum_routes."""

class _Blueprint:
    def __init__(self, name):
        self.name = name
        self._routes = {}

    def route(self, path, **kw):
        def decorator(f):
            self._routes[path] = f
            return f
        return decorator

quantum_bp = _Blueprint("quantum_routes")
