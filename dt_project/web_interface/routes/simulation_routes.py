"""Simulation routes stub."""

class _Blueprint:
    def __init__(self, name):
        self.name = name
        self._routes = {}

    def route(self, path, **kw):
        def decorator(f):
            self._routes[path] = f
            return f
        return decorator

simulation_bp = _Blueprint("simulation_routes")

def create_simulation_routes():
    return {}
