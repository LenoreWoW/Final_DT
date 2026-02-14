"""WebSocket handler stub."""

class MockSocketIO:
    def __init__(self):
        self._handlers = {}

    def on(self, event, handler=None):
        if handler:
            self._handlers[event] = handler
            return handler
        def decorator(f):
            self._handlers[event] = f
            return f
        return decorator

    def emit(self, event, data=None, room=None):
        pass

socketio = MockSocketIO()
