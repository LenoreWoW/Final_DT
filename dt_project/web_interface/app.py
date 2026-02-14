"""Web app stubs — maps to FastAPI app in backend.main."""

def create_app(config=None):
    try:
        from backend.main import app
        return app
    except Exception:
        return None

def configure_app(app, config=None):
    pass
