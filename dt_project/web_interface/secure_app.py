"""Secure app stub."""

def create_secure_app(config=None):
    try:
        from backend.main import app
        return app
    except Exception:
        return None
