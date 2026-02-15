"""
Authentication & Security User Journey Tests.

Tests the complete authentication lifecycle:
    1. Register a new user
    2. Login and receive a JWT token
    3. Access the protected /me endpoint with the token
    4. Duplicate registration is rejected (409)
    5. Wrong password is rejected (401)
    6. Missing token is rejected (401)
    7. Invalid/corrupted token is rejected (401)
    8. Login with a non-existent user is rejected (401)

All tests in TestAuthenticationJourney are sequential: earlier tests
store state on ``self.__class__`` so that later tests can reuse the
registered user and JWT token without re-registering.

Usage:
    cd /Users/hassanalsahli/Desktop/Final_DT
    venv/bin/python -m pytest tests/test_user_journey_auth.py -v
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.main import app
from backend.models.database import Base, get_db


# ---------------------------------------------------------------------------
# Class-scoped fixtures so DB state persists across the journey
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class")
def journey_db_session():
    """Create a single in-memory SQLite DB that lives for the entire class."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="class")
def journey_client(journey_db_session):
    """Class-scoped TestClient sharing the same DB session."""
    def override_get_db():
        yield journey_db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.pop(get_db, None)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestAuthenticationJourney:
    """
    Sequential user-journey tests for authentication and security.

    ORDERING DEPENDENCY: Tests are numbered (test_01_, test_02_, ...) and MUST
    run in lexicographic order. Earlier tests store state on ``self.__class__``
    (e.g. _user_id, _token) that later tests depend on. pytest collects tests
    in definition order by default, and the numbering ensures correct ordering
    even if a plugin sorts alphabetically.

    State is shared across methods via ``self.__class__`` attributes so that
    the register -> login -> profile chain works naturally.
    """

    # Unique suffix so parallel runs never collide
    _suffix: str = uuid.uuid4().hex[:8]
    _username: str = f"journey_user_{_suffix}"
    _email: str = f"journey_{_suffix}@example.com"
    _password: str = "Str0ngP4ss"

    # Populated by earlier tests, consumed by later ones
    _user_id: str = ""
    _token: str = ""

    # ---- 1. Register ---------------------------------------------------

    def test_01_register(self, journey_client):
        """POST /api/auth/register creates a new user and returns 201."""
        resp = journey_client.post("/api/auth/register", json={
            "username": self.__class__._username,
            "email": self.__class__._email,
            "password": self.__class__._password,
        })

        assert resp.status_code == 201, f"Expected 201, got {resp.status_code}: {resp.text}"
        data = resp.json()

        # Response contains the user profile fields
        assert data["username"] == self.__class__._username
        assert data["email"] == self.__class__._email
        assert "id" in data
        assert "created_at" in data

        # Stash the user_id for later tests
        self.__class__._user_id = data["id"]

    # ---- 2. Login ------------------------------------------------------

    def test_02_login(self, journey_client):
        """POST /api/auth/login returns 200 with a JWT token."""
        resp = journey_client.post("/api/auth/login", json={
            "username": self.__class__._username,
            "password": self.__class__._password,
        })

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["username"] == self.__class__._username
        assert data["user_id"] == self.__class__._user_id

        # Stash the token for later tests
        self.__class__._token = data["access_token"]

    # ---- 3. Get profile ------------------------------------------------

    def test_03_get_profile(self, journey_client):
        """GET /api/auth/me with Bearer token returns the correct profile."""
        assert self.__class__._token, "No token — test_02_login must run first"

        resp = journey_client.get("/api/auth/me", headers={
            "Authorization": f"Bearer {self.__class__._token}",
        })

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()

        assert data["username"] == self.__class__._username
        assert data["email"] == self.__class__._email
        assert data["id"] == self.__class__._user_id
        assert "created_at" in data

    # ---- 4. Duplicate registration fails -------------------------------

    def test_04_duplicate_register_fails(self, journey_client):
        """Registering the same username again returns 409 Conflict."""
        resp = journey_client.post("/api/auth/register", json={
            "username": self.__class__._username,
            "email": f"other_{self.__class__._suffix}@example.com",
            "password": self.__class__._password,
        })

        assert resp.status_code == 409, f"Expected 409, got {resp.status_code}: {resp.text}"
        assert "already registered" in resp.json()["detail"].lower()

    # ---- 5. Wrong password fails ---------------------------------------

    def test_05_wrong_password_fails(self, journey_client):
        """Login with the wrong password returns 401 Unauthorized."""
        resp = journey_client.post("/api/auth/login", json={
            "username": self.__class__._username,
            "password": "WrongPassword99",
        })

        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"

    # ---- 6. No token rejected ------------------------------------------

    def test_06_no_token_rejected(self, journey_client):
        """GET /api/auth/me without an Authorization header returns 401."""
        resp = journey_client.get("/api/auth/me")

        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"

    # ---- 7. Invalid token rejected -------------------------------------

    def test_07_invalid_token_rejected(self, journey_client):
        """GET /api/auth/me with a corrupted token returns 401."""
        resp = journey_client.get("/api/auth/me", headers={
            "Authorization": "Bearer this.is.not.a.valid.jwt.token",
        })

        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"

    # ---- 8. Non-existent user login fails ------------------------------

    def test_08_nonexistent_user_login_fails(self, journey_client):
        """Login with a username that was never registered returns 401."""
        resp = journey_client.post("/api/auth/login", json={
            "username": f"ghost_user_{uuid.uuid4().hex[:8]}",
            "password": "doesntmatter",
        })

        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}: {resp.text}"
