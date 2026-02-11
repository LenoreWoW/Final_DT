"""
Authentication dependencies for JWT-based auth.

Provides:
- Password hashing and verification (via passlib)
- JWT token creation and verification (via python-jose)
- FastAPI dependency for extracting the current user from a request
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.models.database import get_db, UserModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "quantum-twin-platform-dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# ---------------------------------------------------------------------------
# Password utilities (passlib with bcrypt)
# ---------------------------------------------------------------------------

try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)

except ImportError:
    # Fallback: use hashlib when passlib is not installed
    import hashlib

    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a plain password against its hash (hashlib fallback)."""
        return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

    def get_password_hash(password: str) -> str:
        """Hash a password using SHA-256 (hashlib fallback)."""
        return hashlib.sha256(password.encode()).hexdigest()

# ---------------------------------------------------------------------------
# JWT utilities (python-jose)
# ---------------------------------------------------------------------------

try:
    from jose import JWTError, jwt

    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a signed JWT token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def _decode_token(token: str) -> dict:
        """Decode and validate a JWT token."""
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    _JWT_ERROR = JWTError

except ImportError:
    # Fallback: use PyJWT when python-jose is not installed
    try:
        import jwt as pyjwt

        def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
            """Create a signed JWT token (PyJWT fallback)."""
            to_encode = data.copy()
            expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
            to_encode.update({"exp": expire})
            return pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

        def _decode_token(token: str) -> dict:
            """Decode and validate a JWT token (PyJWT fallback)."""
            return pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        _JWT_ERROR = pyjwt.PyJWTError

    except ImportError:
        # No JWT library available at all - tokens will be opaque stubs
        import json
        import base64

        class _StubJWTError(Exception):
            pass

        _JWT_ERROR = _StubJWTError

        def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
            """Create a stub token (no JWT library available)."""
            to_encode = data.copy()
            expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
            to_encode.update({"exp": expire.isoformat()})
            return base64.urlsafe_b64encode(json.dumps(to_encode).encode()).decode()

        def _decode_token(token: str) -> dict:
            """Decode a stub token (no JWT library available)."""
            try:
                payload = json.loads(base64.urlsafe_b64decode(token.encode()))
                exp = datetime.fromisoformat(payload["exp"])
                if exp < datetime.utcnow():
                    raise _StubJWTError("Token expired")
                return payload
            except Exception as exc:
                raise _StubJWTError(str(exc)) from exc

# ---------------------------------------------------------------------------
# FastAPI dependency: get_current_user
# ---------------------------------------------------------------------------

async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> UserModel:
    """
    FastAPI dependency that extracts and validates the current user
    from the Authorization header (Bearer token).

    Raises 401 if the token is missing, expired, or the user does not exist.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if token is None:
        raise credentials_exception

    try:
        payload = _decode_token(token)
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except _JWT_ERROR:
        raise credentials_exception

    user = db.query(UserModel).filter(UserModel.username == username).first()
    if user is None:
        raise credentials_exception

    return user
