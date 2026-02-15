"""
Authentication dependencies for JWT-based auth.

Provides:
- Password hashing and verification (via passlib + bcrypt)
- JWT token creation and verification (via python-jose or PyJWT)
- FastAPI dependency for extracting the current user from a request
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.models.database import get_db, UserModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "quantum-twin-platform-dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

if SECRET_KEY == "quantum-twin-platform-dev-secret-key-change-in-production":
    logger.warning("Using default JWT secret key — set JWT_SECRET_KEY env var in production")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)

# ---------------------------------------------------------------------------
# Password utilities — bcrypt via passlib (required dependency)
# ---------------------------------------------------------------------------

from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its bcrypt hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    return pwd_context.hash(password)


# ---------------------------------------------------------------------------
# JWT utilities — python-jose (primary), PyJWT (fallback)
# ---------------------------------------------------------------------------

try:
    from jose import JWTError, jwt

    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a signed JWT token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def _decode_token(token: str) -> dict:
        """Decode and validate a JWT token."""
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    _JWT_ERROR = JWTError

except ImportError:
    import jwt as pyjwt

    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a signed JWT token (PyJWT)."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        return pyjwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    def _decode_token(token: str) -> dict:
        """Decode and validate a JWT token (PyJWT)."""
        return pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

    _JWT_ERROR = pyjwt.PyJWTError

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
