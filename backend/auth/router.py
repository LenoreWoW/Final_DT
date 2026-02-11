"""
Authentication API Router - Register, login, and user profile.

Endpoints:
    POST /api/auth/register  - Create a new user account
    POST /api/auth/login     - Authenticate and receive a JWT token
    GET  /api/auth/me        - Get current user profile
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.models.database import get_db, UserModel
from backend.auth.dependencies import (
    create_access_token,
    get_current_user,
    get_password_hash,
    verify_password,
)

router = APIRouter(tags=["auth"])


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------

class UserRegisterRequest(BaseModel):
    """Request body for user registration."""
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=5, max_length=200)
    password: str = Field(..., min_length=6, max_length=128)


class UserLoginRequest(BaseModel):
    """Request body for login."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    username: str


class UserProfileResponse(BaseModel):
    """Current user profile."""
    id: str
    username: str
    email: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
async def register(request: UserRegisterRequest, db: Session = Depends(get_db)):
    """
    Register a new user.

    Returns the created user profile.  Raises 409 if the username or
    email is already taken.
    """
    # Check for existing username
    if db.query(UserModel).filter(UserModel.username == request.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )

    # Check for existing email
    if db.query(UserModel).filter(UserModel.email == request.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = UserModel(
        id=str(uuid.uuid4()),
        username=request.username,
        email=request.email,
        hashed_password=get_password_hash(request.password),
        created_at=datetime.utcnow(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserProfileResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        created_at=user.created_at,
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: UserLoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate a user and return a JWT access token.

    Raises 401 if the credentials are invalid.
    """
    user = db.query(UserModel).filter(UserModel.username == request.username).first()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user.username})

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id,
        username=user.username,
    )


@router.get("/me", response_model=UserProfileResponse)
async def get_me(current_user: UserModel = Depends(get_current_user)):
    """
    Return the profile of the currently authenticated user.

    Requires a valid Bearer token in the Authorization header.
    """
    return UserProfileResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        created_at=current_user.created_at,
    )
