"""Security helpers for authenticating API requests."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash

from .database import get_db
from .models import User

security = HTTPBasic()


def get_current_user(
    credentials: HTTPBasicCredentials = Depends(security),
    db: Session = Depends(get_db),
) -> User:
    """Authenticate a user via HTTP Basic credentials."""
    user = db.query(User).filter(User.username == credentials.username).first()
    if not user or not check_password_hash(
        user.password,
        credentials.password,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user
