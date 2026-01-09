"""Runtime configuration helpers for the FastAPI backend."""

from __future__ import annotations

import os
from pathlib import Path

# Determine project directories
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Environment selection mirrors the legacy Flask backend behaviour
APP_ENV = os.getenv("APP_ENV", "production").lower()

# SQLite database file shared with the existing application
_default_db_name = "users_dev.db" if APP_ENV == "development" else "users.db"
# Resolve the SQLite file path, allowing overrides via environment variable
_db_override = os.getenv("RPP_DB_PATH")
if _db_override:
    DB_PATH = Path(_db_override).resolve()
else:
    DB_PATH = Path(PROJECT_ROOT / _default_db_name).resolve()

# Parent directory (same level as rpp_results)
PARENT_DIR = PROJECT_ROOT.parent

# Root folder for FITS storage; defaults to <parent>/rpp_data/fits (same level as rpp_results)
FITS_STORAGE_ROOT = Path(
    os.getenv("RPP_FITS_ROOT", PARENT_DIR / "rpp_data" / "fits")
).resolve()

# Ensure the storage root exists without changing existing permissions
FITS_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

# SQLite connection string consumed by SQLAlchemy
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"


__all__ = [
    "APP_ENV",
    "DB_PATH",
    "FITS_STORAGE_ROOT",
    "PARENT_DIR",
    "SQLALCHEMY_DATABASE_URL",
]
