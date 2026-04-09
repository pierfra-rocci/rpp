Backend API
===========

RPP exposes a FastAPI backend in ``api/main.py``. The Streamlit frontend uses
this backend for account management, configuration persistence, and FITS file
storage. A legacy backend still exists for compatibility, but the FastAPI
service is the primary documented interface.

Authentication Model
--------------------

- Registration and login are initiated through JSON requests.
- Authenticated routes use the current backend security layer from ``api/security.py``.
- The Streamlit frontend handles credential storage and request formatting through ``pages/api_client.py``.

Core Endpoints
--------------

Health and Authentication
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``GET /health``
- ``POST /api/register``
- ``POST /api/login``
- ``POST /api/recovery/request``
- ``POST /api/recovery/confirm``

Configuration
^^^^^^^^^^^^^

- ``GET /api/config``
- ``POST /api/config``

FITS Storage
^^^^^^^^^^^^

- ``POST /api/upload/fits``
- ``GET /api/fits``

Behavior Notes
--------------

- FITS uploads are written to user-scoped storage paths under ``rpp_data/fits/``.
- Configuration blobs are stored per user in the database.
- Duplicate FITS uploads are rejected when the backend detects the same stored hash.
- The backend returns schema-driven JSON responses defined in ``api/schemas.py``.

Main Backend Modules
--------------------

``api/main.py``
   FastAPI application, route definitions, upload handling, and response wiring.

``api/models.py``
   SQLAlchemy ORM models for users, recovery codes, uploaded FITS files, and
   related tracking tables.

``api/schemas.py``
   Pydantic request and response models used by the API.

``api/storage.py``
   User-scoped storage-path generation for uploaded FITS files.

``api/database.py``
   Database engine, session creation, and SQLAlchemy base setup.

``api/security.py``
   Authentication helpers and dependency wiring for protected routes.

Programmatic Use
----------------

For frontend-integrated usage, the simplest client surface is the helper layer
in ``pages/api_client.py``. For broader codebase entry points, see
``api_reference.rst`` for the surrounding module map.