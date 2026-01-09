"""Client utilities for talking to the FastAPI backend."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests  # type: ignore[import]
from requests.auth import HTTPBasicAuth  # type: ignore[import]

DEFAULT_API_URL = os.getenv("RPP_API_URL", "http://localhost:8000")
DEFAULT_LEGACY_URL = os.getenv("RPP_LEGACY_URL", "http://localhost:5000")
_REQUEST_TIMEOUT = 15


class ApiError(Exception):
    """Raised when the API returns an error response."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class ApiClient:
    """Thin wrapper around HTTP calls to the FastAPI service."""

    def __init__(self, base_url: Optional[str] = None) -> None:
        root = base_url or DEFAULT_API_URL
        self.base_url = root.rstrip("/")

    # ------------------------------------------------------------------
    # Job management endpoints
    # ------------------------------------------------------------------
    def submit_job(
        self,
        username: str,
        password: str,
        job_type: str,
        fits_file_id: int,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Submit a background job for processing."""
        payload = {
            "job_type": job_type,
            "fits_file_id": fits_file_id,
            "params": params or {},
        }
        response = self._post(
            "/api/jobs/submit",
            json=payload,
            auth=HTTPBasicAuth(username, password),
        )
        return response.json()

    def get_job_status(
        self,
        username: str,
        password: str,
        job_id: int,
    ) -> Dict[str, Any]:
        """Get the status and details of a job."""
        response = self._get(
            f"/api/jobs/{job_id}/status",
            auth=HTTPBasicAuth(username, password),
        )
        return response.json()

    def get_job_events(
        self,
        username: str,
        password: str,
        job_id: int,
    ) -> Dict[str, Any]:
        """Get progress events for a job."""
        response = self._get(
            f"/api/jobs/{job_id}/events",
            auth=HTTPBasicAuth(username, password),
        )
        return response.json()

    def list_jobs(
        self,
        username: str,
        password: str,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """List jobs for the current user."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        response = self._get(
            "/api/jobs",
            auth=HTTPBasicAuth(username, password),
            params=params,
        )
        return response.json()

    def cancel_job(
        self,
        username: str,
        password: str,
        job_id: int,
    ) -> str:
        """Cancel a pending or running job."""
        response = self._delete(
            f"/api/jobs/{job_id}",
            auth=HTTPBasicAuth(username, password),
        )
        return self._extract_message(response)

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------
    def login(self, username: str, password: str) -> str:
        """Validate credentials against the API."""
        payload = {"username": username, "password": password}
        response = self._post("/api/login", json=payload, auth=None)
        return self._extract_message(response)

    def register(self, username: str, password: str, email: str) -> str:
        """Register a new user."""
        payload = {"username": username, "password": password, "email": email}
        response = self._post("/api/register", json=payload, auth=None)
        return self._extract_message(response)

    def request_password_recovery(self, email: str) -> str:
        """Trigger password recovery email."""
        payload = {"email": email}
        response = self._post("/api/recovery/request", json=payload, auth=None)
        return self._extract_message(response)

    def confirm_password_recovery(
        self,
        email: str,
        code: str,
        new_password: str,
    ) -> str:
        """Complete password reset using a recovery code."""
        payload = {
            "email": email,
            "code": code,
            "new_password": new_password,
        }
        response = self._post("/api/recovery/confirm", json=payload, auth=None)
        return self._extract_message(response)

    # ------------------------------------------------------------------
    # Configuration endpoints
    # ------------------------------------------------------------------
    def get_config(
        self,
        username: str,
        password: str,
    ) -> Optional[Dict[str, Any]]:
        """Fetch stored user configuration."""
        response = self._get(
            "/api/config",
            auth=HTTPBasicAuth(username, password),
        )
        data = response.json()
        return data.get("config") if isinstance(data, dict) else None

    def save_config(
        self,
        username: str,
        password: str,
        config: Dict[str, Any],
    ) -> str:
        """Persist user configuration."""
        payload = {"config": config}
        response = self._post(
            "/api/config",
            json=payload,
            auth=HTTPBasicAuth(username, password),
        )
        return self._extract_message(response)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get(
        self,
        path: str,
        auth: Optional[HTTPBasicAuth],
        params: Optional[Dict[str, Any]] = None,
    ) -> requests.Response:
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                auth=auth,
                params=params,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:  # pragma: no cover
            raise ApiError(status_code=0, message=str(exc)) from exc
        self._raise_for_status(response)
        return response

    def _post(
        self,
        path: str,
        json: Optional[Dict[str, Any]],
        auth: Optional[HTTPBasicAuth],
    ) -> requests.Response:
        try:
            response = requests.post(
                f"{self.base_url}{path}",
                json=json,
                auth=auth,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:  # pragma: no cover
            raise ApiError(status_code=0, message=str(exc)) from exc
        self._raise_for_status(response)
        return response

    def _delete(
        self,
        path: str,
        auth: Optional[HTTPBasicAuth],
    ) -> requests.Response:
        try:
            response = requests.delete(
                f"{self.base_url}{path}",
                auth=auth,
                timeout=_REQUEST_TIMEOUT,
            )
        except requests.RequestException as exc:  # pragma: no cover
            raise ApiError(status_code=0, message=str(exc)) from exc
        self._raise_for_status(response)
        return response

    @staticmethod
    def _extract_message(response: requests.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text or ""
        if isinstance(data, dict) and "message" in data:
            return str(data["message"])
        return response.text or ""

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        if response.ok:
            return
        message = ApiClient._extract_error_message(response)
        raise ApiError(status_code=response.status_code, message=message)

    @staticmethod
    def _extract_error_message(response: requests.Response) -> str:
        try:
            data = response.json()
        except ValueError:
            return response.text or "Unexpected error"
        if isinstance(data, dict):
            if "detail" in data:
                detail = data["detail"]
                if isinstance(detail, str):
                    return detail
                if isinstance(detail, dict) and "msg" in detail:
                    return str(detail["msg"])
            if "message" in data:
                return str(data["message"])
        return response.text or "Unexpected error"


def detect_backend(timeout: float = 2.0) -> Dict[str, str]:
    """Determine which backend should be used by probing the API."""
    api_url_env = os.getenv("RPP_API_URL")
    legacy_url_env = os.getenv("RPP_LEGACY_URL")

    api_url = (api_url_env or DEFAULT_API_URL).rstrip("/")
    legacy_url = (legacy_url_env or DEFAULT_LEGACY_URL).rstrip("/")

    health_url = f"{api_url}/health"
    try:
        response = requests.get(health_url, timeout=timeout)
        if response.ok:
            return {
                "mode": "api",
                "api_base_url": api_url,
                "legacy_backend_url": legacy_url,
                "message": "Using API backend",
            }
    except requests.RequestException:
        pass

    return {
        "mode": "legacy",
        "api_base_url": api_url,
        "legacy_backend_url": legacy_url,
        "message": ("API backend not reachable. Using legacy server"),
    }


def check_celery_available(base_url: Optional[str] = None, timeout: float = 2.0) -> bool:
    """Check if Celery workers are available for background processing."""
    api_url = (base_url or DEFAULT_API_URL).rstrip("/")
    try:
        response = requests.get(f"{api_url}/api/jobs/health", timeout=timeout)
        if response.ok:
            data = response.json()
            return data.get("celery_available", False)
    except requests.RequestException:
        pass
    return False
