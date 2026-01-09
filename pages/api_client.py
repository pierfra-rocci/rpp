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
    ) -> requests.Response:
        try:
            response = requests.get(
                f"{self.base_url}{path}",
                auth=auth,
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
