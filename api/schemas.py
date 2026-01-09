"""Pydantic models for FastAPI requests and responses."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from .models import FitsFileStatus


class Message(BaseModel):
    """Generic response message payload."""

    message: str


class RegisterRequest(BaseModel):
    """Registration payload."""

    username: str = Field(min_length=3, max_length=80)
    password: str = Field(min_length=8, max_length=256)
    email: EmailStr


class LoginRequest(BaseModel):
    """Login payload."""

    username: str
    password: str


class RecoveryRequest(BaseModel):
    """Initiate password recovery for a user."""

    email: EmailStr


class RecoveryConfirmRequest(BaseModel):
    """Confirm password reset with a code."""

    email: EmailStr
    code: str = Field(min_length=1, max_length=64)
    new_password: str = Field(min_length=8, max_length=256)


class ConfigResponse(BaseModel):
    """User configuration blob."""

    config: Optional[Dict[str, Any]] = None


class ConfigUpdateRequest(BaseModel):
    """Persisted configuration wrapper."""

    config: Dict[str, Any]


class FitsFileSummary(BaseModel):
    """Metadata describing a stored FITS asset."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    stored_relpath: str
    sha256: str
    size_bytes: int
    status: FitsFileStatus
    upload_completed_at: Optional[datetime]


class FitsUploadResponse(BaseModel):
    """Response returned after a successful FITS upload."""

    id: int
    original_filename: str
    stored_relpath: str
    sha256: str
    size_bytes: int
    status: FitsFileStatus
    upload_completed_at: Optional[datetime]


class FitsFileListResponse(BaseModel):
    """List of FITS files for the current user."""

    files: List[FitsFileSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# WCS FITS Files and ZIP Archives Tracking Schemas (v1.6.0)
# ---------------------------------------------------------------------------


class WcsFitsFileSummary(BaseModel):
    """Metadata describing a WCS-solved FITS file."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    original_filename: str
    stored_filename: str
    has_wcs: bool
    created_at: datetime


class ZipArchiveSummary(BaseModel):
    """Metadata describing a result ZIP archive."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    archive_filename: str
    stored_relpath: str
    created_at: datetime


class WcsFitsFileWithZips(WcsFitsFileSummary):
    """WCS FITS file with its associated ZIP archives."""

    zip_archives: List[ZipArchiveSummary] = Field(default_factory=list)


class ZipArchiveWithFits(ZipArchiveSummary):
    """ZIP archive with its associated FITS files."""

    fits_files: List[WcsFitsFileSummary] = Field(default_factory=list)


class WcsFitsFileListResponse(BaseModel):
    """List of WCS FITS files for the current user."""

    files: List[WcsFitsFileSummary] = Field(default_factory=list)


class ZipArchiveListResponse(BaseModel):
    """List of ZIP archives for the current user."""

    archives: List[ZipArchiveSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis Job Schemas (Celery background tasks)
# ---------------------------------------------------------------------------


class JobType(str, Enum):
    """Types of analysis jobs that can be submitted."""

    PLATE_SOLVE = "plate_solve"
    PHOTOMETRY = "photometry"
    TRANSIENT_DETECTION = "transient_detection"
    FULL_PIPELINE = "full_pipeline"


class JobSubmitRequest(BaseModel):
    """Request to submit a new analysis job."""

    fits_file_id: int = Field(..., description="ID of the FITS file to process")
    job_type: JobType = Field(..., description="Type of analysis to run")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Processing parameters (depends on job type)",
    )


class JobSubmitResponse(BaseModel):
    """Response after submitting a job."""

    job_id: int
    status: str
    message: str


class JobEventSummary(BaseModel):
    """Summary of a job progress event."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    event_type: str
    message: Optional[str]
    created_at: datetime


class JobStatusResponse(BaseModel):
    """Current status of an analysis job."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    fits_file_id: int
    status: str
    parameters: Optional[Dict[str, Any]] = None
    result_relpath: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    events: List[JobEventSummary] = Field(default_factory=list)
    progress: Optional[float] = Field(
        default=None,
        description="Latest progress value (0.0 to 1.0) if available",
    )
    progress_message: Optional[str] = Field(
        default=None,
        description="Latest progress message if available",
    )


class JobListResponse(BaseModel):
    """List of analysis jobs for the current user."""

    jobs: List[JobStatusResponse] = Field(default_factory=list)

