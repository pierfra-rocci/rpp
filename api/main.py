"""FastAPI application exposing RAPAS backend services."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import smtplib
import string
import tempfile
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from fastapi import (  # type: ignore[import]
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import]
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash, generate_password_hash

from .database import Base, engine, get_db
from .models import AnalysisJob, FitsFile, FitsFileStatus, JobEvent, JobStatus, PasswordRecoveryCode, User
from .schemas import (
    ConfigResponse,
    ConfigUpdateRequest,
    FitsFileListResponse,
    FitsFileSummary,
    FitsUploadResponse,
    JobEventSummary,
    JobListResponse,
    JobStatusResponse,
    JobSubmitRequest,
    JobSubmitResponse,
    LoginRequest,
    Message,
    RecoveryConfirmRequest,
    RecoveryRequest,
    RegisterRequest,
)
from .security import get_current_user
from .storage import build_storage_path

app = FastAPI(title="RAPAS API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    """Initialize database schema on startup."""
    Base.metadata.create_all(bind=engine)


def _ensure_password_strength(password: str) -> None:
    """Validate minimal password complexity."""
    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters long.",
        )
    if not any(c.isupper() for c in password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain an uppercase letter.",
        )
    if not any(c.islower() for c in password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain a lowercase letter.",
        )
    if not any(c.isdigit() for c in password):
        raise HTTPException(
            status_code=400,
            detail="Password must contain a digit.",
        )


RECOVERY_CODE_LENGTH = 6
RECOVERY_CODE_LIFETIME_MINUTES = 15

try:  # pragma: no cover - optional SMTP configuration
    import config as smtp_config  # type: ignore[import]
except ImportError:  # pragma: no cover - optional SMTP configuration
    smtp_config = None


def _generate_recovery_code() -> str:
    """Return a numeric recovery token."""
    return "".join(secrets.choice(string.digits) for _ in range(RECOVERY_CODE_LENGTH))


def _cleanup_expired_codes(db: Session) -> None:
    """Remove codes that have passed their validity window."""
    cutoff = datetime.utcnow()
    db.query(PasswordRecoveryCode).filter(
        PasswordRecoveryCode.expires_at < cutoff
    ).delete(synchronize_session=False)


def _send_recovery_email(to_email: str, code: str) -> None:
    """Dispatch the recovery code via SMTP, raising on failure."""
    if smtp_config is None:
        raise HTTPException(
            status_code=500,
            detail="Email service is not configured.",
        )

    smtp_server = getattr(smtp_config, "SMTP_SERVER", None)
    smtp_port = getattr(smtp_config, "SMTP_PORT", None)
    smtp_user = getattr(smtp_config, "SMTP_USER", None)
    encoded_pass = getattr(smtp_config, "SMTP_PASS_ENCODED", "")

    try:
        smtp_pass = base64.b64decode(encoded_pass).decode() if encoded_pass else None
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Email service is not configured.",
        ) from exc

    if (
        smtp_server is None
        or smtp_port is None
        or smtp_user is None
        or smtp_pass is None
    ):
        raise HTTPException(
            status_code=500,
            detail="Email service is not configured.",
        )

    try:
        smtp_server_str = str(smtp_server)
        smtp_port_int = int(smtp_port)
        smtp_user_str = str(smtp_user)
        smtp_pass_str = str(smtp_pass)
    except (TypeError, ValueError) as exc:  # pragma: no cover - misconfig
        raise HTTPException(
            status_code=500,
            detail="Email service is not configured.",
        ) from exc

    message = MIMEMultipart()
    message["From"] = smtp_user_str
    message["To"] = to_email
    message["Subject"] = "Password Recovery Code"
    message.attach(MIMEText(f"Your recovery code is: {code}", "plain"))

    try:
        with smtplib.SMTP(smtp_server_str, smtp_port_int) as server:
            server.starttls()
            server.login(smtp_user_str, smtp_pass_str)
            server.send_message(message)
    except (smtplib.SMTPException, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to send recovery email.",
        ) from exc


@app.get("/health", response_model=Message)
def health() -> Message:
    """Simple health probe."""
    return Message(message="ok")


@app.get("/api/jobs/health")
def jobs_health():
    """Check if Celery workers are available for background job processing."""
    celery_available = False
    worker_count = 0
    
    try:
        from celery_app import celery_app
        # Try to inspect active workers
        inspect = celery_app.control.inspect()
        active = inspect.active()
        if active:
            celery_available = True
            worker_count = len(active)
    except Exception:
        # Celery not available or not configured
        pass
    
    return {
        "celery_available": celery_available,
        "worker_count": worker_count,
        "message": "Celery workers available" if celery_available else "No workers",
    }


@app.post(
    "/api/register",
    response_model=Message,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: RegisterRequest,
    db: Session = Depends(get_db),
) -> Message:
    """Register a new user account."""
    _ensure_password_strength(payload.password)

    username_exists = db.query(User).filter(User.username == payload.username).first()
    if username_exists:
        raise HTTPException(
            status_code=409,
            detail="Username already registered.",
        )

    email_exists = db.query(User).filter(User.email == payload.email).first()
    if email_exists:
        raise HTTPException(
            status_code=409,
            detail="Email already registered.",
        )

    hashed_pw = generate_password_hash(payload.password)
    user = User(
        username=payload.username,
        password=hashed_pw,
        email=payload.email,
    )

    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Unable to register user.",
        ) from exc

    return Message(message="Registered successfully.")


@app.post("/api/login", response_model=Message)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> Message:
    """Validate credentials for downstream HTTP Basic usage."""
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not check_password_hash(user.password, payload.password):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password.",
        )
    return Message(message="Logged in successfully.")


@app.post("/api/recovery/request", response_model=Message)
def recovery_request(
    payload: RecoveryRequest,
    db: Session = Depends(get_db),
) -> Message:
    """Initiate password recovery by emailing a numeric code."""
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not found.")

    _cleanup_expired_codes(db)

    code = _generate_recovery_code()
    hashed_code = generate_password_hash(code)
    expires_at = datetime.utcnow() + timedelta(minutes=RECOVERY_CODE_LIFETIME_MINUTES)

    record = PasswordRecoveryCode(
        email=payload.email,
        code=hashed_code,
        expires_at=expires_at,
    )

    db.add(record)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to store recovery code.",
        ) from exc

    try:
        _send_recovery_email(payload.email, code)
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:  # pragma: no cover - unexpected mail failure
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Failed to send recovery email.",
        ) from exc

    db.commit()
    return Message(message="Recovery code sent to your email.")


@app.post("/api/recovery/confirm", response_model=Message)
def recovery_confirm(
    payload: RecoveryConfirmRequest,
    db: Session = Depends(get_db),
) -> Message:
    """Validate the recovery code and update the password."""
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email not found.")

    _ensure_password_strength(payload.new_password)
    _cleanup_expired_codes(db)

    codes = (
        db.query(PasswordRecoveryCode)
        .filter(PasswordRecoveryCode.email == payload.email)
        .all()
    )

    now = datetime.utcnow()
    valid_code = None
    for candidate in codes:
        if candidate.is_expired(now):
            continue
        if check_password_hash(candidate.code, payload.code):
            valid_code = candidate
            break

    if valid_code is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired recovery code.",
        )

    user.password = generate_password_hash(payload.new_password)
    db.add(user)
    db.delete(valid_code)

    try:
        db.commit()
    except Exception as exc:  # pragma: no cover
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Unable to reset password.",
        ) from exc

    return Message(message="Password updated successfully.")


@app.get("/api/config", response_model=ConfigResponse)
def get_config(
    current_user: User = Depends(get_current_user),
) -> ConfigResponse:
    """Retrieve persisted configuration for the authenticated user."""
    if not current_user.config_json:
        return ConfigResponse(config=None)
    try:
        data = json.loads(current_user.config_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail="Stored configuration is invalid JSON.",
        ) from exc
    return ConfigResponse(config=data)


@app.post("/api/config", response_model=Message)
def save_config(
    payload: ConfigUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Message:
    """Persist configuration blob for the authenticated user."""
    current_user.config_json = json.dumps(payload.config)
    db.add(current_user)
    db.commit()
    return Message(message="Config saved.")


async def _write_temp_file(upload: UploadFile) -> tuple[Path, str, int]:
    """Stream upload contents to a temp file while computing SHA256."""
    sha = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        try:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
                sha.update(chunk)
                size += len(chunk)
        finally:
            await upload.close()
    if size == 0:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return tmp_path, sha.hexdigest(), size


@app.post(
    "/api/upload/fits",
    response_model=FitsUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_fits(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FitsUploadResponse:
    """Persist a FITS upload to disk and store metadata."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    tmp_path, sha_hex, size_bytes = await _write_temp_file(file)
    started_at = datetime.now(timezone.utc)

    final_path, rel_path = build_storage_path(current_user.id, file.filename)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.replace(tmp_path, final_path)
    except Exception as exc:  # pragma: no cover - unexpected filesystem error
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to store uploaded file.",
        ) from exc

    completed_at = datetime.now(timezone.utc)

    fits_record = FitsFile(
        user_id=current_user.id,
        original_filename=file.filename,
        stored_relpath=rel_path,
        sha256=sha_hex,
        size_bytes=size_bytes,
        status=FitsFileStatus.STORED,
        upload_started_at=started_at,
        upload_completed_at=completed_at,
    )

    db.add(fits_record)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        # Clean up stored file to keep disk in sync with DB
        final_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=409,
            detail="A file with the same hash already exists.",
        ) from exc

    db.refresh(fits_record)
    return FitsUploadResponse(
        id=fits_record.id,
        original_filename=fits_record.original_filename,
        stored_relpath=fits_record.stored_relpath,
        sha256=fits_record.sha256,
        size_bytes=fits_record.size_bytes,
        status=fits_record.status,
        upload_completed_at=fits_record.upload_completed_at,
    )


@app.get("/api/fits", response_model=FitsFileListResponse)
def list_fits_files(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FitsFileListResponse:
    """Return all FITS files belonging to the authenticated user."""
    records = (
        db.query(FitsFile)
        .filter(FitsFile.user_id == current_user.id)
        .order_by(FitsFile.upload_completed_at.desc())
        .all()
    )
    summaries = [
        FitsFileSummary.model_validate(rec, from_attributes=True) for rec in records
    ]
    return FitsFileListResponse(files=summaries)


# ---------------------------------------------------------------------------
# Analysis Job Endpoints (Celery background tasks)
# ---------------------------------------------------------------------------


def _extract_progress_from_events(events: list[JobEvent]) -> tuple[float | None, str | None]:
    """Extract latest progress value and message from job events."""
    progress = None
    progress_message = None
    
    # Find the latest progress event
    for event in reversed(events):
        if event.event_type == "progress" and event.message:
            try:
                data = json.loads(event.message)
                progress = data.get("progress")
                progress_message = data.get("message")
                break
            except (json.JSONDecodeError, TypeError):
                continue
    
    return progress, progress_message


def _build_job_status_response(job: AnalysisJob) -> JobStatusResponse:
    """Build a JobStatusResponse from an AnalysisJob model."""
    progress, progress_message = _extract_progress_from_events(job.events)
    
    parameters = None
    if job.parameters_json:
        try:
            parameters = json.loads(job.parameters_json)
        except json.JSONDecodeError:
            parameters = None
    
    return JobStatusResponse(
        id=job.id,
        fits_file_id=job.fits_file_id,
        status=job.status.value,
        parameters=parameters,
        result_relpath=job.result_relpath,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
        events=[
            JobEventSummary.model_validate(e, from_attributes=True)
            for e in job.events
        ],
        progress=progress,
        progress_message=progress_message,
    )


@app.post(
    "/api/jobs/submit",
    response_model=JobSubmitResponse,
    status_code=status.HTTP_201_CREATED,
)
def submit_job(
    payload: JobSubmitRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobSubmitResponse:
    """
    Submit a new analysis job for background processing.
    
    The job will be queued and processed by a Celery worker.
    Use GET /api/jobs/{job_id}/status to monitor progress.
    """
    # Verify the FITS file exists and belongs to the user
    fits_file = (
        db.query(FitsFile)
        .filter(
            FitsFile.id == payload.fits_file_id,
            FitsFile.user_id == current_user.id,
        )
        .first()
    )
    if not fits_file:
        raise HTTPException(
            status_code=404,
            detail="FITS file not found or access denied.",
        )
    
    if fits_file.status != FitsFileStatus.STORED:
        raise HTTPException(
            status_code=400,
            detail="FITS file is not ready for processing.",
        )
    
    # Create the job record
    job = AnalysisJob(
        user_id=current_user.id,
        fits_file_id=payload.fits_file_id,
        parameters_json=json.dumps({
            "job_type": payload.job_type.value,
            **(payload.parameters or {}),
        }),
        status=JobStatus.QUEUED,
    )
    
    db.add(job)
    try:
        db.commit()
        db.refresh(job)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=400,
            detail="Unable to create job.",
        ) from exc
    
    # Add initial event
    event = JobEvent(
        job_id=job.id,
        event_type="queued",
        message=f"Job submitted: {payload.job_type.value}",
    )
    db.add(event)
    db.commit()
    
    # TODO: Step 3 - Dispatch Celery task based on job_type
    # Example:
    # from tasks.pipeline_tasks import run_plate_solve, run_photometry
    # if payload.job_type == JobType.PLATE_SOLVE:
    #     run_plate_solve.delay(job.id, fits_file.stored_relpath, payload.parameters or {})
    # elif payload.job_type == JobType.PHOTOMETRY:
    #     run_photometry.delay(job.id, fits_file.stored_relpath, payload.parameters or {})
    
    return JobSubmitResponse(
        job_id=job.id,
        status=job.status.value,
        message=f"Job {job.id} submitted successfully. Background processing not yet enabled.",
    )


@app.get("/api/jobs/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> JobStatusResponse:
    """Get the current status and progress of an analysis job."""
    job = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.id == job_id,
            AnalysisJob.user_id == current_user.id,
        )
        .first()
    )
    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found or access denied.",
        )
    
    return _build_job_status_response(job)


@app.get("/api/jobs/{job_id}/events", response_model=list[JobEventSummary])
def get_job_events(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[JobEventSummary]:
    """Get all events (progress history) for an analysis job."""
    job = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.id == job_id,
            AnalysisJob.user_id == current_user.id,
        )
        .first()
    )
    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found or access denied.",
        )
    
    return [
        JobEventSummary.model_validate(e, from_attributes=True)
        for e in job.events
    ]


@app.get("/api/jobs", response_model=JobListResponse)
def list_jobs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
    status_filter: str | None = None,
) -> JobListResponse:
    """
    List analysis jobs for the authenticated user.
    
    Args:
        limit: Maximum number of jobs to return (default 50)
        status_filter: Filter by status (queued, running, succeeded, failed)
    """
    query = db.query(AnalysisJob).filter(AnalysisJob.user_id == current_user.id)
    
    if status_filter:
        try:
            status_enum = JobStatus(status_filter)
            query = query.filter(AnalysisJob.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status_filter}. Must be one of: queued, running, succeeded, failed",
            )
    
    jobs = query.order_by(AnalysisJob.created_at.desc()).limit(limit).all()
    
    return JobListResponse(
        jobs=[_build_job_status_response(job) for job in jobs]
    )


@app.delete("/api/jobs/{job_id}", response_model=Message)
def cancel_job(
    job_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Message:
    """
    Cancel a queued job. Running jobs cannot be cancelled.
    """
    job = (
        db.query(AnalysisJob)
        .filter(
            AnalysisJob.id == job_id,
            AnalysisJob.user_id == current_user.id,
        )
        .first()
    )
    if not job:
        raise HTTPException(
            status_code=404,
            detail="Job not found or access denied.",
        )
    
    if job.status != JobStatus.QUEUED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status.value}. Only queued jobs can be cancelled.",
        )
    
    job.status = JobStatus.FAILED
    job.error_message = "Cancelled by user"
    job.completed_at = datetime.now(timezone.utc)
    
    event = JobEvent(
        job_id=job.id,
        event_type="cancelled",
        message="Job cancelled by user",
    )
    db.add(event)
    db.commit()
    
    # TODO: If task was already dispatched to Celery, revoke it
    # from celery_app import celery_app
    # celery_app.control.revoke(task_id, terminate=True)
    
    return Message(message=f"Job {job_id} cancelled.")
