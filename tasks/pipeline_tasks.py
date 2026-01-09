"""
Celery task definitions for RAPAS pipeline operations.

These tasks wrap the core pipeline functions to run as background jobs.
Each task:
1. Updates job status to RUNNING
2. Executes the processing function with progress callbacks
3. Updates job status to SUCCEEDED or FAILED
4. Records events for progress tracking

Note: Full implementation requires Step 3 (callback interface in pipeline functions).
      This is a placeholder structure that will be completed in later steps.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from celery import shared_task
from celery.exceptions import SoftTimeLimitExceeded

# These imports will be used once the callback interface is implemented
# from src.astrometry import solve_with_astrometrynet
# from src.pipeline import run_detection_and_photometry
# from src.transient import find_transient_candidates

from api.database import session_scope
from api.models import AnalysisJob, JobEvent, JobStatus

logger = logging.getLogger(__name__)


def update_job_status(
    job_id: int,
    status: JobStatus,
    error_message: Optional[str] = None,
    result_relpath: Optional[str] = None,
) -> None:
    """Update job status in database."""
    with session_scope() as session:
        job = session.query(AnalysisJob).filter(AnalysisJob.id == job_id).first()
        if job:
            job.status = status
            if error_message:
                job.error_message = error_message
            if result_relpath:
                job.result_relpath = result_relpath
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                job.completed_at = datetime.now(timezone.utc)
            session.commit()


def add_job_event(job_id: int, event_type: str, message: str) -> None:
    """Add a progress event to the job."""
    with session_scope() as session:
        event = JobEvent(
            job_id=job_id,
            event_type=event_type,
            message=message,
        )
        session.add(event)
        session.commit()


def create_progress_callback(job_id: int):
    """
    Create a progress callback function for pipeline operations.
    
    This callback can be passed to pipeline functions to report progress
    without depending on Streamlit.
    
    Returns a callable that accepts (progress: float, message: str).
    """
    def callback(progress: float, message: str) -> None:
        """Report progress (0.0 to 1.0) with a message."""
        add_job_event(
            job_id=job_id,
            event_type="progress",
            message=json.dumps({"progress": progress, "message": message}),
        )
    return callback


@shared_task(bind=True, name="tasks.run_plate_solve")
def run_plate_solve(
    self,
    job_id: int,
    fits_path: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run plate solving (astrometry) as a background task.
    
    Args:
        job_id: Database ID of the AnalysisJob
        fits_path: Path to the FITS file
        params: Processing parameters (ra_hint, dec_hint, radius, etc.)
    
    Returns:
        Dict with success status and result path or error message
    """
    logger.info(f"Starting plate solve task for job {job_id}")
    update_job_status(job_id, JobStatus.RUNNING)
    add_job_event(job_id, "started", "Plate solving started")
    
    try:
        # TODO: Step 3 - Call refactored solve_with_astrometrynet with callback
        # progress_callback = create_progress_callback(job_id)
        # result = solve_with_astrometrynet(
        #     fits_path=fits_path,
        #     progress_callback=progress_callback,
        #     **params
        # )
        
        # Placeholder - will be implemented in Step 3
        raise NotImplementedError(
            "Plate solve task requires callback interface (Step 3)"
        )
        
    except SoftTimeLimitExceeded:
        error_msg = "Plate solving timed out (exceeded 30 minutes)"
        logger.error(f"Job {job_id}: {error_msg}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
        add_job_event(job_id, "failed", error_msg)
        return {"success": False, "error": error_msg}
        
    except Exception as e:
        error_msg = f"Plate solving failed: {str(e)}"
        logger.exception(f"Job {job_id}: {error_msg}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
        add_job_event(job_id, "failed", error_msg)
        return {"success": False, "error": error_msg}


@shared_task(bind=True, name="tasks.run_photometry")
def run_photometry(
    self,
    job_id: int,
    fits_path: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run source detection and photometry as a background task.
    
    Args:
        job_id: Database ID of the AnalysisJob
        fits_path: Path to the FITS file
        params: Processing parameters (fwhm, threshold, apertures, etc.)
    
    Returns:
        Dict with success status and result path or error message
    """
    logger.info(f"Starting photometry task for job {job_id}")
    update_job_status(job_id, JobStatus.RUNNING)
    add_job_event(job_id, "started", "Photometry pipeline started")
    
    try:
        # TODO: Step 3 - Call refactored run_detection_and_photometry with callback
        # progress_callback = create_progress_callback(job_id)
        # result = run_detection_and_photometry(
        #     fits_path=fits_path,
        #     progress_callback=progress_callback,
        #     **params
        # )
        
        # Placeholder - will be implemented in Step 3
        raise NotImplementedError(
            "Photometry task requires callback interface (Step 3)"
        )
        
    except SoftTimeLimitExceeded:
        error_msg = "Photometry timed out (exceeded 30 minutes)"
        logger.error(f"Job {job_id}: {error_msg}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
        add_job_event(job_id, "failed", error_msg)
        return {"success": False, "error": error_msg}
        
    except Exception as e:
        error_msg = f"Photometry failed: {str(e)}"
        logger.exception(f"Job {job_id}: {error_msg}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
        add_job_event(job_id, "failed", error_msg)
        return {"success": False, "error": error_msg}


@shared_task(bind=True, name="tasks.run_transient_detection")
def run_transient_detection(
    self,
    job_id: int,
    fits_path: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run transient candidate detection as a background task.
    
    Args:
        job_id: Database ID of the AnalysisJob
        fits_path: Path to the FITS file
        params: Detection parameters (catalog, magnitude limits, etc.)
    
    Returns:
        Dict with success status and result path or error message
    """
    logger.info(f"Starting transient detection task for job {job_id}")
    update_job_status(job_id, JobStatus.RUNNING)
    add_job_event(job_id, "started", "Transient detection started")
    
    try:
        # TODO: Step 3 - Call refactored find_transient_candidates with callback
        # progress_callback = create_progress_callback(job_id)
        # result = find_transient_candidates(
        #     fits_path=fits_path,
        #     progress_callback=progress_callback,
        #     **params
        # )
        
        # Placeholder - will be implemented in Step 3
        raise NotImplementedError(
            "Transient detection task requires callback interface (Step 3)"
        )
        
    except SoftTimeLimitExceeded:
        error_msg = "Transient detection timed out (exceeded 30 minutes)"
        logger.error(f"Job {job_id}: {error_msg}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
        add_job_event(job_id, "failed", error_msg)
        return {"success": False, "error": error_msg}
        
    except Exception as e:
        error_msg = f"Transient detection failed: {str(e)}"
        logger.exception(f"Job {job_id}: {error_msg}")
        update_job_status(job_id, JobStatus.FAILED, error_message=error_msg)
        add_job_event(job_id, "failed", error_msg)
        return {"success": False, "error": error_msg}
