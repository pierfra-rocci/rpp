"""
RAPAS Background Tasks Module

This package contains Celery task definitions for long-running operations:
- Plate solving (astrometry)
- Source detection and photometry
- PSF photometry
- Transient candidate detection

Tasks are designed to:
1. Update job status in the database
2. Report progress via JobEvent records
3. Store results for later retrieval
"""

from tasks.pipeline_tasks import (
    run_plate_solve,
    run_photometry,
    run_transient_detection,
)

__all__ = [
    "run_plate_solve",
    "run_photometry", 
    "run_transient_detection",
]
