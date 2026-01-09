"""
Celery application configuration for RAPAS background task processing.

This module configures Celery with Redis as the message broker and result backend.
Workers execute long-running tasks (plate-solving, photometry, transient detection)
independently of the Streamlit frontend.

Usage:
    # Start a worker (from project root):
    celery -A celery_app worker --loglevel=info

    # On Windows, you may need:
    celery -A celery_app worker --loglevel=info --pool=solo

Environment Variables:
    REDIS_URL: Redis connection URL (default: redis://localhost:6379/0)
"""

import os
from celery import Celery

# Redis configuration from environment or default to localhost
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery application
celery_app = Celery(
    "rapas_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks.pipeline_tasks"],  # Auto-discover tasks
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    
    # Result expiration (24 hours)
    result_expires=86400,
    
    # Task execution settings
    task_acks_late=True,  # Acknowledge after task completes (safer for long tasks)
    task_reject_on_worker_lost=True,  # Requeue if worker crashes
    
    # Rate limiting (prevent overload)
    worker_prefetch_multiplier=1,  # One task at a time per worker
    
    # Task time limits (seconds)
    task_soft_time_limit=1800,  # 30 minutes soft limit (raises exception)
    task_time_limit=2100,  # 35 minutes hard limit (kills task)
    
    # Task result settings
    task_track_started=True,  # Track when task starts
    task_send_sent_event=True,  # Send event when task is dispatched
)

# Optional: Configure task routes for different queues
# celery_app.conf.task_routes = {
#     "tasks.pipeline_tasks.run_plate_solve": {"queue": "astrometry"},
#     "tasks.pipeline_tasks.run_photometry": {"queue": "photometry"},
# }

if __name__ == "__main__":
    celery_app.start()
