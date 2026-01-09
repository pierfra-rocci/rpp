
Usage Guide
===========

This page shows the minimal steps to run the project locally and use the main features, matching the modern workflow (FastAPI or legacy backend) described in the README and TUTORIAL.

Quick Start (Development)
-------------------------

1. **Create and activate a virtual environment**

   .. code-block:: powershell

      python -m venv .venv
      .\.venv\Scripts\Activate.ps1

   .. code-block:: bash

      python3 -m venv .venv
      source .venv/bin/activate

2. **Install requirements**

   .. code-block:: powershell

      pip install -e .

3. **Start the backend**

   - *FastAPI (recommended):*

     .. code-block:: powershell

        python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

     Or use the provided batch file (Windows):

     .. code-block:: powershell

        run_all_cmd.bat

   - *Legacy Flask backend:*

     .. code-block:: powershell

        python backend.py

4. **Start the frontend (Streamlit)**

   .. code-block:: powershell

      streamlit run frontend.py

   or

   .. code-block:: powershell

      streamlit run pages/app.py

   Visit the URL printed by Streamlit (usually http://localhost:8501).


Production Deployment
---------------------

For production, you can use a WSGI/ASGI server (e.g., uvicorn, gunicorn, waitress) to run the backend. See the README for details. The frontend is always launched with Streamlit.

**With Background Jobs (Recommended)**:

To enable browser-independent analysis with background job processing:

1. **Start Redis** (required for Celery):

   .. code-block:: bash

      # Linux
      sudo systemctl start redis
      
      # macOS
      brew services start redis
      
      # Windows (Docker)
      docker run -d -p 6379:6379 redis

2. **Start all services** using the management script:

   .. code-block:: bash

      # Linux (starts FastAPI + Celery + Streamlit)
      ./run_all_linux.sh
      
      # Or use the management script
      ./manage_streamlit.sh start

3. **Monitor services**:

   .. code-block:: bash

      ./manage_streamlit.sh status    # Check all services
      ./manage_streamlit.sh logs all  # View all logs
      ./manage_streamlit.sh redis status  # Check Redis

**Service Management Commands**:

.. code-block:: bash

   ./manage_streamlit.sh status   # Show service status
   ./manage_streamlit.sh start    # Start all services
   ./manage_streamlit.sh stop     # Stop all services gracefully
   ./manage_streamlit.sh restart  # Restart all services
   ./manage_streamlit.sh logs celery  # View Celery logs
   ./manage_streamlit.sh redis start  # Start Redis


Logging and Troubleshooting
--------------------------

- Backend logs are printed to the console. Check for database initialization messages about the `users` and `recovery_codes` tables.
- If email-based features fail, confirm SMTP environment variables (`SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS_ENCODED`) are set.
- For astrometry/plate solving features, ensure external dependencies (like Astrometry.net indices) are installed and available on PATH.


Next Steps
----------

- See :doc:`examples` for a minimal Python example.
- See :doc:`installation` for more details about optional dependencies and advanced setup.
