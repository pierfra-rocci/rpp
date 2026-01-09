
Installation Guide
==================

This page lists quick, copyable steps to set up the project locally on Windows (PowerShell) and Unix (bash). It covers both FastAPI (recommended) and legacy backend options. External tools (Astrometry.net, SCAMP) are optional for advanced features.


Minimum Requirements
--------------------

- Python 3.12 (exactly) - required for verified dependency compatibility
- pip (latest version recommended)
- 2-3 GB free disk space (more recommended for large image sets and temporary processing files)
- Redis server (optional, required for background job processing)


Quick Install (PowerShell)
-------------------------

.. code-block:: powershell

   git clone https://github.com/your-repo/rpp.git
   cd rpp
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e .

Quick Install (bash)
--------------------

.. code-block:: bash

   git clone https://github.com/your-repo/rpp.git
   cd rpp
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .


Optional External Tools
----------------------

- Astrometry.net: required for blind plate solving (solve-field). Install via package manager or from source and ensure `solve-field` is on PATH.
- SCAMP: optional, for astrometric refinement.
- Redis: required for background job processing (Celery task queue).

  .. code-block:: bash

     # Ubuntu/Debian
     sudo apt-get install redis-server
     sudo systemctl enable redis-server
     sudo systemctl start redis-server

     # macOS with Homebrew
     brew install redis
     brew services start redis

     # Windows (via Docker)
     docker run -d -p 6379:6379 redis


Environment Variables
---------------------

Set SMTP credentials if you want the app to send emails (password can be base64-encoded in `SMTP_PASS_ENCODED`):

.. code-block:: powershell

   $env:SMTP_SERVER = 'smtp.gmail.com'
   $env:SMTP_PORT = '587'
   $env:SMTP_USER = 'you@example.com'
   $env:SMTP_PASS_ENCODED = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('your-app-password'))

Set Redis/Celery configuration for background job processing:

.. code-block:: powershell

   # Windows PowerShell
   $env:REDIS_URL = 'redis://localhost:6379/0'
   $env:CELERY_BROKER_URL = 'redis://localhost:6379/0'
   $env:CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

.. code-block:: bash

   # Linux/macOS
   export REDIS_URL='redis://localhost:6379/0'
   export CELERY_BROKER_URL='redis://localhost:6379/0'
   export CELERY_RESULT_BACKEND='redis://localhost:6379/0'


Verification
------------

After installation, verify Python dependencies and start the app:

.. code-block:: powershell

   python -c "import streamlit, astropy; print('OK')"

   # Start backend (choose one):
   python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
   # or legacy:
   python backend.py

   # Start frontend:
   streamlit run frontend.py

If you plan to use plate solving, verify `solve-field --help` works and download index files appropriate for your typical field scale.


Support
-------

- See :doc:`usage` and :doc:`examples` for simple run instructions and a minimal Python example.
- For Astrometry.net and SCAMP installation follow their project pages.