Usage Guide
===========

This page shows the minimal steps to run the project locally and how to use
the main features. It includes short, copy-paste commands for Windows
PowerShell and Unix shells.

Quick start (development)
-------------------------

1. Create and activate a virtual environment (PowerShell):

   .. code-block:: powershell

      python -m venv .venv
      .\.venv\Scripts\Activate.ps1

   or on Unix/macOS:

   .. code-block:: bash

      python3 -m venv .venv
      source .venv/bin/activate

2. Install requirements:

   .. code-block:: powershell

      pip install -r requirements.txt

3. Start the backend (development server):

   .. code-block:: powershell

      python backend.py

   The backend runs by default on port 5000.

4. Start the frontend (Streamlit):

   .. code-block:: powershell

      streamlit run frontend.py

   Visit the URL printed by Streamlit (usually http://localhost:8501).

Production-ish run
------------------

To run the backend behind a production server you can use `waitress` (Windows
friendly) like the included `backend_prod.py`:

.. code-block:: powershell

   python backend_prod.py

And run the frontend with Streamlit as above.

Logging and troubleshooting
---------------------------

- Backend logs are printed to the console. Check for database initialization
  messages about the `users` and `recovery_codes` tables.
- If email-based features fail, confirm SMTP environment variables
  (`SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS_ENCODED`) are set.
- For astrometry/plate solving features ensure external dependencies (like
  Astrometry.net indices) are installed and available on PATH.

Next steps
----------

- See :doc:`examples` for a tiny local Python example.
- See :doc:`installation` for more details about optional dependencies.
