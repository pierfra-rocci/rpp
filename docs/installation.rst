Installation Guide
=================

This page lists quick, copyable steps to set up the project locally on
Windows (PowerShell) and Unix (bash). It focuses on Python-level
installation; external tools (Astrometry.net, SCAMP) are optional for
advanced features.

Minimum requirements
--------------------

- Python 3.12+
- pip
- 2 GB free disk (more recommended for large image sets)

Quick install (PowerShell)
-------------------------

.. code-block:: powershell

   git clone https://github.com/your-repo/rpp.git
   cd rpp
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -e .

Quick install (bash)
--------------------

.. code-block:: bash

   git clone https://github.com/your-repo/rpp.git
   cd rpp
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .

Optional external tools
-----------------------

- Astrometry.net: required for blind plate solving (solve-field). Install
  via package manager or from source and ensure `solve-field` is on PATH.
- SCAMP: optional, for astrometric refinement.

Environment variables
---------------------

Set SMTP credentials if you want the app to send emails (password can be
base64-encoded in `SMTP_PASS_ENCODED`):

.. code-block:: powershell

   $env:SMTP_SERVER = 'smtp.gmail.com'
   $env:SMTP_PORT = '587'
   $env:SMTP_USER = 'you@example.com'
   $env:SMTP_PASS_ENCODED = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes('your-app-password'))

Verification
------------

After installation, verify Python dependencies and start the app:

.. code-block:: powershell

   python -c "import streamlit, astropy; print('OK')"
   python backend.py
   streamlit run frontend.py

If you plan to use plate solving, verify `solve-field --help` works and
download index files appropriate for your typical field scale.

Support
-------

- See :doc:`usage` and :doc:`examples` for simple run instructions and a
  minimal Python example.
- For Astrometry.net and SCAMP installation follow their project pages.