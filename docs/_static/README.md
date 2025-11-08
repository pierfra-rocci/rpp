# Static Files Directory

This directory contains static files used in the Sphinx documentation, such as images.

## Required Images

Please ensure the following images are present in this directory and are up-to-date for the documentation build:

-   `pfr_logo.png` - The logo for Photometry Factory for RAPAS (used in `index.rst`).
-   `app_layout.png` - Screenshot of the main application layout (used in `user_guide.rst`). *Verify if still accurate.*
-   `pfr_interface.png` - General screenshot of the PFR interface (used in `usage.rst`). *Verify if still accurate.*
-   `basic_example.png` - Screenshot illustrating basic usage (used in `examples.rst`). *Verify if still accurate.*

You can create/update these screenshots by:
1. Running the backend: `python backend_dev.py`
2. Running the frontend: `streamlit run run_frontend.py`
3. Navigating the application in your browser.
4. Taking screenshots of relevant sections.
5. Saving them to this directory with the correct filenames.
