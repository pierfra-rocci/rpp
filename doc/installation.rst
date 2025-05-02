Installation
===========

Requirements
-----------

Photometry Factory for RAPAS requires:

* Python 3.8 or higher
* Several astronomical Python packages (see `requirements.txt`)
* A running instance of the backend server (e.g., `backend_dev.py`)
* An email server configuration (for password recovery) via environment variables (SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASS)

System Dependencies
------------------

Before installing PFR, ensure you have the following system dependencies:

- FITS file support (usually included with astropy)
- Internet connection for catalog access and plate solving
- SQLite3 (usually included with Python)

Installation Steps
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-username/pfr.git
      cd pfr

2. Create a virtual environment (recommended):

   .. code-block:: bash

      python -m venv pfr-env
      source pfr-env/bin/activate  # On Windows: pfr-env\Scripts\activate

3. Install the required packages:

   .. code-block:: bash

      pip install -r requirements.txt

4. Initialize the user database (if running for the first time):
   The backend (`backend_dev.py`) will create `users.db` automatically on first run.

5. Configure Email Server (Optional, for password recovery):
   Set the following environment variables:
   - `SMTP_SERVER`: Your SMTP server address (e.g., smtp.gmail.com)
   - `SMTP_PORT`: Your SMTP server port (e.g., 587)
   - `SMTP_USER`: Your SMTP username/email
   - `SMTP_PASS`: Your SMTP password or app password

6. Run the backend server in a separate terminal:

   .. code-block:: bash

      python backend_dev.py

7. Run the frontend application:

   .. code-block:: bash

      streamlit run run_frontend.py

API Keys
--------

Some features of PFR require API keys:

* **Astrometry.net**: Register at http://nova.astrometry.net to get an API key for plate solving (if using the astrometry.net option, currently Siril is also an option).
* **Astro-Colibri**: Register at https://astro-colibri.science to get a UID key for transient/event cross-matching. Enter this in the sidebar.

Configuration
------------

- User-specific configurations (observatory, analysis parameters, API keys) are saved per user via the backend (`users.db` and `*_config.json` files in the results directory).
- Default results directory is `rpp_results`.