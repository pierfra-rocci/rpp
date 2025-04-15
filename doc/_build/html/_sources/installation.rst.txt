Installation
===========

Requirements
-----------

Photometry Factory for RAPAS requires:

* Python 3.8 or higher
* Several astronomical Python packages

System Dependencies
------------------

Before installing PFR, ensure you have the following system dependencies:

- FITS file support (usually included with astropy)
- Internet connection for catalog access and plate solving

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

4. Run the application:

   .. code-block:: bash

      streamlit run pfr_app.py

API Keys
--------

Some features of PFR require API keys:

* **Astrometry.net**: Register at http://nova.astrometry.net to get an API key for plate solving

Configuration
------------

By default, PFR saves results in the ``pfr_results`` directory. You can change this behavior within the application.