Installation
===========

Requirements
-----------

Photometry Factory for RAPAS requires Python 3.8 or later. It depends on several
astronomical Python packages, particularly:

* astropy
* photutils
* astroquery
* matplotlib
* numpy
* pandas
* streamlit

Basic Installation
-----------------

You can install the required packages using pip:

.. code-block:: bash

   pip install -r requirements.txt

Alternative Installation Options
-------------------------------

Using the executable (Windows)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A standalone Windows executable is available that doesn't require Python installation:

1. Download the latest release
2. Extract all files to a directory of your choice
3. Run ``PfrApp.bat``

Using Streamlit
^^^^^^^^^^^^^^

To run the application through streamlit:

.. code-block:: bash

   cd /path/to/P_F_R
   streamlit run pfr_app.py

Development Installation
-----------------------

For development purposes, clone the repository:

.. code-block:: bash

   git clone https://github.com/yourusername/P_F_R.git
   cd P_F_R
   pip install -e .