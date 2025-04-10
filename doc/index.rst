Photometry Factory for RAPAS Documentation
=========================================

.. image:: _static/pfr_logo.png
   :width: 200px
   :align: right
   :alt: PFR Logo

**Photometry Factory for RAPAS (PFR)** is a comprehensive Python-based application designed for astronomical image processing and photometry with a focus on RAPAS (Réseau Amateur Professionnel pour les Alertes) data.

This tool provides astronomers with a streamlined workflow for image calibration, plate solving, and photometric analysis of astronomical images through an intuitive web interface powered by Streamlit.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   user_guide
   examples
   advanced_features
   api_reference
   troubleshooting
   changelog

Key Features
-----------

* **Image Calibration**: Apply bias, dark, and flat-field corrections to raw astronomical images
* **Automatic Plate Solving**: Determine accurate WCS coordinates using astrometry.net
* **Source Detection**: Identify astronomical sources with configurable detection parameters
* **Photometry**: Perform aperture and PSF photometry on detected sources
* **Zero-point Calibration**: Automatically calibrate magnitudes using GAIA DR3 catalog cross-matching
* **Catalog Integration**: Cross-match detected sources with SIMBAD, SkyBoT, and AAVSO catalogs
* **Interactive Visualization**: Explore results with interactive plots and an embedded Aladin Lite viewer
* **Standardized Outputs**: Export results to CSV formats compatible with other astronomical software

Quick Start
-----------

.. code-block:: bash

   # Install required packages
   pip install -r requirements.txt
   
   # Run the application
   streamlit run pfr_app.py

Acknowledgements
---------------

This software was developed for RAPAS (Réseau Amateur Professionnel pour les Alertes). We thank all contributors and users who have provided valuable feedback.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`