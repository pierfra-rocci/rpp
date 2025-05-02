Photometry Factory for RAPAS
============================

.. image:: _static/pfr_logo.png
   :width: 200px
   :align: right
   :alt: Photometry Factory for RAPAS logo

**Photometry Factory for RAPAS (PFR)** is an interactive web application for analyzing astronomical images and performing photometric analysis, built with Streamlit and Python.

It provides a user-friendly pipeline for:

*   User authentication and configuration management.
*   FITS image display and loading.
*   Optional Cosmic Ray removal.
*   Background estimation and subtraction.
*   Source detection and FWHM estimation.
*   Astrometric plate solving (via Siril) and WCS refinement (via `stdpipe`).
*   Aperture and PSF photometry.
*   Zero point calibration with GAIA DR3.
*   Cross-matching with astronomical catalogs (SIMBAD, SkyBoT, AAVSO, Astro-Colibri, Milliquas).
*   Interactive visualization using Aladin Lite.
*   Downloading comprehensive results (catalogs, plots, logs) in a ZIP archive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   user_guide
   features
   advanced_features
   api
   examples
   troubleshooting
   changelog

Features Overview
-----------------

-   Interactive web-based interface built with Streamlit.
-   Secure user login and persistent configuration.
-   Robust FITS file handling.
-   Optional automated Cosmic Ray removal (`astroscrappy`).
-   Sophisticated background estimation (`photutils.Background2D`).
-   Accurate source detection (`photutils.DAOStarFinder`) and FWHM measurement.
*   Aperture and Empirical PSF photometry (`photutils`).
*   Plate solving using Siril and WCS refinement using `stdpipe`.
*   Photometric calibration against GAIA DR3.
*   Extensive catalog cross-matching (GAIA, SIMBAD, SkyBoT, AAVSO, Astro-Colibri, Milliquas).
*   Interactive sky viewing with Aladin Lite integration.
*   Downloadable results package (CSV, PNG, FITS, TXT, LOG).

Indices and tables
-----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`