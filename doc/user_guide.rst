User Guide
=========

This guide provides an overview of how to use the Photometry Factory for RAPAS.

Getting Started
--------------

The application offers a streamlined workflow for astronomical image processing and photometry:

1. Upload a science image (and calibration frames if needed)
2. Apply calibration steps as needed
3. Run plate solving if WCS is missing
4. Perform photometry and zero-point calibration
5. Export and analyze results

Main Interface
-------------

The interface is divided into several sections:

* **File Upload**: Upload science image and calibration frames
* **Calibration Options**: Apply bias, dark, and flat field corrections
* **Analysis Parameters**: Configure detection and analysis settings
* **Results Display**: View and interact with analysis results

Image Calibration
----------------

1. Upload your science image and calibration frames in FITS format
2. Select which calibration steps to apply (bias, dark, flat)
3. Click "Run Image Calibration"

Plate Solving
------------

If your FITS file is missing WCS coordinates, the application offers:

1. Manual entry of RA/DEC coordinates
2. Automatic plate solving with astrometry.net (requires API key)

Source Detection and Photometry
------------------------------

The application performs:

1. Background estimation and subtraction
2. Source detection with configurable threshold
3. Aperture photometry on all detected sources
4. PSF photometry with automatic PSF modeling

Zero-Point Calibration
---------------------

1. Cross-matching with Gaia DR3
2. Zero-point calculation using matched stars
3. Calibrated magnitude calculation for all detected sources

Output Files
-----------

The application generates several output files in the `pfr_results` directory:

* Photometry catalog (CSV)
* Analysis metadata (TXT)
* FITS header information (TXT)
* PSF model (FITS)
* Log file (LOG)

Interactive Features
------------------

* Embedded Aladin Lite for viewing your field with catalog overlays
* Interactive tables for catalog exploration
* Direct links to online astronomy services

Troubleshooting
--------------

Common issues:

* **WCS determination fails**: Check if your image has sufficient stars or try using astrometry.net
* **No stars detected**: Adjust the detection threshold or seeing estimate
* **Zero-point calibration fails**: Check magnitude range for Gaia stars or adjust the detection mask