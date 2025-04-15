Usage Guide
==========

Getting Started
--------------

Photometry Factory for RAPAS provides a web-based interface that guides you through the process of analyzing astronomical images.

1. Launch the application using:

   .. code-block:: bash

      streamlit run pfr_app.py

2. Open your web browser to the displayed URL (typically http://localhost:8501)

Basic Workflow
-------------

The typical workflow in PFR consists of:

1. **File Upload**: Upload FITS files (science image, bias, dark, flat)
2. **Image Calibration**: Apply bias, dark, and flat field corrections
3. **Photometry**: Detect sources and measure their brightness
4. **Calibration**: Determine the photometric zero point using GAIA references
5. **Results**: Download or explore the resulting catalog

Interface Overview
----------------

.. image:: _static/pfr_interface.png
   :width: 100%
   :alt: PFR interface screenshot

The interface consists of:

* **Left sidebar**: File uploads and configuration options
* **Main area**: Image display and analysis results
* **Result displays**: Tables and plots showing photometry results

Step-by-Step Guide
-----------------

File Upload
~~~~~~~~~~

1. Use the sidebar to upload your FITS files
2. Required: Science image (.fits, .fit, or .fts format)
3. Optional: Master bias, dark, and flat field frames

Image Calibration
~~~~~~~~~~~~~~~

1. Select which calibration steps to apply (bias, dark, flat)
2. Click "Run Image Calibration"
3. Review the calibrated image results

Photometry and Analysis
~~~~~~~~~~~~~~~~~~~~~

1. Configure analysis parameters in the sidebar:
   - Seeing conditions
   - Detection threshold
   - Border mask size
   - Gaia magnitude range

2. Click "Run Zero Point Calibration" to:
   - Detect sources in the image
   - Perform aperture and PSF photometry
   - Cross-match with GAIA catalog
   - Calculate photometric zero point
   - Cross-match with other catalogs

3. Review the results:
   - View the detected sources table
   - Examine the zero point calibration plot
   - Explore the final photometry catalog
   - Interact with the Aladin sky view

Downloading Results
~~~~~~~~~~~~~~~~~

1. All results are saved in the `pfr_results` directory
2. Use the download button to retrieve a ZIP file with all results
3. Files include:
   - Photometry catalog (.csv)
   - Metadata file (.txt)
   - Log file (.log)
   - Generated plots (.png)
