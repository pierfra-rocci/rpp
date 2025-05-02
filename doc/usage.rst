Usage Guide
==========

Getting Started
--------------

Photometry Factory for RAPAS provides a web-based interface that guides you through the process of analyzing astronomical images.

1. Start the backend server (in a separate terminal):

   .. code-block:: bash

      python backend_dev.py

2. Launch the frontend application:

   .. code-block:: bash

      streamlit run run_frontend.py

3. Open your web browser to the displayed URL (typically http://localhost:8501)
4. Log in using your credentials or register a new account.

Basic Workflow
-------------

The typical workflow in PFR consists of:

1. **Login**: Authenticate using your username and password.
2. **File Upload**: Upload FITS files (science image).
3. **Configuration**: Set observatory details, analysis parameters (seeing, threshold, mask), processing options (Astrometry+, CRR), and API keys (Astro-Colibri) in the sidebar.
4. **Image Processing**: Run the pipeline via the "Run Zero Point Calibration" button. This includes:
    - Optional Cosmic Ray Removal
    - Background Estimation
    - Source Detection
    - Optional Astrometry Refinement (Astrometry+)
    - Aperture and PSF Photometry
    - Cross-matching with GAIA DR3
    - Zero Point Calculation
    - Cross-matching with other catalogs (SIMBAD, SkyBoT, AAVSO, Astro-Colibri, Milliquas)
5. **Results**: Review results (plots, tables, Aladin viewer) and download the output files (catalog, logs, plots, headers) as a ZIP archive.

Interface Overview
----------------

.. image:: _static/pfr_interface.png
   :width: 100%
   :alt: PFR interface screenshot

The interface consists of:

* **Login Page**: Initial authentication screen.
* **Main Application Page**:
    * **Left sidebar**: File uploader, processing options (Astrometry+, CRR), observatory settings, analysis parameters, Astro-Colibri key, configuration saving, and quick links.
    * **Main area**: Image display, statistics, processing results (plots, tables), and Aladin viewer.

Step-by-Step Guide
-----------------

Login
~~~~~
1. Navigate to the application URL.
2. Enter your username and password.
3. Click "Login". If you don't have an account, use the "Register" button (requires email).
4. Password recovery is available via email if configured.

File Upload
~~~~~~~~~~

1. Use the sidebar to upload your science image (.fits, .fit, or .fts format).

Configuration
~~~~~~~~~~~~~

1. **Observatory**: Enter or verify observatory name, latitude, longitude, and elevation.
2. **Process Options**: Enable/disable Astrometry+ refinement and Cosmic Ray Removal. Configure CRR parameters if needed.
3. **Analysis Parameters**: Adjust Seeing estimate, Detection Threshold, and Border Mask size.
4. **Photometry Parameters**: Select the GAIA filter band and magnitude limit for calibration.
5. **Astro-Colibri**: Enter your UID key (optional).
6. **Save Configuration**: Click "Save" to store your current settings for future sessions.

Photometry and Analysis
~~~~~~~~~~~~~~~~~~~~~

1. Click "Run Zero Point Calibration" to start the full pipeline. The application will:
   - Load the image and header.
   - Optionally remove cosmic rays.
   - Estimate background.
   - Detect sources using DAOStarFinder.
   - Estimate FWHM.
   - Optionally refine WCS using Astrometry+ (stdpipe).
   - Perform aperture photometry.
   - Build PSF model and perform PSF photometry.
   - Cross-match with GAIA DR3 catalog.
   - Calculate photometric zero point.
   - Enhance catalog with cross-matches (SIMBAD, SkyBoT, AAVSO, Astro-Colibri, Milliquas).

Reviewing Results
~~~~~~~~~~~~~~~~~

1. **Image Display**: View the processed image with background/RMS plots.
2. **FWHM Histogram**: Check the distribution of measured FWHM values.
3. **PSF Model**: View the constructed empirical PSF model.
4. **Zero Point Plot**: Examine the calibration plot (Gaia vs. Instrumental Mags).
5. **Catalogs**:
    - View the table of GAIA cross-matched sources used for calibration.
    - View the summary table of objects matched with external catalogs.
    - Explore the final photometry catalog table.
6. **Aladin Viewer**: Interact with the sky view showing your detected sources overlaid on DSS imagery. Click sources for details.
7. **Log**: Review the processing log messages.

Downloading Results
~~~~~~~~~~~~~~~~~

1. All results are saved in the `rpp_results` directory, organized by the base filename of the input image.
2. Use the "Download All Results (ZIP)" button in the sidebar to retrieve a ZIP file containing:
   - Photometry catalog (.csv)
   - Header file (.txt)
   - Log file (.log)
   - Generated plots (.png)
   - Background model (.fits)
   - PSF model (.fits)
