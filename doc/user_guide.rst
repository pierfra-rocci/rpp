User Guide
=========

Getting Started
--------------

Photometry Factory for RAPAS provides a streamlined workflow for astronomical image analysis through a user-friendly web interface. This guide walks you through the complete process from login and image upload to final photometric catalog generation.

**Prerequisites:** Ensure the backend server (`backend_dev.py`) is running before launching the frontend.

1. Launch the frontend: `streamlit run run_frontend.py`
2. Access the URL (e.g., http://localhost:8501) and log in or register.

Application Layout
-----------------

.. image:: _static/app_layout.png
   :width: 700px
   :alt: Application Layout

The interface is divided into two main sections:

* **Left Sidebar**: Contains login/logout controls, file uploader, processing options (Astrometry+, Cosmic Ray Removal), observatory settings, analysis parameters (Seeing, Threshold, Mask), photometry parameters (Gaia band/limit), Astro-Colibri API key input, configuration saving, and quick links.
* **Main Panel**: Displays the uploaded image, processing status, results (plots, tables), interactive visualizations (Aladin), and logs.

Step 1: Login & Upload
----------------------
1. Authenticate using the login page.
2. Once logged in, use the sidebar file uploader:
   * **Image** (required): Your main astronomical image in FITS format (.fits, .fit, .fts, .fits.gz).

Step 2: Configure Processing
--------------------------
Set up your analysis using the sidebar options:

1.  **Observatory Location**: Verify or input the observatory's name, latitude, longitude, and elevation. This is used for airmass calculation.
2.  **Process Options**:
    *   **Astrometry +**: Check to enable WCS refinement using `stdpipe` and GAIA DR3.
    *   **Remove Cosmic Rays**: Check to enable cosmic ray removal using `astroscrappy`. Configure Gain, Read Noise, and Threshold in the expander if needed.
3.  **Analysis Parameters**:
    *   **Seeing (arcsec)**: Estimate of atmospheric seeing.
    *   **Detection Threshold (σ)**: Sigma level above background for source detection.
    *   **Border Mask (pixels)**: Pixels to ignore around the image edge.
4.  **Photometry Parameters**:
    *   **Filter Band**: Select the GAIA band for zero-point calibration.
    *   **Filter Max Magnitude**: Set the faint limit for GAIA calibration stars.
5.  **Astro-Colibri**: Enter your UID key (optional) for transient cross-matching.
6.  **Save Configuration**: Click "Save" to store these settings for your user account.

Step 3: Run Analysis
--------------------
Click the "Run Zero Point Calibration" button in the main panel (appears after image upload). The application will perform the following steps automatically:

1.  **Load Image**: Reads FITS data and header.
2.  **Cosmic Ray Removal** (if enabled).
3.  **WCS Check/Solve**: Reads WCS from header. If invalid or missing, attempts solving with Siril (if configured/available) or prompts for Astrometry.net.
4.  **Astrometry Refinement** (if Astrometry+ enabled): Refines WCS using `stdpipe`.
5.  **Background Estimation**: Calculates and subtracts 2D background.
6.  **FWHM Estimation**: Estimates the average stellar FWHM.
7.  **Source Detection**: Finds sources using DAOStarFinder.
8.  **Photometry**: Performs both aperture and PSF photometry.
9.  **GAIA Cross-match**: Matches detected sources with GAIA DR3.
10. **Zero Point Calculation**: Determines the photometric zero point.
11. **Catalog Enhancement**: Cross-matches with SIMBAD, SkyBoT, AAVSO VSX, Astro-Colibri, and Milliquas.

Step 4: Review Results
-----------------------
After processing, the main panel displays comprehensive results:

*   **Processed Image**: View the background-subtracted image.
*   **Background/RMS Plots**: Visualize the estimated background model.
*   **FWHM Histogram**: Distribution of measured FWHM values.
*   **PSF Model**: The empirically derived PSF.
*   **Zero Point Plot**: Calibration plot showing GAIA vs. instrumental magnitudes.
*   **Source Catalogs**:
    *   Interactive table of GAIA calibration stars.
    *   Interactive table summarizing external catalog matches.
    *   Full photometry catalog table.
*   **Aladin Viewer**: Interactive sky map with your catalog overlaid. Click sources for details.
*   **Log Output**: Detailed processing log.

Step 5: Export Data
--------------------
Your analysis results are saved to the `rpp_results` directory.

*   Use the **Download All Results (ZIP)** button in the sidebar to get a compressed archive containing:
    *   CSV Catalog: Complete photometry results.
    *   Header File (.txt): Original FITS header.
    *   Log File (.log): Processing log.
    *   Plots (.png): FWHM histogram, Zero Point plot, etc.
    *   Background Model (.fits).
    *   PSF Model (.fits).

Refer to Troubleshooting for common issues and solutions.