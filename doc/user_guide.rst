User Guide
=========

Getting Started
--------------

RAPAS Photometry Pipeline (RPP) provides a streamlined workflow for astronomical image analysis through a user-friendly web interface. This guide walks you through the complete process from login and image upload to final photometric catalog generation.

**Prerequisites:** Ensure the backend server (`backend.py`) is running before launching the frontend.

1. Launch the backend: `python backend.py`
2. Launch the frontend: `streamlit run frontend.py`
3. Access the URL (e.g., http://localhost:8501) and log in or register.

Application Layout
-----------------

.. image:: _static/app_layout.png
   :width: 700px
   :alt: Application Layout

The interface is divided into two main sections:

* **Left Sidebar**: Contains user authentication, file uploader, processing options (Astrometry refinement, Cosmic Ray Removal), observatory settings, analysis parameters (Seeing, Threshold, Mask), photometry parameters (Gaia band/limit), Astro-Colibri API key input, configuration saving, archived results browser, and logout controls.
* **Main Panel**: Displays the uploaded image with dual visualization (ZScale and Histogram Equalization), processing status, results (plots, tables), interactive visualizations (Aladin Lite), ESA Sky links, and download options.

Step 1: Login & Upload
----------------------
1. Authenticate using the login page or register a new account.
2. Once logged in, use the file uploader:
   * **Science Image** (required): Your astronomical image in FITS format (.fits, .fit, .fts, .fits.gz).

Step 2: Configure Processing
--------------------------
Set up your analysis using the sidebar options:

1.  **Observatory Data**: Input or verify the observatory's name, latitude, longitude, and elevation. The application accepts both comma and dot decimal separators for coordinates. Observatory information can be automatically extracted from FITS headers if available.

2.  **Analysis Parameters**:
    *   **Estimated Seeing (FWHM, arcsec)**: Initial atmospheric seeing estimate (1.0-6.0 arcsec).
    *   **Detection Threshold (sigma)**: Sigma level above background for source detection (0.5-4.5).
    *   **Border Mask Size (pixels)**: Pixels to ignore around the image edge (0-200).
    *   **Calibration Filter Band**: Gaia photometric band for calibration (G, BP, RP, or synthetic bands).
    *   **Max Calibration Mag**: Faintest magnitude limit for calibration stars (15.0-21.0).
    *   **Refine Astrometry**: Enable WCS refinement using `stdpipe` and Gaia DR3.
    *   **Remove Cosmic Rays**: Enable cosmic ray removal using `astroscrappy` with configurable parameters.

3.  **API Keys**: Enter your Astro-Colibri UID key for transient event cross-matching (optional).

Step 3: Image Processing
-----------------------
1. Upload your FITS file using the file uploader.
2. The application will automatically:
   * Load and display the image with dual visualization
   * Extract observatory information from FITS header if available
   * Display image statistics and coordinate information
   * Calculate airmass based on observation time and location

3. If WCS information is missing, the application will attempt plate solving using SIRIL.
4. You can optionally force re-solving even if WCS exists.

Step 4: Run Photometric Calibration
----------------------------------
1. Click the "Photometric Calibration" button to start the pipeline.
2. The processing includes:
   * Optional cosmic ray removal (if enabled)
   * Background estimation and modeling
   * Source detection using DAOStarFinder
   * FWHM estimation and refinement
   * Multi-aperture photometry (1.5×, 2.0×, 2.5×, 3.0× FWHM)
   * PSF model construction and PSF photometry
   * Optional WCS refinement using stdpipe and Gaia DR3
   * Cross-matching with Gaia DR3 for photometric calibration
   * Zero-point calculation with outlier rejection
   * Atmospheric extinction correction

Step 5: Catalog Enhancement
--------------------------
After photometric calibration, the pipeline automatically cross-matches detected sources with:

* **Gaia DR3**: For photometric calibration and stellar properties
* **SIMBAD**: For object identification and classification
* **SkyBoT**: For solar system object detection
* **AAVSO VSX**: For variable star identification
* **Astro-Colibri**: For transient event matching (requires API key)
* **VizieR Milliquas**: For quasar identification

Step 6: Results and Visualization
--------------------------------
The application provides comprehensive results including:

1. **Interactive Aladin Lite Viewer**: Explore detected sources overlaid on sky survey images
2. **Magnitude Distribution Plots**: Histograms and error analysis for both aperture and PSF photometry
3. **Photometry Catalog**: Complete source catalog with multi-aperture measurements
4. **Processing Logs**: Detailed execution logs with timestamps
5. **ESA Sky Links**: Direct links to external sky survey viewers

Step 7: Download Results
-----------------------
Use the "Download Results (ZIP)" button to download a complete archive containing:

* **Photometry catalog** (CSV format with all measurements)
* **Processing logs** (detailed execution history)
* **Plots and visualizations** (PNG format)
* **FITS headers** (original and WCS-solved headers)
* **Background models** (FITS format if generated)
* **PSF models** (FITS format if generated)

Configuration Management
-----------------------
* **Save Configuration**: Store current analysis parameters, observatory settings, and API keys
* **Archived Results**: Browse and download previous analysis results
* **Auto-cleanup**: Old files are automatically cleaned up (30-day retention)

Advanced Features
----------------
* **Header Fixing**: Automatic correction of common FITS header issues
* **Multi-extension FITS**: Support for complex FITS file formats
* **Quality Filtering**: Robust filtering for reliable photometric calibration
* **Rate Limiting**: Automatic throttling of catalog queries
* **Error Recovery**: Graceful handling of network timeouts and service unavailability

Troubleshooting Tips
-------------------
* Ensure SIRIL is installed and accessible for plate solving
* Check internet connectivity for catalog queries
* Verify FITS file format and header completeness
* Use manual coordinate entry if header coordinates are missing
* Adjust detection parameters for crowded or sparse fields
* Monitor processing logs for detailed error information

Refer to the Troubleshooting section for common issues and solutions.