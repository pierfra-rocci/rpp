Step-by-Step Photometry Tutorial
--------------------------------

This tutorial will guide you through a typical analysis session from uploading an image to interpreting the results.

### 1. Configure Observatory Data

After logging in, you will see the main application interface. In the sidebar on the left, you need to configure your observatory's details. This information is crucial for accurate time-dependent calculations, such as airmass.

-   **Observatory Name**: A descriptive name for your observing location (e.g., "My Backyard Observatory").
-   **Latitude, Longitude, Elevation**: The precise geographical coordinates (in decimal degrees for latitude/longitude and meters for elevation) of your observatory.

The application will attempt to automatically read these coordinates from the FITS file header if they are present (`SITELAT`, `SITELONG`, `SITEELEV`). However, it is good practice to verify and set them manually in the sidebar to ensure accuracy.

### 2. Upload a FITS file

In the main area of the page, you will find the file uploader.

-   Click "Browse files" or drag and drop your FITS image into the upload area.
-   Supported file extensions are: `.fits`, `.fit`, `.fts`, and `.fits.gz`.

Once uploaded, the application will display a preview of your image, allowing you to visually confirm you have selected the correct file.

### 3. Set Analysis Parameters

Below the image preview and in the sidebar, you will find parameters to control the analysis. Fine-tuning these parameters is key to obtaining high-quality photometric results.

-   **Estimated Seeing (FWHM)**: The Full-Width at Half-Maximum of the stellar profiles in your image, measured in pixels or arcseconds (check the unit in the UI). This is a crucial parameter for accurate source extraction and PSF photometry.
-   **Detection Threshold**: The signal-to-noise ratio threshold for detecting sources above the background noise. A value between 3.0 and 5.0 is typical.
-   **Border Mask**: The width of a border around the image to exclude from source detection. This helps avoid detecting partial sources at the edges.
-   **Calibration Filter Band**: Select the photometric band (e.g., G, Bp, Rp) that matches your observation. This selection determines which GAIA DR3 data is used for photometric calibration.
-   **Max Calibration Mag**: The brightest magnitude for stars from the reference catalog to be used for calibration. This helps exclude saturated stars.
-   **Astrometry check**: A toggle switch. If enabled, the pipeline will attempt to solve the plate (using Astrometry.net via `stdpipe`) and refine the WCS. Use this if your image has no WCS or an inaccurate one.

*Note: Cosmic ray removal is now performed automatically during the processing steps.*

### 4. Transient Candidates (Optional / Beta)

In the sidebar, you will find an expander for "Transient Candidates".
-   **Enable Transient Finder**: Check this to run the transient detection module.
-   **Reference Survey/Filter**: Choose the survey (e.g., PanSTARRS) to compare your image against.

### 5. Run the Analysis

Once you have set the parameters, the analysis runs automatically or upon clicking the processing button (depending on the specific app flow updates). The pipeline will:
1.  Estimate background and noise.
2.  Detect sources and cosmic rays.
3.  Perform Aperture and PSF photometry.
4.  Cross-match with GAIA for calibration.
5.  Cross-match with other catalogs (SIMBAD, SkyBoT, etc.).

### Understanding the Outputs

After the analysis is complete, the results will be available for download as a ZIP archive. Key files include:

-   `*_catalog.csv` / `.vot`: The complete source catalog with instrumental and calibrated magnitudes.
-   `*_psf.fits`: The PSF model used (EPSF or Gaussian).
-   `*_bkg.fits`: The background model.
-   `*_wcs_header.txt`: The solved astrometric header.
-   `*.log`: Detailed processing log.

You can also inspect the results interactively using the embedded Aladin Lite viewer if coordinates are available.

### Support

If you encounter any issues, please check the log file first. For bugs or feedback, contact `rpp_support@saf-astronomie.fr`.