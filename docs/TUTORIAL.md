Step-by-Step Photometry Tutorial
--------------------------------

This tutorial will guide you through a typical analysis session from uploading an image to the final results.

### 0. Upload a FITS file

In the main area of the page, you will find the file uploader.

-   Click "Browse files" or drag and drop your FITS image into the upload area.
-   Supported file extensions are: `.fits`, `.fit`, `.fts`, `.fits.gz` and `.fts.gz`.

Once uploaded, the application will display a preview of your image, allowing you to visually confirm you have selected the correct file.
Then it starts processing the image to extract initial metadata (e.g., WCS, observation time).

### 1. Configure Observatory

After logging in, you will see the main application interface. In the sidebar on the left, you need to configure your observatory's details. This information is crucial for accurate time-dependent calculations, such as airmass.

-   **Observatory Name**: A descriptive name for your observing location (e.g., "Backyard Observatory").
-   **Latitude, Longitude, Elevation**: The precise geographical coordinates (in decimal degrees for latitude/longitude and meters for elevation) of your observatory.

The application will attempt to automatically read these coordinates from the FITS file header if they are present (`SITELAT`, `SITELONG`, `SITEELEV`). However, it is good practice to verify and set them manually in the sidebar to ensure accuracy.

### 2. Set Parameters

Below the image preview and in the sidebar, you will find parameters to control the analysis. Fine-tuning these parameters is key to obtaining high-quality photometric results.

-   **Estimated Seeing (FWHM)**: The Full-Width at Half-Maximum of the stellar profiles in your image, measured in pixels or arcseconds (check the unit in the UI). This is a crucial parameter for accurate source extraction and PSF photometry.
-   **Detection Threshold**: The sigma threshold for detecting sources above the background noise. A value between 2.0 and 4.0 is typical.
-   **Border Mask**: The width of a border around the image to exclude from source detection. This helps avoid detecting partial sources at the edges.
-   **Calibration Filter Band**: Select the photometric band (e.g., G, Bp, Rp) that matches your observation. This selection determines which data is used for photometric calibration.
-   **Max Calibration Mag**: The magnitude limit for stars from the reference catalog to be used for calibration.
-   **Astrometry check**: If enabled, the pipeline will attempt to plate-solving (using Astrometry.net via `stdpipe`) and refine the WCS. Use this if your image has no WCS or an inaccurate one (toggles as default based on initial WCS presence).

*Note: Cosmic ray removal is now performed automatically during the processing steps.*

### 3. Astro_Colibri (Optional)

If you have an Astro_Colibri API key, you can enter it in the designated field in the sidebar. This allows the application to access additional services provided by Astro-Colibri,to enhance catalog queries for transient search.

### 4. Transient Candidates (Optional)

In the sidebar, you will find an expander for "Transient Candidates".
-   **Enable Transient Finder**: Check this to run the transient detection module.
-   **Reference Filter**: Choose the right filter to compare your image against. Automatically set PanSTARRS1 g, r, i, z, y filters for optical images in the north hemisphere, and SkyMapper g, r, i for southern hemisphere images.

### 5. Run the Analysis

Once you have set the parameters, the analysis runs automatically or upon clicking the processing button (depending on the specific app flow updates). The pipeline will:

1.  Estimate background and noise.
2.  Detect sources and cosmic rays.
3.  Perform Aperture and PSF photometry.
4.  Cross-match with GAIA for calibration.
5.  Cross-match with other catalogs (SIMBAD, SkyBoT, AAVSO VSX, 10 Parsec, etc.).

### Understanding the Outputs

After the analysis is complete, the results will be available for download as a ZIP archive. Key files include:

-   `*_catalog.csv` / `.vot`: The complete source catalog with instrumental and calibrated magnitudes.
-   `*_psf.fits`: The PSF model used (EPSF or Gaussian).
-   `*_wcs_header.txt`: The solved astrometric header.
-   `*.log`: Detailed processing log.
-   `.png`: Various diagnostic plots (e.g., source detection, photometric calibration).

You can also inspect the results interactively using the embedded Aladin Lite viewer if coordinates are available, or choose directly to go to ESA SkyView.

### Support

If you encounter any issues, please check the log file first. For bugs or feedback, contact `rpp_support@saf-astronomie.fr`.