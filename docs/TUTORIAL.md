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

Below the image preview, you will find several parameters to control the analysis. Fine-tuning these parameters is key to obtaining high-quality photometric results.

-   **Seeing (FWHM)**: The Full-Width at Half-Maximum of the stellar profiles in your image, measured in pixels. This is a crucial parameter for accurate source extraction and PSF photometry. You can estimate this value using an image inspection tool or by examining a few unsaturated stars in the preview.
-   **Detection Threshold**: The signal-to-noise ratio threshold for detecting sources above the background noise. A value between 3.0 and 5.0 is typical. Lowering this value may detect fainter sources but also risks including more spurious noise detections.
-   **Border Mask (pixels)**: The width of a border around the image to exclude from source detection. This helps avoid detecting partial or distorted sources at the very edges of the frame.
-   **Calibration Band**: Select the photometric band (e.g., V, R, I) that matches the filter used for your image. This selection determines which reference catalog data (from GAIA) is used for photometric calibration.
-   **Max Calibration Magnitude**: The brightest magnitude for stars from the reference catalog to be used for calibration. This helps exclude saturated or non-linear stars from the calibration process, leading to a more reliable zero-point calculation.

### 4. Advanced Options (Optional)

-   **Remove Cosmic Rays**: If your image is affected by cosmic rays, enable this option. For optimal results, you may need to provide the **Camera Gain** (in e-/ADU) and **Read Noise** (in e-) for your detector. This step uses an algorithm (like L.A.Cosmic) to identify and remove cosmic ray hits before source detection.
-   **Refine Astrometry**: If your image's World Coordinate System (WCS) is imprecise or missing, this option can calculate it by matching star patterns to a catalog. This process, known as plate-solving, requires `Astrometry.net` and `SCAMP` to be installed and configured on the server. A successful astrometric solution is essential for cross-matching with photometric catalogs.

### 5. Run the Analysis

Once you have set all the parameters, click the "Photometric Calibration" button to start the analysis. The process may take a few minutes depending on the image size, the number of sources, and the options selected. You can monitor the progress through the application's status indicators.

Understanding the Outputs
-------------------------

After the analysis is complete, the results will be available for download. The results are saved in a user-specific folder on the server (`<username>_results`) to keep your data private.

A `*.zip` file containing all output files will be created for easy download. Key files inside the archive include:

-   `*_catalog.csv`: A CSV file containing the list of all detected sources. It includes instrumental magnitudes, calibrated magnitudes, errors, positions (pixel and celestial coordinates), and other photometric parameters for each source.
-   `*_psf.fits`: A FITS image of the effective Point Spread Function (PSF) model that was derived from bright, non-saturated stars in your image.
-   `*_bkg.fits`: A FITS image of the calculated background model that was subtracted from your original image before source detection.
-   `*.log`: A detailed log file of the entire analysis process, including all parameters used and steps taken. This file is invaluable for troubleshooting any issues or for documenting your analysis.

You can download the ZIP archive to your computer to inspect the results in detail using your preferred FITS viewers and data analysis software.

Support
-------

If you encounter any issues or have questions, please check the `*.log` file first for detailed error messages or warnings. If the problem persists, contact rpp_support@saf-astronomie.fr or open an issue on the project's repository, providing the log file and a description of the problem.


