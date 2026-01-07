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
-   **Astrometry check**: If enabled, the pipeline will perform blind astrometric plate-solving using Astrometry.net through `stdpipe` integration and refine the WCS solution. Use this if your image lacks WCS metadata or has an inaccurate solution. This toggle is set automatically based on the initial WCS presence and quality in your FITS header.

*Note: Cosmic ray removal is automatically performed during preprocessing using the L.A.Cosmic algorithm (astroscrappy). This is now enabled by default to improve data quality.*

### 3. Astro_Colibri (Optional)

If you have an Astro_Colibri API key, you can enter it in the designated field in the sidebar. This enables access to Astro-Colibri's real-time transient alerts and variable source information, enhancing the catalog cross-matching pipeline with up-to-date source classifications and transient annotations.

### 4. Transient Candidates (Beta Feature)

In the sidebar, you will find an expander for "Transient Candidates" which enables the advanced transient detection pipeline.
-   **Enable Transient Finder**: Check this to run the transient detection module, which uses `stdpipe` image subtraction to compare your observation against reference surveys and identify potential transient sources.
-   **Reference Survey & Filter**: Automatically selects the optimal reference survey and filter bands based on your image location and coordinates:
    - **Northern Hemisphere**: PanSTARRS1 g, r, i, z, y filters (covering full optical range)
    - **Southern Hemisphere**: SkyMapper g, r, i filters
    This selection determines which template images and catalogs are used for transient candidate identification.

### 5. Run the Analysis

Once you have configured all parameters, the analysis pipeline executes with the following steps:

1.  **Background & Noise Estimation**: Compute 2D background model and RMS noise maps using SExtractor algorithm
2.  **Source Detection & Cosmic Ray Removal**: Identify astronomical sources using DAOStarFinder; automatically remove cosmic rays with L.A.Cosmic (astroscrappy)
3.  **Photometry**: Perform multi-aperture photometry (1.1×, 1.3× FWHM) and PSF photometry using empirical ePSF modeling with Gaussian fallback. Includes:
    - Background-corrected S/N calculation
    - Proper magnitude error computation: σ_mag = 1.0857 × (σ_flux / flux)
    - Quality flag assignment based on S/N thresholds
4.  **Astrometric Refinement** (if enabled): Apply blind plate-solving via Astrometry.net/stdpipe to solve or refine WCS
5.  **Photometric Calibration**: Cross-match with Catalogs for absolute photometric zero-point determination with outlier rejection
6.  **Multi-Catalog Cross-Matching**: Query and cross-match sources with GAIA DR3 (stellar parameters), SIMBAD (object classification), SkyBoT (solar system objects), AAVSO VSX (variable stars), Milliquas (QSOs/AGN), 10 Parsec Catalog (nearby stars), and optionally Astro-Colibri (transient alerts)
7.  **Transient Detection** (if enabled): Perform image subtraction against reference survey templates and flag candidate transient sources

### Understanding the Outputs

After the analysis is complete, the results will be available for download as a ZIP archive. Key output files include:

-   **`*_catalog.csv` / `.vot`**: Complete source catalog with instrumental magnitudes (aperture & PSF) and GAIA-calibrated absolute magnitudes, including:
    - Photometric errors with proper zero-point uncertainty propagation
    - Quality flags: 'good' (S/N≥5), 'marginal' (3≤S/N<5), 'poor' (S/N<3)
    - Background-corrected flux measurements
    - Cross-match identifications from multiple catalogs
-   **`*_background.fits`**: 2D background model and RMS noise maps used for source detection
-   **`*_psf.fits`**: Empirical Point Spread Function (ePSF) model fitted to stellar sources (or Gaussian model if ePSF fails to converge)
-   **`*_wcs_header.txt`**: Astrometric solution header (WCS) with plate-solve parameters if astrometry was performed
-   **`*.log`**: Comprehensive processing log with timestamps, parameter values, and diagnostic messages for troubleshooting
-   **`*.png`**: Diagnostic plots including FWHM profile analysis, magnitude distributions, zero-point calibration residuals, source detection map, and photometric quality indicators

### Photometric Quality Assessment

The pipeline provides quality flags to help assess the reliability of photometric measurements:

| Quality Flag | S/N Range | Reliability | Recommended Use |
|-------------|-----------|-------------|-----------------|
| `good` | S/N ≥ 5 | High | Science-ready data |
| `marginal` | 3 ≤ S/N < 5 | Moderate | Use with caution |
| `poor` | S/N < 3 | Low | Exclude from analysis |

The magnitude errors include proper error propagation:
- **Instrumental error**: σ_mag = 1.0857 × (σ_flux / flux)
- **Calibrated error**: σ_mag_calib = √(σ_mag_inst² + σ_zp²)

**Interactive Analysis**: You can inspect results in real-time using the embedded Aladin Lite v3 sky viewer for interactive visualization of detections and catalog cross-matches, or export coordinates to ESA SkyView for broader context

### Support

If you encounter any issues, please check the log file first. For bugs or feedback, contact `rpp_support@saf-astronomie.fr`.
