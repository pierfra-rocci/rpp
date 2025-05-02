Features
========

User Authentication & Configuration
---------------------------------
*   **Secure Login**: User registration and login system via a Flask backend.
*   **Password Management**: Password hashing and secure recovery via email (requires SMTP configuration).
*   **User Configuration**: Saves and loads user-specific settings (observatory, analysis parameters, API keys) via the backend database.

Image Processing
--------------

**FITS File Support**
   PFR supports standard FITS file formats (.fits, .fit, .fts, .fits.gz). It handles multi-extension files, data cubes (extracting the first plane), and properly extracts header information.

**Cosmic Ray Removal**
   Optional detection and removal of cosmic rays using the `astroscrappy` package (L.A.Cosmic algorithm). Configurable parameters (gain, readnoise, thresholds).

**Background Estimation**
   Sophisticated 2D background modeling using `photutils.Background2D` with sigma-clipping and the SExtractor background algorithm. Visualizations and FITS output of the background model and RMS map.

Source Detection
--------------

**DAOStarFinder Implementation**
   Uses the DAOPhot algorithm (`photutils.detection.DAOStarFinder`) to detect point sources based on sharpness and roundness.

**FWHM Estimation**
   Automatically estimates the Full Width at Half Maximum (FWHM) of stars by fitting 1D Gaussian models to marginal sums of detected sources. Includes histogram visualization.

**PSF Modeling**
   Builds an empirical Point Spread Function (PSF) model from bright, isolated stars using `photutils.psf.EPSFBuilder`. Saves the model as a FITS file.

Photometry
---------

**Aperture Photometry**
   Measures source brightness using circular apertures (`photutils.aperture.CircularAperture`) scaled to the estimated FWHM. Calculates instrumental magnitudes and Signal-to-Noise Ratio (SNR).

**PSF Photometry**
   Performs PSF photometry using the empirically built PSF model via `photutils.psf.IterativePSFPhotometry`. Calculates fitted fluxes and instrumental magnitudes.

**Error Estimation**
   Propagates background RMS noise into aperture photometry errors. PSF photometry provides fitted flux errors.

Astrometry
---------

**WCS Handling**
   Extracts and validates World Coordinate System (WCS) information from FITS headers using `astropy.wcs`. Handles multi-dimensional WCS by reducing to celestial coordinates.

**Plate Solving**
   *   **Siril Integration**: Option to use an external Siril script (`plate_solve.ps1`/`.sh`) for robust plate solving if WCS is missing or invalid.
   *   **Astrometry.net**: (Mentioned in docs, but Siril/Stdpipe seem primary now) Potential integration for web-based solving.

**Astrometry Refinement (Astrometry+)**
   Optional WCS refinement using `stdpipe` library, cross-matching detected sources with GAIA DR3 to improve WCS accuracy.

**Coordinate Transformations**
   Converts between pixel coordinates and celestial coordinates (RA/Dec) using the validated WCS.

Calibration
----------

**Zero Point Calculation**
   Determines photometric zero point by cross-matching detected sources with the GAIA DR3 catalog using sigma-clipping for robustness. Includes visualization plot.

**Airmass Calculation**
   Calculates airmass based on FITS header time/coordinates and configured observatory location using `astropy.coordinates`.

**Magnitude Systems**
   Calculates instrumental magnitudes and converts them to a calibrated magnitude system tied to GAIA DR3 (currently G-band, BP, RP selectable), applying a basic airmass correction (0.1 * airmass).

Catalog Integration
-----------------

**GAIA DR3**
   Accesses the GAIA Data Release 3 catalog via `astroquery.gaia` for astrometric refinement and photometric calibration. Filters calibration stars based on magnitude, variability flags, color index, and RUWE.

**Astro-Colibri**
   Cross-matches field with the Astro-Colibri database via API (requires UID key) to identify known transients or events.

**SIMBAD**
   Cross-matches with SIMBAD database via `astroquery.simbad` for object identification and classification.

**SkyBoT**
   Queries the SkyBoT service (IMCCE) via web request to identify known solar system objects in the field of view based on observation time and location.

**AAVSO VSX**
   Queries the AAVSO Variable Star Index (via VizieR `B/vsx/vsx`) to check for known variable stars.

**Milliquas (VizieR VII/294)**
   Queries the Million Quasars Catalog (via VizieR `VII/294`) for quasar identification.

Visualization
-----------

**Image Display**
   Displays the FITS image using `matplotlib` with options for scaling (ZScale, Percentile).

**Interactive Plots**
   *   Background and RMS maps.
   *   FWHM distribution histogram.
   *   PSF model visualization.
   *   Zero point calibration plot (Gaia vs. Calibrated Mags) with binned statistics.

**Aladin Integration**
   Embeds an interactive Aladin Lite sky viewer showing the field of view (DSS) with detected sources overlaid. Popups provide source details (coordinates, magnitude, catalog matches).

**Data Tables**
   Uses `streamlit` and `pandas` DataFrames for interactive display of:
   *   GAIA calibration star matches.
   *   Summary of external catalog matches.
   *   Full final photometry catalog with sorting and filtering.

Output and Results
----------------

**Catalog Generation**
   Creates a final CSV catalog containing source positions (pixel and sky), aperture photometry results (flux, error, instrumental mag, calibrated mag, SNR), PSF photometry results (if successful), and cross-match information from all queried catalogs.

**Metadata Logging**
   *   **Log File**: Records processing steps, parameters, warnings, and errors to a `.log` file.
   *   **Header File**: Saves the original FITS header to a `.txt` file.

**Result Downloads**
   Provides a single "Download All Results (ZIP)" button to download a compressed archive containing the CSV catalog, log file, header file, background FITS, PSF FITS, and all generated PNG plots for the processed image. Files are organized in the `rpp_results` directory.
