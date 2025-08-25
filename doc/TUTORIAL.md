# RAPAS Photometry Pipeline — Quick Start Tutorial

This short guide shows the minimal steps to run photometry with the RAPAS app. For full details, consult the project's README.md.

1. Launch the app:
   - streamlit run pages/app.py
   - Open the provided localhost URL in your browser.

2. Login:
   - Use the login page. If browser autocomplete fails, type credentials manually.

3. Configure observatory:
   - Open the sidebar "🔭 Observatory Data".
   - Enter Name, Latitude, Longitude, Elevation.
   - In any case the application will try to read the fits file to automatically fill the fields.

4. Upload a FITS file:
   - Use the "Choose a FITS file for analysis" uploader on the main page.
   - Supported extensions: .fits, .fit, .fts, .fits.gz

5. Main Analysis Parameters:
   - Seeing (FWHM): initial guess in arcseconds. Typical values: 2.5–3.5". The app will refine this automatically.
   - Detection Threshold (sigma): default 3.0. Increase (4–5) to reduce spurious detections in noisy images.
   - Border Mask Size (pixels): excludes image edges from detection (default ~10). Increase for vignetting or cosmetic edges.
   - Calibration Filter Band: choose the GAIA band used for zero-point (e.g., phot_g_mean_mag).
   - Max Calibration Mag: faintest magnitude to use for calibration stars (default ~20.0). Use a brighter limit (e.g., 18) for shallow or crowded fields.
   - Refine Astrometry: toggle to attempt WCS refinement (requires astrometry.net / SCAMP); enable for better catalog cross-matching.
   - Remove Cosmic Rays: enable for long exposures; set camera gain/read-noise parameters in the sidebar when enabled.
   - Force Plate Solve: force re-solving of astrometry even if WCS exists (useful when header WCS is unreliable).
   - Pixel Scale & FWHM (derived): the app computes pixel scale from header (or solved WCS) and converts seeing to pixels for detection.

5.5 Optional preprocessing:
   - In the sidebar "⚙️ Analysis Parameters" enable options such as "Remove Cosmic Rays" or "Refine Astrometry".
   - When "Remove Cosmic Rays" is enabled, set:
     - CRR Gain (e-/ADU)
     - CRR Read Noise (e-)
     - CRR Sigma Clip (detection threshold)
   - When "Refine Astrometry" is enabled, ensure astrometry.net / SCAMP are available on the system.

6. Run photometry:
   - Click "Photometric Calibration" to start detection, cross-match and zero-point computation.
   - Wait for status messages and check the log in the output directory.

7. Inspect results:
   - View image, statistics, and the generated photometry catalog in the main page.
   - Use the "📦 Download Results (ZIP)" button or the archived results browser in the sidebar.

8. Troubleshooting & details:
   - See README.md in the project root for installation, dependencies, and troubleshooting tips.

If anything is unclear, open the README or check the log files saved to your output directory.
