Troubleshooting
=============

This section addresses common issues and provides solutions for RAPAS Photometry Pipeline.

Installation & Setup Issues
-------------------------

**Error: ImportError: No module named '...'** (e.g., `astropy`, `photutils`, `flask`, `astroscrappy`)

*   **Problem**: Required Python packages are not installed or not found in the current environment.
*   **Solution**:
    1.  Ensure your virtual environment is activated (`source .venv/bin/activate` or `.venv\Scripts\activate`).
    2.  Run `pip install -r requirements.txt` from the application's root directory.

**Error: Could not connect to backend at http://localhost:5000** (on Login page)

*   **Problem**: The Flask backend server (`backend.py`) is not running or is inaccessible.
*   **Solution**:
    1.  Open a separate terminal in the application's root directory.
    2.  Activate the virtual environment.
    3.  Run `python backend.py`.
    4.  Check the terminal output for errors. Ensure no other process is using port 5000.

**Error: DLL load failed while importing...** (Windows)

*   **Problem**: Missing C++ runtime libraries required by some Python packages (e.g., `numpy`, `scipy`).
*   **Solution**: Install the latest Microsoft Visual C++ Redistributable packages for Visual Studio from the official Microsoft website.

**Streamlit Version Issues**

*   **Problem**: Errors related to Streamlit functions or behavior might indicate an incompatible version.
*   **Solution**: Check the `requirements.txt` file for the recommended Streamlit version and install it specifically: `pip install streamlit==<version>`.

Login & Registration Issues
-------------------------

**Error: "Username or email is already taken."** (Registration)

*   **Problem**: The chosen username or email address already exists in the database.
*   **Solution**: Choose a different username and/or email address.

**Error: "Invalid username or password."** (Login)

*   **Problem**: Incorrect username or password entered.
*   **Solution**: Verify your credentials. Use the password recovery option if you forgot your password.

**Password Recovery Fails / No Email Received**

*   **Problem**: Email sending is not configured correctly on the backend.
*   **Solution**:
    1.  Ensure the backend server (`backend.py`) is running.
    2.  Verify that the SMTP environment variables (`SMTP_SERVER`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`) are correctly set in the environment where the backend is running.
    3.  Check the backend terminal output for email sending errors.
    4.  Check your email spam folder.
    5.  Ensure your email provider allows sending via SMTP (e.g., Gmail might require an "App Password").

Image Loading & Processing Issues
-------------------------------

**Error: Could not read FITS file / No image data found**

*   **Problem**: The file is corrupted, not a valid FITS file, or uses an unsupported format/compression.
*   **Solution**:
    *   Verify the file opens correctly in other FITS viewers (e.g., SAOImage DS9, Astap).
    *   Ensure the file has a standard extension (.fits, .fit, .fts, .fits.gz).
    *   If it's a multi-extension FITS, ensure image data exists in at least one HDU.

**Error: Missing required WCS keywords / WCS creation error**

*   **Problem**: The FITS header lacks the necessary keywords (like CTYPE, CRVAL, CRPIX) for `astropy.wcs` to build a coordinate system.
*   **Solution**:
    *   Enable plate solving using Astrometry.net (via stdpipe). Ensure Astrometry.net is correctly installed and accessible.
    *   Alternatively, use external software to add WCS information to the FITS header before uploading.

**Astrometry+ / WCS Refinement Fails**

*   **Problem**: The `stdpipe` refinement process could not find enough matches between detected sources and the GAIA catalog, or the initial WCS was too inaccurate.
*   **Solution**:
    *   Ensure the initial WCS (from header or Astrometry.net) is roughly correct (within a few arcminutes).
    *   Check internet connectivity (required for GAIA query).
    *   Try adjusting the "Seeing" parameter, as it affects detection.
    *   If the field is very sparse or very crowded, refinement might struggle. Consider disabling Astrometry+.

**Cosmic Ray Removal Issues**

*   **Problem**: Too many or too few cosmic rays detected; artifacts introduced.
*   **Solution**:
    *   Adjust the CRR parameters in the sidebar expander ("Gain", "Read Noise", "Detection Threshold"). Lowering the threshold detects more CRs, raising it detects fewer.
    *   Ensure Gain and Read Noise values roughly match your detector's characteristics.
    *   Visually inspect the cleaned image; disable CRR if it causes issues.

**Error Estimating Background / Background estimation error**

*   **Problem**: `photutils.Background2D` failed, possibly due to image size, extreme saturation, or unusual image structure.
*   **Solution**:
    *   Ensure the image is not completely saturated or empty.
    *   The tool automatically adjusts box size for small images, but very small images (< ~40 pixels wide) might still fail.

**No Sources Found / Few Sources Found**

*   **Problem**: Detection parameters might be unsuitable for the image.
*   **Solution**:
    *   Lower the "Detection Threshold (σ)" in the sidebar to detect fainter sources.
    *   Ensure the "Seeing (arcsec)" estimate is reasonable; an incorrect FWHM estimate for detection can hinder results.
    *   Check if the "Border Mask" is too large.
    *   Verify the image isn't blank or excessively noisy.

**Zero Point Calculation Fails / No Gaia Matches**

*   **Problem**: Could not match detected sources to the GAIA catalog or calculate a reliable zero point.
*   **Solution**:
    *   Verify WCS accuracy. If WCS is significantly off, matching will fail. Try enabling Astrometry+ or using Astrometry.net solving.
    *   Check internet connection for GAIA query.
    *   Adjust GAIA "Filter Max Magnitude" – if set too bright (low number), might exclude usable calibration stars.
    *   Ensure the selected "Filter Band" is appropriate for the image filter (if known).
    *   The field might genuinely lack suitable GAIA stars in the specified magnitude range.

Catalog Query Issues (SIMBAD, SkyBoT, etc.)
-----------------------------------------

**Warning: "Query failed: Network error / Timeout"**

*   **Problem**: Temporary network issue or the external catalog service is down or slow.
*   **Solution**: Wait and try processing the image again later. Check your internet connection.

**Warning: "No sources found..." for a specific catalog**

*   **Problem**: The catalog genuinely contains no known objects matching the criteria within your image field.
*   **Solution**: This is expected for many fields. The message is informational.

**Astro-Colibri: No matches or Error**

*   **Problem**: Invalid or missing API key, network issue, or no recent events in the field.
*   **Solution**:
    *   Ensure you have entered the correct Astro-Colibri UID key in the sidebar.
    *   Check internet connection.
    *   Check the Astro-Colibri website for service status.
