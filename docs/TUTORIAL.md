
# Step-by-Step Photometry Tutorial

This tutorial guides you through a typical analysis session, from launching the backend to downloading your results.

## 0. Launch the Application

**Backend:**

- *FastAPI backend (recommended):*
    - Activate your virtual environment:
        ```powershell
        .venv\Scripts\Activate.ps1
        ```
    - Start the backend:
        ```powershell
        python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
        ```
    - Or use the provided batch file for Windows:
        ```powershell
        run_all_cmd.bat
        ```
- *Legacy backend (Flask):*
    ```powershell
    python backend.py
    ```

**Frontend:**

In a new terminal (with the virtual environment activated):
```powershell
streamlit run frontend.py
# OR
streamlit run pages/app.py
```
Visit the URL printed by Streamlit (usually http://localhost:8501).

## 1. Login or Register

Create an account or log in. You can also run in anonymous mode if the backend is not configured.

## 2. Upload a FITS File

In the main area, use the file uploader to select your FITS image. Supported extensions: `.fits`, `.fit`, `.fts`, `.fits.gz`, `.fts.gz`.
The app will preview your image and extract initial metadata (WCS, observation time).

## 3. Configure Observatory

In the sidebar, set your observatory details:
- **Name**: e.g., "Backyard Observatory"
- **Latitude, Longitude, Elevation**: Decimal degrees/meters. These may be auto-filled from the FITS header, but always verify.

## 4. Set Analysis Parameters

In the sidebar, adjust:
- **Estimated Seeing (FWHM)**: Initial guess in arcseconds
- **Detection Threshold**: Sigma threshold for source detection
- **Border Mask**: Pixels to exclude at the image edge
- **Calibration Filter Band**: Choose the photometric band for calibration
- **Max Calibration Mag**: Faintest magnitude for calibration stars
- **Astrometry Check**: Enable to force plate solving/WCS refinement

*Cosmic ray removal is always performed automatically using the L.A.Cosmic algorithm (astroscrappy).* 

## 5. (Optional) Astro-Colibri API Key

Enter your Astro-Colibri API key in the sidebar to enable real-time transient alerts and variable source cross-matching.

## 6. (Optional) Transient Candidates (Beta)

Expand the "Transient Candidates" section in the sidebar:
- **Enable Transient Finder**: Activates transient detection using image subtraction against reference surveys (PanSTARRS1 for north, SkyMapper for south).
- **Reference Filter**: Select the filter band for template comparison.

## 7. Run the Analysis

Click the button to start the photometry pipeline. The following steps are performed:
1. **Background & Noise Estimation**
2. **Source Detection & Cosmic Ray Removal**
3. **Photometry**: Multi-aperture and PSF photometry, S/N and error calculation, quality flag assignment
4. **Astrometric Refinement** (if enabled)
5. **Photometric Calibration**: Cross-match with catalogs for zero-point
6. **Multi-Catalog Cross-Matching**: GAIA DR3, SIMBAD, SkyBoT, AAVSO VSX, Milliquas, 10 Parsec, Astro-Colibri
7. **Transient Detection** (if enabled)

## 8. Download and Interpret Results

After processing, download the ZIP archive containing:
- `*_catalog.csv` / `.vot`: Source catalog with photometry, errors, flags, and cross-matches
- `*_background.fits`: 2D background and RMS maps
- `*_psf.fits`: Empirical PSF model
- `*_wcs_header.txt`: Astrometric solution header
- `*_wcs.fits`: Original image with refined WCS header (when astrometry is performed)
- `*.log`: Processing log
- `*.png`: Diagnostic plots

**Note on WCS-Solved FITS Files**: When astrometry is performed, the pipeline saves your original image with the updated WCS solution in two locations:
1. **In the ZIP archive** (`*_wcs.fits`) - included with your download
2. **In `rpp_data/fits/`** - a permanent copy that gets overwritten if you reprocess the same file

**Analysis Tracking**: All analysis results are automatically tracked in the database:
- Each WCS-solved FITS file is recorded with your username
- Each ZIP archive is linked to its source FITS file(s)
- You can have multiple analysis runs from the same FITS file (different parameters)
- Query your analysis history programmatically using `src/db_tracking.py` functions

### Photometric Quality Flags

| Quality Flag | S/N Range | Reliability | Recommended Use |
|-------------|-----------|-------------|-----------------|
| `good`      | S/N ≥ 5   | High        | Science-ready   |
| `marginal`  | 3 ≤ S/N < 5 | Moderate   | Use with caution|
| `poor`      | S/N < 3   | Low         | Exclude         |

Magnitude errors are propagated as:
- **Instrumental error**: σ_mag = 1.0857 × (σ_flux / flux)
- **Calibrated error**: σ_mag_calib = √(σ_mag_inst² + σ_zp²)

**Interactive Analysis**: Use the embedded Aladin Lite viewer for real-time exploration, or export coordinates to ESA SkyView.

## Support

If you encounter issues, check the log file. For bugs or feedback, contact `rpp_support@saf-astronomie.fr`.
