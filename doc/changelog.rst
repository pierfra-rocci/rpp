Changelog
=========

This document records all notable changes to Photometry Factory for RAPAS.

Version 1.1.0 (Upcoming)
----------------------

**New Features**
- Cross-matching with multiple astronomical catalogs (SIMBAD, SkyBot, AAVSO VSX)
- Interactive Aladin Lite sky viewer integration
- Automatic PSF model construction and PSF photometry
- Enhanced zero-point calibration with sigma clipping

**Improvements**
- Improved WCS refinement using Gaia DR3
- More robust source detection in crowded fields
- Better error handling for network services
- Performance optimizations for large images

**Bug Fixes**
- Fixed memory leak when processing multiple large images
- Resolved issue with coordinate transformations near celestial poles
- Corrected zero point calculation for images with small number of reference stars
- Fixed display errors in magnitude histograms

Version 1.0.0 (2023-09-01)
----------------------

**Features**
- FITS image calibration (bias, dark, flat)
- Automatic plate solving with astrometry.net
- Source detection and aperture photometry
- Zero-point calibration with Gaia DR3
- CSV catalog export
- Image visualization with matplotlib
- User-friendly Streamlit interface

**Requirements**
- Python 3.8+
- Required packages: astropy, photutils, astroquery, streamlit, numpy, pandas, matplotlib

Version 0.9.0 (2023-07-15)
----------------------

**Initial Beta Release**
- Core functionality implemented
- Basic calibration and photometry workflow
- Preliminary documentation

Version 0.2.0 (2025-05-02)
--------------------------

**Major Features & Changes**

*   **User Authentication**: Implemented user login, registration, and password recovery using a Flask backend (`backend_dev.py`) and SQLite database (`users.db`). Email recovery requires SMTP configuration via environment variables.
*   **User Configuration**: User-specific settings (observatory, analysis parameters, API keys) are now saved and loaded via the backend.
*   **Astrometry**:
    *   Added Siril integration (`plate_solve.ps1`/`.sh`) for robust plate solving when WCS is missing.
    *   Added optional WCS refinement ("Astrometry+") using the `stdpipe` library and GAIA DR3.
*   **Image Processing**:
    *   Added optional Cosmic Ray Removal using `astroscrappy` (L.A.Cosmic algorithm) with configurable parameters.
    *   Improved background estimation using `photutils.Background2D` and SExtractor algorithm, including visualization and FITS output.
    *   Improved FWHM estimation using Gaussian fitting on marginal sums, with histogram visualization.
*   **Photometry**:
    *   Implemented PSF Photometry using `photutils.psf.EPSFBuilder` and `IterativePSFPhotometry`. PSF model is visualized and saved as FITS.
    *   Added Signal-to-Noise Ratio (SNR) calculation to aperture photometry results.
*   **Catalog Integration**:
    *   Added cross-matching with Astro-Colibri API (requires UID key) for transient identification.
    *   Added cross-matching with Milliquas catalog (VizieR VII/294) for quasar identification.
    *   Refined GAIA DR3 cross-matching for calibration (filtering by variability, color, RUWE).
*   **UI & Workflow**:
    *   Restructured sidebar for better organization of settings (Observatory, Process Options, Analysis Params, etc.).
    *   Added "Save Configuration" button.
    *   Added "Download All Results (ZIP)" button for convenient output retrieval.
    *   Improved logging with timestamps and levels, saved to `.log` file.
    *   Added Aladin Lite viewer for interactive source exploration.
    *   Application entry point changed to `run_frontend.py`.
*   **Documentation**: Updated all documentation files (`.rst`) to reflect current features and usage. Added new sections and refined existing ones.

**Minor Changes & Fixes**

*   Refactored code into `pages/app.py`, `pages/login.py`, `tools.py`, and `backend_dev.py`.
*   Improved error handling in various processing steps.
*   Standardized figure sizes using `tools.FIGURE_SIZES`.
*   Added `__version__.py` for version tracking.
*   Updated `requirements.txt`.
*   Added cleanup for temporary files.

Version 0.1.0 (2024-XX-XX) - *Placeholder for previous state*
-------------------------------------------------------------
**Initial Release Candidate**
*   Core photometry pipeline based on `photutils`.
*   Basic FITS loading and display.
*   Aperture photometry.
*   Zero-point calibration with Gaia DR3.
*   CSV catalog export.
*   Image visualization with matplotlib.
*   User-friendly Streamlit interface (single script `pfr_app.py`).

**Requirements**
*   Python 3.8+
*   Required packages: astropy, photutils, astroquery, streamlit, numpy, pandas, matplotlib

Version 0.9.0 (2023-07-15) - *Placeholder*
------------------------------------------
**Initial Beta Release**
*   Core functionality implemented.
*   Basic calibration and photometry workflow.
*   Preliminary documentation.
