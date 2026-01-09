# RAPAS Photometry Pipeline

A comprehensive astronomical photometry pipeline built with Streamlit, featuring local astrometric plate solving, multi-catalog cross-matching, and advanced photometric analysis capabilities.

## Features

### 🔭 Core Photometry
- **Multi-aperture photometry**: Automatic aperture photometry with multiple radii (1.1×, 1.3× FWHM)
- **PSF photometry**: Effective Point Spread Function (ePSF) modeling.
- **Background estimation**: Advanced 2D background modeling with SExtractor algorithm
- **Source detection**: DAOStarFinder and Sep with configurable detection thresholds
- **FWHM estimation**: Automatic seeing measurement with Gaussian profile fitting

### 🌌 Astrometric Solutions
- **Local plate solving**: Integration with Astrometry.net via `stdpipe` for blind astrometric solving
- **WCS refinement**: Advanced astrometric refinement using standard pipe tools
- **Header validation**: Automatic WCS header fixing and validation
- **Coordinate systems**: Support for multiple coordinate reference frames

### 📊 Photometric Calibration
- **Catalogs integration**: Automatic cross-matching with standard star catalogs
- **Zero-point calculation**: Robust photometric calibration with outlier rejection
- **Multiple filter bands**: Support for GAIA, synthetic, PanStarrs et SkyMapper photometry bands

### 🛰️ Multi-Catalog Cross-Matching
- **GAIA DR3**: Stellar parameters, proper motions, and photometry
- **SIMBAD**: Object identifications and classifications
- **Astro-Colibri**: Transient and variable source alerts
- **SkyBoT**: Solar system object identification
- **AAVSO VSX**: Variable star catalog
- **Milliquas**: Quasar and AGN catalog
- **10 Parsec Catalog**: Nearby stars (within 10 pc)

### 🔧 Advanced Processing
- **Cosmic ray removal**: Automatic L.A.Cosmic algorithm implementation via astroscrappy
- **Image enhancement**: Multiple visualization modes (ZScale, histogram equalization)
- **Quality filtering**: Automated source quality assessment with S/N-based flags
- **Error propagation**: Comprehensive photometric error calculation with zero-point uncertainty
- **Transient Detection**: Identification of transient candidates using survey templates

### 🖥️ User Interface
- **Interactive web interface**: Streamlit-based GUI with real-time processing
- **Parameter configuration**: Adjustable detection and photometry parameters
- **Progress tracking**: Real-time status updates and logging
- **Results visualization**: Interactive plots and Aladin Lite v3 sky viewer integration

## Installation

### Prerequisites
- Python 3.12+
- Astrometry.net local installation (`solve-field` binary must be in PATH)
- Index files for Astrometry.net appropriate for your field of view

### Required Python Packages
The required packages are listed in the `pyproject.toml` file. You can install them using pip:
```bash
pip install -e .
```
This will install the project in editable mode and all the dependencies.

### System Dependencies
- **Astrometry.net**: Required for local plate solving (accessed via `stdpipe`)
  ```bash
  # Ubuntu/Debian
  sudo apt-get install astrometry.net
  
  # macOS with Homebrew
  brew install astrometry-net
  ```


## Quick Start

### 1. Start the Backend

You must start the backend server before launching the frontend. Two modes are supported:

- **FastAPI (recommended, modern):**
  ```bash
  # Windows PowerShell
  .venv\Scripts\Activate.ps1
  python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
  # Or use run_all_cmd.bat for one-click launch
  run_all_cmd.bat
  ```
- **Legacy Flask backend:**
  ```bash
  python backend.py
  ```

### 2. Start the Frontend

In a new terminal (with the virtual environment activated):
```bash
streamlit run frontend.py
# OR
streamlit run pages/app.py
```
Visit the URL printed by Streamlit (usually http://localhost:8501).

### 3. Login or Register

Create an account or login with existing credentials. You can also run in anonymous mode if the backend is not configured.

### 4. Configure Observatory

Set your observatory location and parameters in the sidebar (name, latitude, longitude, elevation).

### 5. Upload FITS File

Upload your astronomical image to the main area. Supported extensions: `.fits`, `.fit`, `.fts`, `.fits.gz`, `.fts.gz`.

### 6. Run Analysis

Adjust analysis parameters as needed (seeing, detection threshold, border mask, filter band, etc.) and run the photometry pipeline. Results and logs will be available for download as a ZIP archive.

---


## Configuration

### Observatory Parameters
- **Name**: Observatory identifier
- **Latitude/Longitude**: Geographic coordinates (decimal degrees)
- **Elevation**: Height above sea level (meters)

### Analysis Parameters
- **Estimated Seeing (FWHM)**: Initial estimate in arcseconds
- **Detection Threshold**: Source detection sigma threshold
- **Border Mask**: Pixel border exclusion size
- **Filter Band**: GAIA or synthetic magnitude band for calibration
- **Max Calibration Mag**: Faintest magnitude for calibration stars
- **Astrometry Check**: Toggle to force a plate solve/WCS refinement

### Output Files
- **Photometry Catalog**: CSV and VOTable with multi-aperture and PSF photometry
- **Background Model**: 2D background and RMS maps (FITS)
- **PSF Model**: Empirical PSF (FITS)
- **WCS Header**: Astrometric solution (TXT)
- **WCS-Solved FITS**: Original image with updated WCS header (when astrometry is performed)
- **Plots**: FWHM, magnitude distributions, zero-point calibration
- **Log File**: Detailed processing log

---

### Transient Candidates (Beta)
- **Enable Transient Finder**: Activates the transient search pipeline
- **Reference Survey**: Survey to use for template comparison (e.g., PanSTARRS)

## Astrometric Pipeline

### Local Plate Solving (stdpipe + Astrometry.net)
The pipeline uses `stdpipe` as a Python wrapper around a local Astrometry.net installation for robust astrometric solutions:

1. **Source Detection**: Automated star detection using photutils
2. **Blind Matching**: `stdpipe.astrometry.blind_match_objects` for initial WCS
3. **Parameter Optimization**: Automatic scale and position hint extraction from header if available
4. **Solution Validation**: Coordinate range and transformation validation

## Output Files

The pipeline generates comprehensive output files, available as a ZIP download:

- **Photometry Catalog** (CSV and VOTable): Complete source catalog with multi-aperture and PSF photometry
- **Log File**: Detailed processing log with timestamps
- **Background Model** (FITS): 2D background and RMS maps
- **PSF Model** (FITS): Empirical PSF (or Gaussian fallback) for the field
- **Plots**: FWHM analysis, magnitude distributions, zero-point calibration
- **WCS Header** (TXT): Updated astrometric solution
- **WCS-Solved FITS**: Original image with refined WCS embedded in header (saved to `rpp_data/fits/`)

## Technical Details

### Algorithms
- **Background**: `photutils.Background2D` with `SExtractorBackground`
- **Detection**: `photutils.DAOStarFinder` with configurable parameters
- **PSF Modeling**: `photutils.EPSFBuilder` with Gaussian fallback
- **Photometry**: Circular apertures and PSF fitting
- **Astrometry**: `stdpipe` / Astrometry.net
- **Cosmic Rays**: `astroscrappy` (L.A.Cosmic)

### Performance
- **Caching**: Streamlit caching for improved performance
- **Memory Management**: Efficient handling of large FITS files

## Troubleshooting

### Browser Compatibility Issues
**Firefox Aladin Lite Issues**:
Firefox may have compatibility issues with Aladin Lite v3 due to WebAssembly loading.
- **Refresh the page**: Sometimes resolves the issue.
- **Clear Firefox cache**: Clear cookies and cached web content.
- **Alternative browsers**: Use Chrome, Edge, or Safari for best Aladin Lite compatibility.

### Common Issues
**Plate Solving Fails**:
- Ensure `solve-field` is in your system PATH.
- Check that astrometry.net index files are installed for your image scale.
- Verify sufficient stars in the field (>5-10 recommended).

**No Catalog Matches**:
- Check internet connectivity for catalog queries (GAIA, SIMBAD, etc.).
- Verify coordinate system and field center are correct.

## Recent changes / Changelog (last update: 2026-01-07)

### Current Version (1.5.3)
- **Photometry Calculation Improvements**: 
  - Fixed critical PSF S/N calculation (was using `sqrt(flux_err)` instead of `flux_err`)
  - Removed S/N rounding to preserve precision and avoid divide-by-zero errors
  - Added proper error propagation for calibrated magnitudes: `σ_mag_calib = √(σ_mag_inst² + σ_zp²)`
  - S/N now uses background-corrected flux for more accurate estimation
  - Added quality flags for photometry: 'good' (S/N≥5), 'marginal' (3≤S/N<5), 'poor' (S/N<3)
- **Documentation**: Comprehensive mathematical documentation of S/N and magnitude error formulas
- **Testing**: New unit test suite for photometric calculations (`tests/test_photometry.py`)

### Version 1.5.1
- **Python 3.12 Compatibility**: Project upgraded to require Python 3.12 with verified dependency resolution
- **Enhanced Dependency Management**: Updated to latest stable versions:
  - astropy 7.1.1
  - astropy-iers-data 0.2025.11.3.0.38.37
  - aiohttp 3.11.18
  - certifi 2025.1.31
- **Improved astrometry support**: Via `stdpipe` integration for robust local plate solving
- **Transient Detection (Beta)**: Identification of transient candidates using template comparison and catalog filtering
- **Multi-catalog cross-matching**: GAIA DR3, SIMBAD, SkyBoT, AAVSO VSX, Milliquas, 10 Parsec Catalog, and Astro-Colibri integration

### Version History

#### Version 1.4.7
- Python 3.12 Compatibility: Project upgraded to require Python 3.12 with verified dependency resolution
- Enhanced Dependency Management: Updated to latest stable versions (astropy 7.1.1, aiohttp 3.11.18, certifi 2025.1.31)

#### Version 1.4.5
- Integrated `stdpipe` for robust local astrometric solving via Astrometry.net
- Added **Transient Finder** (Beta) utilizing `stdpipe` image subtraction and catalog filtering
- Added **10 Parsec Catalog** to multi-catalog cross-matching pipeline
- Improved PSF Photometry with empirical ePSF and Gaussian fallback mechanisms
- Automatic Cosmic Ray rejection enabled by default using `astroscrappy` (L.A.Cosmic algorithm)
- UI/UX enhancements: Astrometry check toggle, Aladin Lite v3 integration for interactive sky viewing
- Refactored header handling and WCS creation for improved stability and accuracy

## Contributing
- Please open issues or pull requests for bugs, feature requests or improvements.
- Tests and clear reproducible examples are appreciated for complex changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) for details.
