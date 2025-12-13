# RAPAS Photometry Pipeline

A comprehensive astronomical photometry pipeline built with Streamlit, featuring local astrometric plate solving, multi-catalog cross-matching, and advanced photometric analysis capabilities.

## Features

### 🔭 Core Photometry
- **Multi-aperture photometry**: Automatic aperture photometry with multiple radii (1.5×, 2.0× FWHM)
- **PSF photometry**: Effective Point Spread Function (ePSF) modeling with Gaussian fallback
- **Background estimation**: Advanced 2D background modeling with SExtractor algorithm
- **Source detection**: DAOStarFinder with configurable detection thresholds
- **FWHM estimation**: Automatic seeing measurement with Gaussian profile fitting

### 🌌 Astrometric Solutions
- **Local plate solving**: Integration with Astrometry.net via `stdpipe` for blind astrometric solving
- **WCS refinement**: Advanced astrometric refinement using standard pipe tools
- **Header validation**: Automatic WCS header fixing and validation
- **Coordinate systems**: Support for multiple coordinate reference frames

### 📊 Photometric Calibration
- **GAIA DR3 integration**: Automatic cross-matching with GAIA Data Release 3
- **Zero-point calculation**: Robust photometric calibration with outlier rejection
- **Extinction correction**: Atmospheric extinction correction using airmass
- **Multiple filter bands**: Support for GAIA G, BP, RP and synthetic photometry bands

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
- **Quality filtering**: Automated source quality assessment
- **Error propagation**: Comprehensive photometric error calculation
- **Transient Detection (Beta)**: Identification of transient candidates using survey templates

### 🖥️ User Interface
- **Interactive web interface**: Streamlit-based GUI with real-time processing
- **Parameter configuration**: Adjustable detection and photometry parameters
- **Progress tracking**: Real-time status updates and logging
- **Results visualization**: Interactive plots and Aladin Lite v3 sky viewer integration

## Installation

### Prerequisites
- Python 3.10+
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

## Environment Variables

To enable email-related features such as registration confirmation and password recovery, you must configure the following environment variable:

- `SMTP_PASS`: The password for the SMTP server account specified in `config.py`.

Example:
```bash
export SMTP_PASS="your_smtp_password"
```

## Quick Start

1. **Start the application**:
   ```bash
   streamlit run frontend.py
   # OR
   streamlit run pages/app.py
   ```

2. **Login**: Create an account or login with existing credentials (or run in anonymous mode if backend is not configured).

3. **Configure observatory**: Set your observatory location and parameters in the sidebar.

4. **Upload FITS file**: Upload your astronomical image to the main area.

## Configuration

### Observatory Parameters
- **Name**: Observatory identifier
- **Latitude/Longitude**: Geographic coordinates (decimal degrees)
- **Elevation**: Height above sea level (meters)

### Analysis Parameters
- **Estimated Seeing (FWHM)**: Initial estimate in arcseconds
- **Detection Threshold**: Source detection sigma threshold
- **Border Mask**: Pixel border exclusion size
- **Filter Band**: GAIA magnitude band for calibration
- **Max Calibration Mag**: Faintest magnitude for calibration stars
- **Astrometry Check**: Toggle to force a plate solve/WCS refinement

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

## Recent changes / Changelog (last update: 2025-11-29)

- **Version 1.4.6**
- Maintenance release with minor improvements and dependency updates.

- **Version 1.4.5**
- Integrated `stdpipe` for robust local astrometric solving.
- Added **Transient Finder** (Beta) utilizing `stdpipe` image subtraction and catalog filtering.
- Added **10 Parsec Catalog** to cross-matching.
- Improved PSF Photometry with Gaussian fallback if EPSF building fails.
- Automatic Cosmic Ray rejection enabled by default using `astroscrappy`.
- UI improvements in Streamlit app (Astrometry check toggle, Aladin Lite v3 integration).
- Refactored header handling and WCS creation for better stability.

## Contributing
- Please open issues or pull requests for bugs, feature requests or improvements.
- Tests and clear reproducible examples are appreciated for complex changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) for details.