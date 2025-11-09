# RAPAS Photometry Pipeline

A comprehensive astronomical photometry pipeline built with Streamlit, featuring local astrometric plate solving, multi-catalog cross-matching, and advanced photometric analysis capabilities.

## Features

### 🔭 Core Photometry
- **Multi-aperture photometry**: Automatic aperture photometry with multiple radii (1.5×, 2.0×, 2.5× FWHM)
- **PSF photometry**: Effective Point Spread Function (ePSF) modeling and fitting
- **Background estimation**: Advanced 2D background modeling with SExtractor algorithm
- **Source detection**: DAOStarFinder with configurable detection thresholds
- **FWHM estimation**: Automatic seeing measurement with Gaussian profile fitting

### 🌌 Astrometric Solutions
- **Local plate solving**: Integration with Astrometry.net via stdpipe for blind astrometric solving
- **WCS refinement**: Advanced astrometric refinement using SCAMP and GAIA DR3 catalog
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

### 🔧 Advanced Processing
- **Cosmic ray removal**: L.A.Cosmic algorithm implementation via astroscrappy
- **Image enhancement**: Multiple visualization modes (ZScale, histogram equalization)
- **Quality filtering**: Automated source quality assessment
- **Error propagation**: Comprehensive photometric error calculation

### 🖥️ User Interface
- **Interactive web interface**: Streamlit-based GUI with real-time processing
- **Parameter configuration**: Adjustable detection and photometry parameters
- **Progress tracking**: Real-time status updates and logging
- **Results visualization**: Interactive plots and Aladin sky viewer integration

## Installation

### Prerequisites
- Python 3.10+
- Astrometry.net installation with solve-field binary
- SCAMP installation (for astrometric refinement)

### Required Python Packages
The required packages are listed in the `pyproject.toml` file. You can install them using pip:
```bash
pip install -e .
```
This will install the project in editable mode and all the dependencies.

### System Dependencies
- **Astrometry.net**: For local plate solving
  ```bash
  # Ubuntu/Debian
  sudo apt-get install astrometry.net
  
  # macOS with Homebrew
  brew install astrometry-net
  ```

- **SCAMP**: For astrometric refinement
  ```bash
  # Ubuntu/Debian
  sudo apt-get install scamp
  
  # macOS with Homebrew
  brew install scamp
  ```

## Quick Start

1. **Start the application**:
   ```bash
   streamlit run frontend.py
   ```

2. **Login**: Create an account or login with existing credentials

3. **Configure observatory**: Set your observatory location and parameters

4. **Upload FITS file**: Upload your astronomical image

5. **Run photometry**: Click "Photometric Calibration" to start the pipeline

## Configuration

### Observatory Parameters
- **Name**: Observatory identifier
- **Latitude/Longitude**: Geographic coordinates (decimal degrees)
- **Elevation**: Height above sea level (meters)

### Analysis Parameters
- **Seeing (FWHM)**: Initial estimate in arcseconds
- **Detection Threshold**: Source detection sigma threshold
- **Border Mask**: Pixel border exclusion size
- **Filter Band**: GAIA magnitude band for calibration
- **Max Calibration Mag**: Faintest magnitude for calibration stars

### Advanced Options
- **Refine Astrometry**: Enable SCAMP-based WCS refinement
- **Remove Cosmic Rays**: Apply L.A.Cosmic algorithm
- **Force Re-solve**: Override existing WCS solution

## Astrometric Pipeline

### Local Plate Solving (stdpipe + Astrometry.net)
The pipeline uses stdpipe as a Python wrapper around Astrometry.net for robust astrometric solutions:

1. **Source Detection**: Automated star detection using photutils
2. **Blind Matching**: stdpipe.astrometry.blind_match_objects for initial WCS
3. **Parameter Optimization**: Automatic scale and position hint extraction
4. **Solution Validation**: Coordinate range and transformation validation

### WCS Refinement (SCAMP + GAIA DR3)
For images with existing WCS, the pipeline can refine the solution:

1. **Source Extraction**: SEP-based source detection via stdpipe
2. **Catalog Matching**: Cross-match with GAIA DR3 reference catalog
3. **SCAMP Fitting**: High-order polynomial distortion correction
4. **Quality Assessment**: Astrometric residual analysis

## Output Files

The pipeline generates comprehensive output files:

- **Photometry Catalog** (CSV and VOTable): Complete source catalog with multi-aperture photometry
- **Log File**: Detailed processing log with timestamps
- **Background Model** (FITS): 2D background and RMS maps
- **PSF Model** (FITS): Empirical PSF for the field
- **Plots**: FWHM analysis, magnitude distributions, zero-point calibration
- **WCS Header** (TXT): Updated astrometric solution

## API Integration

### Astro-Colibri
Real-time transient alerts and multi-messenger astronomy events:
- Requires API key from https://www.astro-colibri.science
- Configurable time windows around observation date
- Automatic coordinate-based event matching

### GAIA Archive
Direct integration with ESA GAIA DR3:
- Cone search around field center
- Quality filtering (variability, astrometry, photometry)
- Synthetic photometry for non-standard bands

## Technical Details

### Algorithms
- **Background**: photutils.Background2D with SExtractorBackground
- **Detection**: photutils.DAOStarFinder with configurable parameters
- **PSF Modeling**: photutils.EPSFBuilder with iterative improvement
- **Photometry**: Circular apertures with local background subtraction
- **Astrometry**: Astrometry.net via stdpipe with SCAMP refinement

### Data Processing
- **Image Formats**: FITS with support for multi-extension and compressed files
- **Coordinate Systems**: Full WCS support with SIP distortion corrections
- **Error Propagation**: Poisson noise plus read noise modeling
- **Quality Control**: Automated outlier rejection and validation

### Performance
- **Caching**: Streamlit caching for improved performance
- **Memory Management**: Efficient handling of large FITS files
- **Processing Time**: Typical 2-5 minutes for 2K×2K images

## Troubleshooting

### Browser Compatibility Issues

**Firefox Autocomplete Problems**:
Firefox may not properly handle autocomplete with Streamlit's login form. Try these solutions:

1. **Clear Browser Data**:
   - Go to `Settings > Privacy & Security > Cookies and Site Data`
   - Click "Clear Data" and select both cookies and cached web content
   - Alternatively, press `Ctrl+Shift+Delete` and clear data for localhost

2. **Disable Enhanced Tracking Protection** (for localhost):
   - Click the shield icon in the address bar when on the login page
   - Toggle off "Enhanced Tracking Protection" for this site

3. **Check Autocomplete Settings**:
   - Go to `Settings > Privacy & Security > Forms and Autofill`
   - Ensure "Autofill logins and passwords" is enabled
   - Check that localhost is not in the exceptions list

4. **Alternative Solutions**:
   - Use the "🔄 Refresh Fields" button if autocomplete doesn't populate fields
   - Try manually typing credentials instead of using autocomplete
   - Consider using Chrome or Edge as alternative browsers

**Streamlit Connection Issues**:
- Default port `localhost:8501` may conflict with other services
- Try running on different port: `streamlit run frontend.py --server.port 8502`
- Check Windows Firewall settings for localhost connections

### Common Issues

**Plate Solving Fails**:
- Ensure solve-field is in PATH
- Check astrometry.net index files are installed
- Verify sufficient stars in field (>10 recommended)

**WCS Refinement Fails**:
- Ensure SCAMP is installed and accessible
- Check GAIA catalog connectivity
- Verify initial WCS is reasonable

**No Catalog Matches**:
- Check internet connectivity for catalog queries
- Verify coordinate system and field center
- Ensure reasonable search radius

### Performance Optimization
- Use SSD storage for temporary files
- Ensure adequate RAM (8GB+ recommended)
- Close other applications during processing

## Recent changes / Changelog (last update: 2025-11-09)

- **Version 1.3.3**
- Updated `README.md` to reflect the current state of the project.
- Updated `pyproject.toml` with the dependencies from `requirements.txt`.
- The `src` and `pages` directories have been updated with the latest changes.
- The `astrometry.py` script now uses `stdpipe` for local astrometric plate solving.
- The `pipeline.py` script has been updated with the latest changes in the photometry workflow.
- The `psf.py` script has been updated with the latest changes in the PSF photometry workflow.
- The `tools.py` script has been updated with the latest changes in the utility functions.
- The `utils_common.py` script has been updated with the latest changes in the utility functions.
- The `xmatch_catalogs.py` script has been updated with the latest changes in the cross-matching workflow.
- The `app.py` script has been updated with the latest changes in the Streamlit application.
- The `login.py` script has been updated with the latest changes in the login page.

## Quick run (minimal)
1. Ensure system dependencies:
   - Python 3.10+
   - Astrometry.net (solve-field) and required index files
   - SCAMP (optional, for high-order WCS refinement)
2. Install Python packages:
   - pip install -e .
3. Start the Streamlit app:
   - streamlit run pages/app.py
   - Default Streamlit port: 8501 (change with --server.port)
4. Backend (optional):
   - Local backend assumed at http://localhost:5000 for login/config endpoints. If not used, the app continues in anonymous mode and saves config locally to output directory.

## Files & outputs
- <username>_rpp_results/
  - *_catalog.csv  (CSV catalog)
  - *_catalog.xml  (VOTable, when conversion succeeds)
  - *_bkg.fits     (background model and RMS)
  - *_psf.fits     (fitted PSF model)
  - *_image.png    (saved visualization)
  - *.log          (processing log)
  - *.zip          (archived results created by the app)

## Notes & Known issues
- Plate solving:
  - Requires local astrometry.net solve-field binary in PATH and appropriate index files for your scale.
  - If plate solving fails, the app will fall back to instrumental photometry without WCS (coordinates limited).
- WCS and header variability:
  - Many FITS headers from telescopes/cameras use non-standard keywords. fix_header and safe_wcs_create attempt automatic fixes, but manual RA/DEC entry may still be necessary.
- Catalog queries:
  - GAIA / Vizier / SIMBAD queries require internet access and may be rate-limited.
  - Synthetic photometry queries are limited in size; the code restricts to a safe number of source_ids when querying GAIA synthetic photometry tables.
- Transient finder:
  - May require additional dependencies and internet access to fetch survey reference images.
- VOTable creation:
  - Some DataFrame columns with mixed types or objects may be removed prior to VOTable conversion; the app warns when it strips problematic columns.

## Contributing
- Please open issues or pull requests for bugs, feature requests or improvements.
- Tests and clear reproducible examples are appreciated for complex changes (WCS, PSF, astrometry).

## Support / Contact
- For quick support, open an issue on the project repository.
- For the Astro-Colibri API, see: https://www.astro-colibri.science
- App version is declared in src/__version__.py (displayed in the app sidebar).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) for details.
The MIT License allows you to use, modify, and distribute this software, provided you include the original copyright notice and disclaimer.

## Acknowledgements

### Core Dependencies

- [Astropy](https://www.astropy.org/) - Core astronomical functionality
- [Photutils](https://photutils.readthedocs.io/) - Photometry algorithms
- [Astroquery](https://astroquery.readthedocs.io/) - Astronomical database access
- [Streamlit](https://streamlit.io/) - Web interface framework
- [stdpipe](https://github.com/karpov-sv/stdpipe) - Astrometric calibration
- [astroscrappy](https://github.com/astropy/astroscrappy) - Cosmic ray removal

### External Services

- [RAPAS](https://rapas.imcce.fr/) - Project support and feedback
- [Astro-Colibri](https://astro-colibri.science/) - Transient event data API
- [Gaia DR3](https://www.cosmos.esa.int/web/gaia/dr3) - Stellar catalog
- [SIMBAD](http://simbad.u-strasbg.fr/simbad/) - Astronomical database
- [VizieR](https://vizier.u-strasbg.fr/) - Catalog service
- [SkyBoT](https://ssp.imcce.fr/webservices/skybot/) - Solar system objects
- [AAVSO VSX](https://www.aavso.org/vsx/) - Variable star catalog

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the pipeline.

## Support

For questions, issues, or feature requests, please contact the development team or create an issue in the project repository.

**Firefox Aladin Lite Issues**:
Firefox may have compatibility issues with Aladin Lite v3 due to WebAssembly loading problems:

1. **Refresh the page**: Sometimes a simple refresh resolves the issue
2. **Clear Firefox cache**: 
   - Go to `Settings > Privacy & Security > Cookies and Site Data > Clear Data`
   - Select both cookies and cached web content
3. **Check WebAssembly support**:
   - Type `about:config` in Firefox address bar
   - Search for `javascript.options.wasm` and ensure it's set to `true`
4. **Disable Enhanced Tracking Protection** for localhost:
   - Click the shield icon in address bar → Toggle off protection
5. **Alternative browsers**: Use Chrome, Edge, or Safari for best Aladin Lite compatibility
6. **Check browser console**: Press F12 and look for specific error messages