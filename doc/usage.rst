Usage Guide
==========

Getting Started
--------------

RAPAS Photometry Pipeline (RPP) provides a web-based interface that guides you through the process of analyzing astronomical images with comprehensive photometric analysis and catalog cross-matching.

1. Start the backend server (in a separate terminal):

   .. code-block:: bash

      python backend.py

2. Launch the frontend application:

   .. code-block:: bash

      streamlit run frontend.py

3. Open your web browser to the displayed URL (typically http://localhost:8501)
4. Log in using your credentials or register a new account.

Basic Workflow
-------------

The typical workflow in RPP consists of:

1. **Authentication**: Log in with your username and password or register a new account.
2. **Observatory Configuration**: Set your observatory location, coordinates, and elevation.
3. **File Upload**: Upload FITS science images (supports .fits, .fit, .fts, .fits.gz formats).
4. **Parameter Configuration**: Adjust analysis parameters in the sidebar.
5. **Image Processing**: Run the comprehensive photometry pipeline.

Detailed Step-by-Step Guide
---------------------------

**Step 1: Authentication**

- Register a new account with username and password
- Login credentials are stored securely with password hashing
- Each user gets their own workspace and configuration

**Step 2: Observatory Setup**

Configure your observatory details in the sidebar:

- **Observatory Name**: Identifier for your observing site
- **Latitude**: Geographic latitude in decimal degrees (North positive)
- **Longitude**: Geographic longitude in decimal degrees (East positive)  
- **Elevation**: Height above sea level in meters

The application accepts both comma and dot as decimal separators for international users.

**Step 3: Analysis Parameters**

Set the following parameters in the sidebar:

- **Seeing (FWHM)**: Initial estimate in arcseconds (1.0-6.0)
- **Detection Threshold**: Source detection sigma threshold (0.5-4.5)
- **Border Mask**: Pixel border exclusion size (0-200 pixels)
- **Calibration Filter Band**: GAIA magnitude band for photometric calibration
- **Max Calibration Mag**: Faintest magnitude for calibration stars (15.0-21.0)

**Step 4: Processing Options**

Enable optional processing features:

- **Refine Astrometry**: Use SCAMP and GAIA DR3 for WCS refinement
- **Remove Cosmic Rays**: Apply L.A.Cosmic algorithm with configurable parameters
- **Force Re-solve**: Override existing WCS solution with new plate solving

**Step 5: File Upload and Processing**

1. Upload your FITS file using the file uploader
2. The application will display the image with dual visualization modes
3. Click "Photometric Calibration" to start the pipeline
4. Monitor progress through real-time status updates

**Step 6: Results and Analysis**

The pipeline generates:

- **Source Detection**: Automatic detection using DAOStarFinder
- **Background Modeling**: 2D background estimation and visualization
- **FWHM Analysis**: Automatic seeing measurement with plots
- **Multi-Aperture Photometry**: Four different aperture radii
- **PSF Photometry**: Empirical PSF model construction and fitting
- **Astrometric Solution**: Local plate solving and optional refinement
- **Catalog Cross-matching**: Multiple astronomical database queries
- **Photometric Calibration**: Zero-point calculation with GAIA DR3

**Step 7: Interactive Exploration**

- **Aladin Lite Viewer**: Interactive sky map with detected sources
- **Magnitude Distributions**: Histograms and scatter plots
- **Source Tables**: Comprehensive catalogs with cross-match results
- **ESA Sky Integration**: External sky viewer links

**Step 8: Data Export**

- **Download Results**: Complete ZIP archive with all outputs
- **Archived Results**: Browse previous analysis results
- **Individual Files**: Access specific output files as needed

Processing Pipeline Details
--------------------------

**Image Loading and Validation**:
   - Automatic FITS header validation and fixing
   - Support for multi-extension and compressed FITS files
   - Data cube handling (extracts 2D planes)
   - RGB image support (uses first channel)

**Optional Cosmic Ray Removal**:
   - L.A.Cosmic algorithm via astroscrappy
   - Configurable gain, read noise, and sigma clipping parameters
   - Before/after comparison display

**Background Estimation**:
   - photutils.Background2D with SExtractor algorithm
   - Sigma clipping and 2D interpolation
   - Background and RMS map visualization
   - FITS output for further analysis

**Source Detection**:
   - DAOStarFinder with configurable parameters
   - Border masking to avoid edge effects
   - Quality filtering based on shape parameters
   - Detection threshold optimization

**FWHM Estimation**:
   - Gaussian fitting on marginal sums
   - Multiple source averaging for robust estimates
   - FWHM vs flux scatter plots
   - Outlier rejection and validation

**Astrometric Processing**:
   - **Blind Plate Solving**: Astrometry.net via stdpipe
     * Automatic source extraction
     * Scale and position hint optimization
     * Index file selection and solving
     * WCS validation and header updates
   
   - **WCS Refinement**: SCAMP with GAIA DR3
     * High-order polynomial fitting
     * Catalog matching and residual analysis
     * Distortion correction modeling
     * Quality assessment and validation

**Multi-Aperture Photometry**:
   - Four aperture radii: 1.5×, 2.0×, 2.5×, 3.0× FWHM
   - Local background estimation with annular apertures
   - Background-corrected and raw flux measurements
   - Error propagation including Poisson and read noise
   - Signal-to-noise ratio calculation for each aperture

**PSF Photometry**:
   - Empirical PSF model construction using EPSFBuilder
   - Star selection based on shape and brightness criteria
   - Iterative PSF fitting with IterativePSFPhotometry
   - PSF model visualization and FITS output
   - Comparison with aperture photometry results

**Photometric Calibration**:
   - GAIA DR3 cross-matching with quality filters
   - Robust zero-point calculation with sigma clipping
   - Atmospheric extinction correction using airmass
   - Calibration star filtering (variability, color, astrometry)
   - Zero-point uncertainty estimation

**Catalog Cross-matching**:
   - **GAIA DR3**: Stellar parameters and calibration
   - **SIMBAD**: Object identification and classification
   - **Astro-Colibri**: Transient alerts and events
   - **SkyBoT**: Solar system object identification
   - **AAVSO VSX**: Variable star catalog
   - **Milliquas**: Quasar and AGN identification

Configuration Management
------------------------

**Saving Configuration**:
   - Click "Save Configuration" to store current settings
   - Settings saved both locally and to backend database
   - User-specific configuration persistence

**Loading Configuration**:
   - Configurations automatically loaded on login
   - Override with manual parameter adjustments
   - Reset to defaults if needed

**API Keys**:
   - Astro-Colibri UID key for transient queries
   - Stored securely with encryption
   - Optional but recommended for full functionality

File Management
--------------

**Input Files**:
   - FITS files with standard extensions
   - Automatic format detection and validation
   - Temporary file handling with cleanup
   - Multi-user file isolation

**Output Files**:
   - Comprehensive CSV catalog with all measurements
   - Processing log with timestamps and status
   - Background and PSF model FITS files
   - Visualization plots in PNG format
   - Updated WCS header in text format
   - Complete ZIP archive for distribution

**Archived Results**:
   - Automatic archiving of completed analyses
   - User-specific result directories
   - Automatic cleanup of old files (30+ days)
   - Browse and download previous results

Error Handling and Troubleshooting
----------------------------------

**Common Issues**:

- **No WCS found**: Enable force plate solving option
- **Plate solving fails**: Check Astrometry.net installation and index files
- **No catalog matches**: Verify internet connection and coordinates
- **Memory errors**: Process smaller images or increase available RAM
- **Permission errors**: Check file system permissions and disk space

**Recovery Options**:

- **Automatic fallbacks**: Graceful degradation when components fail
- **Parameter adjustment**: Modify detection and calibration thresholds
- **Manual coordinates**: Enter RA/DEC if header values are missing
- **Skip optional steps**: Disable cosmic ray removal or astrometry refinement

**Getting Help**:

- Check detailed log files for error diagnosis
- Review troubleshooting section in documentation
- Verify all dependencies are properly installed
- Contact support with specific error messages
**Diagnostic Information**:
   - Detailed processing logs with timestamps and log levels
   - Error messages with context and suggested solutions
   - Processing parameter documentation and help text
   - Version information and system requirements
