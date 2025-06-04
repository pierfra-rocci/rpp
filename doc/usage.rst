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
2. **File Upload**: Upload FITS science images (supports .fits, .fit, .fts, .fits.gz formats).
3. **Configuration**: Set observatory details, analysis parameters, processing options, and API keys in the sidebar.
4. **Image Processing**: Run the comprehensive photometry pipeline including:
   - Automatic FITS header validation and fixing
   - Optional cosmic ray removal using L.A.Cosmic algorithm
   - 2D background estimation with SExtractor algorithm
   - Source detection using DAOStarFinder
   - FWHM estimation through Gaussian fitting
   - Multi-aperture photometry (1.5×, 2.0×, 2.5×, 3.0× FWHM radii)
   - Empirical PSF model construction and PSF photometry
   - Optional astrometric refinement using stdpipe and Gaia DR3
   - Photometric calibration against Gaia DR3 with zero-point calculation
   - Atmospheric extinction correction based on airmass
5. **Catalog Enhancement**: Automatic cross-matching with multiple astronomical databases
6. **Visualization**: Interactive results display with Aladin Lite integration
7. **Results Export**: Download comprehensive results as ZIP archives

Observatory Configuration
------------------------

Observatory settings are used for accurate airmass calculations and can be configured in several ways:

**Manual Entry**:
   - Observatory name, latitude, longitude, and elevation
   - Supports both comma and dot decimal separators for international users
   - Coordinate validation ensures proper ranges (±90° for latitude, ±180° for longitude)

**Automatic Extraction**:
   - Observatory data is automatically extracted from FITS headers when available
   - Searches for standard keywords: TELESCOP, SITELAT, SITELONG, SITEELEV
   - Updates session configuration with header values

**Persistent Storage**:
   - Observatory settings are saved per user account
   - Configuration persists across sessions
   - Can be updated and saved at any time

Analysis Parameters
-----------------

**Source Detection**:
   - **Seeing (FWHM)**: Initial atmospheric seeing estimate in arcseconds (1.0-6.0)
   - **Detection Threshold**: Source detection sigma threshold (0.5-4.5)
   - **Border Mask**: Edge exclusion zone in pixels (0-200)

**Photometric Calibration**:
   - **Filter Band**: Gaia photometric band for calibration (G, BP, RP, synthetic bands)
   - **Maximum Magnitude**: Faint limit for calibration stars (15.0-21.0)

**Processing Options**:
   - **Refine Astrometry**: Enable WCS improvement using stdpipe and Gaia DR3
   - **Remove Cosmic Rays**: Enable L.A.Cosmic cosmic ray removal with configurable parameters:
     - Gain (e-/ADU): Camera gain setting (0.1-10.0)
     - Read Noise (e-): Camera read noise (1.0-20.0)
     - Sigma Clip: Detection threshold (4.0-10.0)

Image Processing Pipeline
------------------------

**1. Image Loading and Validation**:
   - Robust FITS file parsing with multi-extension support
   - Automatic handling of data cubes and RGB images
   - FITS header validation and automatic fixing of common issues
   - Pixel scale extraction and coordinate system validation

**2. Pre-processing** (optional):
   - Cosmic ray detection and removal using astroscrappy
   - Configurable parameters for different detector types
   - Visual feedback on cosmic ray detection results

**3. Background Estimation**:
   - 2D background modeling using photutils.Background2D
   - SExtractor background algorithm with sigma clipping
   - Automatic box size adjustment for small images
   - Background visualization and FITS output

**4. Source Detection**:
   - DAOStarFinder implementation with optimized parameters
   - Border masking to exclude edge effects
   - Quality filtering based on source shape parameters
   - FWHM estimation using marginal Gaussian fitting

**5. Astrometric Solution**:
   - Automatic plate solving using SIRIL for missing WCS
   - Optional WCS refinement using stdpipe and Gaia DR3
   - Coordinate system validation and transformation testing
   - Support for various coordinate reference frames

**6. Photometry**:
   - Multi-aperture photometry with four different radii
   - Empirical PSF model construction using bright, isolated stars
   - PSF photometry using IterativePSFPhotometry
   - Error estimation including background and Poisson noise
   - Signal-to-noise ratio calculation for all measurements

**7. Calibration**:
   - Cross-matching with Gaia DR3 catalog
   - Quality filtering (variability, color, astrometric quality)
   - Robust zero-point calculation with sigma clipping
   - Atmospheric extinction correction using calculated airmass

Catalog Cross-matching
---------------------

RPP automatically cross-matches detected sources with multiple astronomical catalogs:

**Gaia DR3**:
   - Primary photometric calibration source
   - Quality filtering for reliable calibration stars
   - Astrometric and photometric data extraction

**SIMBAD**:
   - Object identification and classification
   - Cross-match within field of view
   - Object type and identifier retrieval

**SkyBoT** (Solar System Bodies):
   - Moving object detection and identification
   - Ephemeris-based position calculation
   - Solar system object classification

**AAVSO VSX** (Variable Star Index):
   - Variable star identification
   - Variability type and period information
   - Cross-match with known variable stars

**Astro-Colibri**:
   - Transient event detection and classification
   - Requires user API key registration
   - Recent astronomical event correlation

**VizieR Milliquas**:
   - Quasar and QSO identification
   - Large-scale structure object detection
   - Redshift and classification information

Results and Outputs
------------------

**Interactive Visualization**:
   - Aladin Lite integration for interactive sky viewing
   - Source overlay with detailed popup information
   - Multiple sky survey options (DSS, 2MASS, etc.)
   - ESA Sky integration for external viewing

**Data Products**:
   - **Photometry Catalog**: Complete source measurements in CSV format
   - **Processing Logs**: Detailed execution history with timestamps
   - **Visualizations**: Magnitude distributions, error plots, image displays
   - **FITS Products**: Background models, PSF models, WCS-solved headers
   - **Archive Files**: Complete ZIP packages for easy distribution

**Catalog Structure**:
   - Multi-aperture photometry results (1.5×, 2.0×, 2.5×, 3.0× FWHM)
   - PSF photometry measurements and errors
   - Astrometric coordinates and uncertainties
   - Cross-match results from all queried catalogs
   - Quality flags and measurement metadata
   - Zero-point and atmospheric correction information

Configuration Management
-----------------------

**User-Specific Settings**:
   - Observatory configuration per user account
   - Analysis parameter preferences
   - API key storage (encrypted)
   - Processing history and archived results

**Session Persistence**:
   - Configuration automatically saved and loaded
   - Settings persist across browser sessions
   - Manual save/load functionality available

**Multi-User Support**:
   - Isolated user workspaces
   - Secure authentication and authorization
   - Individual result directories and configurations

File Management
--------------

**Input Support**:
   - Standard FITS formats (.fits, .fit, .fts)
   - Compressed FITS (.fits.gz)
   - Multi-extension FITS files
   - Data cubes (extracts first 2D plane)

**Output Organization**:
   - User-specific result directories
   - Timestamped file naming
   - Automatic cleanup of old temporary files
   - ZIP archive creation for complete result sets

**Archive Management**:
   - Automatic 30-day retention for result files
   - Archived ZIP file browser in sidebar
   - Selective download of previous results
   - Configuration files preserved indefinitely

Error Handling and Troubleshooting
---------------------------------

**Robust Processing**:
   - Graceful degradation when external services are unavailable
   - Automatic retry mechanisms for network requests
   - Comprehensive error logging and user feedback
   - Fallback options for critical processing steps

**Common Issues**:
   - Missing WCS: Automatic plate solving with SIRIL
   - Network timeouts: Retry mechanisms and user notification
   - Invalid coordinates: Manual entry options and validation
   - Catalog service unavailability: Continued processing with available services

**Diagnostic Information**:
   - Detailed processing logs with timestamps and log levels
   - Error messages with context and suggested solutions
   - Processing parameter documentation and help text
   - Version information and system requirements
