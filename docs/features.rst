Features
========


User Authentication & Multi-User Support
----------------------------------------

**Secure User Management**:
   - User registration and login system via FastAPI (recommended) or legacy Flask backend
   - Password hashing with secure storage in SQLite or SQL database
   - Password recovery via email (requires SMTP configuration)
   - Individual user workspaces with isolated data storage
   - Session management with automatic logout options

**Configuration Persistence**:
   - User-specific settings stored in backend database
   - Observatory parameters, analysis settings, and API keys
   - Automatic loading of saved configurations on login
   - Local and remote configuration synchronization

Advanced Image Processing
------------------------

**FITS File Support**:
   - Standard FITS formats (.fits, .fit, .fts, .fits.gz)
   - Multi-extension FITS (MEF) file handling
   - Data cubes with automatic 2D plane extraction
   - RGB astronomical images (uses first color channel)
   - Automatic header validation and fixing

**Header Processing**:
   - Automatic WCS keyword validation and correction
   - Removal of problematic or conflicting keywords
   - Observatory information extraction from headers
   - Coordinate system standardization and validation
   - Missing keyword interpolation and defaults

**Cosmic Ray Removal**:
   - L.A.Cosmic algorithm implementation via `astroscrappy` package
   - Automatic cosmic ray detection and removal enabled by default
   - Configurable parameters (gain, read noise, sigma clipping thresholds)
   - Before/after comparison for quality assessment
   - Seamless integration into photometry pipeline preprocessing

**Background Estimation**:
   - photutils.Background2D with SExtractor algorithm
   - 2D background modeling with sigma clipping
   - Automatic box size adjustment for image dimensions
   - Background and RMS map visualization
   - FITS output for background and noise models

Astrometric Solutions
--------------------

**Local Plate Solving**:
   - Astrometry.net integration via stdpipe wrapper
   - Blind astrometric solving for images without WCS
   - Automatic source detection using photutils
   - Scale estimation from focal length and pixel size
   - Solution validation and coordinate range checking

**WCS Refinement**:
   - SCAMP integration for high-precision astrometry
   - GAIA DR3 catalog matching for reference stars
   - High-order polynomial distortion correction
   - Residual analysis and quality assessment
   - Optional refinement for existing WCS solutions

**Coordinate Systems**:
   - Support for multiple coordinate reference frames
   - SIP distortion correction handling
   - Coordinate transformation validation
   - Edge case handling for problematic coordinates
   - Force re-solve option for existing solutions

Source Detection & Photometry
-----------------------------

**Source Detection**:
   - DAOStarFinder implementation with configurable parameters
   - Border masking to avoid edge artifacts
   - Quality filtering based on shape parameters (roundness, sharpness)
   - Automatic threshold optimization
   - Detection validation and source counting

**FWHM Estimation**:
   - Gaussian profile fitting on stellar marginal sums
   - Multiple source averaging for robust estimates
   - FWHM vs relative flux analysis
   - Outlier rejection with sigma clipping
   - Seeing estimation from stellar profiles

**Multi-Aperture Photometry**:
   - Two primary aperture radii: 1.1× and 1.3× FWHM (optimized for most use cases)
   - Circular apertures with local background estimation using annular regions
   - Per-aperture background-corrected and raw flux measurements
   - Signal-to-noise ratio (S/N) calculation using background-corrected flux
   - Magnitude error calculation: ``σ_mag = 1.0857 × (σ_flux / flux)``
   - Quality flags: 'good' (S/N≥5), 'marginal' (3≤S/N<5), 'poor' (S/N<3)
   - Magnitude calibration against GAIA DR3 zero-point
   - Support for aperture correction and PSF comparison

**PSF Photometry**:
   - Empirical PSF model construction using EPSFBuilder
   - Star selection with quality filtering criteria
   - Iterative PSF fitting with IterativePSFPhotometry
   - Correct S/N calculation: ``S/N = flux_fit / flux_err``
   - PSF-specific quality flags for reliability assessment
   - PSF model visualization and FITS output
   - Comparison metrics with aperture photometry

**Error Propagation**:
   - Comprehensive photometric error calculation
   - Poisson noise plus read noise modeling
   - Background uncertainty propagation
   - Total error estimation with calc_total_error
   - Zero-point error propagation: ``σ_mag_calib = √(σ_mag_inst² + σ_zp²)``
   - Full precision S/N (no rounding) to avoid divide-by-zero

Photometric Calibration
-----------------------

**GAIA DR3 Integration**:
   - Automatic cross-matching with GAIA Data Release 3
   - Quality filtering (variability, color, astrometric quality)
   - Multiple photometric bands including synthetic photometry
   - Cone search with configurable radius
   - Proper motion and parallax data inclusion

**Zero-Point Calculation**:
   - Robust calibration with sigma clipping outlier rejection
   - Multiple aperture zero-point determination
   - Atmospheric extinction correction using airmass
   - Calibration star quality assessment
   - Zero-point uncertainty estimation and propagation
   - Correct column name lookup for error columns

**Magnitude Systems**:
   - GAIA G, BP, RP photometric bands
   - Johnson-Cousins UBVRI synthetic photometry
   - SDSS ugriz synthetic photometry
   - Instrumental to standard magnitude conversion
   - Multi-band calibration support

Multi-Catalog Cross-Matching
----------------------------

**GAIA DR3 Catalog**:
   - Stellar parameters, proper motions, and photometry
   - Quality filtering for reliable calibration sources
   - Synthetic photometry for non-standard bands
   - Astrometric quality indicators (RUWE)
   - Variability and color index filtering

**SIMBAD Database**:
   - Object identification and classification
   - Multiple identifier retrieval
   - Astronomical object types and properties
   - B and V magnitude data when available
   - Cross-reference with other catalogs

**Astro-Colibri Transient Alerts**:
   - Real-time transient and variable source alerts
   - Multi-messenger astronomy event correlation
   - Configurable time windows around observation date
   - API key authentication for full access
   - Event classification and discovery information

**Solar System Objects (SkyBoT)**:
   - Moving object identification and ephemeris
   - Asteroid and comet detection
   - Magnitude predictions and orbital elements
   - Time-dependent position calculations
   - Integration with observation timestamps

**Variable Star Catalog (AAVSO VSX)**:
   - Variable star identification and classification
   - Period information when available
   - Variable star types and characteristics
   - Cross-matching with detected sources
   - Integration with photometric analysis

**Quasar Catalog (Milliquas)**:
   - Quasar and AGN identification
   - Redshift information when available
   - R-band magnitude data
   - Large-scale structure studies support
   - Cross-matching with point sources

**Nearby Stars (10 Parsec Catalog)**:
   - Identification of stars within 10 parsecs
   - G, BP, RP magnitude data
   - Object type classification
   - Cross-matching with field sources

Transient Detection (Beta)
-------------------------

**Transient Finder**:
   - Automated search for transient candidates
   - Image subtraction and catalog filtering
   - Integration with PanSTARRS (Northern) and SkyMapper (Southern) surveys
   - Template image retrieval and masking
   - Candidate visualization with cutouts (Science, Template, Difference)

Data Visualization & Analysis
-----------------------------

**Image Display**:
   - Dual visualization modes (ZScale and histogram equalization)
   - Automatic contrast optimization
   - Color-mapped display with customizable scales
   - Interactive zoom and pan capabilities
   - Multi-panel layout for comparison

**Interactive Sky Viewer**:
   - Aladin Lite integration for source exploration
   - Overlay of detected sources with catalog information
   - Clickable source markers with detailed information
   - Survey selection and coordinate display
   - Export to external sky viewers (ESA Sky)

**Photometric Plots**:
   - FWHM analysis with scatter plots and statistics
   - Magnitude distribution histograms
   - Zero-point calibration plots with residuals
   - Background model visualization
   - PSF model display and analysis

**Source Catalogs**:
   - Comprehensive tabular display of all measurements
   - Multi-aperture photometry results
   - Cross-match results from all catalogs
   - Coordinate information and quality indicators
   - Export capabilities in multiple formats

User Interface & Workflow
-------------------------

**Streamlit Web Interface**:
   - Modern, responsive design accessible via web browser
   - Real-time processing progress indicators
   - Interactive parameter adjustment with immediate feedback
   - Organized sidebar with collapsible sections
   - Mobile-friendly responsive layout

**Configuration Management**:
   - Persistent parameter storage across sessions
   - Save/load configuration with single click
   - Default parameter sets for common use cases
   - Parameter validation with helpful error messages
   - Reset options for troubleshooting

**File Management**:
   - Drag-and-drop file upload interface
   - Automatic file format detection and validation
   - Temporary file handling with automatic cleanup
   - Archive browser for previous results
   - Batch download options for result sets

**Error Handling**:
   - Comprehensive error messages with recovery suggestions
   - Graceful degradation when optional components fail
   - Automatic fallback options for common failure modes
   - Detailed logging for troubleshooting
   - User-friendly error reporting

Output Generation & Export
--------------------------

**Comprehensive Result Files**:
   - CSV catalogs with complete photometric measurements
   - FITS files for background and PSF models
   - PNG plots for all analysis visualizations
   - Text files for headers and processing logs
   - ZIP archives for convenient result distribution

**Data Organization**:
   - User-specific result directories
   - Timestamped file naming for version control
   - Automatic file categorization by type
   - Metadata preservation in output files
   - Cross-referenced file relationships

**Archive Management**:
   - Automatic result archiving with compression
   - Old file cleanup with configurable retention
   - Browse interface for historical results
   - Bulk download options for research workflows
   - Secure multi-user data isolation

Performance & Scalability
-------------------------

**Efficient Processing**:
   - Streamlit caching for improved response times
   - Memory-efficient handling of large FITS files
   - Optimized algorithms for typical image sizes
   - Background processing for time-consuming operations
   - Resource monitoring and management

**Multi-User Support**:
   - Concurrent user session handling
   - Isolated workspaces for data security
   - Shared configuration options when appropriate
   - Load balancing for multiple simultaneous analyses
   - Scalable backend architecture

**Quality Assurance**:
   - Comprehensive input validation
   - Automatic quality checks throughout pipeline
   - Robust error recovery mechanisms
   - Extensive testing with real astronomical data
   - Performance benchmarking and optimization
