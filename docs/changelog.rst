Changelog
=========

This document records all notable changes to RAPAS Photometry Pipeline.

Version 1.5.3 (Current)
-----------------------

**Released: January 7, 2026**

**Photometry Calculation Improvements**

*   **Critical Bug Fix**: Fixed PSF S/N calculation that was incorrectly using ``sqrt(flux_err)`` instead of ``flux_err``. This was causing incorrect S/N values and magnitude errors for all PSF photometry.
*   **Precision Enhancement**: Removed S/N rounding to preserve full floating-point precision and avoid divide-by-zero errors in magnitude error calculations.
*   **Improved S/N Accuracy**: S/N now uses background-corrected flux (``aperture_sum_bkg_corr``) when available, providing more accurate estimates especially for faint sources.
*   **Error Propagation**: Added proper zero-point error propagation in calibrated magnitudes using quadrature: ``σ_mag_calib = √(σ_mag_inst² + σ_zp²)``
*   **Quality Flags**: Added photometric quality flags for both aperture and PSF photometry:
    
    - ``'good'``: S/N ≥ 5 (reliable photometry)
    - ``'marginal'``: 3 ≤ S/N < 5 (use with caution)
    - ``'poor'``: S/N < 3 (unreliable)
    - ``'unknown'``: missing data

**Documentation & Testing**

*   Comprehensive mathematical documentation of S/N and magnitude error formulas
*   New documentation file: ``docs/PHOTOMETRY_FORMULAS.md``
*   Complete unit test suite: ``tests/test_photometry.py``
*   Inline formula documentation in source code

**Technical Details**

*   S/N Formula: ``S/N = flux / flux_error``
*   Magnitude Error Formula: ``σ_mag = 1.0857 × (σ_flux / flux)`` where 1.0857 = 2.5/ln(10)
*   Calibrated Error: ``σ_mag_calib = √(σ_mag_inst² + σ_zp²)``

Version 1.5.1
-------------

**Released: January 3, 2026**

**Current Status**
*   All features from version 1.4.7 maintained and stabilized
*   Python 3.12 compatibility confirmed
*   All dependencies verified and up-to-date
*   Production-ready release with comprehensive testing

Version 1.4.7
-------------

**Released: December 30, 2025**

**Python 3.12 Upgrade & Dependency Management**
*   Upgraded to Python 3.12 with verified dependency resolution
*   Updated astropy to 7.1.1 with latest IERS data (0.2025.11.3.0.38.37)
*   Updated aiohttp to 3.11.18 with latest security patches
*   Updated certifi to 2025.1.31 for current CA certificate bundle
*   All dependencies verified compatible with Python 3.12

**Maintenance & Stability**
*   Comprehensive dependency audit and updates
*   Improved compatibility with latest Python ecosystem
*   Enhanced system library integration

Version 1.4.5
-------------

**Major Features**

*   **Advanced Astrometric Pipeline**: 
    *   Integrated `stdpipe` for robust local astrometric solving
    *   Added WCS refinement using SCAMP (via stdpipe) and GAIA DR3 catalog
    *   Automatic header validation and fixing for problematic WCS keywords

*   **Transient Detection (Beta)**:
    *   New **Transient Finder** module utilizing image subtraction and catalog filtering
    *   Integration with PanSTARRS and SkyMapper surveys for template comparison
    *   Automatic candidate filtering against cataloged sources
    *   Hemisphere-based automatic survey selection (PanSTARRS north, SkyMapper south)

*   **Enhanced Cross-Matching**:
    *   Added **10 Parsec Catalog** for nearby star identification
    *   Improved filtering for existing catalogs
    *   Multi-band photometric support (GAIA G, BP, RP)

*   **Photometry Improvements**:
    *   Improved PSF Photometry with Gaussian fallback if EPSF building fails
    *   Automatic Cosmic Ray rejection enabled by default using `astroscrappy` (L.A.Cosmic algorithm)
    *   Multi-aperture photometry with 1.5×, 2.0× FWHM radii

*   **User Interface**:
    *   Astrometry check toggle for automatic/manual plate solving
    *   Aladin Lite v3 integration for improved interactive visualization
    *   Enhanced progress tracking and status updates

Version 0.9.8
-----------------------

**Major Features & Enhancements**

*   **Advanced Astrometric Pipeline**: 
    *   Integrated stdpipe for robust local plate solving with Astrometry.net
    *   Added WCS refinement using SCAMP and GAIA DR3 catalog
    *   Automatic header validation and fixing for problematic WCS keywords
    *   Force re-solve option for overriding existing WCS solutions
    *   Comprehensive coordinate validation and transformation checking

*   **Multi-Aperture Photometry**:
    *   Implemented four aperture radii (1.5×, 2.0×, 2.5×, 3.0× FWHM)
    *   Local background estimation with annular apertures for each radius
    *   Background-corrected and raw flux measurements
    *   Signal-to-noise ratio calculation for all apertures
    *   Backward compatibility with legacy single-aperture columns

*   **Enhanced PSF Photometry**:
    *   Empirical PSF model construction using EPSFBuilder
    *   Advanced star selection with quality filtering criteria
    *   Iterative PSF fitting with IterativePSFPhotometry
    *   PSF model visualization and FITS output
    *   Integration with multi-aperture photometry workflow

*   **Comprehensive Catalog Cross-Matching**:
    *   GAIA DR3 with quality filtering (variability, color, astrometry)
    *   SIMBAD for object identification and classification
    *   Astro-Colibri API for transient event correlation
    *   SkyBoT for solar system object identification
    *   AAVSO VSX for variable star catalog matching
    *   VizieR Milliquas for quasar and AGN identification

*   **Interactive Visualization**:
    *   Aladin Lite integration for interactive sky viewing
    *   Source overlay with detailed popup information
    *   Multiple survey options and coordinate display
    *   ESA Sky external viewer integration
    *   Real-time catalog cross-match results display

*   **Robust Error Handling**:
    *   Comprehensive error recovery mechanisms
    *   Graceful degradation when optional components fail
    *   Detailed error messages with recovery suggestions
    *   Automatic fallback options for common failure modes
    *   Enhanced logging with severity levels and timestamps

**Image Processing Improvements**:
    *   Advanced FITS header validation and automatic fixing
    *   Support for multi-extension FITS and data cubes
    *   RGB astronomical image handling (first channel extraction)
    *   Cosmic ray removal integration with L.A.Cosmic algorithm
    *   Enhanced background modeling with visualization and FITS output

**User Interface Enhancements**:
    *   Streamlined sidebar organization with collapsible sections
    *   Real-time parameter validation with helpful error messages
    *   Improved file upload handling with format detection
    *   Enhanced progress indicators and status updates
    *   Mobile-friendly responsive design improvements

**Data Management**:
    *   Automatic result archiving with timestamped ZIP files
    *   User-specific workspace isolation and management
    *   Archive browser for accessing previous results
    *   Automatic cleanup of old files (30+ day retention)
    *   Comprehensive output file generation and organization

**Performance Optimizations**:
    *   Streamlit caching for improved response times
    *   Memory-efficient handling of large FITS files
    *   Optimized algorithms for typical astronomical image sizes
    *   Background processing for time-consuming operations
    *   Resource monitoring and management improvements

**Configuration Management**:
    *   Enhanced parameter persistence across user sessions
    *   Observatory information extraction from FITS headers
    *   API key management with secure storage
    *   Default parameter sets for common use cases
    *   Configuration validation and error checking

**Minor Enhancements & Bug Fixes**

*   Improved coordinate system handling with multiple keyword support
*   Enhanced pixel scale calculation with multiple methods
*   Better FWHM estimation with outlier rejection
*   Robust airmass calculation with multiple date format support
*   Enhanced zero-point calculation with atmospheric extinction correction
*   Improved catalog query error handling and retry mechanisms
*   Better temporary file management and cleanup
*   Enhanced logging with structured format and severity levels

**Documentation Updates**:
    *   Comprehensive API reference documentation
    *   Updated installation guide with current dependencies
    *   Enhanced user guide with step-by-step workflows
    *   Feature documentation with detailed descriptions
    *   Troubleshooting guide with common issues and solutions

Version 1.0.0 (2023-09-01) - Previous Release
--------------------------------------------

**Features**:
- FITS image calibration (bias, dark, flat)
- Basic plate solving with astrometry.net
- Source detection and aperture photometry
- Zero-point calibration with Gaia DR3
- CSV catalog export
- Image visualization with matplotlib
- User-friendly Streamlit interface

**Requirements**:
- Python 3.8+
- Core packages: astropy, photutils, astroquery, streamlit, numpy, pandas, matplotlib

Version 0.9.0 (2023-07-15) - Beta Release
-----------------------------------------

**Initial Beta Features**:
- Core functionality implemented
- Basic calibration and photometry workflow
- Preliminary documentation
- Single-user desktop application

**Technical Foundation**:
- photutils-based photometry pipeline
- Basic FITS loading and display
- Simple aperture photometry
- Gaia DR3 cross-matching
- Basic CSV export functionality

Development Roadmap
------------------

**Planned Features for v1.0**:
- Enhanced PSF modeling with multiple PSF types
- Advanced astrometric distortion correction
- Batch processing capabilities for image sequences
- Machine learning-based source classification
- Advanced visualization tools and interactive plots
- Integration with additional astronomical catalogs
- Performance optimizations for large surveys
- Enhanced documentation and tutorials

**Future Enhancements**:
- Multi-band photometry across different filters
- Time-series analysis for variable source monitoring
- Advanced statistical analysis tools
- Integration with observatory control systems
- Cloud-based processing and storage options
- Advanced machine learning features
- Professional data reduction pipeline integration

**Known Issues & Limitations**:
- SCAMP refinement requires proper installation and configuration
- Large images (>8K×8K) may require increased memory allocation
- Some catalog services may have temporary availability issues
- Complex WCS solutions with high-order distortions may need manual validation
- Network-dependent features require stable internet connection

**Migration Notes**:
- Configuration files from previous versions are automatically migrated
- Output file formats are backward compatible
- API changes are minimal and documented in the API reference
- Legacy single-aperture photometry columns are maintained for compatibility
