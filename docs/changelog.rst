Changelog
=========

This document records all notable changes to RAPAS Photometry Pipeline.

Version 1.4.6 (Current)
-----------------------

**Maintenance & Stability**
*   Minor improvements and bug fixes.
*   Updated dependencies.

Version 1.4.5
-------------

**Major Features**

*   **Advanced Astrometric Pipeline**: 
    *   Integrated `stdpipe` for robust local astrometric solving.
    *   Added WCS refinement using SCAMP (via stdpipe) and GAIA DR3 catalog.
    *   Automatic header validation and fixing for problematic WCS keywords.

*   **Transient Detection (Beta)**:
    *   New **Transient Finder** module utilizing image subtraction and catalog filtering.
    *   Integration with PanSTARRS and SkyMapper surveys for template comparison.
    *   Automatic candidate filtering against cataloged sources.

*   **Enhanced Cross-Matching**:
    *   Added **10 Parsec Catalog** for nearby star identification.
    *   Improved filtering for existing catalogs.

*   **Photometry Improvements**:
    *   Improved PSF Photometry with Gaussian fallback if EPSF building fails.
    *   Automatic Cosmic Ray rejection enabled by default using `astroscrappy`.

*   **User Interface**:
    *   Astrometry check toggle.
    *   Aladin Lite v3 integration for improved visualization.

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
