RAPAS Photometry Pipeline (RPP)
==============================

.. image:: _static/rpp_logo.png
   :width: 200px
   :align: right
   :alt: RAPAS Photometry Pipeline logo

**RAPAS Photometry Pipeline (RPP)** is a comprehensive web-based astronomical photometry system built with Streamlit and Python. It provides an end-to-end solution for processing astronomical images, performing precise photometry, and cross-matching results with major astronomical catalogs.

The pipeline is designed for both amateur and professional astronomers, offering a user-friendly interface while maintaining the rigor and precision required for scientific analysis.

Key Features
-----------

**Complete Photometry Pipeline**:
   - Multi-aperture photometry with four different radii (1.5×, 2.0×, 2.5×, 3.0× FWHM)
   - Empirical PSF model construction and PSF photometry
   - Robust background estimation with SExtractor algorithm
   - Automatic source detection using DAOStarFinder
   - FWHM estimation through Gaussian fitting of stellar profiles

**Advanced Astrometric Solutions**:
   - Local plate solving using Astrometry.net via stdpipe for blind astrometric solving
   - WCS refinement using SCAMP and GAIA DR3 catalog
   - Automatic WCS header validation and fixing
   - Support for multiple coordinate reference frames
   - Force re-solve option for existing WCS solutions

**Comprehensive Photometric Calibration**:
   - Photometric calibration against Gaia DR3 catalog
   - Robust zero-point calculation with sigma clipping and outlier rejection
   - Atmospheric extinction correction based on airmass calculation
   - Quality filtering for reliable calibration stars
   - Support for multiple Gaia photometric bands and synthetic photometry

**Multi-Catalog Cross-matching**:
   - Gaia DR3 for stellar properties, proper motions, and calibration
   - SIMBAD for object identification and classification
   - Astro-Colibri for transient event correlation and alerts
   - SkyBoT for solar system object detection
   - AAVSO VSX for variable star identification
   - VizieR Milliquas for quasar and AGN identification

**Advanced Image Processing**:
   - Cosmic ray removal using L.A.Cosmic algorithm (astroscrappy)
   - Support for multi-extension FITS files and data cubes
   - Multiple visualization modes (ZScale, histogram equalization)
   - Automatic FITS header validation and fixing
   - Quality filtering and automated source assessment

**User-Friendly Interface**:
   - Web-based interface accessible through any modern browser
   - Interactive Aladin Lite integration for source exploration
   - Real-time processing progress and status updates
   - Comprehensive error handling and user feedback
   - Multi-user support with individual accounts and workspaces
   - Configuration persistence across sessions

**Data Management**:
   - Automated result archiving and organization
   - ZIP package generation for easy data distribution
   - User-specific workspaces with automatic cleanup
   - Detailed processing logs with timestamps
   - Comprehensive output file generation

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   user_guide
   features
   advanced_features
   api
   examples
   troubleshooting
   changelog

Quick Start
----------

1. **Installation**: Install Python dependencies, Astrometry.net, and SCAMP
2. **Launch**: Start backend (`python backend.py`) and frontend (`streamlit run frontend.py`)
3. **Register**: Create user account and configure observatory settings
4. **Upload**: Load FITS image file for analysis
5. **Process**: Run photometric calibration pipeline
6. **Explore**: View results in interactive Aladin viewer
7. **Download**: Export complete analysis as ZIP archive

System Architecture
------------------

**Frontend (Streamlit)**:
   - Interactive web interface for user interaction
   - Real-time visualization and progress monitoring
   - Session state management and configuration persistence
   - File upload handling and result presentation

**Backend (Flask)**:
   - User authentication and authorization
   - Configuration storage and retrieval
   - Multi-user workspace management
   - API endpoints for frontend communication

**Processing Pipeline**:
   - Modular design with independent processing stages
   - Robust error handling and recovery mechanisms
   - Configurable parameters for different observation types
   - Integration with external tools and services

**External Integration**:
   - Astrometry.net via stdpipe for local plate solving
   - SCAMP for astrometric refinement
   - Multiple astronomical catalog services
   - Aladin Lite for interactive sky visualization

Supported Data Formats
---------------------

**Input Formats**:
   - FITS files (.fits, .fit, .fts)
   - Compressed FITS (.fits.gz)
   - Multi-extension FITS (MEF)
   - Data cubes (automatically extracts 2D planes)
   - RGB astronomical images (uses first channel)

**Output Formats**:
   - CSV catalogs with comprehensive source measurements
   - PNG visualizations and plots
   - FITS files for background and PSF models
   - Text files for headers and processing logs
   - ZIP archives for complete result packages

Scientific Applications
----------------------

**Photometric Studies**:
   - Precise magnitude measurements for stellar photometry
   - Variable star monitoring and light curve generation
   - Color-magnitude diagram construction
   - Multi-aperture photometric analysis

**Astrometric Analysis**:
   - Accurate source position determination
   - WCS solution validation and refinement
   - Coordinate system calibration
   - Multi-epoch analysis support

**Survey Work**:
   - Large-scale source catalog generation
   - Cross-identification with existing catalogs
   - Quality assessment and filtering
   - Transient detection and classification

**Educational Use**:
   - Teaching astronomical data analysis techniques
   - Demonstrating photometric principles
   - Hands-on experience with professional tools
   - Research project development

Performance Characteristics
--------------------------

**Processing Speed**:
   - Typical 2K×2K image: 2-5 minutes complete analysis
   - Multi-aperture photometry with background modeling
   - Efficient memory management for large files
   - Streamlit caching for improved performance

**Accuracy**:
   - Sub-arcsecond astrometric precision with plate solving
   - Millimagnitude photometric precision with proper calibration
   - Robust error propagation and uncertainty estimation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`