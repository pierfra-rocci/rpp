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

**Advanced Image Processing**:
   - Automatic FITS header validation and fixing
   - Cosmic ray removal using L.A.Cosmic algorithm (astroscrappy)
   - Support for multi-extension FITS files and data cubes
   - Automatic plate solving using SIRIL integration
   - Optional WCS refinement using stdpipe and Gaia DR3

**Comprehensive Calibration**:
   - Photometric calibration against Gaia DR3 catalog
   - Robust zero-point calculation with outlier rejection
   - Atmospheric extinction correction based on airmass calculation
   - Quality filtering for reliable calibration stars
   - Support for multiple Gaia photometric bands

**Multi-Catalog Cross-matching**:
   - Gaia DR3 for stellar properties and calibration
   - SIMBAD for object identification and classification
   - SkyBoT for solar system object detection
   - AAVSO VSX for variable star identification
   - Astro-Colibri for transient event correlation
   - VizieR Milliquas for quasar identification

**User-Friendly Interface**:
   - Web-based interface accessible through any modern browser
   - Interactive Aladin Lite integration for source exploration
   - Real-time processing progress and status updates
   - Comprehensive error handling and user feedback
   - Multi-user support with individual accounts and workspaces

**Data Management**:
   - Automated result archiving and organization
   - ZIP package generation for easy data distribution
   - Configuration persistence across sessions
   - Automatic cleanup of temporary and old files
   - Detailed processing logs with timestamps

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

1. **Installation**: Install Python dependencies and SIRIL
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
   - SIRIL for astrometric plate solving
   - Multiple astronomical catalog services
   - Email services for password recovery
   - File system for result storage and archival

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
   - Photometric redshift estimation preparation

**Astrometric Analysis**:
   - Accurate source position determination
   - Proper motion measurement preparation
   - Coordinate system calibration and validation
   - Multi-epoch analysis support

**Survey Work**:
   - Large-scale source catalog generation
   - Cross-identification with existing catalogs
   - Quality assessment and filtering
   - Data reduction pipeline integration

**Educational Use**:
   - Teaching astronomical data analysis techniques
   - Demonstrating photometric principles
   - Hands-on experience with professional tools
   - Research project development

Performance Characteristics
--------------------------

**Processing Speed**:
   - Typical 2K×2K image: 2-5 minutes complete analysis
   - 4K×4K image: 5-15 minutes depending on source density
   - Network-dependent catalog queries: 30 seconds - 2 minutes
   - Concurrent user support with isolated workspaces

**Accuracy**:
   - Photometric precision: ~0.01-0.05 mag (depending on SNR)
   - Astrometric accuracy: ~0.1-0.5 arcsec (WCS-dependent)
   - Calibration stability: Limited by Gaia DR3 systematics
   - Background estimation: Adaptive to image characteristics

**Scalability**:
   - Multi-user concurrent operation
   - Configurable resource usage limits
   - Automatic cleanup and memory management
   - Extensible catalog integration framework

Community and Support
--------------------

**Development**:
   - Open-source development model
   - Community contributions welcome
   - Regular updates and feature additions
   - Comprehensive testing and validation

**Documentation**:
   - Complete user guides and tutorials
   - API documentation for developers
   - Troubleshooting guides and FAQ
   - Example workflows and use cases

**Collaboration**:
   - Integration with RAPAS project workflows
   - Compatible with standard astronomical tools
   - Export formats for external analysis software
   - Professional astronomy community support

Indices and Tables
-----------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`