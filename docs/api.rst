API Reference
============

This page documents the key functions and modules used within the RAPAS Photometry Pipeline application. The pipeline is organized into several modules handling different aspects of astronomical image analysis.

Core Application Modules
------------------------

Frontend Application (pages/app.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pages.app.initialize_session_state
   
   Initialize all session state variables for the application. Sets up default values for user authentication, data storage, analysis parameters, and configuration management.

.. autofunction:: pages.app.load_fits_data
   
   Load image data and header from a FITS file with robust error handling. Supports multi-extension files, data cubes, RGB images, and automatic header fixing.

.. autofunction:: pages.app.display_catalog_in_aladin
   
   Display a DataFrame catalog in an embedded Aladin Lite interactive sky viewer. Creates interactive astronomical visualization with catalog overlay and source exploration.

.. autofunction:: pages.app.provide_download_buttons
   
   Creates download buttons for ZIP archives containing all analysis results. Handles file compression and user-friendly download interface.

.. autofunction:: pages.app.update_observatory_from_fits_header
   
   Extract and update observatory information from FITS header keywords including telescope name, site coordinates, and elevation.

Pipeline Processing (src/pipeline.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Astrometric Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: src.pipeline.solve_with_astrometrynet
   
   Solve astrometric plate using local Astrometry.Net installation via stdpipe. Performs blind plate solving with automatic parameter optimization and WCS validation.

.. autofunction:: src.pipeline.refine_astrometry_with_stdpipe
   
   Perform astrometry refinement using stdpipe SCAMP and GAIA DR3 catalog. Provides high-precision astrometric solutions with distortion correction.

Image Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: src.pipeline.detect_remove_cosmic_rays
   
   Detect and remove cosmic rays from astronomical images using astroscrappy L.A.Cosmic algorithm. Configurable parameters for different detector types.

.. autofunction:: src.pipeline.estimate_background
   
   Estimate sky background and RMS using photutils.Background2D with SExtractor algorithm. Includes visualization and FITS output generation.

.. autofunction:: src.pipeline.make_border_mask
   
   Create binary mask for excluding border regions from analysis. Supports flexible border specifications and handles edge artifacts.

Photometry Functions
~~~~~~~~~~~~~~~~~~

.. autofunction:: src.pipeline.detection_and_photometry
   
   Complete photometry workflow including source detection, multi-aperture photometry, PSF photometry, and coordinate transformation. Main processing pipeline function.

.. autofunction:: src.pipeline.perform_psf_photometry
   
   Perform PSF photometry using empirically-constructed PSF model. Includes star selection, PSF building, and iterative fitting.

.. autofunction:: src.pipeline.fwhm_fit
   
   Estimate FWHM of stars using Gaussian fitting on marginal sums. Provides robust seeing estimates with quality filtering.

Calibration Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: src.pipeline.cross_match_with_gaia
   
   Cross-match detected sources with GAIA DR3 catalog. Includes quality filtering and support for synthetic photometry bands.

.. autofunction:: src.pipeline.calculate_zero_point
   
   Calculate photometric zero point from GAIA-matched sources. Robust calibration with outlier rejection and atmospheric extinction correction.

.. autofunction:: src.pipeline.airmass
   
   Calculate airmass for celestial objects from observation parameters. Handles multiple coordinate formats and observatory locations.

.. autofunction:: src.pipeline.enhance_catalog
   
   Enhance photometric catalog with cross-matches from multiple astronomical databases including SIMBAD, Astro-Colibri, SkyBoT, AAVSO VSX, and Milliquas.

Utility Functions (src/tools.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

File and Data Handling
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: src.tools.get_base_filename
   
   Extract base filename without extension from file objects. Handles double extensions and special cases.

.. autofunction:: src.tools.ensure_output_directory
   
   Create output directory if it doesn't exist. Handles permission errors and provides fallback options.

.. autofunction:: src.tools.cleanup_temp_files
   
   Remove temporary FITS files created during processing. Automatic cleanup with error handling.

Header and WCS Utilities
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: src.tools.safe_wcs_create
   
   Safely create WCS object from FITS header with validation and error handling. Removes problematic keywords and validates transformations.

.. autofunction:: src.tools.fix_header
   
   Fix common issues in FITS headers for better WCS compatibility. Removes problematic keywords and standardizes coordinate systems.

.. autofunction:: src.tools.get_header_value
   
   Extract values from FITS headers using multiple possible keywords. Provides fallback options for varying keyword conventions.

.. autofunction:: src.tools.extract_coordinates
   
   Extract celestial coordinates (RA, Dec) from FITS headers. Validates coordinate ranges and handles multiple formats.

.. autofunction:: src.tools.extract_pixel_scale
   
   Extract or calculate pixel scale from FITS headers. Supports multiple methods including CD matrix, focal length calculation, and direct keywords.

Catalog and Network Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: src.tools.get_json
   
   Fetch JSON data from URLs with robust error handling. Used for catalog queries and API interactions.

.. autofunction:: src.tools.safe_catalog_query
   
   Execute catalog query functions with comprehensive error handling. Wrapper for astroquery and other catalog access functions.

Logging and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: src.tools.initialize_log
   
   Initialize in-memory logging buffer with standard header format. Creates structured logs for processing workflows.

.. autofunction:: src.tools.write_to_log
   
   Write formatted messages to log buffer with timestamps and severity levels. Structured logging for debugging and analysis.

.. autofunction:: src.tools.create_figure
   
   Create matplotlib figures with predefined sizes and DPI settings. Standardized figure creation for consistent output.

Version Information (src/__version__.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autodata:: src.__version__.version
   
   Current version string of the RAPAS Photometry Pipeline.

.. autodata:: src.__version__.version_info
   
   Version tuple for programmatic version comparison.

.. autodata:: src.__version__.author
   
   Author information and contact details.

Configuration Constants
-----------------------

Figure Sizes (src/tools.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autodata:: src.tools.FIGURE_SIZES
   
   Dictionary defining standard figure sizes for consistent plot generation:
   
   - "small": (6, 5) - For compact plots
   - "medium": (8, 6) - Standard size plots  
   - "large": (10, 8) - Detailed analysis plots
   - "wide": (12, 6) - Wide aspect ratio plots
   - "stars_grid": (10, 8) - PSF star grid displays

Photometric Bands (src/tools.py)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autodata:: src.tools.GAIA_BANDS
   
   List of supported GAIA and synthetic photometry bands:
   
   - GAIA DR3: phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
   - Johnson-Cousins: u_jkc_mag, v_jkc_mag, b_jkc_mag, r_jkc_mag, i_jkc_mag
   - SDSS: u_sdss_mag, g_sdss_mag, r_sdss_mag, i_sdss_mag, z_sdss_mag

Backend API (backend.py)
^^^^^^^^^^^^^^^^^^^^^^^

User Management Endpoints
~~~~~~~~~~~~~~~~~~~~~~~~

.. http:post:: /register
   
   Register new user account with username and password validation.

.. http:post:: /login
   
   Authenticate user with username and password.

.. http:post:: /recover_request
   
   Request password recovery via email.

.. http:post:: /recover_confirm
   
   Confirm password recovery with verification code.

Configuration Management
~~~~~~~~~~~~~~~~~~~~~~~

.. http:post:: /save_config
   
   Save user configuration including observatory settings, analysis parameters, and API keys.

.. http:get:: /get_config
   
   Retrieve saved user configuration from database.

Error Handling
~~~~~~~~~~~~~

.. http:error:: 404
   
   Page not found error handler with user-friendly response.

.. http:error:: 500
   
   Internal server error handler with logging and recovery options.

Usage Examples
--------------

Basic Pipeline Usage
^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.pipeline import detection_and_photometry, cross_match_with_gaia
   from src.tools import safe_wcs_create, extract_pixel_scale
   
   # Load and process image
   wcs_obj, wcs_error = safe_wcs_create(header)
   pixel_scale, scale_source = extract_pixel_scale(header)
   
   # Run photometry pipeline
   phot_table, epsf_table, daofind, bkg, refined_wcs = detection_and_photometry(
       image_data, header, fwhm_pixels, threshold_sigma, 
       border_mask, filter_band
   )
   
   # Cross-match with GAIA
   matched_table = cross_match_with_gaia(
       phot_table, header, pixel_scale, fwhm_pixels,
       filter_band, max_magnitude, refined_wcs
   )

Custom Processing Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from src.tools import initialize_log, write_to_log, create_figure
   from src.pipeline import estimate_background, airmass
   
   # Initialize logging
   log_buffer = initialize_log("my_analysis")
   write_to_log(log_buffer, "Starting custom analysis", "INFO")
   
   # Background estimation
   background, error = estimate_background(image_data)
   if background is not None:
       write_to_log(log_buffer, "Background estimation successful")
       
   # Calculate airmass
   air = airmass(header, observatory_data)
   write_to_log(log_buffer, f"Airmass: {air:.2f}")
   
   # Create standardized plots
   fig = create_figure("large")
   # ... plotting code ...

Notes
-----

- All functions include comprehensive error handling and validation
- Functions are designed to work together in the complete pipeline
- Individual functions can be used for custom analysis workflows
- Extensive logging and progress reporting throughout the pipeline
- Streamlit integration for real-time user feedback and interaction
