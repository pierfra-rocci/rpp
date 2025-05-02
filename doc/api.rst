API Reference
============

This page documents the key functions used within the Photometry Factory for RAPAS application, primarily located in `pages/app.py` and `tools.py`. These functions handle the core processing steps.

Initialization & Setup
----------------------
.. autofunction:: pages.app.initialize_session_state
.. autofunction:: tools.ensure_output_directory
.. autofunction:: tools.initialize_log
.. autofunction:: tools.write_to_log

File Handling & Loading
-----------------------
.. autofunction:: pages.app.load_fits_data
.. autofunction:: tools.get_base_filename
.. autofunction:: pages.app.save_header_to_txt

Image Processing & Calibration
-----------------------------
.. autofunction:: pages.app.detect_remove_cosmic_rays
.. autofunction:: pages.app.estimate_background
.. autofunction:: tools.make_border_mask

Astrometry
---------
.. autofunction:: tools.safe_wcs_create
.. autofunction:: pages.app.solve_with_siril
.. autofunction:: tools.extract_coordinates
.. autofunction:: tools.extract_pixel_scale
.. autofunction:: tools.airmass
.. note:: Astrometry refinement using `stdpipe` happens within `pages.app.detection_and_photometry`.

Source Detection & Photometry
-----------------------------
.. autofunction:: pages.app.fwhm_fit
.. autofunction:: pages.app.detection_and_photometry
.. autofunction:: pages.app.perform_psf_photometry

Catalog Operations & Calibration
--------------------------------
.. autofunction:: pages.app.cross_match_with_gaia
.. autofunction:: pages.app.calculate_zero_point
.. autofunction:: pages.app.enhance_catalog
.. autofunction:: tools.safe_catalog_query
.. autofunction:: tools.get_json

Visualization & Output
----------------------
.. autofunction:: tools.create_figure
.. autofunction:: pages.app.display_catalog_in_aladin
.. autofunction:: pages.app.provide_download_buttons
.. autofunction:: tools.cleanup_temp_files
.. autofunction:: tools.zip_rpp_results_on_exit

Backend Interaction (Login/Config)
----------------------------------
.. note:: Functions interacting with the Flask backend (`login`, `register`, `save_config`, `get_config`) are primarily handled within `pages/login.py` using `requests` to call the backend API endpoints defined in `backend_dev.py`.
