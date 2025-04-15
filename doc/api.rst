API Reference
============

This page documents the key functions and modules in the Photometry Factory for RAPAS.

Image Handling
-------------

.. autofunction:: pfr_app.load_fits_data

.. autofunction:: pfr_app.calibrate_image_streamlit

.. autofunction:: pfr_app.estimate_background

Source Detection
---------------

.. autofunction:: pfr_app.find_sources_and_photometry_streamlit

.. autofunction:: pfr_app.fwhm_fit

Photometry
---------

.. autofunction:: pfr_app.perform_epsf_photometry

Astrometry
---------

.. autofunction:: pfr_app.solve_with_astrometry_net

.. autofunction:: pfr_app.safe_wcs_create

.. autofunction:: pfr_app.extract_coordinates

.. autofunction:: pfr_app.extract_pixel_scale

Catalog Operations
----------------

.. autofunction:: pfr_app.cross_match_with_gaia_streamlit

.. autofunction:: pfr_app.enhance_catalog_with_crossmatches

.. autofunction:: pfr_app.calculate_zero_point_streamlit

.. autofunction:: pfr_app.getJson

Visualization
------------

.. autofunction:: pfr_app.create_figure

.. autofunction:: pfr_app.display_catalog_in_aladin

Utilities
--------

.. autofunction:: pfr_app.airmass

.. autofunction:: pfr_app.ensure_output_directory

.. autofunction:: pfr_app.make_border_mask

.. autofunction:: pfr_app.write_to_log

.. autofunction:: pfr_app.initialize_session_state
