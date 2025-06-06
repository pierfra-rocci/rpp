// filepath: c:\Users\pierf\P_F_R\doc\api_reference.rst
API Reference
=============

This section documents the key functions and modules in Photometry Factory for RAPAS. While the application is primarily designed as an integrated tool, its components can be used programmatically.

Core Functions
-------------

Image Processing
^^^^^^^^^^^^^^

.. py:function:: load_fits_data(file)

   Load image data and header from a FITS file with robust error handling.
   
   :param file: FITS file object from Streamlit file uploader
   :type file: StreamlitUploadedFile
   :return: Tuple of (image_data, header)
   :rtype: tuple(numpy.ndarray, astropy.io.fits.Header)

.. py:function:: calibrate_image_streamlit(science_data, science_header, bias_data, dark_data, flat_data, exposure_time_science, exposure_time_dark, apply_bias, apply_dark, apply_flat)

   Calibrates an astronomical image using bias, dark, and flat-field frames.
   
   :param science_data: Raw science image data
   :type science_data: numpy.ndarray
   :param science_header: FITS header
   :type science_header: dict or astropy.io.fits.Header
   :param bias_data: Bias frame data
   :type bias_data: numpy.ndarray or None
   :param dark_data: Dark frame data
   :type dark_data: numpy.ndarray or None
   :param flat_data: Flat field data
   :type flat_data: numpy.ndarray or None
   :param exposure_time_science: Exposure time of science image in seconds
   :type exposure_time_science: float
   :param exposure_time_dark: Exposure time of dark frame in seconds
   :type exposure_time_dark: float
   :param apply_bias: Whether to apply bias subtraction
   :type apply_bias: bool
   :param apply_dark: Whether to apply dark subtraction
   :type apply_dark: bool
   :param apply_flat: Whether to apply flat field correction
   :type apply_flat: bool
   :return: Tuple of (calibrated_data, header)
   :rtype: tuple(numpy.ndarray, dict)

Astrometry
^^^^^^^^^

.. py:function:: solve_with_astrometry_net(image_data, header=None, api_key=None)

   Solve astrometric plate using the astrometry.net web API.
   
   :param image_data: The 2D image array to solve
   :type image_data: numpy.ndarray
   :param header: FITS header with metadata to help the solver
   :type header: astropy.io.fits.Header or dict, optional
   :param api_key: Astrometry.net API key
   :type api_key: str, optional
   :return: Tuple of (wcs_object, updated_header, status_message)
   :rtype: tuple(astropy.wcs.WCS, dict, str)

.. py:function:: safe_wcs_create(header)

   Create a WCS object from a FITS header with robust error handling.
   
   :param header: FITS header containing WCS information
   :type header: dict or astropy.io.fits.Header
   :return: Tuple of (wcs_object, error_message)
   :rtype: tuple(astropy.wcs.WCS, str)

Source Detection & Photometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: find_sources_and_photometry_streamlit(image_data, _science_header, mean_fwhm_pixel, threshold_sigma, detection_mask)

   Find astronomical sources and perform both aperture and PSF photometry.
   
   :param image_data: Science image data
   :type image_data: numpy.ndarray
   :param _science_header: FITS header information
   :type _science_header: dict or astropy.io.fits.Header
   :param mean_fwhm_pixel: Estimated FWHM in pixels
   :type mean_fwhm_pixel: float
   :param threshold_sigma: Detection threshold in sigma units
   :type threshold_sigma: float
   :param detection_mask: Border size to mask in pixels
   :type detection_mask: int
   :return: Tuple of (phot_table, epsf_table, daofind, bkg)
   :rtype: tuple(astropy.table.Table, astropy.table.Table, object, object)

.. py:function:: perform_epsf_photometry(img, phot_table, fwhm, daostarfind, mask=None)

   Perform PSF photometry using an empirically-constructed PSF model.
   
   :param img: Image with sky background subtracted
   :type img: numpy.ndarray
   :param phot_table: Table containing source positions
   :type phot_table: astropy.table.Table
   :param fwhm: Full Width at Half Maximum in pixels
   :type fwhm: float
   :param daostarfind: Star detection function
   :type daostarfind: callable
   :param mask: Mask to exclude image areas
   :type mask: numpy.ndarray, optional
   :return: Tuple of (phot_epsf_result, epsf)
   :rtype: tuple(astropy.table.Table, photutils.psf.EPSFModel)

.. py:function:: fwhm_fit(img, fwhm, pixel_scale, mask=None, std_lo=0.5, std_hi=0.5)

   Estimate the Full Width at Half Maximum (FWHM) of stars in an image.
   
   :param img: The 2D image array
   :type img: numpy.ndarray
   :param fwhm: Initial FWHM estimate in pixels
   :type fwhm: float
   :param pixel_scale: Pixel scale in arcseconds per pixel
   :type pixel_scale: float
   :param mask: Boolean mask array for pixels to ignore
   :type mask: numpy.ndarray, optional
   :param std_lo: Lower bound for flux filtering
   :type std_lo: float
   :param std_hi: Upper bound for flux filtering
   :type std_hi: float
   :return: Median FWHM value in pixels or None if failed
   :rtype: float or None

Catalog Operations
^^^^^^^^^^^^^^^^

.. py:function:: cross_match_with_gaia_streamlit(_phot_table, _science_header, pixel_size_arcsec, mean_fwhm_pixel, gaia_band, gaia_min_mag, gaia_max_mag)

   Cross-match detected sources with the GAIA DR3 star catalog.
   
   :param _phot_table: Table with detected source positions
   :type _phot_table: astropy.table.Table
   :param _science_header: FITS header with WCS information
   :type _science_header: dict or astropy.io.fits.Header
   :param pixel_size_arcsec: Pixel scale in arcseconds per pixel
   :type pixel_size_arcsec: float
   :param mean_fwhm_pixel: FWHM in pixels
   :type mean_fwhm_pixel: float
   :param gaia_band: GAIA magnitude band to use
   :type gaia_band: str
   :param gaia_min_mag: Minimum magnitude for GAIA filtering
   :type gaia_min_mag: float
   :param gaia_max_mag: Maximum magnitude for GAIA filtering
   :type gaia_max_mag: float
   :return: DataFrame with matched sources or None
   :rtype: pandas.DataFrame or None

.. py:function:: calculate_zero_point_streamlit(_phot_table, _matched_table, gaia_band, air)

   Calculate photometric zero point from matched sources with GAIA.
   
   :param _phot_table: Photometry results table
   :type _phot_table: astropy.table.Table or pandas.DataFrame
   :param _matched_table: Table of GAIA cross-matched sources
   :type _matched_table: pandas.DataFrame
   :param gaia_band: GAIA magnitude band used
   :type gaia_band: str
   :param air: Airmass value for atmospheric extinction
   :type air: float
   :return: Tuple of (zero_point_value, zero_point_std, matplotlib_figure)
   :rtype: tuple(float, float, matplotlib.figure.Figure)

.. py:function:: enhance_catalog_with_crossmatches(final_table, matched_table, header, pixel_scale_arcsec, search_radius_arcsec=6.0)

   Enhance a photometric catalog with cross-matches from multiple databases.
   
   :param final_table: Final photometry catalog
   :type final_table: pandas.DataFrame
   :param matched_table: Table of matched GAIA sources
   :type matched_table: pandas.DataFrame
   :param header: FITS header with observation information
   :type header: dict or astropy.io.fits.Header
   :param pixel_scale_arcsec: Pixel scale in arcseconds
   :type pixel_scale_arcsec: float
   :param search_radius_arcsec: Search radius for matching
   :type search_radius_arcsec: float
   :return: Enhanced DataFrame with catalog information
   :rtype: pandas.DataFrame

Utilities
^^^^^^^^

.. py:function:: airmass(_header, observatory=None, return_details=False)

   Calculate the airmass for a celestial object from observation parameters.
   
   :param _header: FITS header with observation information
   :type _header: dict
   :param observatory: Observatory information dictionary
   :type observatory: dict, optional
   :param return_details: Whether to return additional information
   :type return_details: bool
   :return: Airmass value or tuple with details
   :rtype: float or tuple(float, dict)

.. py:function:: extract_pixel_scale(header)

   Extract the pixel scale from a FITS header.
   
   :param header: FITS header with metadata
   :type header: dict or astropy.io.fits.Header
   :return: Tuple of (pixel_scale_value, source_description)
   :rtype: tuple(float, str)

.. py:function:: extract_coordinates(header)

   Extract celestial coordinates from a FITS header.
   
   :param header: FITS header with coordinate information
   :type header: dict or astropy.io.fits.Header
   :return: Tuple of (ra, dec, source_description)
   :rtype: tuple(float, float, str)

.. py:function:: display_catalog_in_aladin(final_table, ra_center, dec_center, fov=0.5, ra_col='ra', dec_col='dec', mag_col='calib_mag', alt_mag_col='aperture_calib_mag', catalog_col='catalog_matches', id_cols=['simbad_main_id', 'skybot_NAME', 'aavso_Name'], fallback_id_prefix="Source", survey="CDS/P/DSS2/color")

   Display a DataFrame catalog in an embedded Aladin Lite interactive sky viewer.
   
   :param final_table: DataFrame containing catalog data
   :type final_table: pandas.DataFrame
   :param ra_center: Right Ascension center coordinate in degrees
   :type ra_center: float
   :param dec_center: Declination center coordinate in degrees
   :type dec_center: float
   :param fov: Field of view in degrees
   :type fov: float, optional
   :param ra_col: Column name for Right Ascension
   :type ra_col: str, optional
   :param dec_col: Column name for Declination
   :type dec_col: str, optional
   :param mag_col: Column name for magnitude
   :type mag_col: str, optional
   :param alt_mag_col: Column name for alternative magnitude
   :type alt_mag_col: str, optional
   :param catalog_col: Column name for catalog matches
   :type catalog_col: str, optional
   :param id_cols: Column names for source identifiers
   :type id_cols: list[str], optional
   :param fallback_id_prefix: Prefix for unnamed sources
   :type fallback_id_prefix: str, optional
   :param survey: Initial sky survey to display
   :type survey: str, optional
   :rtype: None

Usage Examples
-------------

Example: Load and calibrate an image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from astropy.io import fits
    
    # Load science and calibration frames
    science_data, science_header = fits.getdata('science.fits', header=True)
    bias_data, _ = fits.getdata('bias.fits', header=True)
    dark_data, dark_header = fits.getdata('dark.fits', header=True)
    flat_data, _ = fits.getdata('flat.fits', header=True)
    
    # Get exposure times
    exposure_time_science = science_header.get('EXPTIME', 1.0)
    exposure_time_dark = dark_header.get('EXPTIME', 1.0)
    
    # Calibrate image
    calibrated_data, calibrated_header = calibrate_image_streamlit(
        science_data, science_header,
        bias_data, dark_data, flat_data,
        exposure_time_science, exposure_time_dark,
        apply_bias=True, apply_dark=True, apply_flat=True
    )

Example: Perform photometry and zero-point calibration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import numpy as np
    
    # Set parameters
    pixel_size_arcsec = 1.2
    mean_fwhm_pixel = 3.0
    threshold_sigma = 3.0
    detection_mask = 50
    
    # Find sources and perform photometry
    phot_table, epsf_table, daofind, bkg = find_sources_and_photometry_streamlit(
        calibrated_data, calibrated_header, 
        mean_fwhm_pixel, threshold_sigma, detection_mask
    )
    
    # Get catalog matches
    matched_table = cross_match_with_gaia_streamlit(
        phot_table, calibrated_header,
        pixel_size_arcsec, mean_fwhm_pixel,
        "phot_g_mean_mag", 10.0, 18.0
    )
    
    # Calculate zero point (using airmass=1.0)
    zp, zp_std, _ = calculate_zero_point_streamlit(
        phot_table, matched_table, "phot_g_mean_mag", 1.0
    )
    
    # Apply zero point to get calibrated magnitudes
    calibrated_mags = phot_table['instrumental_mag'] + zp + 0.1*1.0  # Apply extinction