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

.. py:function:: detect_remove_cosmic_rays(image_data, gain=1.0, readnoise=6.5, sigclip=4.5, sigfrac=0.3, objlim=5.0, verbose=True)

   Detect and remove cosmic rays from an astronomical image using astroscrappy.
   
   :param image_data: The 2D image array
   :type image_data: numpy.ndarray
   :param gain: CCD gain (electrons/ADU)
   :type gain: float
   :param readnoise: CCD read noise (electrons)
   :type readnoise: float
   :param sigclip: Detection sigma threshold
   :type sigclip: float
   :param sigfrac: Fractional detection threshold
   :type sigfrac: float
   :param objlim: Minimum contrast between cosmic ray and underlying object
   :type objlim: float
   :param verbose: Whether to print verbose output
   :type verbose: bool
   :return: Tuple of (cleaned_image, mask)
   :rtype: tuple(numpy.ndarray, numpy.ndarray)

.. py:function:: make_border_mask(image, border=50, invert=True, dtype=bool)

   Create a binary mask for an image excluding one or more border regions.
   
   :param image: The input image as a NumPy array
   :type image: numpy.ndarray
   :param border: Border size(s) to exclude from the mask
   :type border: int or tuple of int
   :param invert: If True, the mask will be inverted (False for border regions)
   :type invert: bool
   :param dtype: Data type of the output mask
   :type dtype: numpy.dtype
   :return: Binary mask with the same height and width as the input image
   :rtype: numpy.ndarray

Astrometry
^^^^^^^^^

.. py:function:: solve_with_astrometrynet(file_path)

   Solve astrometric plate using local Astrometry.Net installation via stdpipe.
   
   :param file_path: Path to the FITS image file
   :type file_path: str
   :return: Tuple of (wcs_object, updated_header)
   :rtype: tuple(astropy.wcs.WCS, astropy.io.fits.Header)

.. py:function:: safe_wcs_create(header)

   Create a WCS object from a FITS header with robust error handling.
   
   :param header: FITS header containing WCS information
   :type header: dict or astropy.io.fits.Header
   :return: Tuple of (wcs_object, error_message)
   :rtype: tuple(astropy.wcs.WCS, str)

Source Detection & Photometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: detection_and_photometry(image_data, science_header, mean_fwhm_pixel, threshold_sigma, detection_mask, filter_band)

   Perform a complete photometry workflow on an astronomical image.
   
   :param image_data: 2D array of the image data.
   :type image_data: numpy.ndarray
   :param science_header: FITS header or dictionary with image metadata.
   :type science_header: dict or astropy.io.fits.Header
   :param mean_fwhm_pixel: Estimated FWHM in pixels, used for aperture and PSF sizing
   :type mean_fwhm_pixel: float
   :param threshold_sigma: Detection threshold in sigma above background.
   :type threshold_sigma: float
   :param detection_mask: Border size in pixels to mask during detection.
   :type detection_mask: int
   :param filter_band: Photometric band for catalog matching and calibration.
   :type filter_band: str
   :return: Tuple of (phot_table, epsf_table, daofind, bkg, wcs_obj)
   :rtype: tuple(astropy.table.Table, astropy.table.Table, photutils.detection.DAOStarFinder, photutils.background.Background2D, astropy.wcs.WCS)

.. py:function:: perform_psf_photometry(img, photo_table, fwhm, daostarfind, mask=None, error=None)

   Perform PSF photometry using an empirically-constructed PSF model.
   
   :param img: Image with sky background subtracted
   :type img: numpy.ndarray
   :param photo_table: Table containing source positions
   :type photo_table: astropy.table.Table
   :param fwhm: Full Width at Half Maximum in pixels
   :type fwhm: float
   :param daostarfind: Star detection function
   :type daostarfind: callable
   :param mask: Mask to exclude image areas
   :type mask: numpy.ndarray, optional
   :param error: Error array for the image
   :type error: numpy.ndarray, optional
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

.. py:function:: cross_match_with_gaia(_phot_table, science_header, pixel_size_arcsec, mean_fwhm_pixel, filter_band, filter_max_mag, refined_wcs=None)

   Cross-match detected sources with the star catalogs.
   
   :param _phot_table: Table with detected source positions
   :type _phot_table: astropy.table.Table
   :param science_header: FITS header with WCS information
   :type science_header: dict or astropy.io.fits.Header
   :param pixel_size_arcsec: Pixel scale in arcseconds per pixel
   :type pixel_size_arcsec: float
   :param mean_fwhm_pixel: FWHM in pixels
   :type mean_fwhm_pixel: float
   :param filter_band: GAIA magnitude band to use
   :type filter_band: str
   :param filter_max_mag: Maximum magnitude for GAIA filtering
   :type filter_max_mag: float
   :param refined_wcs: Refined WCS object
   :type refined_wcs: astropy.wcs.WCS, optional
   :return: DataFrame with matched sources or None
   :rtype: pandas.DataFrame or None

.. py:function:: calculate_zero_point(_phot_table, _matched_table, filter_band, air)

   Calculate photometric zero point from matched sources with GAIA.
   
   :param _phot_table: Photometry results table
   :type _phot_table: astropy.table.Table or pandas.DataFrame
   :param _matched_table: Table of GAIA cross-matched sources
   :type _matched_table: pandas.DataFrame
   :param filter_band: GAIA magnitude band used
   :type filter_band: str
   :param air: Airmass value for atmospheric extinction
   :type air: float
   :return: Tuple of (zero_point_value, zero_point_std, matplotlib_figure)
   :rtype: tuple(float, float, matplotlib.figure.Figure)

.. py:function:: enhance_catalog(api_key, final_table, matched_table, header, pixel_scale_arcsec, search_radius_arcsec=60)

   Enhance a photometric catalog with cross-matches from multiple databases.
   
   :param api_key: API key for Astro-Colibri
   :type api_key: str
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

.. py:function:: estimate_background(image_data, box_size=100, filter_size=5, figure=True)

   Estimate the background and background RMS of an astronomical image.
   
   :param image_data: The 2D image array
   :type image_data: numpy.ndarray
   :param box_size: The box size in pixels for the local background estimation
   :type box_size: int
   :param filter_size: Size of the filter for smoothing the background
   :type filter_size: int
   :param figure: Whether to display a figure of the background
   :type figure: bool
   :return: Tuple of (background_2d_object, error_message)
   :rtype: tuple(photutils.background.Background2D, str)

.. py:function:: refine_astrometry_with_stdpipe(image_data, science_header, fwhm_estimate, pixel_scale, filter_band)

   Perform astrometry refinement using stdpipe SCAMP and GAIA DR3 catalog.
   
   :param image_data: The 2D image array
   :type image_data: numpy.ndarray
   :param science_header: FITS header with WCS information
   :type science_header: dict
   :param fwhm_estimate: FWHM estimate in pixels
   :type fwhm_estimate: float
   :param pixel_scale: Pixel scale in arcseconds per pixel
   :type pixel_scale: float
   :param filter_band: Gaia magnitude band to use for catalog matching
   :type filter_band: str
   :return: Refined WCS object if successful, None otherwise
   :rtype: astropy.wcs.WCS or None

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