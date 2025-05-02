# Standard Library Imports
import sys
import os
import zipfile
import base64
import json
import tempfile
import warnings
import atexit
import subprocess
from datetime import datetime, timedelta
from urllib.parse import quote
from functools import partial
from io import StringIO, BytesIO
from typing import Union, Any, Optional, Dict, Tuple, List

# Third-Party Imports
import streamlit as st # Keep streamlit for st.write/error/warning/spinner etc. used within functions
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astroscrappy
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.stats import sigma_clip, SigmaClip, sigma_clipped_stats
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
from astropy.modeling import models, fitting
from astropy.nddata import NDData
from astropy.visualization import (ZScaleInterval, ImageNormalize,
                                   PercentileInterval, simple_norm)
import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, SExtractorBackground
from photutils.psf import EPSFBuilder, extract_stars, IterativePSFPhotometry
# Assuming stdpipe is installed or available in the environment
# from stdpipe import photometry, astrometry, catalogs, pipeline

# Local Application Imports
# Assuming tools.py and __version__.py are in the same root directory
from tools import (FIGURE_SIZES, URL, GAIA_BAND, extract_coordinates,
                   extract_pixel_scale, get_base_filename, safe_catalog_query,
                   safe_wcs_create, ensure_output_directory, cleanup_temp_files,
                   initialize_log, write_to_log, zip_rpp_results_on_exit)
# from __version__ import version # Version not typically needed in function files

warnings.filterwarnings("ignore")


def solve_with_siril(file_path):
    """
    Solve astrometric plate using Siril through a PowerShell script.
    This function sends an image file path to a PowerShell script that uses Siril
    to determine accurate WCS (World Coordinate System) information for the image.
    It then reads the resulting solved file to extract the WCS data.

    file_path : str
        Path to the image file that needs astrometric solving
        FITS header parameter (not used in current implementation but kept
        for interface compatibility)
    Returns
    -------
    - wcs_object: astropy.wcs.WCS object containing the WCS solution
    - updated_header: Header with WCS keywords from the solved image

    The function expects a PowerShell script named 'plate_solve.ps1' to be available
    in the current directory. The solved image will be saved with '_solved' appended
    to the original filename.
    """
    try:
        if os.path.exists(file_path):
            command = [
                "powershell.exe",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                "plate_solve.ps1",
                "-filepath",
                f"{file_path}",
            ]
            subprocess.run(command, check=True)
        else:
            st.warning(f"File {file_path} does not exist.")

    except Exception as e:
        st.error(f"Error solving with Siril: {str(e)}")
        return None
    try:
        file_path_list = file_path.split(".")
        file_path = file_path_list[0] + "_solved.fits"
        hdu = fits.open(file_path, mode="readonly")
        head = hdu[0].header
        wcs_obj = WCS(head)

        return wcs_obj, head

    except Exception as e:
        st.error(f"Error reading solved file: {str(e)}")


def detect_remove_cosmic_rays(
    image_data,
    gain=1.0,
    readnoise=6.5,
    sigclip=4.5,
    sigfrac=0.3,
    objlim=5.0,
    verbose=True,
):
    """
    Detect and remove cosmic rays from an astronomical image using astroscrappy.

    Parameters
    ----------
    image_data : numpy.ndarray
        The 2D image array
    gain : float, optional
        CCD gain (electrons/ADU), default=1.0
    readnoise : float, optional
        CCD read noise (electrons), default=6.5
    sigclip : float, optional
        Detection sigma threshold, default=4.5
    sigfrac : float, optional
        Fractional detection threshold, default=0.3
    objlim : float, optional
        Minimum contrast between cosmic ray and underlying object, default=5.0
    verbose : bool, optional
        Whether to print verbose output, default=False

    Returns
    -------
    tuple
        (cleaned_image, mask, num_cosmic_rays) where:
        - cleaned_image: numpy.ndarray with cosmic rays removed
        - mask: boolean numpy.ndarray showing cosmic ray locations (True where cosmic rays were detected)
        - num_cosmic_rays: int, number of cosmic rays detected

    Notes
    -----
    Uses the L.A.Cosmic algorithm implemented in astroscrappy.
    The algorithm detects cosmic rays using Laplacian edge detection.
    """
    try:
        # Ensure the image is in the correct format
        image_data = image_data.astype(np.float32)

        # Detect and remove cosmic rays using astroscrappy's implementation of L.A.Cosmic
        mask, cleaned_image = astroscrappy.detect_cosmics(
            image_data,
            gain=gain,
            readnoise=readnoise,
            sigclip=sigclip,
            sigfrac=sigfrac,
            objlim=objlim,
            verbose=verbose,
        )

        mask = mask.astype(bool)

        return cleaned_image, mask

    except ImportError:
        st.error("astroscrappy package is not installed. Cannot remove cosmic rays.")
        return image_data, None, 0
    except Exception as e:
        st.error(f"Error during cosmic ray removal: {str(e)}")
        return image_data, None, 0


def make_border_mask(
    image: np.ndarray,
    border: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 50,
    invert: bool = True,
    dtype: np.dtype = bool,
) -> np.ndarray:
    """
    Create a binary mask for an image excluding one or more border regions.

    This function creates a mask that can be used to exclude border pixels
    from analysis, which is useful for operations that might be affected
    by edge artifacts.

    Parameters
    ----------
    image : numpy.ndarray
        The input image as a NumPy array
    border : int or tuple of int, default=50
        Border size(s) to exclude from the mask:
        - If int: same border size on all sides
        - If tuple of 2 elements: (vertical, horizontal) borders
        - If tuple of 4 elements: (top, bottom, left, right) borders
    invert : bool, default=True
        If True, the mask will be inverted (False for border regions)
        If False, the mask will be True for non-border regions
    dtype : numpy.dtype, default=bool
        Data type of the output mask

    Returns
    -------
    numpy.ndarray
        Binary mask with the same height and width as the input image

    Raises
    ------
    TypeError
        If image is not a NumPy array or border is not of the correct type
    ValueError
        If image is empty, borders are negative, borders are larger than the image,
        or border tuple has an invalid length

    Examples
    --------
    >>> img = np.ones((100, 100))
    >>> # Create a mask with 10px border on all sides
    >>> mask = make_border_mask(img, 10)
    >>> # Create a mask with different vertical/horizontal borders
    >>> mask = make_border_mask(img, (20, 30))
    >>> # Create a mask with custom borders for each side
    >>> mask = make_border_mask(img, (10, 20, 30, 40), invert=False)
    """
    if image is None:
        raise ValueError("Image cannot be None")
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy.ndarray")

    if image.size == 0:
        raise ValueError("Image cannot be empty")

    height, width = image.shape[:2]

    if isinstance(border, (int, float)):
        border = int(border)
        top = bottom = left = right = border
    elif isinstance(border, tuple):
        if len(border) == 2:
            vert, horiz = border
            top = bottom = vert
            left = right = horiz
        elif len(border) == 4:
            top, bottom, left, right = border
        else:
            raise ValueError("border must be an int or a tuple of 2 or 4 el")
    else:
        raise TypeError("border must be an int or a tuple")

    if any(b < 0 for b in (top, bottom, left, right)):
        raise ValueError("Borders cannot be negative")

    if top + bottom >= height or left + right >= width:
        raise ValueError("Borders are larger than the image")

    mask = np.zeros(image.shape[:2], dtype=dtype)
    mask[top : height - bottom, left : width - right] = True

    return ~mask if invert else mask


def estimate_background(image_data, box_size=128, filter_size=7):
    """
    Estimate the background and background RMS of an astronomical image.

    Uses photutils.Background2D to create a 2D background model with sigma-clipping
    and the SExtractor background estimation algorithm. Includes error handling
    and automatic adjustment for small images.

    Parameters
    ----------
    image_data : numpy.ndarray
        The 2D image array
    box_size : int, optional
        The box size in pixels for the local background estimation.
        Will be automatically adjusted if the image is small.
    filter_size : int, optional
        Size of the filter for smoothing the background.

    Returns
    -------
    tuple
        (background_2d_object, error_message) where:
        - background_2d_object: photutils.Background2D object if successful, None if failed
        - error_message: None if successful, string describing the error if failed

    Notes
    -----
    The function automatically adjusts the box_size and filter_size parameters
    if the image is too small, and handles various edge cases to ensure robust
    background estimation.
    """
    if image_data is None:
        return None, "No image data provided"

    if not isinstance(image_data, np.ndarray):
        return None, f"Image data must be a numpy array, got {type(image_data)}"

    if len(image_data.shape) != 2:
        return None, f"Image must be 2D, got shape {image_data.shape}"

    height, width = image_data.shape
    adjusted_box_size = min(box_size, height // 4, width // 4)
    adjusted_filter_size = min(filter_size, adjusted_box_size // 2)

    if adjusted_box_size < 10:
        return None, f"Image too small ({height}x{width}) for background estimation"

    try:
        sigma_clip = SigmaClip(sigma=3)
        bkg_estimator = SExtractorBackground()

        bkg = Background2D(
            data=image_data,
            box_size=adjusted_box_size,
            filter_size=adjusted_filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )

        # Plot the background model with ZScale and save as FITS
        try:
            # Create a figure with two subplots side by side for background and RMS
            fig_bkg, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZES["wide"])

            # Use ZScaleInterval for better visualization
            zscale = ZScaleInterval()
            vmin, vmax = zscale.get_limits(bkg.background)

            # Plot the background model
            im1 = ax1.imshow(bkg.background, origin='lower', cmap='viridis',
                           vmin=vmin, vmax=vmax)
            ax1.set_title('Estimated Background')
            fig_bkg.colorbar(im1, ax=ax1, label='Flux')

            # Plot the background RMS
            vmin_rms, vmax_rms = zscale.get_limits(bkg.background_rms)
            im2 = ax2.imshow(bkg.background_rms, origin='lower', cmap='viridis',
                           vmin=vmin_rms, vmax=vmax_rms)
            ax2.set_title('Background RMS')
            fig_bkg.colorbar(im2, ax=ax2, label='Flux')

            fig_bkg.tight_layout()
            st.pyplot(fig_bkg)

            # Save background as FITS file
            base_filename = st.session_state.get("base_filename", "photometry")
            output_dir = ensure_output_directory("rpp_results")
            bkg_filename = f"{base_filename}_bkg.fits"
            bkg_filepath = os.path.join(output_dir, bkg_filename)

            # Create FITS HDU and save background model
            hdu_bkg = fits.PrimaryHDU(data=bkg.background)
            hdu_bkg.header['COMMENT'] = 'Background model created with photutils.Background2D'
            hdu_bkg.header['BOXSIZE'] = (adjusted_box_size, 'Box size for background estimation')
            hdu_bkg.header['FILTSIZE'] = (adjusted_filter_size, 'Filter size for background smoothing')

            # Add RMS as extension
            hdu_rms = fits.ImageHDU(data=bkg.background_rms)
            hdu_rms.header['EXTNAME'] = 'BACKGROUND_RMS'

            hdul = fits.HDUList([hdu_bkg, hdu_rms])
            hdul.writeto(bkg_filepath, overwrite=True)

            # Write to log if available
            log_buffer = st.session_state.get("log_buffer")
            if log_buffer is not None:
                write_to_log(log_buffer, f"Background model saved to {bkg_filename}")

        except Exception as e:
            st.warning(f"Error creating or saving background plot: {str(e)}")

        return bkg, None
    except Exception as e:
        return None, f"Background estimation error: {str(e)}"


def airmass(
    _header: Dict, observatory: Optional[Dict] = None, return_details: bool = False
) -> Union[float, Tuple[float, Dict]]:
    """
    Calculate the airmass for a celestial object from observation parameters.

    Airmass is a measure of the optical path length through Earth's atmosphere.
    This function calculates it from header information and observatory location,
    handling multiple coordinate formats and performing physical validity checks.

    Parameters
    ----------
    _header : Dict
        FITS header or dictionary containing observation information.
        Must include:
        - Coordinates (RA/DEC or OBJRA/OBJDEC)
        - Observation date (DATE-OBS)
    observatory : Dict, optional
        Information about the observatory. If not provided, uses the default (TJMS).
        Format: {
            'name': str,           # Observatory name
            'latitude': float,     # Latitude in degrees
            'longitude': float,    # Longitude in degrees
            'elevation': float     # Elevation in meters
        }
    return_details : bool, optional
        If True, returns additional information about the observation.

    Returns
    -------
    Union[float, Tuple[float, Dict]]
        - If return_details=False: airmass (float)
        - If return_details=True: (airmass, observation_details)
          where observation_details is a dictionary with:
          - observatory name
          - datetime
          - target coordinates
          - altitude/azimuth
          - sun altitude
          - observation type (night/twilight/day)

    Notes
    -----
    Airmass is approximately sec(z) where z is the zenith angle.
    The function enforces physical constraints (airmass ≥ 1.0) and
    displays warnings for extreme values. It also determines whether
    the observation was taken during night, twilight or day.
    """

    def get_observation_type(sun_alt):
        if sun_alt < -18:
            return "night"
        if sun_alt < 0:
            return "twilight"
        return "day"

    obs_data = observatory

    try:
        ra = _header.get("RA") or _header.get("OBJRA") or _header.get("CRVAL1")
        dec = _header.get("DEC") or _header.get("OBJDEC") or _header.get("CRVAL2")
        obstime_str = _header.get("DATE-OBS") or _header.get("DATE")

        if any(v is None for v in [ra, dec, obstime_str]):
            missing = []
            if ra is None:
                missing.append("RA")
            if dec is None:
                missing.append("DEC")
            if obstime_str is None:
                missing.append("DATE-OBS")
            raise KeyError(f"Missing required header keywords: {', '.join(missing)}")

        coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame="icrs")
        obstime = Time(obstime_str)
        location = EarthLocation(
            lat=obs_data["latitude"] * u.deg,
            lon=obs_data["longitude"] * u.deg,
            height=obs_data["elevation"] * u.m,
        )

        altaz_frame = AltAz(obstime=obstime, location=location)
        altaz = coord.transform_to(altaz_frame)
        airmass_value = float(altaz.secz)

        if airmass_value < 1.0:
            st.warning("Calculated airmass is less than 1 (impossible)")
            airmass_value = 1.0
        elif airmass_value > 30.0:
            st.warning("Extremely high airmass (>30), object near horizon")

        sun_altaz = get_sun(obstime).transform_to(altaz_frame)
        sun_alt = float(sun_altaz.alt.deg)

        details = {
            "observatory": obs_data["name"],
            "datetime": obstime.iso,
            "target_coords": {
                "ra": coord.ra.to_string(unit=u.hour),
                "observation_type": get_observation_type(sun_alt),
            },
            "sun_altitude": round(sun_alt, 2),
            "observation_type": get_observation_type(sun_alt),
            "altaz": {
                "altitude": round(float(altaz.alt.deg), 2),
                "azimuth": round(float(altaz.az.deg), 2),
            },
        }

        st.write(f"Date & Local-Time: {obstime.iso}")
        st.write(
            f"Altitude: {details['altaz']['altitude']}°, "
            f" Azimuth: {details['altaz']['azimuth']}°"
        )

        if return_details:
            return round(airmass_value, 2), details
        return round(airmass_value, 2)

    except Exception as e:
        st.warning(f"Error calculating airmass: {str(e)}")
        if return_details:
            return 0.0, {}
        return 0.0


@st.cache_data
def load_fits_data(file):
    """
    Load image data and header from a FITS file with robust error handling.

    Handles multiple FITS formats including multi-extension files, data cubes,
    and RGB images. For multi-dimensional data, extracts a 2D plane.

    Parameters
    ----------
    file : Streamlit UploadedFile
        FITS file object from Streamlit file uploader

    Returns
    -------
    tuple
        (image_data, header) where:
        - image_data: 2D numpy array of pixel values, or None if loading failed
        - header: FITS header dictionary, or None if loading failed

    Notes
    -----
    - For multi-extension FITS files, uses the primary HDU if it contains data,
      otherwise uses the first HDU with valid data.
    - For 3D data (RGB or data cube), extracts the first 2D plane with appropriate warnings.
    - For higher dimensional data, takes the first slice along all extra dimensions.
    """
    if file is not None:
        file_content = file.read()
        hdul = fits.open(BytesIO(file_content), mode="readonly")
        try:
            data = hdul[0].data
            hdul.verify("fix")
            header = hdul[0].header

            if data is None:
                for i, hdu in enumerate(hdul[1:], 1):
                    if hasattr(hdu, "data") and hdu.data is not None:
                        data = hdu.data
                        header = hdu.header
                        st.info(f"Primary HDU has no data. Using data from HDU #{i}.")
                        break

            if data is None:
                st.warning("No image data found in the FITS file.")
                return None, None

            if len(data.shape) == 3:
                if data.shape[0] == 3 or data.shape[0] == 4:
                    st.info(
                        f"Detected RGB data with shape {data.shape}. Using first color channel."
                    )
                    data = data[0]
                elif data.shape[2] == 3 or data.shape[2] == 4:
                    st.info(
                        f"Detected RGB data with shape {data.shape}. Using first color channel."
                    )
                    data = data[:, :, 0]
                else:
                    st.info(
                        f"Detected 3D data with shape {data.shape}. Using first plane."
                    )
                    data = data[0]
            elif len(data.shape) > 3:
                st.warning(
                    f"Data has {len(data.shape)} dimensions. Using first slice only."
                )
                sliced_data = data
                while len(sliced_data.shape) > 2:
                    sliced_data = sliced_data[0]
                data = sliced_data

            norm = ImageNormalize(data, interval=PercentileInterval(99.9))
            normalized_data = norm(data)

            return normalized_data, data, header

        except Exception as e:
            st.error(f"Error loading FITS file: {str(e)}")
            return None, None, None
        finally:
            hdul.close()

    return None, None, None


def fwhm_fit(
    _img: np.ndarray,
    fwhm: float,
    mask: Optional[np.ndarray] = None,
    std_lo: float = 1.0,
    std_hi: float = 1.0,
) -> Optional[float]:
    """Estimate the FWHM from an image using a fitting method.

    (Implementation details are missing in the provided code snippet.)

    Parameters
    ----------
    _img : np.ndarray
        Input image data.
    fwhm : float
        Initial guess for the FWHM.
    mask : Optional[np.ndarray], optional
        Mask to exclude regions from the fit, by default None.
    std_lo : float, optional
        Lower standard deviation threshold, by default 1.0.
    std_hi : float, optional
        Upper standard deviation threshold, by default 1.0.

    Returns
    -------
    Optional[float]
        Estimated FWHM value in pixels, or None if the fit fails.
    """
    # Implementation missing in provided snippet
    # Placeholder: return initial guess or a simple estimate if possible
    st.warning("fwhm_fit function implementation is missing. Returning initial guess.")
    # Example simple estimate (replace with actual implementation):
    try:
        # A very basic estimate - replace with proper fitting
        mean, median, std = sigma_clipped_stats(_img[~mask] if mask is not None else _img)
        # This is NOT a real FWHM calculation, just a placeholder
        estimated_fwhm = 2.355 * std
        return estimated_fwhm
    except Exception:
        return fwhm # Fallback to initial guess


def perform_psf_photometry(
    img: np.ndarray,
    phot_table: Table,
    fwhm: float,
    daostarfind: Any,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Table, Any]:
    """Perform PSF photometry on detected sources.

    (Implementation details are missing in the provided code snippet.)

    Parameters
    ----------
    img : np.ndarray
        Image data.
    phot_table : astropy.table.Table
        Table containing initial source detections (positions).
    fwhm : float
        Estimated FWHM of sources in pixels.
    daostarfind : Any
        DAOStarFinder instance or similar object used for detection.
    mask : Optional[np.ndarray], optional
        Mask to exclude regions, by default None.

    Returns
    -------
    Tuple[astropy.table.Table, Any]
        A tuple containing the photometry results table and potentially
        other PSF-related outputs (e.g., PSF model).
    """
    # Implementation missing in provided snippet
    st.warning("perform_psf_photometry function implementation is missing. Returning None.")
    return None, None


def detection_and_photometry(
    image_data: np.ndarray,
    _science_header: Optional[dict | fits.Header],
    data_not_normalized: Optional[np.ndarray], # Keep for potential future use
    mean_fwhm_pixel: float,
    threshold_sigma: float,
    detection_mask: int,
    filter_band: str, # Keep for potential future use
    gb: str = "Gmag", # Keep for potential future use
) -> Tuple[Optional[Table], Optional[Table], Optional[Background2D], Optional[float]]:
    """Perform source detection and aperture photometry on an image.

    Steps include background estimation, FWHM estimation, source detection
    using DAOStarFinder, optional astrometric refinement, and aperture photometry.

    Parameters
    ----------
    image_data : np.ndarray
        The background-subtracted or processed image data for photometry.
    _science_header : Optional[dict | fits.Header]
        FITS header (unused in current snippet, underscore suggests potential caching).
    data_not_normalized : Optional[np.ndarray]
        Original image data before normalization (unused in current snippet).
    mean_fwhm_pixel : float
        Initial estimate or guess for the FWHM in pixels.
    threshold_sigma : float
        Detection threshold in units of background standard deviation.
    detection_mask : int
        Size of the border mask in pixels to exclude from detection.
    filter_band : str
        Filter band name (unused in current snippet).
    gb : str, optional
        Gaia band identifier (unused in current snippet), by default "Gmag".

    Returns
    -------
    Tuple[Optional[Table], Optional[Table], Optional[Background2D], Optional[float]]
        - Aperture photometry results as an Astropy Table, or None on failure.
        - PSF photometry results (implementation missing), currently None.
        - Background2D object from photutils, or None on failure.
        - Estimated FWHM in pixels, or None on failure.

    Notes
    -----
    - Uses photutils for background estimation and aperture photometry.
    - Uses DAOStarFinder for source detection.
    - Includes basic error handling and logging via Streamlit.
    - PSF photometry part seems incomplete.
    - Relies on `st.session_state` for astrometry check option.
    """
    st.write("Estimating background...")
    bkg, bkg_error = estimate_background(image_data, box_size=128, filter_size=7)
    if bkg is None:
        st.error(f"Background estimation failed: {bkg_error}")
        return None, None, None, None
    st.success("Background estimated.")

    st.write("Estimating FWHM...")
    mask = make_border_mask(image_data, detection_mask)
    fwhm_estimate = fwhm_fit(image_data - bkg.background, mean_fwhm_pixel, mask)

    if fwhm_estimate is None:
        st.warning("FWHM estimation failed, using initial guess.")
        fwhm_estimate = mean_fwhm_pixel # Fallback
    else:
        st.success(f"Estimated FWHM: {fwhm_estimate:.2f} pixels")


    st.write("Detecting sources...")
    try:
        mean, median, std = sigma_clipped_stats(image_data, sigma=3.0, mask=mask)
        daofind = DAOStarFinder(fwhm=fwhm_estimate, threshold=threshold_sigma * std)
        sources = daofind(image_data - bkg.background, mask=mask)
    except Exception as e:
        st.error(f"Source detection failed: {e}")
        return None, None, bkg, fwhm_estimate

    if sources is None or len(sources) == 0:
        st.warning("No sources detected.")
        return None, None, bkg, fwhm_estimate
    st.success(f"Detected {len(sources)} sources.")

    # Optional Astrometry Check (using session state)
    if st.session_state.get("analysis_parameters", {}).get("astrometry_check", False):
        st.write("Performing astrometry check/refinement...")
        # Placeholder for astrometry check logic using _science_header and sources
        # This might involve calling solve_with_siril or another astrometry tool
        # and potentially updating the WCS or source positions.
        st.info("Astrometry check/refinement step is currently a placeholder.")


    st.write("Performing aperture photometry...")
    try:
        positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
        apertures = CircularAperture(positions, r=2.0 * fwhm_estimate) # Aperture radius based on FWHM
        phot_table = aperture_photometry(image_data - bkg.background, apertures, error=bkg.background_rms)

        # Calculate instrumental magnitude
        # Avoid log(0) or log(<0) errors
        valid_flux = phot_table['aperture_sum'] > 0
        phot_table['instrumental_mag'] = np.nan # Initialize column
        phot_table['instrumental_mag'][valid_flux] = -2.5 * np.log10(phot_table['aperture_sum'][valid_flux])

        st.success("Aperture photometry complete.")

        # Placeholder for PSF photometry call
        st.write("Performing PSF photometry (placeholder)...")
        epsf_table, epsf_model = perform_psf_photometry(
            image_data - bkg.background, phot_table, fwhm_estimate, daofind, mask
        )
        if epsf_table is not None:
             st.success("PSF photometry complete.")
             st.session_state['epsf_model'] = epsf_model # Store PSF model if created
             st.session_state['epsf_photometry_result'] = epsf_table # Store PSF results
        else:
             st.warning("PSF photometry skipped or failed.")


        return phot_table, epsf_table, bkg, fwhm_estimate
    except Exception as e:
        st.error(f"Aperture photometry failed: {e}")
        return None, None, bkg, fwhm_estimate


def cross_match_with_gaia(
    _phot_table: Table,
    _science_header: Optional[dict | fits.Header],
    pixel_size_arcsec: float,
    mean_fwhm_pixel: float,
    filter_band: str,
    filter_max_mag: float,
) -> Optional[pd.DataFrame]:
    """
    Cross-match detected sources with the Gaia DR3 star catalog.

    Queries Gaia for the field of view defined by the WCS in the header,
    filters the Gaia catalog based on the specified magnitude band and limit,
    and matches detected sources to Gaia sources based on celestial coordinates.

    Parameters
    ----------
    _phot_table : astropy.table.Table
        Table containing detected source positions ('xcentroid', 'ycentroid')
        and instrumental magnitudes ('instrumental_mag').
        (Underscore suggests potential caching).
    _science_header : Optional[dict | fits.Header]
        FITS header containing WCS information needed to convert pixel coordinates
        to celestial coordinates (RA, Dec). (Underscore suggests potential caching).
    pixel_size_arcsec : float
        Pixel scale in arcseconds per pixel (unused in current snippet, but potentially
        useful for calculating search radius if WCS is missing).
    mean_fwhm_pixel : float
        Mean FWHM of sources in pixels. Used to determine the matching radius
        (currently set to 2 * FWHM in arcseconds).
    filter_band : str
        Gaia magnitude band to use for filtering (e.g., 'phot_g_mean_mag').
    filter_max_mag : float
        Maximum (faintest) magnitude to include from the Gaia catalog.

    Returns
    -------
    Optional[pd.DataFrame]
        A pandas DataFrame containing the matched sources. Includes columns from
        both the input `_phot_table` and the Gaia catalog for the matched stars.
        Returns None if WCS creation fails, the Gaia query fails, or no matches
        are found.

    Notes
    -----
    - Requires a valid WCS in `_science_header` to work correctly.
    - Uses `astroquery.gaia` to query the Gaia archive.
    - Matching is performed using `astropy.coordinates.match_to_catalog_sky`.
    - The maximum separation for a match is `2 * mean_fwhm_pixel * pixel_size_arcsec` arcseconds.
    - Includes basic filtering for Gaia sources (magnitude limit, non-variable).
    - Displays progress and errors using Streamlit (`st.write`, `st.error`).
    """
    st.write("Cross-matching with Gaia DR3...")

    if _science_header is None:
        st.error("Cannot cross-match: FITS header is missing.")
        return None
    if _phot_table is None or len(_phot_table) == 0:
        st.error("Cannot cross-match: Input photometry table is empty.")
        return None
    if 'xcentroid' not in _phot_table.colnames or 'ycentroid' not in _phot_table.colnames:
        st.error("Cannot cross-match: Photometry table missing 'xcentroid' or 'ycentroid'.")
        return None

    # Get WCS
    wcs_obj, wcs_error = safe_wcs_create(_science_header)
    if wcs_obj is None:
        st.error(f"Cannot cross-match: Failed to create WCS object - {wcs_error}")
        return None

    # Get field center and radius for Gaia query
    try:
        center_coord = wcs_obj.pixel_to_world(
            _science_header['NAXIS1'] / 2, _science_header['NAXIS2'] / 2
        )
        # Estimate radius (diagonal semi-major axis)
        corner_coord = wcs_obj.pixel_to_world(0, 0)
        radius = center_coord.separation(corner_coord)
    except Exception as e:
        st.error(f"Cannot determine field center/radius from WCS: {e}")
        return None

    # Query Gaia
    st.write(f"Querying Gaia around RA={center_coord.ra.deg:.4f}, Dec={center_coord.dec.deg:.4f} with radius={radius.arcmin:.2f} arcmin...")
    try:
        gaia_table = safe_catalog_query(
            center_coord,
            radius,
            catalog_name="Gaia",
            band=filter_band,
            mag_limit=filter_max_mag,
        )
    except Exception as e:
        st.error(f"Gaia query failed: {e}")
        return None

    if gaia_table is None or len(gaia_table) == 0:
        st.warning("No Gaia sources found in the field of view matching criteria.")
        return None
    st.success(f"Found {len(gaia_table)} Gaia sources.")

    # Perform cross-matching
    st.write("Matching detected sources to Gaia catalog...")
    try:
        # Convert pixel coordinates to SkyCoord
        source_coords = wcs_obj.pixel_to_world(_phot_table['xcentroid'], _phot_table['ycentroid'])
        gaia_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit=(u.deg, u.deg))

        # Perform cross-matching
        max_separation = 2 * mean_fwhm_pixel * pixel_size_arcsec * u.arcsec
        st.write(f"Using matching radius: {max_separation:.2f}")
        idx, d2d, _ = source_coords.match_to_catalog_sky(gaia_coords)
        sep_constraint = d2d < max_separation

        matched_phot = _phot_table[sep_constraint]
        matched_gaia = gaia_table[idx[sep_constraint]]

        if len(matched_phot) == 0:
            st.warning("No matches found between detected sources and Gaia catalog within the specified radius.")
            return None

        # Combine matched tables
        matched_phot_df = matched_phot.to_pandas()
        matched_gaia_df = matched_gaia.to_pandas()

        # Ensure unique columns before merging, prefix Gaia columns
        gaia_cols_renamed = {col: f"gaia_{col}" for col in matched_gaia_df.columns if col in matched_phot_df.columns and col != 'index'} # Avoid renaming index if it exists
        matched_gaia_df.rename(columns=gaia_cols_renamed, inplace=True)

        # Reset index for safe concatenation
        matched_phot_df.reset_index(drop=True, inplace=True)
        matched_gaia_df.reset_index(drop=True, inplace=True)

        combined_table = pd.concat([matched_phot_df, matched_gaia_df], axis=1)

        # Add separation distance
        combined_table['match_separation_arcsec'] = d2d[sep_constraint].to(u.arcsec).value

        st.success(f"Found {len(combined_table)} matches with Gaia.")
        return combined_table

    except Exception as e:
        st.error(f"Error during cross-matching: {e}")
        return None


def calculate_zero_point(
    _phot_table: Union[Table, pd.DataFrame],
    _matched_table: Optional[pd.DataFrame],
    filter_band: str,
    air: float
) -> Tuple[Optional[float], Optional[float], Optional[plt.Figure]]:
    """
    Calculate the photometric zero point using cross-matched Gaia sources.

    Determines the offset between instrumental magnitudes and standard Gaia
    magnitudes for a given filter band, accounting for airmass.

    Parameters
    ----------
    _phot_table : astropy.table.Table or pd.DataFrame
        Photometry results table containing instrumental magnitudes
        (e.g., 'instrumental_mag'). (Underscore suggests potential caching).
    _matched_table : Optional[pd.DataFrame]
        DataFrame of sources cross-matched with Gaia, containing both
        instrumental magnitudes and Gaia standard magnitudes (e.g., 'phot_g_mean_mag').
        Must contain the `filter_band` column and the instrumental magnitude column.
        (Underscore suggests potential caching).
    filter_band : str
        The Gaia magnitude band used for calibration (e.g., 'phot_g_mean_mag').
        This column must exist in `_matched_table`.
    air : float
        Airmass of the observation, used for a simple extinction correction (0.1 * air).

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[plt.Figure]]
        - zero_point_value: The calculated median zero point, or None if calculation fails.
        - zero_point_std: The standard deviation of the zero point calculation, or None.
        - matplotlib_figure: A plot comparing Gaia magnitudes to calibrated
          instrumental magnitudes, or None if plotting fails.

    Notes
    -----
    - Requires `_matched_table` to be not None and non-empty.
    - Assumes the instrumental magnitude column in `_phot_table` and `_matched_table`
      is named 'instrumental_mag'.
    - Uses sigma clipping (`astropy.stats.sigma_clipped_stats`) to robustly calculate
      the median zero point and its standard deviation.
    - Applies a simple atmospheric extinction correction: `calib_mag = inst_mag + zp + 0.1 * air`.
    - Updates `_phot_table` with a 'calib_mag' column.
    - Stores the updated `_phot_table` in `st.session_state['final_phot_table']`.
    - Generates a plot showing Gaia magnitude vs. calibrated magnitude, including binned statistics.
    - Displays results and errors using Streamlit.
    """
    if _matched_table is None or len(_matched_table) == 0:
        st.error("Cannot calculate zero point: No matched Gaia sources provided.")
        return None, None, None

    # Find the Gaia magnitude column (prefixed during crossmatch)
    gaia_mag_col = f"gaia_{filter_band}"
    if gaia_mag_col not in _matched_table.columns:
         # Check if original name exists (maybe crossmatch failed to rename?)
         if filter_band in _matched_table.columns:
             gaia_mag_col = filter_band
             st.warning(f"Using non-prefixed Gaia column '{filter_band}'. Cross-match might have had issues.")
         else:
             st.error(f"Cannot calculate zero point: Gaia filter band column '{gaia_mag_col}' or '{filter_band}' missing from matched table.")
             return None, None, None

    # Find instrumental magnitude column
    if 'instrumental_mag' not in _matched_table.columns:
        # Try to find another mag column if 'instrumental_mag' is missing
        mag_cols = [col for col in _matched_table.columns if 'mag' in col.lower() and 'gaia' not in col.lower() and 'calib' not in col.lower()]
        if not mag_cols:
            st.error(f"Cannot calculate zero point: Instrumental magnitude column ('instrumental_mag' or similar) missing from matched table.")
            return None, None, None
        inst_mag_col = mag_cols[0]
        st.warning(f"Using '{inst_mag_col}' as instrumental magnitude column.")
    else:
        inst_mag_col = 'instrumental_mag'


    try:
        # Calculate the difference: Gaia_mag - instrumental_mag
        # This difference represents the zero point (ignoring extinction for now)
        zp_diff = _matched_table[gaia_mag_col] - _matched_table[inst_mag_col]

        # Filter out NaNs or Infs before sigma clipping
        valid_zp_diff = zp_diff[np.isfinite(zp_diff)]
        if len(valid_zp_diff) < 3: # Need at least 3 points for robust stats
             st.error("Not enough valid magnitude difference values to calculate zero point.")
             return None, None, None

        # Use sigma clipping to get a robust estimate of the zero point
        mean_zp, median_zp, std_zp = sigma_clipped_stats(valid_zp_diff, sigma=3.0)

        zero_point_value = median_zp # Use median as the robust zero point
        zero_point_std = std_zp

        # Apply zero point and extinction correction to the full photometry table
        # Ensure _phot_table is a DataFrame for easier column addition
        if isinstance(_phot_table, Table):
            _phot_table_df = _phot_table.to_pandas()
        elif isinstance(_phot_table, pd.DataFrame):
            _phot_table_df = _phot_table.copy() # Avoid modifying original if it's passed around
        else:
            st.error("Input _phot_table must be an Astropy Table or Pandas DataFrame.")
            return None, None, None

        # Find instrumental magnitude in the full table
        if inst_mag_col not in _phot_table_df.columns:
             # Try to find it again if it wasn't 'instrumental_mag'
             mag_cols_full = [col for col in _phot_table_df.columns if 'mag' in col.lower() and 'calib' not in col.lower()]
             if not mag_cols_full:
                 st.error(f"Cannot apply calibration: Instrumental magnitude column '{inst_mag_col}' or similar missing from full photometry table.")
                 return None, None, None
             inst_mag_col_full = mag_cols_full[0]
             st.warning(f"Using '{inst_mag_col_full}' as instrumental magnitude in full table.")
        else:
            inst_mag_col_full = inst_mag_col

        _phot_table_df["calib_mag"] = (
            _phot_table_df[inst_mag_col_full] + zero_point_value + 0.1 * air
        )
        _phot_table_df["calib_mag_err"] = np.nan # Placeholder for error propagation

        # Update the matched table as well for plotting
        _matched_table_plot = _matched_table.copy()
        _matched_table_plot["calib_mag"] = (
             _matched_table_plot[inst_mag_col] + zero_point_value + 0.1 * air
        )

        st.session_state["final_phot_table"] = _phot_table_df # Store the calibrated table

        # --- Plotting --- #
        fig = plt.figure(figsize=FIGURE_SIZES.get("medium", (10, 6)), dpi=100) # Use plt.figure
        ax = fig.add_subplot(111)

        # Calculate residuals for plotting
        _matched_table_plot["residual"] = (
            _matched_table_plot[gaia_mag_col] - _matched_table_plot["calib_mag"]
        )

        # Create bins for magnitude ranges
        bin_width = 0.5
        min_mag = _matched_table_plot[gaia_mag_col].min()
        max_mag = _matched_table_plot[gaia_mag_col].max()
        if pd.isna(min_mag) or pd.isna(max_mag) or min_mag == max_mag:
             st.warning("Could not determine valid magnitude range for plotting bins.")
             bins = np.array([])
        else:
             # Ensure bins cover the range, handle edge cases
             start_bin = np.floor(min_mag / bin_width) * bin_width
             end_bin = np.ceil(max_mag / bin_width) * bin_width
             bins = np.arange(start_bin, end_bin + bin_width, bin_width)


        # Group data by magnitude bins if bins exist
        if len(bins) > 1:
            try:
                # Use the Gaia magnitude for binning
                grouped = _matched_table_plot.groupby(pd.cut(_matched_table_plot[gaia_mag_col], bins, right=False)) # Use gaia mag for binning
                bin_centers = [(bin.left + bin.right) / 2 for bin in grouped.groups.keys()]
                # Calculate stats on the *calibrated* magnitude within each bin
                bin_means = grouped["calib_mag"].mean().values
                bin_stds = grouped["calib_mag"].std().values
                valid_bins = ~np.isnan(bin_means) & ~np.isnan(bin_stds) & (grouped.size() > 0) # Ensure bins are not empty
            except Exception as plot_err:
                 st.warning(f"Could not group data for binned plot: {plot_err}")
                 valid_bins = np.array([], dtype=bool)
                 bin_centers = []
                 bin_means = []
                 bin_stds = []

        else:
            valid_bins = np.array([], dtype=bool)
            bin_centers = []
            bin_means = []
            bin_stds = []

        # Plot individual points
        ax.scatter(
            _matched_table_plot[gaia_mag_col],
            _matched_table_plot["calib_mag"],
            alpha=0.5,
            label="Matched sources",
            color="blue",
            s=10 # Smaller points
        )

        # Plot binned means with error bars
        if np.any(valid_bins):
            ax.errorbar(
                np.array(bin_centers)[valid_bins],
                bin_means[valid_bins],
                yerr=bin_stds[valid_bins],
                fmt="ro-",
                label="Mean Calib Mag ± StdDev (binned by Gaia Mag)",
                capsize=5,
            )

        # Add a diagonal line for reference (y=x)
        if not pd.isna(min_mag) and not pd.isna(max_mag):
            plot_min = min(min_mag, _matched_table_plot["calib_mag"].min())
            plot_max = max(max_mag, _matched_table_plot["calib_mag"].max())
            if pd.isfinite(plot_min) and pd.isfinite(plot_max):
                ideal_mag = np.linspace(plot_min, plot_max, 10)
                ax.plot(ideal_mag, ideal_mag, "k--", alpha=0.7, label="Gaia Mag = Calib Mag")

        ax.set_xlabel(f"Gaia {filter_band}")
        ax.set_ylabel("Calibrated Instrumental Mag")
        ax.set_title(f"Zero Point Calibration (ZP = {zero_point_value:.2f} ± {zero_point_std:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Invert y-axis for magnitudes
        ax.invert_yaxis()
        ax.invert_xaxis() # Also invert x-axis for consistency

        st.success(
            f"Calculated Zero Point: {zero_point_value:.2f} ± {zero_point_std:.2f} using {len(valid_zp_diff)} stars."
        )

        # Save the plot
        try:
            output_dir = st.session_state.get("output_dir", ".")
            base_filename = st.session_state.get("base_filename", "photometry")
            plot_filename = os.path.join(output_dir, f"{base_filename}_zero_point_plot.png")
            fig.savefig(plot_filename, bbox_inches="tight")
            st.image(plot_filename, caption="Zero Point Calibration Plot")
            write_to_log(st.session_state.get("log_buffer"), f"Saved zero point plot to {plot_filename}")
        except Exception as e:
            st.warning(f"Could not save zero point plot: {e}")

        return zero_point_value, zero_point_std, fig

    except Exception as e:
        st.error(f"Error calculating zero point: {e}")
        import traceback
        st.error(traceback.format_exc()) # Print traceback for debugging
        return None, None, None


def run_zero_point_calibration(
    image_to_process: np.ndarray,
    header: Optional[dict | fits.Header],
    pixel_size_arcsec: float,
    mean_fwhm_pixel: float,
    threshold_sigma: float,
    detection_mask: int,
    filter_band: str,
    filter_max_mag: float,
    air: float,
) -> Tuple[Optional[float], Optional[float], Optional[pd.DataFrame]]:
    """
    Orchestrate the full photometric calibration pipeline.

    This function executes the sequence:
    1. Source detection and aperture photometry.
    2. Cross-matching detected sources with the Gaia catalog.
    3. Calculating the photometric zero point using the matched stars.

    Parameters
    ----------
    image_to_process : np.ndarray
        The image data array ready for analysis.
    header : Optional[dict | fits.Header]
        FITS header containing WCS and other metadata.
    pixel_size_arcsec : float
        Pixel scale of the image in arcseconds per pixel.
    mean_fwhm_pixel : float
        Estimated or initial guess for the FWHM in pixels.
    threshold_sigma : float
        Detection threshold in sigma units above the background noise.
    detection_mask : int
        Width of the border mask in pixels to exclude from detection.
    filter_band : str
        Gaia magnitude band to use for calibration (e.g., 'phot_g_mean_mag').
    filter_max_mag : float
        Maximum (faintest) magnitude for Gaia sources used in calibration.
    air : float
        Airmass of the observation for extinction correction.

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[pd.DataFrame]]
        - zero_point_value: The calculated zero point, or None on failure.
        - zero_point_std: The standard deviation of the zero point, or None.
        - final_phot_table: A pandas DataFrame containing the full photometry
          catalog with calibrated magnitudes, or None if the process fails early.
          This table is also stored in `st.session_state['final_phot_table']`.

    Notes
    -----
    - Manages the workflow using Streamlit spinners and status messages.
    - Calls `detection_and_photometry`, `cross_match_with_gaia`, and `calculate_zero_point`.
    - Handles potential failures at each step and returns None values accordingly.
    - Displays intermediate results (matched table head) in the Streamlit interface.
    """
    zero_point_value = None
    zero_point_std = None
    final_phot_table = None # Initialize

    with st.spinner("Finding sources and performing photometry..."):
        # Assuming data_not_normalized and gb are not strictly needed based on usage
        # Pass None for data_not_normalized as it wasn't used previously
        phot_table_qtable, epsf_table, _, fwhm_estimate = detection_and_photometry(
            image_to_process,
            header,
            None, # data_not_normalized placeholder
            mean_fwhm_pixel,
            threshold_sigma,
            detection_mask,
            filter_band, # Pass filter_band, though not used in current detection func
        )

        if phot_table_qtable is None:
            st.error("Photometry failed. Cannot proceed with calibration.")
            return None, None, None
        else:
            st.success("Photometry complete.")
            # Update FWHM if it was estimated
            if fwhm_estimate is not None and fwhm_estimate != mean_fwhm_pixel:
                 st.info(f"Using estimated FWHM ({fwhm_estimate:.2f} px) for cross-matching.")
                 mean_fwhm_pixel = fwhm_estimate # Use the estimated FWHM going forward

    with st.spinner("Cross-matching with Gaia..."):
        matched_table = cross_match_with_gaia(
            phot_table_qtable,
            header,
            pixel_size_arcsec,
            mean_fwhm_pixel, # Use potentially updated FWHM
            filter_band,
            filter_max_mag,
        )

    if matched_table is None:
        st.error("Cross-matching failed. Cannot proceed with calibration.")
        return None, None, st.session_state.get("final_phot_table") # Return potentially uncalibrated table

    st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
    st.dataframe(matched_table.head(10))

    with st.spinner("Calculating zero point..."):
        zero_point_value, zero_point_std, _ = calculate_zero_point(
            phot_table_qtable, # Pass the original table for calibration
            matched_table,
            filter_band,
            air,
        )

    if zero_point_value is None:
        st.error("Zero point calculation failed.")
        # Return the uncalibrated table stored during ZP calculation attempt
        final_phot_table = st.session_state.get("final_phot_table")
    else:
        st.success("Zero point calibration complete.")
        # Retrieve the calibrated table stored by calculate_zero_point
        final_phot_table = st.session_state.get("final_phot_table")

    return zero_point_value, zero_point_std, final_phot_table


def enhance_catalog(
    api_key: Optional[str],
    final_table: Optional[pd.DataFrame],
    matched_table: Optional[pd.DataFrame],
    header: Optional[dict | fits.Header],
    pixel_scale_arcsec: Optional[float],
    search_radius_arcsec: float = 60.0,
) -> Optional[pd.DataFrame]:
    """
    Enhance the final photometry catalog with additional information.

    (Currently a placeholder - implementation details missing)

    Parameters
    ----------
    api_key : Optional[str]
        API key for external services (e.g., Colibri).
    final_table : Optional[pd.DataFrame]
        The final photometry table (potentially calibrated).
    matched_table : Optional[pd.DataFrame]
        Table of sources matched with Gaia.
    header : Optional[dict | fits.Header]
        FITS header.
    pixel_scale_arcsec : Optional[float]
        Pixel scale.
    search_radius_arcsec : float, optional
        Search radius for external catalogs, by default 60.0.

    Returns
    -------
    Optional[pd.DataFrame]
        The enhanced DataFrame, or the original if enhancement fails or is skipped.
    """
    st.info("Catalog enhancement step is currently a placeholder.")
    # Potential enhancements:
    # - Query Simbad/Vizier for known objects near detected sources
    # - Add flags for variability, galaxy contamination etc. based on Gaia data
    # - Query specific catalogs (e.g., GRB catalogs using Colibri API if api_key provided)
    # - Calculate additional parameters (e.g., ellipticity)

    if final_table is None:
        st.warning("Cannot enhance catalog: final_table is None.")
        return None

    # Example: Add Simbad object type if matched
    if matched_table is not None and 'gaia_source_id' in matched_table.columns and 'main_id' in matched_table.columns:
         # Merge Simbad info based on Gaia ID or coordinates if available in matched_table
         # This requires matched_table to have Simbad query results included earlier
         # or performing a new Simbad query here based on coordinates.
         st.write("Adding Simbad info (placeholder)...")
         # final_table = pd.merge(final_table, matched_table[['gaia_source_id', 'simbad_otype']], on='gaia_source_id', how='left')


    return final_table # Return table, possibly unmodified
