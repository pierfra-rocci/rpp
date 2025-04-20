import sys
import subprocess

if getattr(sys, "frozen", False):
    import importlib.metadata
    importlib.metadata.distributions = lambda **kwargs: []

import os
import zipfile
from datetime import datetime, timedelta
import base64
import json
import requests
import tempfile
from urllib.parse import quote

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun
from typing import Union, Any, Optional, Dict, Tuple

from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from astropy.modeling import models, fitting
import streamlit as st
import streamlit.components.v1 as components

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, SigmaClip

import astropy.units as u
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, SExtractorBackground
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.visualization import ZScaleInterval, ImageNormalize, simple_norm
from io import StringIO, BytesIO
from astropy.wcs import WCS

from photutils.psf import EPSFBuilder, extract_stars, IterativePSFPhotometry
from astropy.nddata import NDData

from stdpipe import photometry, astrometry, catalogs, pipeline

from __version__ import version

import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Photometry Factory for RAPAS", page_icon="🔭", layout="wide"
)

# Add application version to the sidebar
st.sidebar.markdown(f"**App Version:** {version}")

# Custom CSS to control plot display size
st.markdown(
    """
<style>
    .stPlot > div {
        display: flex;
        justify-content: center;
        min-height: 400px;
    }
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .element-container img {
        max-width: 100% !important;
        height: auto !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

FIGURE_SIZES = {
    "small": (6, 5),  # For small plots
    "medium": (8, 6),  # For medium plots
    "large": (10, 8),  # For large plots
    "wide": (12, 6),  # For wide plots
    "stars_grid": (10, 8),  # For grid of stars
}

URL = "https://astro-colibri.science"


def getJson(url: str) -> json:
    """
    Fetch JSON data from a given URL and handle errors gracefully.

    Parameters
    ----------
    url : str
        The URL to fetch JSON data from

    Returns
    -------
    json
        Parsed JSON data if successful, or error message in JSON format if failed

    Notes
    -----
    - Handles network errors and JSON parsing errors.
    - Returns an error message in JSON format if the request fails or the response is empty.
    """
    if not url.startswith("http"):
        return json.dumps({"error": "invalid URL"})
    try:
        # Send a GET request to the provided URL using the requests library
        req = requests.get(url)
        # Raise an error if the request was not successful
        req.raise_for_status()
        # Check if the response has any content
        if not req.content:
            # If the response is empty, return an error message as JSON
            return json.dumps({"error": "empty response"})
        # If the response has content, parse it as JSON and return it
        return req.json()
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": "request exception", "message": str(e)})
    except json.decoder.JSONDecodeError as e:
        return json.dumps({"error": "invalid json", "message": str(e)})


def solve_with_siril(file_path, header=None):
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
                "-ExecutionPolicy", "Bypass",
                "-File", "plate_solve.ps1",
                "-filepath",
                f"{file_path}"
            ]
            subprocess.run(command, check=True)
        else:
            st.warning(f"File {file_path} does not exist.")

    except Exception as e:
        st.error(f"Error solving with Siril: {str(e)}")
        return None
    try:
        file_path_list = file_path.split(".")
        file_path = file_path_list[0]+"_solved.fits"
        hdu = fits.open(file_path, mode="readonly")
        head = hdu[0].header
        wcs_obj = WCS(head)

        return wcs_obj, head
    
    except Exception as e:
        st.error(f"Error reading solved file: {str(e)}")


def ensure_output_directory(directory="pfr_results"):
    """
    Create an output directory if it doesn't exist.

    Parameters
    ----------
    directory : str, optional
        Path to the directory to create. Default is "pfr_results".

    Returns
    -------
    str
        Path to the created/existing output directory, or "." (curr directory)
        if creation failed

    Notes
    -----
    Will attempt to create the directory and display a warning in Streamlit
    if creation fails. In case of failure, returns the current directory.
    """
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return directory
        except Exception as e:
            st.warning(f"Could not create directory '{directory}': {e}")
            return "."
    return directory


def safe_wcs_create(header):
    """
    Create a WCS (World Coordinate System) object from a FITS header with error handling.

    This function validates the header contains required WCS keywords before
    create a WCS object, and properly handles various edge cases and errors.

    Parameters
    ----------
    header : dict or astropy.io.fits.Header
        FITS header containing WCS information

    Returns
    -------
    tuple
        (wcs_object, None) if successful, (None, error_message) if failed :
        - wcs_object: astropy.wcs.WCS object
        - error_message: String describing the error if WCS creation failed

    Notes
    -----
    The function checks for required WCS keywords and validates the resulting
    WCS object. For higher dimensional data, it will reduce the WCS to celestial
    coordinates only.
    """
    if not header:
        return None, "No header provided"

    required_keys = ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2"]
    missing_keys = [key for key in required_keys if key not in header]

    try:
        header.remove('XPIXELSZ')
        header.remove('YPIXELSZ')
        header.remove('CDELTM1')
        header.remove('CDELTM2')
    except KeyError:
        pass

    if missing_keys:
        return None, f"Missing required WCS keywords: {', '.join(missing_keys)}"

    try:
        wcs_obj = WCS(header)

        if wcs_obj is None:
            return None, "WCS creation returned None"

        if wcs_obj.pixel_n_dim > 2:
            wcs_obj = wcs_obj.celestial

        if not hasattr(wcs_obj, "wcs"):
            return None, "Created WCS object has no transformation attributes"

        return wcs_obj, None
    except Exception as e:
        return None, f"WCS creation error: {str(e)}"


def estimate_background(image_data, box_size=100, filter_size=5):
    """
    Estimate the background and background RMS of an astronomical image.

    Uses photutils.Background2D to create a 2D background model with sigma-clipping
    and the SExtractor background estimation algorithm. Includes error handling
    and automatic adjustment for small images.

    Parameters
    ----------
    image_data : numpy.ndarray
        The 2D science image array
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

        return bkg, None
    except Exception as e:
        return None, f"Background estimation error: {str(e)}"


def get_header_value(header, keys, default=None):
    """
    Extract a value from a FITS header by trying multiple possible keywords.

    This is useful for handling FITS files with different keyword conventions.
    The function tries each key in the provided list in order of preference.

    Parameters
    ----------
    header : dict or astropy.io.fits.Header
        The FITS header dictionary
    keys : list of str
        List of possible header keywords to try in order of preference
    default : any, optional
        Default value to return if none of the keys are found

    Returns
    -------
    any
        The header value corresponding to the first found key,
        or the default value if no keys are found

    Examples
    --------
    >>> # Try different exposure time keywords
    >>> exposure = get_header_value(header, ['EXPTIME', 'EXPOSURE', 'EXP'], 0.0)
    """
    if header is None:
        return default

    for key in keys:
        if key in header:
            return header[key]
    return default


def extract_pixel_scale(header):
    """
    Extract the pixel scale (arcsec/pixel) from a FITS header.

    Tries multiple approaches to determine the pixel scale:
    1. Direct pixel scale keywords (PIXSIZE, PIXSCALE, etc.)
    2. WCS CDELT keywords
    3. Calculation from physical pixel size and focal length

    Parameters
    ----------
    header : dict or astropy.io.fits.Header
        FITS header with metadata

    Returns
    -------
    tuple
        (pixel_scale_value, source_description) where:
        - pixel_scale_value: Float value of pixel scale in arcseconds per pixel
        - source_description: String describing how the value was determined

    Notes
    -----
    Returns a default value of 1.0 arcsec/pixel if no pixel scale information
    can be determined from the header.
    """
    if header is None:
        return 1.0, "default (no header)"

    for key in ["PIXSIZE", "PIXSCALE", "PIXELSCAL"]:
        if key in header:
            return header[key], f"from {key}"

    for key in ["CDELT2", "CDELT1"]:
        if key in header:
            scale = abs(header[key]) * 3600.0
            return scale, f"from {key}"

    if "XPIXSZ" in header:
        if "FOCALLEN" in header:
            focal_length_mm = header["FOCALLEN"]
            pixel_size = header["XPIXSZ"]

            xpixsz_unit = header.get("XPIXSZU", "").strip().lower()

            if xpixsz_unit in ["arcsec", "as"]:
                return pixel_size, "from XPIXSZ (in arcsec)"
            elif xpixsz_unit == "mm":
                scale = (pixel_size * 1000) / focal_length_mm * 206.265
                return scale, "calculated from XPIXSZ (mm) and FOCALLEN"
            else:
                scale = pixel_size / focal_length_mm * 206.265
                return scale, "calculated from XPIXSZ (μm) and FOCALLEN"
        else:
            return header["XPIXSZ"], "from XPIXSZ (assumed arcsec)"

    return 1.0, "default fallback value"


def extract_coordinates(header):
    """
    Extract celestial coordinates (RA/Dec) from a FITS header.

    Tries multiple possible coordinate keywords and validates the values
    are within reasonable ranges.

    Parameters
    ----------
    header : dict or astropy.io.fits.Header
        FITS header containing coordinate information

    Returns
    -------
    tuple
        (ra, dec, source_description) where:
        - ra: Right Ascension value in degrees or None if not found/invalid
        - dec: Declination value in degrees or None if not found/invalid
        - source_description: String describing the source keywords or error message

    Notes
    -----
    Coordinates are validated to ensure RA is between -360 and 360 degrees
    and DEC is between -90 and 90 degrees. Values outside these ranges will
    be rejected with an appropriate error message.
    """
    if header is None:
        return None, None, "No header available"

    ra_keys = ["RA", "OBJRA", "RA---", "CRVAL1"]
    dec_keys = ["DEC", "OBJDEC", "DEC---", "CRVAL2"]

    ra = get_header_value(header, ra_keys)
    dec = get_header_value(header, dec_keys)

    if ra is not None and dec is not None:
        ra_source = next((k for k in ra_keys if k in header), "unknown")
        dec_source = next((k for k in dec_keys if k in header), "unknown")
        source = f"{ra_source}/{dec_source}"

        try:
            ra_val = float(ra)
            dec_val = float(dec)

            if not (-360 <= ra_val <= 360):
                return None, None, f"Invalid RA value: {ra_val}"
            if not (-90 <= dec_val <= 90):
                return None, None, f"Invalid DEC value: {dec_val}"

            return ra_val, dec_val, source
        except (ValueError, TypeError):
            return None, None, f"Non-numeric coordinates: RA={ra}, DEC={dec}"

    return None, None, "Coordinates not found in header"


def safe_catalog_query(query_func, error_msg, *args, **kwargs):
    """
    Execute an astronomical catalog query with comprehensive error handling.

    This function wraps catalog query functions (like those from astroquery)
    with standardized error handling to catch network issues, timeouts,
    and other common problems when querying online services.

    Parameters
    ----------
    query_func : callable
        The catalog query function to call
    error_msg : str
        Base error message to prepend to any caught exception message
    *args, **kwargs
        Arguments to pass through to query_func

    Returns
    -------
    tuple
        (result, error_message) where:
        - result: Query result object if successful, None if failed
        - error_message: None if successful, string describing the error if failed

    Examples
    --------
    >>> from astroquery.simbad import Simbad
    >>> result, error = safe_catalog_query(
    ...     Simbad.query_object,
    ...     "Failed to query SIMBAD",
    ...     "M31"
    ... )
    >>> if error:
    ...     print(f"Query failed: {error}")
    >>> else:
    ...     print(result)
    """
    try:
        result = query_func(*args, **kwargs)
        return result, None
    except requests.exceptions.RequestException as e:
        return None, f"{error_msg}: Network error - {str(e)}"
    except requests.exceptions.Timeout:
        return None, f"{error_msg}: Query timed out"
    except ValueError as e:
        return None, f"{error_msg}: Value error - {str(e)}"
    except Exception as e:
        return None, f"{error_msg}: {str(e)}"


def create_figure(size="medium", dpi=100):
    """
    Create a matplotlib figure with standardized size for consistent visualizations.

    Parameters
    ----------
    size : str, optional
        Predefined size key: 'small', 'medium', 'large', 'wide', or 'stars_grid'.
        The sizes are defined in the FIGURE_SIZES global dictionary.
    dpi : int, optional
        Dots per inch (resolution) for the figure

    Returns
    -------
    matplotlib.figure.Figure
        Figure object with the specified dimensions

    Notes
    -----
    Uses the FIGURE_SIZES global dictionary to map size names to actual dimensions.
    If an invalid size is provided, falls back to 'medium'.
    """
    if size in FIGURE_SIZES:
        figsize = FIGURE_SIZES[size]
    else:
        figsize = FIGURE_SIZES["medium"]
    return plt.figure(figsize=figsize, dpi=dpi)


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
    DEFAULT_OBSERVATORY = {
        "name": "TJMS",
        "latitude": 48.29166,
        "longitude": 2.43805,
        "elevation": 94.0,
    }

    obs_data = observatory or DEFAULT_OBSERVATORY

    try:
        ra = _header.get("RA", _header.get("OBJRA", _header.get("RA---")))
        dec = _header.get("DEC", _header.get("OBJDEC", _header.get("DEC---")))
        obstime_str = _header.get("DATE-OBS", _header.get("DATE"))

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
        elif airmass_value > 40.0:
            st.warning("Extremely high airmass (>40), object near horizon")

        sun_altaz = get_sun(obstime).transform_to(altaz_frame)
        sun_alt = float(sun_altaz.alt.deg)

        details = {
            "observatory": obs_data["name"],
            "datetime": obstime.iso,
            "target_coords": {
                "ra": coord.ra.to_string(unit=u.hour),
                "dec": coord.dec.to_string(unit=u.deg),
            },
            "altaz": {
                "altitude": round(float(altaz.alt.deg), 2),
                "azimuth": round(float(altaz.az.deg), 2),
            },
            "sun_altitude": round(sun_alt, 2),
            "observation_type": "night"
            if sun_alt < -18
            else "twilight"
            if sun_alt < 0
            else "day",
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

            return data, header
        except Exception as e:
            st.error(f"Error loading FITS file: {str(e)}")
            return None, None
        finally:
            hdul.close()
    return None, None


def calibrate_image_streamlit(
    science_data,
    science_header,
    bias_data,
    dark_data,
    flat_data,
    exposure_time_science,
    exposure_time_dark,
    apply_bias,
    apply_dark,
    apply_flat,
):
    """
    Calibrate an astronomical science image using bias, dark, and flat-field frames.

    This function performs standard CCD calibration steps with status updates in Streamlit:
    1. Bias subtraction (optional)
    2. Dark frame subtraction with exposure time scaling (optional)
    3. Flat field correction using normalized flat (optional)

    Parameters
    ----------
    science_data : numpy.ndarray
        The raw science image data to be calibrated
    science_header : dict or astropy.io.fits.Header
        Header information from the science image
    bias_data : numpy.ndarray or None
        Bias frame data for zero-level correction
    dark_data : numpy.ndarray or None
        Dark frame data for thermal noise correction
    flat_data : numpy.ndarray or None
        Flat field data for sensitivity/vignetting correction
    exposure_time_science : float
        Exposure time of the science image in seconds
    exposure_time_dark : float
        Exposure time of the dark frame in seconds
    apply_bias : bool
        Whether to apply bias subtraction
    apply_dark : bool
        Whether to apply dark frame subtraction
    apply_flat : bool
        Whether to apply flat field correction

    Returns
    -------
    tuple
        (calibrated_science, science_header) where:
        - calibrated_science is the processed science image as numpy.ndarray
        - science_header is the unchanged header information

    Notes
    -----
    - If bias correction is applied, it's also applied to dark and flat frames before they're used.
    - Dark frames are scaled according to the exposure time ratio if different from science exposure.
    - The flat field is normalized by its median value before division.
    - Progress updates are shown in the Streamlit interface.
    """
    if not apply_bias and not apply_dark and not apply_flat:
        st.write("Calibration steps are disabled. Returning raw science data.")
        return science_data, science_header

    calibrated_science = science_data.copy()
    steps_applied = []

    dark_data_corrected = dark_data
    flat_data_corrected = flat_data

    if apply_bias and bias_data is not None:
        st.write("Application de la soustraction du bias...")
        calibrated_science -= bias_data
        steps_applied.append("Soustraction du Bias")
        if dark_data is not None:
            dark_data_corrected = dark_data - bias_data
        if flat_data is not None:
            flat_data_corrected = flat_data - bias_data

    if apply_dark and dark_data_corrected is not None:
        st.write("Application de la soustraction du dark...")
        if exposure_time_science != exposure_time_dark:
            dark_scale_factor = exposure_time_science / exposure_time_dark
            scaled_dark = dark_data_corrected * dark_scale_factor
        else:
            scaled_dark = dark_data_corrected
        calibrated_science -= scaled_dark
        steps_applied.append("Soustraction du Dark")

    if apply_flat and flat_data_corrected is not None:
        st.write("Application de la correction du flat field...")
        normalized_flat = flat_data_corrected / np.median(flat_data_corrected)
        calibrated_science /= normalized_flat
        steps_applied.append("Correction du Flat Field")

    if not steps_applied:
        st.write(
            "No calibration steps were applied because files are missing or options are disabled."
        )
        return science_data, science_header

    st.success(f"Calibration steps applied: {', '.join(steps_applied)}")
    return calibrated_science, science_header


@st.cache_data
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
            raise ValueError("border must be an int or a tuple of 2 or 4 elements")
    else:
        raise TypeError("border must be an int or a tuple")

    if any(b < 0 for b in (top, bottom, left, right)):
        raise ValueError("Borders cannot be negative")

    if top + bottom >= height or left + right >= width:
        raise ValueError("Borders are larger than the image")

    mask = np.zeros(image.shape[:2], dtype=dtype)
    mask[top : height - bottom, left : width - right] = True

    return ~mask if invert else mask


@st.cache_data
def fwhm_fit(
    img: np.ndarray,
    fwhm: float,
    pixel_scale: float,
    mask: Optional[np.ndarray] = None,
    std_lo: float = 0.5,
    std_hi: float = 0.5,
) -> Optional[float]:
    """
    Estimate the Full Width at Half Maximum (FWHM) of stars in an astronomical image.

    This function detects sources in the image using DAOStarFinder, filters them based on
    flux values, and fits 1D Gaussian models to the marginal distributions of each source
    to calculate FWHM values.

    Parameters
    ----------
    img : numpy.ndarray
        The 2D image array containing astronomical sources
    fwhm : float
        Initial FWHM estimate in pixels, used for source detection
    pixel_scale : float
        The pixel scale of the image in arcseconds per pixel
    mask : numpy.ndarray, optional
        Boolean mask array where True indicates pixels to ignore during source detection
    std_lo : float, default=0.5
        Lower bound for flux filtering, in standard deviations below median flux
    std_hi : float, default=0.5
        Upper bound for flux filtering, in standard deviations above median flux

    Returns
    -------
    float or None
        The median FWHM value in pixels (rounded to nearest integer) if successful,
        or None if no valid sources could be measured

    Notes
    -----
    The function uses marginal sums along the x and y dimensions and fits 1D Gaussian
    profiles to estimate FWHM. For each source, a box of size ~6×FWHM is extracted,
    and profiles are fit independently in both dimensions, with the results averaged.

    Progress updates and error messages are displayed using Streamlit.
    """

    def compute_fwhm_marginal_sums(image_data, center_row, center_col, box_size):
        half_box = box_size // 2

        if box_size < 5:
            return None

        row_start = center_row - half_box
        row_end = center_row + half_box + 1
        col_start = center_col - half_box
        col_end = center_col + half_box + 1

        if (
            row_start < 0
            or row_end > image_data.shape[0]
            or col_start < 0
            or col_end > image_data.shape[1]
        ):
            return None

        box_data = image_data[row_start:row_end, col_start:col_end]

        if box_data.shape[0] < 5 or box_data.shape[1] < 5:
            return None

        sum_rows = np.sum(box_data, axis=1)
        sum_cols = np.sum(box_data, axis=0)

        if np.max(sum_rows) < 5 * np.median(sum_rows) or np.max(
            sum_cols
        ) < 5 * np.median(sum_cols):
            return None

        row_indices = np.arange(box_data.shape[0])
        col_indices = np.arange(box_data.shape[1])

        fitter = fitting.LevMarLSQFitter()

        try:
            row_max_idx = np.argmax(sum_rows)
            row_max_val = sum_rows[row_max_idx]

            model_row = models.Gaussian1D(
                amplitude=row_max_val, mean=row_max_idx, stddev=box_size / 6
            )

            fitted_row = fitter(model_row, row_indices, sum_rows)
            center_row_fit = fitted_row.mean.value + row_start
            fwhm_row = 2 * np.sqrt(2 * np.log(2)) * fitted_row.stddev.value
        except Exception:
            return None

        try:
            col_max_idx = np.argmax(sum_cols)
            col_max_val = sum_cols[col_max_idx]

            model_col = models.Gaussian1D(
                amplitude=col_max_val, mean=col_max_idx, stddev=box_size / 6
            )

            fitted_col = fitter(model_col, col_indices, sum_cols)
            center_col_fit = fitted_col.mean.value + col_start
            fwhm_col = 2 * np.sqrt(2 * np.log(2)) * fitted_col.stddev.value
        except Exception:
            return None

        return fwhm_row, fwhm_col, center_row_fit, center_col_fit

    try:
        daofind = DAOStarFinder(fwhm=1.5 * fwhm, threshold=6 * np.std(img))
        sources = daofind(img, mask=mask)
        if sources is None:
            st.warning("No sources found !")
            return None

        flux = sources["flux"]
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        mask_flux = (flux > median_flux - std_lo * std_flux) & (
            flux < median_flux + std_hi * std_flux
        )
        filtered_sources = sources[mask_flux]

        filtered_sources = filtered_sources[~np.isnan(filtered_sources["flux"])]

        st.write(f"Number of sources after flux filtering: {len(filtered_sources)}")

        if len(filtered_sources) == 0:
            msg = "No valid sources for fitting found after filtering."
            st.error(msg)
            raise ValueError(msg)

        box_size = int(6 * round(fwhm))
        if box_size % 2 == 0:
            box_size += 1

        fwhm_values = []
        skipped_sources = 0

        for source in filtered_sources:
            try:
                x_cen = int(source["xcentroid"])
                y_cen = int(source["ycentroid"])

                fwhm_results = compute_fwhm_marginal_sums(img, y_cen, x_cen, box_size)
                if fwhm_results is None:
                    skipped_sources += 1
                    continue

                fwhm_row, fwhm_col, _, _ = fwhm_results

                fwhm_source = np.mean([fwhm_row, fwhm_col])
                fwhm_values.append(fwhm_source)

            except Exception:
                skipped_sources += 1
                continue

        if skipped_sources > 0:
            st.write(
                f"FWHM calculation failed for {skipped_sources} sources out of {len(filtered_sources)} and were skipped."
            )

        if len(fwhm_values) == 0:
            msg = "No valid sources for FWHM fitting after marginal sums adjustment."
            st.error(msg)
            raise ValueError(msg)

        fwhm_values_arr = np.array(fwhm_values)
        valid = ~np.isnan(fwhm_values_arr) & ~np.isinf(fwhm_values_arr)
        if not np.any(valid):
            msg = "All FWHM values are NaN or infinite after marginal sums calculation."
            st.error(msg)
            raise ValueError(msg)

        mean_fwhm = np.median(fwhm_values_arr[valid])
        st.write(f"FWHM estimate based on Gaussian model: {round(mean_fwhm)} pixels")

        return round(mean_fwhm)
    except ValueError as e:
        raise e
    except Exception as e:
        st.error(f"Unexpected error in fwhm_fit: {e}")
        raise ValueError(f"Unexpected error in fwhm_fit: {e}")


def perform_epsf_photometry(
    img: np.ndarray,
    phot_table: Table,
    fwhm: float,
    daostarfind: Any,
    mask: Optional[np.ndarray] = None,
) -> Tuple[Table, Any]:
    """
    Perform PSF (Point Spread Function) photometry using an empirically-constructed PSF model.

    This function builds an empirical PSF (EPSF) from bright stars in the image
    and then uses this model to perform PSF photometry on all detected sources.

    Parameters
    ----------
    img : numpy.ndarray
        Image with sky background subtracted
    phot_table : astropy.table.Table
        Table containing source positions
    fwhm : float
        Full Width at Half Maximum in pixels, used to define extraction and fitting sizes
    daostarfind : callable
        Star detection function used as "finder" in PSF photometry
    mask : numpy.ndarray, optional
        Mask to exclude certain image areas from analysis

    Returns
    -------
    Tuple[astropy.table.Table, photutils.psf.EPSFModel]
        - phot_epsf_result : Table containing PSF photometry results, including fitted fluxes and positions
        - epsf : The fitted EPSF model used for photometry

    Notes
    -----
    The function displays the extracted stars used for PSF model building and the
    resulting PSF model in the Streamlit interface. It also saves the PSF model
    as a FITS file in the output directory.

    If successful, the photometry results are also stored in the Streamlit session state
    under 'epsf_photometry_result'.
    """
    try:
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError(
                "Invalid image data provided. Ensure the image is a non-empty numpy array."
            )

        nddata = NDData(data=img)
    except Exception as e:
        st.error(f"Error creating NDData: {e}")
        raise

    try:
        stars_table = Table()
        stars_table["x"] = phot_table["xcenter"]
        stars_table["y"] = phot_table["ycenter"]
        st.write("Star positions table prepared.")
    except Exception as e:
        st.error(f"Error preparing star positions table: {e}")
        if fwhm <= 0:
            raise ValueError("FWHM must be a positive number.")
        fit_shape = 2 * round(fwhm) + 1

    try:
        fit_shape = 2 * round(fwhm) + 1
        st.write(f"Fitting shape: {fit_shape} pixels.")
    except Exception as e:
        st.error(f"Error calculating fitting shape: {e}")
        raise

    try:
        stars = extract_stars(nddata, stars_table, size=fit_shape)
        st.write(f"{len(stars)} stars extracted.")
    except Exception as e:
        st.error(f"Error extracting stars: {e}")
        raise

    try:
        nrows, ncols = 5, 5
        fig_stars, ax_stars = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=FIGURE_SIZES["stars_grid"], squeeze=False
        )
        ax_stars = ax_stars.ravel()
        n_disp = min(len(stars), nrows * ncols)
        for i in range(n_disp):
            norm = simple_norm(stars[i].data, "log", percent=99.0)
            ax_stars[i].imshow(stars[i].data, norm=norm, origin="lower", cmap="viridis")
        plt.tight_layout()
        st.pyplot(fig_stars)
    except Exception as e:
        st.warning(f"Error displaying extracted stars: {e}")

    try:
        epsf_builder = EPSFBuilder(oversampling=2, maxiters=5, progress_bar=True)
        epsf, _ = epsf_builder(stars)
        st.session_state["epsf_model"] = epsf
    except Exception as e:
        st.error(f"Error fitting PSF model: {e}")
        raise

    if epsf.data is not None and epsf.data.size > 0:
        try:
            hdu = fits.PrimaryHDU(data=epsf.data)

            hdu.header["COMMENT"] = "PSF model created with photutils.EPSFBuilder"
            hdu.header["FWHMPIX"] = (fwhm, "FWHM in pixels used for extraction")
            hdu.header["OVERSAMP"] = (2, "Oversampling factor")

            psf_filename = (
                f"{st.session_state.get('base_filename', 'psf_model')}_psf.fits"
            )
            psf_filepath = os.path.join(
                st.session_state.get("output_dir", "."), psf_filename
            )

            hdu.writeto(psf_filepath, overwrite=True)
            st.success("PSF model saved as FITS file")

            norm_epsf = simple_norm(epsf.data, "log", percent=99.0)
            fig_epsf_model, ax_epsf_model = plt.subplots(
                figsize=FIGURE_SIZES["medium"], dpi=100
            )
            ax_epsf_model.imshow(
                epsf.data,
                norm=norm_epsf,
                origin="lower",
                cmap="viridis",
                interpolation="nearest",
            )
            ax_epsf_model.set_title("Fitted PSF Model")
            st.pyplot(fig_epsf_model)
        except Exception as e:
            st.warning(f"Error working with PSF model: {e}")
    else:
        st.warning("EPSF data is empty or invalid. Cannot display the PSF model.")
        if not callable(daostarfind):
            raise ValueError(
                "The 'finder' parameter must be a callable star finder, such as DAOStarFinder."
            )

    psfphot = IterativePSFPhotometry(
        psf_model=epsf,
        fit_shape=fit_shape,
        finder=daostarfind,
        aperture_radius=fit_shape / 2,
        maxiters=3,
        progress_bar=True,
    )

    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError(
            "Invalid image data provided. Ensure the image is a non-empty numpy array."
        )
    if mask is not None and (
        not isinstance(mask, np.ndarray) or mask.shape != img.shape
    ):
        raise ValueError(
            "Invalid mask provided. Ensure the mask is a numpy array with the same shape as the image."
        )

    psfphot.x = phot_table["xcenter"]
    psfphot.y = phot_table["ycenter"]

    try:
        st.write("Performing PSF photometry...")
        phot_epsf_result = psfphot(img, mask=mask)
        st.session_state["epsf_photometry_result"] = phot_epsf_result
        st.write("PSF photometry completed successfully.")
    except Exception as e:
        st.error(f"Error executing PSF photometry: {e}")
        raise

    return phot_epsf_result, epsf


@st.cache_data
def find_sources_and_photometry_streamlit(
    image_data, _science_header, mean_fwhm_pixel, threshold_sigma, detection_mask
):
    """
    Find astronomical sources and perform both aperture and PSF photometry.

    This comprehensive function handles the full photometry workflow:
    1. Background estimation
    2. Source detection
    3. WCS refinement with GAIA DR3
    4. Aperture photometry
    5. PSF/EPSF photometry

    Parameters
    ----------
    image_data : numpy.ndarray
        Science image data (2D array)
    _science_header : dict or astropy.io.fits.Header
        Header information from FITS file (underscore prevents caching issues)
    mean_fwhm_pixel : float
        Estimated FWHM in pixels for aperture sizing
    threshold_sigma : float
        Detection threshold in sigma units above background
    detection_mask : int
        Border size in pixels to mask during detection

    Returns
    -------
    tuple
        (phot_table, epsf_table, daofind, bkg) where:
        - phot_table: Table with aperture photometry results
        - epsf_table: Table with PSF photometry results
        - daofind: The DAOStarFinder object used for detection
        - bkg: Background2D object with background model

    Notes
    -----
    - Uses stdpipe for astrometry refinement with GAIA DR3
    - Automatically adds celestial coordinates (RA/Dec) to the photometry tables if WCS is available
    - Both aperture and PSF photometry include instrumental magnitude calculations
    - Shows progress and status updates in the Streamlit interface
    """
    daofind = None

    try:
        w, wcs_error = safe_wcs_create(_science_header)
        if w is None:
            st.error(f"Error creating WCS: {wcs_error}")
            return None, None, daofind, None
    except Exception as e:
        st.error(f"Error creating WCS: {e}")
        return None, None, daofind, None

    pixel_scale = _science_header.get(
        "PIXSCALE",
        _science_header.get("PIXSIZE", _science_header.get("PIXELSCAL", 1.0)),
    )

    bkg, bkg_error = estimate_background(image_data, box_size=100, filter_size=5)
    if bkg is None:
        st.error(f"Error estimating background: {bkg_error}")
        return None, None, daofind, None

    mask = make_border_mask(image_data, border=detection_mask)

    total_error = np.sqrt(bkg.background_rms**2 + bkg.background_median) / np.sqrt(
        bkg.background_median
    )

    st.write("Estimating FWHM...")
    fwhm_estimate = fwhm_fit(
        image_data - bkg.background, mean_fwhm_pixel, pixel_scale, mask
    )

    if fwhm_estimate is None:
        st.warning("Failed to estimate FWHM. Using the initial estimate.")
        fwhm_estimate = mean_fwhm_pixel

    daofind = DAOStarFinder(
        fwhm=1.5 * fwhm_estimate,
        threshold=threshold_sigma * np.std(image_data - bkg.background),
    )

    sources = daofind(image_data - bkg.background, mask=mask)

    obj = photometry.get_objects_sep(image_data - bkg.background, mask=mask,
                                     aper=fwhm_estimate, gain=1, edge=25
                                     )

    if sources is None or len(sources) == 0:
        st.warning("No sources found!")
        return None, None, daofind, bkg

    st.info("Doing astrometry refinement...")
    ra0, dec0, sr0 = astrometry.get_frame_center(wcs=w,
                                                 width=image_data.shape[1],
                                                 height=image_data.shape[0])
    cat = catalogs.get_cat_vizier(ra0, dec0, sr0, 'gaiaedr3',
                                  filters={'RPmag': '<19'})
    cat_col_mag = 'RPmag'
    try:
        wcs = pipeline.refine_astrometry(obj, cat,
                                         1.5*fwhm_estimate*pixel_scale/3600,
                                         wcs=w, order=0,
                                         cat_col_mag=cat_col_mag,
                                         cat_col_mag_err=None,
                                         verbose=True)
        if wcs:
            st.info("Refined WCS successfully.")
            astrometry.clear_wcs(_science_header, remove_comments=True,
                                 remove_underscored=True, remove_history=True)
            _science_header.update(wcs.to_header(relax=True))
        else:
            st.warning("WCS refinement failed.")
    except Exception as e:
        st.warning(f"Skipping WCS refinement: {str(e)}")

    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))
    apertures = CircularAperture(positions, r=1.5 * fwhm_estimate)

    try:
        wcs_obj = None
        if "CTYPE1" in _science_header:
            try:
                wcs_obj = WCS(_science_header)
                if wcs_obj.pixel_n_dim > 2:
                    wcs_obj = wcs_obj.celestial
                    st.info("Reduced WCS to 2D celestial coordinates for photometry")
            except Exception as e:
                st.warning(f"Error creating WCS object: {e}")
                wcs_obj = None

        phot_table = aperture_photometry(
            image_data - bkg.background, apertures, error=total_error, wcs=wcs_obj
        )

        phot_table["xcenter"] = sources["xcentroid"]
        phot_table["ycenter"] = sources["ycentroid"]

        instrumental_mags = -2.5 * np.log10(phot_table["aperture_sum"])
        phot_table["instrumental_mag"] = instrumental_mags

        try:
            epsf_table, _ = perform_epsf_photometry(
                image_data - bkg.background, phot_table, fwhm_estimate, daofind, mask
            )

            epsf_instrumental_mags = -2.5 * np.log10(epsf_table["flux_fit"])
            epsf_table["instrumental_mag"] = epsf_instrumental_mags
        except Exception as e:
            st.error(f"Error performing EPSF photometry: {e}")
            epsf_table = None

        valid_sources = (phot_table["aperture_sum"] > 0) & np.isfinite(
            phot_table["instrumental_mag"]
        )
        phot_table = phot_table[valid_sources]

        if epsf_table is not None:
            epsf_valid_sources = (epsf_table["flux_fit"] > 0) & np.isfinite(
                epsf_table["instrumental_mag"]
            )

            epsf_table = epsf_table[epsf_valid_sources]

        try:
            if wcs_obj is not None:
                ra, dec = wcs_obj.pixel_to_world_values(
                    phot_table["xcenter"], phot_table["ycenter"]
                )
                phot_table["ra"] = ra * u.deg
                phot_table["dec"] = dec * u.deg

                if epsf_table is not None:
                    try:
                        epsf_ra, epsf_dec = wcs_obj.pixel_to_world_values(
                            epsf_table["x_fit"], epsf_table["y_fit"]
                        )
                        epsf_table["ra"] = epsf_ra * u.deg
                        epsf_table["dec"] = epsf_dec * u.deg
                    except Exception as e:
                        st.warning(f"Could not add coordinates to EPSF table: {e}")
            else:
                if all(
                    key in _science_header for key in ["RA", "DEC", "NAXIS1", "NAXIS2"]
                ):
                    st.info("Using simple linear approximation for RA/DEC coordinates")
                    center_ra = _science_header["RA"]
                    center_dec = _science_header["DEC"]
                    width = _science_header["NAXIS1"]
                    height = _science_header["NAXIS2"]

                    pix_scale = pixel_scale or 1.0

                    center_x = width / 2
                    center_y = height / 2

                    for i, row in enumerate(phot_table):
                        x = row["xcenter"]
                        y = row["ycenter"]
                        dx = (x - center_x) * pix_scale / 3600.0
                        dy = (y - center_y) * pix_scale / 3600.0
                        phot_table[i]["ra"] = center_ra + dx / np.cos(
                            np.radians(center_dec)
                        )
                        phot_table[i]["dec"] = center_dec + dy
        except Exception as e:
            st.warning(
                f"WCS transformation failed: {e}. RA and Dec not added to tables."
            )

        st.success(f"Found {len(phot_table)} sources and performed photometry.")
        return phot_table, epsf_table, daofind, bkg
    except Exception as e:
        st.error(f"Error performing aperture photometry: {e}")
        return None, None, daofind, bkg


@st.cache_data
def cross_match_with_gaia_streamlit(
    _phot_table,
    _science_header,
    pixel_size_arcsec,
    mean_fwhm_pixel,
    gaia_band,
    gaia_min_mag,
    gaia_max_mag,
):
    """
    Cross-match detected sources with the GAIA DR3 star catalog.

    This function queries the GAIA catalog for a region matching the image field of view,
    applies filtering based on magnitude range, and matches GAIA sources to the detected sources.

    Parameters
    ----------
    _phot_table : astropy.table.Table
        Table containing detected source positions (underscore prevents caching issues)
    _science_header : dict or astropy.io.fits.Header
        FITS header with WCS information (underscore prevents caching issues)
    pixel_size_arcsec : float
        Pixel scale in arcseconds per pixel
    mean_fwhm_pixel : float
        FWHM in pixels, used to determine matching radius
    gaia_band : str
        GAIA magnitude band to use for filtering (e.g., 'phot_g_mean_mag')
    gaia_min_mag : float
        Minimum magnitude for GAIA source filtering
    gaia_max_mag : float
        Maximum magnitude for GAIA source filtering

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing matched sources with both measured and GAIA catalog data,
        or None if the cross-match failed or found no matches

    Notes
    -----
    - The maximum separation for matching is set to twice the FWHM in arcseconds
    - Includes filtering to exclude variable stars if the information is available
    - Progress and status updates are displayed in the Streamlit interface
    """
    st.write("Cross-matching with Gaia DR3...")

    if _science_header is None:
        st.warning("No header information available. Cannot cross-match with Gaia.")
        return None

    try:
        w = WCS(_science_header)
    except Exception as e:
        st.error(f"Error creating WCS: {e}")
        return None

    try:
        source_positions_pixel = np.transpose(
            (_phot_table["xcenter"], _phot_table["ycenter"])
        )
        source_positions_sky = w.pixel_to_world(
            source_positions_pixel[:, 0], source_positions_pixel[:, 1]
        )
    except Exception as e:
        st.error(f"Error converting pixel positions to sky coordinates: {e}")
        return None

    try:
        image_center_ra_dec = w.pixel_to_world(
            _science_header["NAXIS1"] // 2, _science_header["NAXIS2"] // 2
        )
        gaia_search_radius_arcsec = (
            max(_science_header["NAXIS1"], _science_header["NAXIS2"])
            * pixel_size_arcsec
            / 2.0
        )
        radius_query = gaia_search_radius_arcsec * u.arcsec

        st.write(
            f"Querying Gaia DR3 in a radius of {round(radius_query.value / 60.0, 2)} arcmin."
        )
        job = Gaia.cone_search(image_center_ra_dec, radius=radius_query)
        gaia_table = job.get_results()
    except Exception as e:
        st.error(f"Error querying Gaia: {e}")
        return None

    if gaia_table is None or len(gaia_table) == 0:
        st.warning("No Gaia sources found within search radius.")
        return None

    try:
        mag_filter = (gaia_table[gaia_band] < gaia_max_mag) & (
            gaia_table[gaia_band] > gaia_min_mag
        )

        if "phot_variable_flag" in gaia_table.colnames:
            var_filter = gaia_table["phot_variable_flag"] != "VARIABLE"
            combined_filter = mag_filter & var_filter
        else:
            combined_filter = mag_filter

        gaia_table_filtered = gaia_table[combined_filter]

        if len(gaia_table_filtered) == 0:
            st.warning(
                f"No Gaia sources found within magnitude range {gaia_min_mag} < {gaia_band} < {gaia_max_mag}."
            )
            return None

        st.write(f"Filtered Gaia catalog to {len(gaia_table_filtered)} sources.")
    except Exception as e:
        st.error(f"Error filtering Gaia catalog: {e}")
        return None

    try:
        gaia_skycoords = SkyCoord(
            ra=gaia_table_filtered["ra"], dec=gaia_table_filtered["dec"], unit="deg"
        )
        idx, d2d, _ = source_positions_sky.match_to_catalog_sky(gaia_skycoords)

        max_sep_constraint = 2 * mean_fwhm_pixel * pixel_size_arcsec * u.arcsec
        gaia_matches = d2d <= max_sep_constraint

        matched_indices_gaia = idx[gaia_matches]
        matched_indices_phot = np.where(gaia_matches)[0]

        if len(matched_indices_gaia) == 0:
            st.warning("No Gaia matches found within the separation constraint.")
            return None

        matched_table_qtable = _phot_table[matched_indices_phot]

        matched_table = matched_table_qtable.to_pandas()
        matched_table["gaia_index"] = matched_indices_gaia
        matched_table["gaia_separation_arcsec"] = d2d[gaia_matches].arcsec
        matched_table[gaia_band] = gaia_table_filtered[gaia_band][matched_indices_gaia]

        valid_gaia_mags = np.isfinite(matched_table[gaia_band])
        matched_table = matched_table[valid_gaia_mags]

        st.success(f"Found {len(matched_table)} Gaia matches after filtering.")
        return matched_table
    except Exception as e:
        st.error(f"Error during cross-matching: {e}")
        return None


@st.cache_data
def calculate_zero_point_streamlit(_phot_table, _matched_table, gaia_band, air):
    """
    Calculate photometric zero point from matched sources with GAIA.

    The zero point transforms instrumental magnitudes to the standard GAIA magnitude system,
    accounting for atmospheric extinction using the provided airmass.

    Parameters
    ----------
    _phot_table : astropy.table.Table or pandas.DataFrame
        Photometry results table (underscore prevents caching issues)
    _matched_table : pandas.DataFrame
        Table of GAIA cross-matched sources (underscore prevents caching issues)
    gaia_band : str
        GAIA magnitude band used (e.g., 'phot_g_mean_mag')
    air : float
        Airmass value for atmospheric extinction correction

    Returns
    -------
    tuple
        (zero_point_value, zero_point_std, matplotlib_figure) where:
        - zero_point_value: Float value of the calculated zero point
        - zero_point_std: Standard deviation of the zero point
        - matplotlib_figure: Figure object showing GAIA vs. calibrated magnitudes

    Notes
    -----
    - Uses sigma clipping to remove outliers from zero point calculation
    - Applies a simple atmospheric extinction correction of 0.1*airmass
    - Stores the calibrated photometry table in session state as 'final_phot_table'
    - Creates and saves a plot showing the relation between GAIA and calibrated magnitudes
    """
    st.write("Calculating zero point...")

    if _matched_table is None or len(_matched_table) == 0:
        st.warning("No matched sources to calculate zero point.")
        return None, None, None

    try:
        zero_points = _matched_table[gaia_band] - _matched_table["instrumental_mag"]
        _matched_table["zero_point"] = zero_points
        _matched_table["zero_point_error"] = np.std(zero_points)

        clipped_zero_points = sigma_clip(zero_points, sigma=3)
        zero_point_value = np.mean(clipped_zero_points)
        zero_point_std = np.std(clipped_zero_points)

        _matched_table["calib_mag"] = (
            _matched_table["instrumental_mag"] + zero_point_value + 0.1 * air
        )

        if not isinstance(_phot_table, pd.DataFrame):
            _phot_table = _phot_table.to_pandas()

        _phot_table["calib_mag"] = (
            _phot_table["instrumental_mag"] + zero_point_value + 0.1 * air
        )

        st.session_state["final_phot_table"] = _phot_table

        fig, ax = plt.subplots(figsize=FIGURE_SIZES["medium"], dpi=100)

        # Calculate residuals
        _matched_table["residual"] = _matched_table[gaia_band] - _matched_table["calib_mag"]
        
        # Create bins for magnitude ranges
        bin_width = 0.5  # 0.5 magnitude width bins
        min_mag = _matched_table[gaia_band].min()
        max_mag = _matched_table[gaia_band].max()
        bins = np.arange(np.floor(min_mag), np.ceil(max_mag) + bin_width,
                         bin_width)
        
        # Group data by magnitude bins
        grouped = _matched_table.groupby(pd.cut(_matched_table[gaia_band],
                                                bins))
        bin_centers = [(bin.left + bin.right) / 2 for bin in grouped.groups.keys()]
        bin_means = grouped["calib_mag"].mean().values
        bin_stds = grouped["calib_mag"].std().values

        # Plot individual points
        ax.scatter(
            _matched_table[gaia_band],
            _matched_table["calib_mag"],
            alpha=0.5,
            label="Matched sources",
            color='blue'
        )

        # Plot binned means with error bars showing standard deviation
        valid_bins = ~np.isnan(bin_means) & ~np.isnan(bin_stds)
        ax.errorbar(
            np.array(bin_centers)[valid_bins],
            bin_means[valid_bins],
            yerr=bin_stds[valid_bins],
            fmt='ro-',
            label='Mean ± StdDev (binned)',
            capsize=5
        )

        # Add a diagonal line for reference
        ideal_mag = np.linspace(min_mag, max_mag, 10)
        ax.plot(ideal_mag, ideal_mag, 'k--', alpha=0.7, label="y=x")

        ax.set_xlabel(f"Gaia {gaia_band}")
        ax.set_ylabel("Calibrated magnitude")
        ax.set_title("Gaia magnitude vs Calibrated magnitude")
        ax.legend()
        ax.grid(True, alpha=0.5)

        st.success(
            f"Calculated Zero Point: {zero_point_value:.3f} ± {zero_point_std:.3f}"
        )

        try:
            base_name = st.session_state.get('base_filename', 'photometry')
            zero_point_plot_path = os.path.join(output_dir, f"{base_name}_zero_point_plot.png")
            plt.savefig(zero_point_plot_path)
        except Exception as e:
            st.warning(f"Could not save plot to file: {e}")

        return zero_point_value, zero_point_std, fig
    except Exception as e:
        st.error(f"Error calculating zero point: {e}")
        return None, None, None


def run_zero_point_calibration(
    header,
    pixel_size_arcsec,
    mean_fwhm_pixel,
    threshold_sigma,
    detection_mask,
    gaia_band,
    gaia_min_mag,
    gaia_max_mag,
    air,
):
    """
    Run the complete photometric zero point calibration workflow.

    This function orchestrates the full photometric analysis pipeline:
    1. Source detection and photometry
    2. GAIA catalog cross-matching
    3. Zero point determination
    4. Results display and catalog creation

    Parameters
    ----------
    header : dict or astropy.io.fits.Header
        FITS header with observation metadata
    pixel_size_arcsec : float
        Pixel scale in arcseconds per pixel
    mean_fwhm_pixel : float
        FWHM estimate in pixels
    threshold_sigma : float
        Detection threshold in sigma units
    detection_mask : int
        Border mask size in pixels
    gaia_band : str
        GAIA magnitude band to use
    gaia_min_mag, gaia_max_mag : float
        Magnitude limits for GAIA sources
    air : float
        Airmass value for extinction correction

    Returns
    -------
    tuple
        (zero_point_value, zero_point_std, final_phot_table) where:
        - zero_point_value: Calculated zero point value
        - zero_point_std: Standard deviation of the zero point
        - final_phot_table: DataFrame with the complete photometry catalog

    Notes
    -----
    This function manages the workflow with appropriate Streamlit spinners and messages,
    creates downloadable output files, logs progress, and provides interactive data views.

    If PSF photometry results are available, they're added to the final catalog alongside
    aperture photometry results.
    """
    with st.spinner("Finding sources and performing photometry..."):
        phot_table_qtable, epsf_table, _, _ = (
            find_sources_and_photometry_streamlit(
                image_to_process,
                header_to_process,
                mean_fwhm_pixel,
                threshold_sigma,
                detection_mask,
            )
        )

        if phot_table_qtable is None:
            st.error("Failed to perform photometry - no sources found")
            return None, None, None

    with st.spinner("Cross-matching with Gaia..."):
        matched_table = cross_match_with_gaia_streamlit(
            phot_table_qtable,
            header,
            pixel_size_arcsec,
            mean_fwhm_pixel,
            gaia_band,
            gaia_min_mag,
            gaia_max_mag,
        )

        if matched_table is None:
            st.error("Failed to cross-match with Gaia")
            return None, None, phot_table_qtable

    st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
    st.dataframe(matched_table.head(10))

    with st.spinner("Calculating zero point..."):
        zero_point_value, zero_point_std, zp_plot = calculate_zero_point_streamlit(
            phot_table_qtable, matched_table, gaia_band, air
        )

        if zero_point_value is not None:
            st.pyplot(zp_plot)
            write_to_log(
                log_buffer, f"Zero point: {zero_point_value:.3f} ± {zero_point_std:.3f}"
            )
            write_to_log(log_buffer, f"Airmass: {air:.2f}")

            try:
                if (
                    "epsf_photometry_result" in st.session_state
                    and epsf_table is not None
                ):
                    epsf_df = (
                        epsf_table.to_pandas()
                        if not isinstance(epsf_table, pd.DataFrame)
                        else epsf_table
                    )

                    epsf_df["match_id"] = (
                        epsf_df["xcenter"].round(2).astype(str)
                        + "_"
                        + epsf_df["ycenter"].round(2).astype(str)
                    )
                    final_table["match_id"] = (
                        final_table["xcenter"].round(2).astype(str)
                        + "_"
                        + final_table["ycenter"].round(2).astype(str)
                    )

                    epsf_cols = {
                        "match_id": "match_id",
                        "flux_fit": "psf_flux_fit",
                        "flux_unc": "psf_flux_unc",
                        "instrumental_mag": "psf_instrumental_mag",
                    }

                    if (
                        len(epsf_cols) > 1
                        and "match_id" in epsf_df.columns
                        and "match_id" in final_table.columns
                    ):
                        epsf_subset = epsf_df[
                            [col for col in epsf_cols.keys() if col in epsf_df.columns]
                        ].rename(columns=epsf_cols)

                    final_table = pd.merge(
                        final_table, epsf_subset, on="match_id", how="left"
                    )

                    if "epsf_instrumental_mag" in final_table.columns:
                        final_table["psf_calib_mag"] = (
                            final_table["psf_instrumental_mag"]
                            + zero_point_value
                            + 0.1 * air
                        )

                    final_table.drop("match_id", axis=1, inplace=True)

                    st.success("Added PSF photometry results to the catalog")

                csv_buffer = StringIO()

                cols_to_drop = []
                for col_name in ["sky_center.ra", "sky_center.dec"]:
                    if col_name in final_table.columns:
                        cols_to_drop.append(col_name)

                if cols_to_drop:
                    final_table = final_table.drop(columns=cols_to_drop)

                if "match_id" in final_table.columns:
                    final_table.drop("match_id", axis=1, inplace=True)

                final_table["zero_point"] = zero_point_value
                final_table["zero_point_std"] = zero_point_std
                final_table["airmass"] = air

                st.subheader("Final Photometry Catalog")
                st.dataframe(final_table.head(10))

                final_table.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                filename = (
                    catalog_name
                    if catalog_name.endswith(".csv")
                    else f"{catalog_name}.csv"
                )
                catalog_path = os.path.join(output_dir, filename)

                with open(catalog_path, "w") as f:
                    f.write(csv_data)

                metadata_filename = f"{base_catalog_name}_metadata.txt"
                metadata_path = os.path.join(output_dir, metadata_filename)

                with open(metadata_path, "w") as f:
                    f.write("RAPAS Photometry Analysis Metadata\n")
                    f.write("================================\n\n")
                    f.write(
                        f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    )
                    f.write(f"Input File: {science_file.name}\n")
                    f.write(f"Catalog File: {filename}\n\n")

                    f.write(
                        f"Zero Point: {zero_point_value:.5f} ± {zero_point_std:.5f}\n"
                    )
                    f.write(f"Airmass: {air:.3f}\n")
                    f.write(f"Pixel Scale: {pixel_size_arcsec:.3f} arcsec/pixel\n")
                    f.write(
                        f"FWHM (estimated): {mean_fwhm_pixel:.2f} pixels ({seeing:.2f} arcsec)\n\n"
                    )

                    f.write("Detection Parameters:\n")
                    f.write(f"  Threshold: {threshold_sigma} sigma\n")
                    f.write(f"  Edge Mask: {detection_mask} pixels\n")
                    f.write(
                        f"  Gaia Magnitude Range: {gaia_min_mag:.1f} - {gaia_max_mag:.1f}\n"
                    )

                write_to_log(log_buffer, "Saved catalog metadata")

            except Exception as e:
                st.error(f"Error preparing download: {e}")
                st.exception(e)

        return zero_point_value, zero_point_std, final_table


def enhance_catalog_with_crossmatches(api_key, final_table, matched_table, 
                                      header, pixel_scale_arcsec,
                                      search_radius_arcsec=60):
    """
    Enhance a photometric catalog with cross-matches from multiple 
    astronomical databases.

    This function queries several catalogs and databases including:
    - GAIA DR3 (using previously matched sources)
    - Astro-Colibri (for known astronomical sources)
    - SIMBAD (for object identifications and classifications)
    - SkyBoT (for solar system objects)
    - AAVSO VSX (for variable stars)
    - VizieR VII/294 (for quasars)

    Parameters
    ----------
    final_table : pandas.DataFrame
        Final photometry catalog with RA/Dec coordinates
    matched_table : pandas.DataFrame
        Table of already matched GAIA sources (used for calibration)
    header : dict or astropy.io.fits.Header
        FITS header with observation information
    pixel_scale_arcsec : float
        Pixel scale in arcseconds per pixel
    search_radius_arcsec : float, optional
        Search radius for cross-matching in arcseconds, default=60

    Returns
    -------
    pandas.DataFrame
        Input dataframe with added catalog information including:
        - GAIA data for calibration stars
        - Astro-Colibri source identifications
        - SIMBAD identifications and object types
        - Solar system object identifications from SkyBoT
        - Variable star information from AAVSO VSX
        - Quasar information from VizieR VII/294
        - A summary column 'catalog_matches' listing all catalog matches

    Notes
    -----
    The function shows progress updates in the Streamlit interface and creates
    a summary display of matched objects. Queries are made with appropriate
    error handling to prevent failures if any catalog service is unavailable.
    API requests are processed in batches to avoid overwhelming servers.
    """
    if final_table is None or len(final_table) == 0:
        st.warning("No sources to cross-match with catalogs.")
        return final_table

    status_text = st.empty()
    status_text.write("Starting cross-match process...")

    if matched_table is not None and len(matched_table) > 0:
        status_text.write("Adding Gaia calibration matches...")

        if "xcenter" in final_table.columns and "ycenter" in final_table.columns:
            final_table["match_id"] = (
                final_table["xcenter"].round(2).astype(str)
                + "_"
                + final_table["ycenter"].round(2).astype(str)
            )

        if "xcenter" in matched_table.columns and "ycenter" in matched_table.columns:
            matched_table["match_id"] = (
                matched_table["xcenter"].round(2).astype(str)
                + "_"
                + matched_table["ycenter"].round(2).astype(str)
            )

            gaia_cols = [
                col
                for col in matched_table.columns
                if any(x in col for x in ["gaia", "phot_"])
            ]
            gaia_cols.append("match_id")

            gaia_subset = matched_table[gaia_cols].copy()

            rename_dict = {}
            for col in gaia_subset.columns:
                if col != "match_id" and not col.startswith("gaia_"):
                    rename_dict[col] = f"gaia_{col}"

            if rename_dict:
                gaia_subset = gaia_subset.rename(columns=rename_dict)

            final_table = pd.merge(final_table, gaia_subset, on="match_id", how="left")

            final_table["gaia_calib_star"] = final_table["match_id"].isin(
                matched_table["match_id"]
            )

            st.success(f"Added {len(matched_table)} Gaia calibration stars to catalog")

    st.info("Querying Astro-Colibri API...")

    if api_key is None:
        api_key = os.environ.get("ASTROCOLIBRI_API")
        if api_key is None:
            st.warning("No API key for ASTRO-COLIBRI provided or found")
            pass

    try:
        field_center_ra = None
        field_center_dec = None

        if "CRVAL1" in header and "CRVAL2" in header:
            field_center_ra = float(header["CRVAL1"])
            field_center_dec = float(header["CRVAL2"])
        elif "RA" in header and "DEC" in header:
            field_center_ra = float(header["RA"])
            field_center_dec = float(header["DEC"])
        elif "OBJRA" in header and "OBJDEC" in header:
            field_center_ra = float(header["OBJRA"])
            field_center_dec = float(header["OBJDEC"])

        try:
            # Base URL of the API
            url = 'https://astro-colibri.science/cone_search'

            # Request parameters (headers, body)
            headers = {"Content-Type": "application/json"}
            
            # Define date range for the query
            observation_date = None
            if header is not None:
                if "DATE-OBS" in header:
                    observation_date = header["DATE-OBS"]
                elif "DATE" in header:
                    observation_date = header["DATE"]
            
            # Set time range to ±7 days from observation date or current date
            if observation_date:
                try:
                    base_date = datetime.fromisoformat(observation_date.replace('T', ' ').split('.')[0])
                except (ValueError, TypeError):
                    base_date = datetime.now()
            else:
                base_date = datetime.now()
                
            date_min = (base_date - timedelta(days=14)).isoformat()
            date_max = (base_date + timedelta(days=7)).isoformat()
            
            body = {
                "uid": api_key,
                "filter": None,
                "time_range": {
                    "max": date_max,
                    "min": date_min,
                },
                "properties": {
                    "type": "cone",
                    "position": {"ra": field_center_ra, 
                                 "dec": field_center_dec},
                    "radius": search_radius_arcsec,
                }
            }

            # Perform the POST request
            response = requests.post(url, headers=headers,
                                     data=json.dumps(body))

            # Process the response
            try:
                if response.status_code == 200:
                    st.write("Response successfully received.")
                    events = response.json()['voevents']
                    st.success(f'number of events: {len(events)}')
                    # st.success(json.dumps(events, indent=4))
                else:
                    st.warning(f"Request failed with status code: {response.status_code}")

            except json.JSONDecodeError:
                st.error("Request did NOT succeed : ", response.status_code)
                st.error("Error message : ", response.content.decode('UTF-8'))

        except Exception as e:
            st.error(f"Error querying Astro-Colibri API: {str(e)}")
            # Continue with function instead of returning None

        if response is not None and response.status_code == 200:
            sources = {
                    "ra": [],
                    "dec": [],
                    "discoverer_internal_name": [],
                    "type": [],
                    "classification": []
                    }

            # astrostars = pd.DataFrame(source)
            final_table["astrocolibri_name"] = None
            final_table["astrocolibri_type"] = None
            final_table["astrocolibri_classification"] = None
            for event in events:
                if "ra" in event and "dec" in event:
                    sources["ra"].append(event["ra"])
                    sources["dec"].append(event["dec"])
                    sources["discoverer_internal_name"].append(event["discoverer_internal_name"])
                    sources["type"].append(event["type"])
                    sources["classification"].append(event["classification"])
            astrostars = pd.DataFrame(sources)
            st.write("Astro-Colibri sources found in field:")
            st.dataframe(astrostars)
            st.success(f"Found {len(astrostars)} Astro-Colibri sources in field.")

            source_coords = SkyCoord(
                ra=final_table["ra"].values,
                dec=final_table["dec"].values,
                unit="deg",
            )

            astro_colibri_coords = SkyCoord(
                ra=astrostars["ra"],
                dec=astrostars["dec"],
                unit=(u.deg, u.deg),
            )

            if not isinstance(search_radius_arcsec, (int, float)):
                raise ValueError("Search radius must be a number")

            idx, d2d, _ = source_coords.match_to_catalog_sky(astro_colibri_coords)
            matches = d2d < (search_radius_arcsec * u.arcsec)

            for i, (match, match_idx) in enumerate(zip(matches, idx)):
                if match:
                    final_table.loc[i, "astro_colibri_name"] = astrostars["discoverer_internal_name"][match_idx]

            matches = []
            if len(matches) == 0:
                st.write(f"No matched Astro-colibri objects in field.")
            else:
                st.success(f"{len(matches)} Astro-Colibri matched objects in field.")
        else:
            st.write("No Astro-Colibri sources found in the field.")

    except Exception as e:
        st.error(f"Error querying Astro-Colibri: {str(e)}")
        st.write("No Astro-Colibri sources found.")

    status_text.write("Querying SIMBAD for object identifications...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            if not (-360 <= field_center_ra <= 360) or not (
                -90 <= field_center_dec <= 90
            ):
                st.warning(
                    f"Invalid coordinates: RA={field_center_ra}, DEC={field_center_dec}"
                )
            else:
                if "NAXIS1" in header and "NAXIS2" in header:
                    field_width_arcmin = (
                        max(header.get("NAXIS1", 1000), header.get("NAXIS2", 1000))
                        * pixel_scale_arcsec
                        / 60.0
                    )
                else:
                    field_width_arcmin = 30.0

                custom_simbad = Simbad()
                custom_simbad.add_votable_fields("otype", "main_id", "ids", "B", "V")

                st.info("Querying SIMBAD")

                try:
                    center_coord = SkyCoord(
                        ra=field_center_ra, dec=field_center_dec, unit="deg"
                    )
                    simbad_result, error = safe_catalog_query(
                        custom_simbad.query_region,
                        "SIMBAD query failed",
                        center_coord,
                        radius=field_width_arcmin * u.arcmin,
                    )
                    if error:
                        st.warning(error)
                    else:
                        if simbad_result is not None and len(simbad_result) > 0:
                            final_table["simbad_main_id"] = None
                            final_table["simbad_otype"] = None
                            final_table["simbad_ids"] = None
                            final_table["simbad_B"] = None
                            final_table["simbad_V"] = None

                            source_coords = SkyCoord(
                                ra=final_table["ra"].values,
                                dec=final_table["dec"].values,
                                unit="deg",
                            )

                            if all(
                                col in simbad_result.colnames for col in ["ra", "dec"]
                            ):
                                try:
                                    simbad_coords = SkyCoord(
                                        ra=simbad_result["ra"],
                                        dec=simbad_result["dec"],
                                        unit=(u.hourangle, u.deg),
                                    )

                                    idx, d2d, _ = source_coords.match_to_catalog_sky(
                                        simbad_coords
                                    )
                                    matches = d2d < (search_radius_arcsec * u.arcsec)

                                    for i, (match, match_idx) in enumerate(
                                        zip(matches, idx)
                                    ):
                                        if match:
                                            final_table.loc[i, "simbad_main_id"] = (
                                                simbad_result["main_id"][match_idx]
                                            )
                                            final_table.loc[i, "simbad_otype"] = (
                                                simbad_result["otype"][match_idx]
                                            )
                                            final_table.loc[i, "simbad_B"] = (
                                                simbad_result["B"][match_idx]
                                            )
                                            final_table.loc[i, "simbad_V"] = (
                                                simbad_result["V"][match_idx]
                                            )
                                            if "ids" in simbad_result.colnames:
                                                final_table.loc[i, "simbad_ids"] = (
                                                    simbad_result["ids"][match_idx]
                                                )

                                    st.success(
                                        f"Found {sum(matches)} SIMBAD objects in field."
                                    )
                                except Exception as e:
                                    st.error(
                                        f"Error creating SkyCoord objects from SIMBAD data: {str(e)}"
                                    )
                                    st.write(
                                        f"Available SIMBAD columns: {simbad_result.colnames}"
                                    )
                            else:
                                available_cols = ", ".join(simbad_result.colnames)
                                st.error(
                                    f"SIMBAD result missing required columns. Available columns: {available_cols}"
                                )
                        else:
                            st.write("No SIMBAD objects found in the field.")
                except Exception as e:
                    st.error(f"SIMBAD query execution failed: {str(e)}")
        else:
            st.warning("Could not extract field center coordinates from header")
    except Exception as e:
        st.error(f"Error in SIMBAD processing: {str(e)}")

    status_text.write("Querying SkyBoT for solar system objects...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            if "DATE-OBS" in header:
                obs_date = header["DATE-OBS"]
            elif "DATE" in header:
                obs_date = header["DATE"]
            else:
                obs_date = Time.now().isot

            obs_time = Time(obs_date).isot

            sr_value = min(field_width_arcmin / 60.0, 1.0)
            skybot_url = (
                f"http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?"
                f"RA={field_center_ra}&DEC={field_center_dec}&SR={sr_value}&"
                f"EPOCH={quote(obs_time)}&mime=json"
            )

            st.info("Querying SkyBoT for solar system objects...")

            try:
                final_table["skybot_NAME"] = None
                final_table["skybot_OBJECT_TYPE"] = None
                final_table["skybot_MAGV"] = None

                response = requests.get(skybot_url, timeout=15)

                if response.status_code == 200:
                    response_text = response.text.strip()

                    if response_text.startswith("{") or response_text.startswith("["):
                        try:
                            skybot_result = response.json()

                            if "data" in skybot_result and skybot_result["data"]:
                                skybot_coords = SkyCoord(
                                    ra=[
                                        float(obj["RA"])
                                        for obj in skybot_result["data"]
                                    ],
                                    dec=[
                                        float(obj["DEC"])
                                        for obj in skybot_result["data"]
                                    ],
                                    unit=u.deg,
                                )

                                source_coords = SkyCoord(
                                    ra=final_table["ra"].values,
                                    dec=final_table["dec"].values,
                                    unit=u.deg,
                                )

                                idx, d2d, _ = source_coords.match_to_catalog_sky(
                                    skybot_coords
                                )
                                matches = d2d < (search_radius_arcsec * u.arcsec)

                                for i, (match, match_idx) in enumerate(
                                    zip(matches, idx)
                                ):
                                    if match:
                                        obj = skybot_result["data"][match_idx]
                                        final_table.loc[i, "skybot_NAME"] = obj["NAME"]
                                        final_table.loc[i, "skybot_OBJECT_TYPE"] = obj[
                                            "OBJECT_TYPE"
                                        ]
                                        if "MAGV" in obj:
                                            final_table.loc[i, "skybot_MAGV"] = obj[
                                                "MAGV"
                                            ]

                                st.success(
                                    f"Found {sum(matches)} solar system objects in field."
                                )
                            else:
                                st.write("No solar system objects found in the field.")
                        except ValueError as e:
                            st.write(
                                f"No solar system objects found (no valid JSON data returned). {str(e)}"
                            )
                    else:
                        st.write("No solar system objects found in the field.")
                else:
                    st.warning(
                        f"SkyBoT query failed with status code {response.status_code}"
                    )

            except requests.exceptions.RequestException as req_err:
                st.warning(f"Request to SkyBoT failed: {req_err}")
        else:
            st.warning("Could not determine field center for SkyBoT query")
    except Exception as e:
        st.error(f"Error in SkyBoT processing: {str(e)}")

    st.info("Querying AAVSO VSX for variable stars...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            Vizier.ROW_LIMIT = -1
            vizier_result = Vizier.query_region(
                SkyCoord(ra=field_center_ra, dec=field_center_dec, unit=u.deg),
                radius=field_width_arcmin * u.arcmin,
                catalog=["B/vsx/vsx"],
            )

            if (
                vizier_result
                and "B/vsx/vsx" in vizier_result.keys()
                and len(vizier_result["B/vsx/vsx"]) > 0
            ):
                vsx_table = vizier_result["B/vsx/vsx"]

                vsx_coords = SkyCoord(
                    ra=vsx_table["RAJ2000"], dec=vsx_table["DEJ2000"], unit=u.deg
                )
                source_coords = SkyCoord(
                    ra=final_table["ra"].values,
                    dec=final_table["dec"].values,
                    unit=u.deg,
                )

                idx, d2d, _ = source_coords.match_to_catalog_sky(vsx_coords)
                matches = d2d < (search_radius_arcsec * u.arcsec)

                final_table["aavso_Name"] = None
                final_table["aavso_Type"] = None
                final_table["aavso_Period"] = None

                for i, (match, match_idx) in enumerate(zip(matches, idx)):
                    if match:
                        final_table.loc[i, "aavso_Name"] = vsx_table["Name"][match_idx]
                        final_table.loc[i, "aavso_Type"] = vsx_table["Type"][match_idx]
                        if "Period" in vsx_table.colnames:
                            final_table.loc[i, "aavso_Period"] = vsx_table["Period"][
                                match_idx
                            ]

                st.success(f"Found {sum(matches)} variable stars in field.")
            else:
                st.write("No variable stars found in the field.")
    except Exception as e:
        st.error(f"Error querying AAVSO VSX: {e}")

    st.info("Querying Milliquas Catalog for quasars...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            # Set columns to retrieve from the quasar catalog
            v = Vizier(columns=["RAJ2000", "DEJ2000", "Name", "z", "Rmag"])
            v.ROW_LIMIT = -1  # No row limit

            # Query the VII/294 catalog around the field center
            result = v.query_region(
                SkyCoord(ra=field_center_ra, dec=field_center_dec, unit=(u.deg, u.deg)),
                width=field_width_arcmin * u.arcmin,
                catalog="VII/294",
            )

            if result and len(result) > 0:
                qso_table = result[0]

                # Convert to pandas DataFrame for easier matching
                qso_df = qso_table.to_pandas()
                qso_coords = SkyCoord(
                    ra=qso_df["RAJ2000"], dec=qso_df["DEJ2000"], unit=u.deg
                )

                # Create source coordinates for matching
                source_coords = SkyCoord(
                    ra=final_table["ra"], dec=final_table["dec"], unit=u.deg
                )

                # Perform cross-matching
                idx, d2d, _ = source_coords.match_to_catalog_3d(qso_coords)
                matches = d2d.arcsec <= search_radius_arcsec

                # Add matched quasar information to the final table
                final_table["qso_name"] = None
                final_table["qso_redshift"] = None
                final_table["qso_Rmag"] = None

                # Initialize catalog_matches column if it doesn't exist
                if "catalog_matches" not in final_table.columns:
                    final_table["catalog_matches"] = ""

                matched_sources = np.where(matches)[0]
                matched_qsos = idx[matches]

                for source_idx, qso_idx in zip(matched_sources, matched_qsos):
                    final_table.loc[source_idx, "qso_name"] = qso_df.iloc[qso_idx][
                        "Name"
                    ]
                    final_table.loc[source_idx, "qso_redshift"] = qso_df.iloc[qso_idx][
                        "z"
                    ]
                    final_table.loc[source_idx, "qso_Rmag"] = qso_df.iloc[qso_idx][
                        "Rmag"
                    ]

                # Update the catalog_matches column for matched quasars
                has_qso = final_table["qso_name"].notna()
                final_table.loc[has_qso, "catalog_matches"] += "QSO; "

                st.info(f"Found {sum(has_qso)} quasars in field from Milliquas catalog.")
                write_to_log(
                    st.session_state.get("log_buffer"),
                    f"Found {sum(has_qso)} quasar matches in Milliquas catalog",
                    "INFO",
                )
            else:
                st.write("No quasars found in field from Milliquas catalog.")
                write_to_log(
                    st.session_state.get("log_buffer"),
                    "No quasars found in field from Milliquas catalog",
                    "INFO",
                )
    except Exception as e:
        st.error(f"Error querying VizieR Milliquas: {str(e)}")
        write_to_log(
            st.session_state.get("log_buffer"),
            f"Error in Milliquas catalog processing: {str(e)}",
            "ERROR",
        )

    status_text.write("Cross-matching complete")
    final_table["catalog_matches"] = ""

    if "gaia_calib_star" in final_table.columns:
        is_calib = final_table["gaia_calib_star"]
        final_table.loc[is_calib, "catalog_matches"] += "GAIA (calib); "

    if "simbad_main_id" in final_table.columns:
        has_simbad = final_table["simbad_main_id"].notna()
        final_table.loc[has_simbad, "catalog_matches"] += "SIMBAD; "

    if "skybot_NAME" in final_table.columns:
        has_skybot = final_table["skybot_NAME"].notna()
        final_table.loc[has_skybot, "catalog_matches"] += "SkyBoT; "

    if "aavso_Name" in final_table.columns:
        has_aavso = final_table["aavso_Name"].notna()
        final_table.loc[has_aavso, "catalog_matches"] += "AAVSO; "

    if "qso_name" in final_table.columns:
        has_qso = final_table["qso_name"].notna()
        final_table.loc[has_qso, "catalog_matches"] += "QSO; "

    final_table["catalog_matches"] = final_table["catalog_matches"].str.rstrip("; ")
    final_table.loc[final_table["catalog_matches"] == "", "catalog_matches"] = None

    matches_count = final_table["catalog_matches"].notna().sum()
    if matches_count > 0:
        st.subheader(f"Matched Objects Summary ({matches_count} sources)")
        matched_df = final_table[final_table["catalog_matches"].notna()].copy()

        display_cols = [
            "xcenter",
            "ycenter",
            "ra",
            "dec",
            "aperture_calib_mag",
            "catalog_matches",
        ]
        display_cols = [col for col in display_cols if col in matched_df.columns]

        st.dataframe(matched_df[display_cols])

    if "match_id" in final_table.columns:
        final_table.drop("match_id", axis=1, inplace=True)

    return final_table


def save_header_to_txt(header, filename):
    """
    Save a FITS header to a formatted text file.

    Parameters
    ----------
    header : dict or astropy.io.fits.Header
        FITS header dictionary or object
    filename : str
        Base filename to use (without extension)

    Returns
    -------
    str or None
        Full path to the saved file, or None if saving failed

    Notes
    -----
    The function formats the header with each keyword-value pair on a separate line,
    includes a header section, and saves the file to the current output directory
    (retrieved from session state).
    """
    if header is None:
        return None

    header_txt = "FITS Header\n"
    header_txt += "==========\n\n"

    for key, value in header.items():
        header_txt += f"{key:8} = {value}\n"

    output_dir = st.session_state.get("output_dir", ".")
    output_filename = os.path.join(output_dir, f"{filename}.txt")

    with open(output_filename, "w") as f:
        f.write(header_txt)

    return output_filename


def display_catalog_in_aladin(
    final_table: pd.DataFrame,
    ra_center: float,
    dec_center: float,
    fov: float = 0.5,
    ra_col: str = "ra",
    dec_col: str = "dec",
    mag_col: str = "calib_mag",
    alt_mag_col: str = "aperture_calib_mag",
    catalog_col: str = "catalog_matches",
    id_cols: list[str] = ["simbad_main_id", "skybot_NAME", "aavso_Name"],
    fallback_id_prefix: str = "Source",
    survey: str = "CDS/P/DSS2/color",
) -> None:
    """
    Display a DataFrame catalog in an embedded Aladin Lite interactive sky viewer.

    This function creates an interactive astronomical image with catalog overlay
    that allows exploring detected sources and their cross-matches.

    Parameters
    ----------
    final_table : pandas.DataFrame
        DataFrame containing catalog data with coordinates and photometry
    ra_center : float
        Right Ascension center coordinate for the Aladin view (degrees)
    dec_center : float
        Declination center coordinate for the Aladin view (degrees)
    fov : float, optional
        Initial field of view in degrees, default=0.5
    ra_col : str, optional
        Name of the column containing Right Ascension values, default='ra'
    dec_col : str, optional
        Name of the column containing Declination values, default='dec'
    mag_col : str, optional
        Name of the primary magnitude column, default='calib_mag'
    alt_mag_col : str, optional
        Name of an alternative (preferred) magnitude column, default='aperture_calib_mag'
    catalog_col : str, optional
        Name of the column containing catalog match information, default='catalog_matches'
    id_cols : list[str], optional
        List of column names (in order of preference) to use for source identifiers
    fallback_id_prefix : str, optional
        Prefix to use for source names if no ID is found, default="Source"
    survey : str, optional
        The initial sky survey to display in Aladin Lite, default="CDS/P/DSS2/color"

    Notes
    -----
    The function creates an interactive HTML component embedded in Streamlit that
    shows a DSS image of the field with catalog sources overlaid as interactive markers.
    Each marker shows a popup with detailed source information when clicked.

    Handles errors gracefully and provides feedback in the Streamlit interface
    if any issues occur.
    """
    if not (
        isinstance(ra_center, (int, float)) and isinstance(dec_center, (int, float))
    ):
        st.error("Missing or invalid center coordinates (RA/Dec) for Aladin display.")
        return

    if not isinstance(final_table, pd.DataFrame) or final_table.empty:
        st.warning("Input table is empty or not a DataFrame. Cannot display in Aladin.")
        return

    if ra_col not in final_table.columns or dec_col not in final_table.columns:
        st.error(
            f"Required columns '{ra_col}' or '{dec_col}' not found in the DataFrame."
        )
        return

    catalog_sources = []
    required_cols = {ra_col, dec_col}
    optional_cols = {mag_col, alt_mag_col, catalog_col}.union(set(id_cols))
    available_cols = set(final_table.columns)

    present_optional_cols = optional_cols.intersection(available_cols)
    cols_to_iterate = list(required_cols.union(present_optional_cols))

    for idx, row in final_table[cols_to_iterate].iterrows():
        ra_val = row[ra_col]
        dec_val = row[dec_col]
        if pd.notna(ra_val) and pd.notna(dec_val):
            try:
                source = {"ra": float(ra_val), "dec": float(dec_val)}
            except (ValueError, TypeError):
                continue

            mag_to_use = None
            if alt_mag_col in present_optional_cols and pd.notna(row[alt_mag_col]):
                try:
                    mag_to_use = float(row[alt_mag_col])
                except (ValueError, TypeError):
                    pass
            elif mag_col in present_optional_cols and pd.notna(row[mag_col]):
                try:
                    mag_to_use = float(row[mag_col])
                except (ValueError, TypeError):
                    pass

            if mag_to_use is not None:
                source["mag"] = mag_to_use

            if catalog_col in present_optional_cols and pd.notna(row[catalog_col]):
                source["catalog"] = str(row[catalog_col])

            source_id = f"{fallback_id_prefix} {idx + 1}"
            if id_cols:
                for id_col in id_cols:
                    if id_col in present_optional_cols and pd.notna(row[id_col]):
                        source_id = str(row[id_col])
                        break
            source["name"] = source_id

            catalog_sources.append(source)

    if not catalog_sources:
        st.warning("No valid sources with RA/Dec found in the table to display.")
        return

    with st.spinner("Loading Aladin Lite viewer..."):
        try:
            sources_json_b64 = base64.b64encode(
                json.dumps(catalog_sources).encode("utf-8")
            ).decode("utf-8")

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Aladin Lite</title>
            </head>
            <body>
                <div id="aladin-lite-div" style="width:100%;height:550px;"></div>
                <script type="text/javascript" src="https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.js" charset="utf-8"></script>
                <script type="text/javascript">
                    document.addEventListener("DOMContentLoaded", function(event) {{
                        try {{
                            let aladin = A.aladin('#aladin-lite-div', {{
                                target: '{ra_center} {dec_center}',
                                fov: {fov},
                                survey: '{survey}',
                                cooFrame: 'J2000',
                                showReticle: false,
                                showZoomControl: true,
                                showFullscreenControl: true,
                                showLayersControl: true,
                                showGotoControl: true,
                                showSimbadPointerControl: true
                            }});

                            let cat = A.catalog({{
                                name: 'Photometry Results',
                                sourceSize: 12,
                                shape: 'circle',
                                color: '#00ff88',
                                onClick: 'showPopup'
                            }});
                            aladin.addCatalog(cat);

                            let sourcesData = JSON.parse(atob("{sources_json_b64}"));
                            let aladinSources = [];

                            sourcesData.forEach(function(source) {{
                                let popupContent = '<div style="padding:5px;">';
                                if(source.name) {{
                                    popupContent += '<b>' + source.name + '</b><br/>';
                                }}
                                popupContent += 'RA: ' + (typeof source.ra === 'number' ? source.ra.toFixed(6) : source.ra) + '<br/>';
                                popupContent += 'Dec: ' + (typeof source.dec === 'number' ? source.dec.toFixed(6) : source.dec) + '<br/>';

                                if(source.mag) {{
                                    popupContent += 'Mag: ' + (typeof source.mag === 'number' ? source.mag.toFixed(2) : source.mag) + '<br/>';
                                }}
                                if(source.catalog) {{
                                    popupContent += 'Catalogs: ' + source.catalog + '<br/>';
                                }}
                                popupContent += '</div>';

                                let aladinSource = A.source(
                                    source.ra,
                                    source.dec,
                                    {{ description: popupContent }}
                                );
                                aladinSources.push(aladinSource);
                            }});

                            if (aladinSources.length > 0) {{
                                cat.addSources(aladinSources);
                            }}

                        }} catch (error) {{
                            console.error("Error initializing Aladin Lite or adding sources:", error);
                            document.getElementById('aladin-lite-div').innerHTML = '<p style="color:red;">Error loading Aladin viewer. Check console.</p>';
                        }}
                    }});
                </script>
            </body>
            </html>
            """

            components.html(
                html_content,
                height=600,  # Explicitly set a height
                scrolling=True,
            )

        except Exception as e:
            st.error(f"Streamlit failed to render the Aladin HTML component: {str(e)}")
            st.exception(e)


def initialize_session_state():
    """
    Initialize all session state variables for the application.

    This function ensures all required session state variables are properly
    initialized with default values when the application starts or reloads.

    The session state includes:
    - Image data storage (calibrated data, headers, photometry tables)
    - Log handling (buffer and filenames)
    - Coordinate storage (manual input and validated coordinates)
    - Analysis parameters (seeing, thresholds, calibration options)
    - File tracking (loaded files)

    This centralized initialization ensures consistent state management
    throughout the application lifecycle.
    """
    if "calibrated_data" not in st.session_state:
        st.session_state["calibrated_data"] = None
    if "calibrated_header" not in st.session_state:
        st.session_state["calibrated_header"] = None
    if "final_phot_table" not in st.session_state:
        st.session_state["final_phot_table"] = None
    if "epsf_model" not in st.session_state:
        st.session_state["epsf_model"] = None
    if "epsf_photometry_result" not in st.session_state:
        st.session_state["epsf_photometry_result"] = None

    if "log_buffer" not in st.session_state:
        st.session_state["log_buffer"] = None
    if "base_filename" not in st.session_state:
        st.session_state["base_filename"] = "photometry"

    if "manual_ra" not in st.session_state:
        st.session_state["manual_ra"] = ""
    if "manual_dec" not in st.session_state:
        st.session_state["manual_dec"] = ""
    if "valid_ra" not in st.session_state:
        st.session_state["valid_ra"] = None
    if "valid_dec" not in st.session_state:
        st.session_state["valid_dec"] = None

    if "analysis_parameters" not in st.session_state:
        st.session_state["analysis_parameters"] = {
            "seeing": 3.5,
            "threshold_sigma": 3.0,
            "detection_mask": 50,
            "gaia_band": "phot_g_mean_mag",
            "gaia_min_mag": 11.0,
            "gaia_max_mag": 19.0,
            "calibrate_bias": False,
            "calibrate_dark": False,
            "calibrate_flat": False,
        }

    if "files_loaded" not in st.session_state:
        st.session_state["files_loaded"] = {
            "science_file": None,
            "bias_file": None,
            "dark_file": None,
            "flat_file": None,
        }


def get_base_filename(file_obj):
    """
    Extract the base filename from an uploaded file object without extension.

    Parameters
    ----------
    file_obj : UploadedFile
        The uploaded file object from Streamlit's file_uploader

    Returns
    -------
    str
        Base filename without extension(s)

    Notes
    -----
    - Returns "photometry" as default if file_obj is None
    - Handles multiple extensions (e.g., .fits.fz) by applying splitext twice
    """
    if file_obj is None:
        return "photometry"

    original_name = file_obj.name
    base_name = os.path.splitext(original_name)[0]
    base_name = os.path.splitext(base_name)[0]

    return base_name


def initialize_log(base_filename):
    """
    Initialize a log buffer for the current processing session.

    Parameters
    ----------
    base_filename : str
        Base name for the log file (usually from the science file)

    Returns
    -------
    StringIO
        Log buffer object with header information

    Notes
    -----
    Creates a formatted log header with timestamp and input filename,
    which will be written to a file at the end of processing.
    """
    log_buffer = StringIO()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_buffer.write("Photometry Factory for RAPAS Log\n")
    log_buffer.write("===============================\n")
    log_buffer.write(f"Processing started: {timestamp}\n")
    log_buffer.write(f"Input file: {base_filename}\n\n")

    return log_buffer


def write_to_log(log_buffer, message, level="INFO"):
    """
    Write a message to the log buffer with timestamp and severity level.

    Parameters
    ----------
    log_buffer : StringIO
        The log buffer to write to
    message : str
        The message to log
    level : str, optional
        Log severity level (INFO, WARNING, ERROR), default="INFO"

    Notes
    -----
    - Does nothing if log_buffer is None
    - Prepends each log entry with a timestamp and level indicator
    - Format: [HH:MM:SS] LEVEL: Message
    """
    if log_buffer is None:
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    log_buffer.write(f"[{timestamp}] {level}: {message}\n")


def provide_download_buttons(folder_path):
    """
    Creates a single download button for a zip file containing all files in the specified folder.

    This function compresses all files in the given folder into a single zip archive
    and provides a download button for the archive.

    Args:
        folder_path (str): The path to the folder containing files to be zipped and made downloadable

    Returns:
        None: The function creates a Streamlit download button directly in the app interface
    """
    try:
        base_filename = st.session_state.get("base_filename", "")
        
        # Filter files to only include those starting with the base filename prefix
        files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f)) and
            f.startswith(base_filename)
        ]
        if not files:
            st.write("No files found in output directory")
            return

        # Create a timestamp for the zip filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = st.session_state.get("base_filename", "pfr_results")
        zip_filename = f"{base_name}_{timestamp}.zip"

        # Create in-memory zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                file_path = os.path.join(folder_path, file)
                zip_file.write(file_path, arcname=file)

        # Reset buffer position to the beginning
        zip_buffer.seek(0)

        # Create download button for the zip file
        st.download_button(
            label="📦 Download All Results (ZIP)",
            data=zip_buffer,
            file_name=zip_filename,
            mime="application/zip",
            on_click="ignore",
        )

        # Show number of files included
        st.caption(f"Archive contains {len(files)} files")

    except Exception as e:
        st.error(f"Error creating zip archive: {str(e)}")


# def update_observatory_inputs(science_header=None):
#     """Update observatory inputs with values from science header or defaults"""
#     st.header("Observatory Location")

#     # Default values
#     defaults = {
#         "name": "",
#         "latitude": 0.,
#         "longitude": 0.,
#         "elevation": 0.,
#     }

#     # Get values from header if available
#     if science_header is not None:
#         defaults["name"] = science_header.get("OBSERVAT", defaults["name"])
#         defaults["latitude"] = float(
#             science_header.get(
#                 "LATITUDE", science_header.get("LAT-OBS", defaults["latitude"])
#             )
#         )
#         defaults["longitude"] = float(
#             science_header.get(
#                 "LONGITUD", science_header.get("LONG-OBS", defaults["longitude"])
#             )
#         )
#         defaults["elevation"] = float(
#             science_header.get(
#                 "ELEVATIO", science_header.get("ALT-OBS", defaults["elevation"])
#             )
#         )

#     # Create a unique widget ID for this call
#     if "observatory_widget_id" not in st.session_state:
#         st.session_state.observatory_widget_id = 0
#     else:
#         st.session_state.observatory_widget_id += 1
    
#     widget_id = st.session_state.observatory_widget_id
    
#     # Use a unique prefix for widget keys
#     prefix = f"obs_{widget_id}_"

#     # Create the input widgets with the determined values and unique keys
#     observatory_name = st.text_input(
#         "Observatory",
#         value=defaults["name"],
#         help="Name of the observatory",
#         key=f"{prefix}observatory_name",
#     )

#     latitude = st.number_input(
#         "Latitude (°)",
#         value=defaults["latitude"],
#         min_value=-90.0,
#         max_value=90.0,
#         help="Observatory latitude in degrees (-90 to 90)",
#         key=f"{prefix}latitude",
#     )

#     longitude = st.number_input(
#         "Longitude (°)",
#         value=defaults["longitude"],
#         min_value=-180.0,
#         max_value=180.0,
#         help="Observatory longitude in degrees (-180 to 180)",
#         key=f"{prefix}longitude",
#     )

#     elevation = st.number_input(
#         "Elevation (m)",
#         value=defaults["elevation"],
#         min_value=0.0,
#         help="Observatory elevation in meters above sea level",
#         key=f"{prefix}elevation",
#     )

#     return {
#         "name": observatory_name,
#         "latitude": latitude,
#         "longitude": longitude,
#         "elevation": elevation,
#     }


def cleanup_temp_files():
    """
    Remove temporary files created during processing.

    Cleans up:
    1. The temporary files created when uploading the science image
    2. The solved files created during plate solving
    """
    # Clean up temp science file
    if "science_file_path" in st.session_state and st.session_state["science_file_path"]:
        try:
            temp_file = st.session_state["science_file_path"]
            if os.path.exists(temp_file):
                base_dir = os.path.dirname(temp_file)
                temp_dir_files = [f for f in os.listdir(base_dir)
                                  if os.path.isfile(os.path.join(base_dir, f)) and
                                  f.lower().endswith(('.fits', '.fit', '.fts'))]
                for file in temp_dir_files:
                    try:
                        os.remove(os.path.join(base_dir, file))
                    except Exception as e:
                        st.warning(f"Could not remove {file}: {str(e)}")
                st.success("Removed temporary file")
        except Exception as e:
            st.warning(f"Could not remove temporary files: {str(e)}")


###################################################################


# Main Streamlit app
initialize_session_state()

st.title("🔭 _Photometry Factory for RAPAS_")

with st.sidebar:
    st.sidebar.header("Upload FITS Files")

    bias_file = st.file_uploader(
        "Master Bias (optional)", type=["fits", "fit", "fts"],
        key="bias_uploader"
    )
    if bias_file is not None:
        st.session_state.files_loaded["bias_file"] = bias_file

    dark_file = st.file_uploader(
        "Master Dark (optional)", type=["fits", "fit", "fts"], key="dark_uploader"
    )
    if dark_file is not None:
        st.session_state.files_loaded["dark_file"] = dark_file

    flat_file = st.file_uploader(
        "Master Flat (optional)", type=["fits", "fit", "fts"], key="flat_uploader"
    )
    if flat_file is not None:
        st.session_state.files_loaded["flat_file"] = flat_file

    # File uploader for science image
    science_file = st.file_uploader(
        "Science Image (required)", type=["fits", "fit", "fts"], key="science_uploader"
    )
    
    # Get absolute path if we need it (for tools like Siril that need direct file access)
    science_file_path = None
    if science_file is not None:
        # Save the uploaded file to a temporary location to get an absolute path
        
        # Create temporary file with the same extension as the uploaded file
        suffix = os.path.splitext(science_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            # Write the uploaded file content to the temporary file
            tmp_file.write(science_file.getvalue())
            science_file_path = tmp_file.name
        
        st.session_state["science_file_path"] = science_file_path
        # st.info(f"File saved temporarily at: {science_file_path}")
    if science_file is not None:
        st.session_state.files_loaded["science_file"] = science_file

        base_filename = get_base_filename(science_file)
        st.session_state["base_filename"] = base_filename

        st.session_state["log_buffer"] = initialize_log(science_file.name)

    st.header("Calibration Options")
    calibrate_bias = st.checkbox(
        "Apply Bias", value=False, help="Subtract bias frame from science image"
    )
    calibrate_dark = st.checkbox(
        "Apply Dark", value=False, help="Subtract dark frame from science image"
    )
    calibrate_flat = st.checkbox(
        "Apply Flat Field", value=False, help="Divide science image by flat field"
    )

    st.sidebar.header("Astro-Colibri")
    colibri_api_key = st.text_input(
        "API Key", value=None, help="Enter your Astro-Colibri API key", type="password"
    )
    st.caption("[Get your Key](https://astro-colibri.science)")

    st.header("Analysis Parameters")
    seeing = st.slider(
        "Seeing (arcsec)",
        1.0,
        6.0,
        3.0,
        0.5,
        help="Estimate of the atmospheric seeing in arcseconds",
    )
    threshold_sigma = st.slider(
        "Detection Threshold (σ)",
        0.5,
        4.0,
        3.0,
        0.5,
        help="Source detection threshold in sigma above background",
    )
    detection_mask = st.slider(
        "Border Mask (pixels)",
        25,
        200,
        50,
        25,
        help="Size of border to exclude from source detection",
    )

    st.header("Gaia Parameters")
    gaia_band = st.selectbox(
        "Gaia Band",
        ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
        index=0,
        help="Gaia magnitude band to use for calibration",
    )
    gaia_min_mag = st.slider(
        "Gaia Min Magnitude",
        6.0,
        12.0,
        10.0,
        0.5,
        help="Minimum magnitude for Gaia sources",
    )
    gaia_max_mag = st.slider(
        "Gaia Max Magnitude",
        15.0,
        20.0,
        19.0,
        0.5,
        help="Maximum magnitude for Gaia sources",
    )

    st.header("Output Options")
    default_catalog_name = f"{st.session_state['base_filename']}_phot.csv"
    catalog_name = st.text_input("Output Catalog Filename", default_catalog_name)

    st.header("Quick Links")
    col1, col2 = st.columns(2)

    with col1:
        st.link_button("GAIA-Archive", "https://gea.esac.esa.int/archive/")
        st.link_button("Simbad", "http://simbad.u-strasbg.fr/simbad/")
        st.link_button("SkyBoT", "https://ssp.imcce.fr/webservices/skybot/")
        st.link_button("VizieR", "http://vizier.u-strasbg.fr/viz-bin/VizieR")

    with col2:
        st.link_button("Astro-Colibri", "https://astro-colibri.com/")
        st.link_button("X-Match", "http://cdsxmatch.u-strasbg.fr/")
        st.link_button("AAVSO", "https://www.aavso.org/vsx/")
        st.link_button("NED", "https://ned.ipac.caltech.edu/")


output_dir = ensure_output_directory("pfr_results")
st.session_state["output_dir"] = output_dir

with st.sidebar:
    st.header("Observatory Location")
    
    # Initialize default values if not in session state
    if "observatory_name" not in st.session_state:
        st.session_state.observatory_name = ""
    if "observatory_latitude" not in st.session_state:
        st.session_state.observatory_latitude = 0.
    if "observatory_longitude" not in st.session_state:
        st.session_state.observatory_longitude = 0.
    if "observatory_elevation" not in st.session_state:
        st.session_state.observatory_elevation = 0.
    
    # Create the input widgets with permanent keys
    observatory_name = st.text_input(
        "Observatory/Telescope",
        value=st.session_state.observatory_name,
        help="Name of the observatory",
        key="obs_name_input"
    )
    
    latitude = st.number_input(
        "Latitude (°)",
        value=st.session_state.observatory_latitude,
        min_value=-90.0,
        max_value=90.0,
        help="Observatory latitude in degrees (-90 to 90)",
        key="obs_lat_input"
    )
    
    longitude = st.number_input(
        "Longitude (°)",
        value=st.session_state.observatory_longitude,
        min_value=-180.0,
        max_value=180.0,
        help="Observatory longitude in degrees (-180 to 180)",
        key="obs_lon_input"
    )
    
    elevation = st.number_input(
        "Elevation (m)",
        value=st.session_state.observatory_elevation,
        min_value=0.0,
        help="Observatory elevation in meters above sea level",
        key="obs_elev_input"
    )
    
    # Update session state with current values
    st.session_state.observatory_name = observatory_name
    st.session_state.observatory_latitude = latitude
    st.session_state.observatory_longitude = longitude
    st.session_state.observatory_elevation = elevation
    
    # Store in the observatory_data dict for consistency with existing code
    st.session_state.observatory_data = {
        "name": observatory_name,
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
    }

if science_file is not None:
    science_data, science_header = load_fits_data(science_file)
    bias_data, _ = load_fits_data(bias_file)
    dark_data, dark_header = load_fits_data(dark_file)
    flat_data, _ = load_fits_data(flat_file)

    # Update observatory values from header if available, but don't recreate widgets
    if science_header is not None:
        # Only update if values aren't already set by the user (non-default)
        if st.session_state.observatory_name == "":
            try:
                obs_name = science_header.get("TELESCOP", science_header.get("OBSERVER", ""))
                st.session_state.observatory_name = obs_name
                st.experimental_rerun()
            except Exception:
                pass
        if st.session_state.observatory_latitude == 0.:
            try:
                lat = float(science_header.get("SITELAT", science_header.get("LAT-OBS", 0.0)))
                st.session_state.observatory_latitude = lat
                st.experimental_rerun()
            except Exception:
                pass
        if st.session_state.observatory_longitude == 0.:
            try:
                lon = float(science_header.get("SITELONG", science_header.get("LONG-OBS", 0.0)))
                st.session_state.observatory_longitude = lon
                st.experimental_rerun()
            except Exception:
                pass
        if st.session_state.observatory_elevation == 0.:
            try:
                elev = float(science_header.get("ELEVATIO", science_header.get("ALT-OBS", 0.0)))
                st.session_state.observatory_elevation = elev
                st.experimental_rerun()
            except Exception:
                pass
        
        # Update the dictionary
        st.session_state.observatory_data = {
            "name": st.session_state.observatory_name,
            "latitude": st.session_state.observatory_latitude,
            "longitude": st.session_state.observatory_longitude,
            "elevation": st.session_state.observatory_elevation,
        }

    wcs_obj, wcs_error = safe_wcs_create(science_header)
    if wcs_obj is None:
        st.warning(f"No valid WCS found in the FITS header: {wcs_error}")

        use_astrometry = st.checkbox(
            "Attempt plate solving with Siril?",
            value=True,
            help="Uses the Siril platesolve to determine WCS coordinates",
        )

        if use_astrometry:
            with st.spinner(
                "Running plate solve (this may take a while)..."
            ):
                wcs_obj, science_header = solve_with_siril(
                    science_file_path, header=science_header)

                log_buffer = st.session_state["log_buffer"]

                if wcs_obj is not None:
                    st.success("Siril plate solving successful!")
                    write_to_log(
                        log_buffer,
                        "Solved plate with Siril",
                    )

                    wcs_header_filename = (
                        f"{st.session_state['base_filename']}_wcs_header"
                    )
                    wcs_header_file_path = save_header_to_txt(
                        science_header, wcs_header_filename
                    )
                    if wcs_header_file_path:
                        st.info("Updated WCS header saved")
                else:
                    st.error(
                        "Plate solving failed"
                    )
                    write_to_log(
                        log_buffer,
                        "Failed to solve plat",
                        level="ERROR",
                    )
        else:
            st.info(
                "Please enter your astrometry.net API key to proceed with plate solving."
            )
    else:
        st.success("Valid WCS found in the FITS header.")
        log_buffer = st.session_state["log_buffer"]
        write_to_log(log_buffer, "Valid WCS found in header")

    if science_header is not None:
        log_buffer = st.session_state["log_buffer"]

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        header_filename = f"{st.session_state['base_filename']}_header"
        header_file_path = os.path.join(output_dir, f"{header_filename}.txt")

        header_file = save_header_to_txt(science_header, header_filename)
        if header_file:
            write_to_log(log_buffer, f"Saved header to {header_file}")

    log_buffer = st.session_state["log_buffer"]
    write_to_log(log_buffer, f"Loaded science image: {science_file.name}")
    if bias_file:
        write_to_log(log_buffer, f"Loaded bias frame: {bias_file.name}")
    if dark_file:
        write_to_log(log_buffer, f"Loaded dark frame: {dark_file.name}")
    if flat_file:
        write_to_log(log_buffer, f"Loaded flat field: {flat_file.name}")

    if science_header is None:
        science_header = {}
    if dark_header is None:
        dark_header = {}

    st.header("Science Image", anchor="science-image")

    if science_data is not None:
        try:
            fig_preview, ax_preview = plt.subplots(figsize=FIGURE_SIZES["medium"])

            try:
                norm = ImageNormalize(science_data, interval=ZScaleInterval())
                im = ax_preview.imshow(
                    science_data, norm=norm, origin="lower", cmap="viridis"
                )
            except Exception as norm_error:
                st.warning(
                    f"ZScale normalization failed: {norm_error}. Using simple normalization."
                )
                vmin, vmax = np.percentile(science_data, [1, 99])
                im = ax_preview.imshow(
                    science_data, vmin=vmin, vmax=vmax, origin="lower", cmap="viridis"
                )

            fig_preview.colorbar(im, ax=ax_preview, label="Pixel Value")
            ax_preview.set_title(f"{science_file.name}")
            ax_preview.axis("off")

            st.pyplot(fig_preview)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{st.session_state['base_filename']}_image.png"
            image_path = os.path.join(output_dir, image_filename)

            try:
                fig_preview.savefig(image_path, dpi=150, bbox_inches="tight")
                write_to_log(log_buffer, "Saved image plot")
            except Exception as save_error:
                write_to_log(
                    log_buffer,
                    f"Failed to save image plot: {str(save_error)}",
                    level="ERROR",
                )
                st.error(f"Error saving image: {str(save_error)}")

            plt.close(fig_preview)

        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            st.exception(e)
    else:
        st.error(
            "No image data available to display. Check if the file was loaded correctly."
        )

    with st.expander("Science Image Header"):
        if science_header:
            st.text(repr(science_header))
        else:
            st.warning("No header information available for science image.")

    st.subheader("Science Image Statistics")
    if science_data is not None:
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("Mean", f"{np.mean(science_data):.3f}")
        stats_col2.metric("Median", f"{np.median(science_data):.3f}")
        stats_col3.metric("Std Dev", f"{np.std(science_data):.3f}")

        pixel_size_arcsec, pixel_scale_source = extract_pixel_scale(science_header)
        st.metric(
            "Pixel Scale (arcsec/pixel)",
            f"{pixel_size_arcsec:.2f} (header estimation)",
        )
        write_to_log(
            log_buffer,
            f"Pixel scale: {pixel_size_arcsec:.2f} arcsec/pixel ({pixel_scale_source})",
        )

        mean_fwhm_pixel = seeing / pixel_size_arcsec
        st.metric(
            "Mean FWHM (pixels)", f"{mean_fwhm_pixel:.2f} (seeing estimation)"
        )
        write_to_log(
            log_buffer,
            f"Seeing FWHM: {seeing:.2f} arcsec ({mean_fwhm_pixel:.2f} pixels)",
        )

        ra_val, dec_val, coord_source = extract_coordinates(science_header)
        if ra_val is not None and dec_val is not None:
            st.write(f"Target: RA={round(ra_val, 4)}°, DEC={round(dec_val, 4)}°")
            write_to_log(
                log_buffer,
                f"Target coordinates: RA={ra_val}°, DEC={dec_val}° ({coord_source})",
            )
            ra_missing = dec_missing = False
        else:
            st.warning(f"Coordinate issue: {coord_source}")
            write_to_log(
                log_buffer, f"Coordinate issue: {coord_source}", level="WARNING"
            )
            ra_missing = dec_missing = True

        if "manual_ra" not in st.session_state:
            st.session_state["manual_ra"] = ""
        if "manual_dec" not in st.session_state:
            st.session_state["manual_dec"] = ""

        if ra_missing or dec_missing:
            st.warning(
                "Target coordinates (RA/DEC) not found in FITS header. Please enter them manually:"
            )

            coord_col1, coord_col2 = st.columns(2)

            with coord_col1:
                default_ra = st.session_state["manual_ra"]
                if (
                    not default_ra
                    and "NAXIS1" in science_header
                    and "CRPIX1" in science_header
                    and "CD1_1" in science_header
                ):
                    default_ra = str(science_header.get("CRVAL1", ""))

                manual_ra = st.text_input(
                    "Right Ascension (degrees)",
                    value=default_ra,
                    help="Enter RA in decimal degrees (0-360)",
                    key="ra_input",
                )
                st.session_state["manual_ra"] = manual_ra

            with coord_col2:
                default_dec = st.session_state["manual_dec"]
                if (
                    not default_dec
                    and "NAXIS2" in science_header
                    and "CRPIX2" in science_header
                    and "CD2_2" in science_header
                ):
                    default_dec = str(science_header.get("CRVAL2", ""))

                manual_dec = st.text_input(
                    "Declination (degrees)",
                    value=default_dec,
                    help="Enter DEC in decimal degrees (-90 to +90)",
                    key="dec_input",
                )
                st.session_state["manual_dec"] = manual_dec

            if manual_ra and manual_dec:
                try:
                    ra_val = float(manual_ra)
                    dec_val = float(manual_dec)

                    if not (0 <= ra_val < 360):
                        st.error("RA must be between 0 and 360 degrees")
                    elif not (-90 <= dec_val <= 90):
                        st.error("DEC must be between -90 and +90 degrees")
                    else:
                        science_header["RA"] = ra_val
                        science_header["DEC"] = dec_val

                        st.session_state["valid_ra"] = ra_val
                        st.session_state["valid_dec"] = dec_val

                        st.success(
                            f"Using manual coordinates: RA={ra_val}°, DEC={dec_val}°"
                        )
                except ValueError:
                    st.error("RA and DEC must be valid numbers")
            else:
                st.warning("Please enter both RA and DEC coordinates")
        else:
            for ra_key in ["RA", "OBJRA", "RA---", "CRVAL1"]:
                if ra_key in science_header:
                    st.session_state["valid_ra"] = science_header[ra_key]
                    break

            for dec_key in ["DEC", "OBJDEC", "DEC---", "CRVAL2"]:
                if dec_key in science_header:
                    st.session_state["valid_dec"] = science_header[dec_key]
                    break

        if "valid_ra" in st.session_state and "valid_dec" in st.session_state:
            science_header["RA"] = st.session_state["valid_ra"]
            science_header["DEC"] = st.session_state["valid_dec"]

        try:
            # Create observatory dictionary using user inputs
            observatory_data = {
                "name": st.session_state.observatory_data["name"],
                "latitude": st.session_state.observatory_data["latitude"],
                "longitude": st.session_state.observatory_data["longitude"],
                "elevation": st.session_state.observatory_data["elevation"],
            }

            air = airmass(science_header, observatory=observatory_data)
            st.write(f"Airmass: {air:.2f}")
        except Exception as e:
            st.warning(f"Error calculating airmass: {e}")
            air = 1.0
            st.write(f"Using default airmass: {air:.2f}")

        calibration_disabled = not (calibrate_bias or calibrate_dark or calibrate_flat)
        exposure_time_science = science_header.get(
            "EXPOSURE", science_header.get("EXPTIME", 1.0)
        )
        exposure_time_dark = dark_header.get(
            "EXPOSURE", dark_header.get("EXPTIME", exposure_time_science)
        )

        if st.button("Run Image Calibration", disabled=calibration_disabled):
            with st.spinner("Calibrating image..."):
                try:
                    calibrated_data, calibrated_header = calibrate_image_streamlit(
                        science_data,
                        science_header,
                        bias_data,
                        dark_data,
                        flat_data,
                        exposure_time_science,
                        exposure_time_dark,
                        calibrate_bias,
                        calibrate_dark,
                        calibrate_flat,
                    )
                    st.session_state["calibrated_data"] = calibrated_data
                    st.session_state["calibrated_header"] = calibrated_header

                    if calibrated_data is not None:
                        st.header("Calibrated Science Image")
                        norm_calibrated = ImageNormalize(
                            calibrated_data, interval=ZScaleInterval()
                        )
                        fig_calibrated, ax_calibrated = plt.subplots(
                            figsize=FIGURE_SIZES["medium"], dpi=100
                        )
                        im_calibrated = ax_calibrated.imshow(
                            calibrated_data,
                            norm=norm_calibrated,
                            origin="lower",
                            cmap="viridis",
                        )
                        fig_calibrated.colorbar(
                            im_calibrated, ax=ax_calibrated, label="pixel value"
                        )
                        ax_calibrated.set_title("Calibrated Image (zscale)")
                        ax_calibrated.axis("off")
                        st.pyplot(fig_calibrated)
                except Exception as e:
                    st.error(f"Error during calibration: {e}")

        zero_point_button_disabled = science_file is None
        if st.button(
            "Run Zero Point Calibration",
            disabled=zero_point_button_disabled,
            key="run_zp",
        ):
            image_to_process = science_data
            header_to_process = science_header

            if st.session_state["calibrated_data"] is not None:
                image_to_process = st.session_state["calibrated_data"]
                header_to_process = st.session_state["calibrated_header"]

            if image_to_process is not None:
                try:
                    with st.spinner(
                        "Background Extraction, Find Sources and Perform Photometry..."
                    ):
                        phot_table_qtable, epsf_table, daofind, bkg = (
                            find_sources_and_photometry_streamlit(
                                image_to_process,
                                header_to_process,
                                mean_fwhm_pixel,
                                threshold_sigma,
                                detection_mask,
                            )
                        )

                        if phot_table_qtable is not None:
                            phot_table_df = phot_table_qtable.to_pandas().copy(
                                deep=True
                            )
                        else:
                            st.error("No sources detected in the image.")
                            phot_table_df = None

                    if phot_table_df is not None:
                        with st.spinner("Cross-matching with Gaia..."):
                            matched_table = cross_match_with_gaia_streamlit(
                                phot_table_qtable,
                                header_to_process,
                                pixel_size_arcsec,
                                mean_fwhm_pixel,
                                gaia_band,
                                gaia_min_mag,
                                gaia_max_mag,
                            )

                        if matched_table is not None:
                            st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
                            st.dataframe(matched_table.head(10))

                            with st.spinner("Calculating zero point..."):
                                zero_point_value, zero_point_std, zp_plot = (
                                    calculate_zero_point_streamlit(
                                        phot_table_qtable, matched_table, gaia_band, air
                                    )
                                )

                                if zero_point_value is not None:
                                    st.pyplot(zp_plot)
                                    write_to_log(
                                        log_buffer,
                                        f"Zero point: {zero_point_value:.3f} ± {zero_point_std:.3f}",
                                    )
                                    write_to_log(log_buffer, f"Airmass: {air:.2f}")

                                    if "final_phot_table" in st.session_state:
                                        final_table = st.session_state[
                                            "final_phot_table"
                                        ]

                                        try:
                                            if (
                                                "epsf_photometry_result"
                                                in st.session_state
                                                and epsf_table is not None
                                            ):
                                                epsf_df = (
                                                    epsf_table.to_pandas()
                                                    if not isinstance(
                                                        epsf_table, pd.DataFrame
                                                    )
                                                    else epsf_table
                                                )

                                                epsf_x_col = (
                                                    "x_fit"
                                                    if "x_fit" in epsf_df.columns
                                                    else "xcenter"
                                                )
                                                epsf_y_col = (
                                                    "y_fit"
                                                    if "y_fit" in epsf_df.columns
                                                    else "ycenter"
                                                )

                                                final_x_col = (
                                                    "xcenter"
                                                    if "xcenter" in final_table.columns
                                                    else "x_0"
                                                )
                                                final_y_col = (
                                                    "ycenter"
                                                    if "ycenter" in final_table.columns
                                                    else "y_0"
                                                )

                                                if (
                                                    epsf_x_col in epsf_df.columns
                                                    and epsf_y_col in epsf_df.columns
                                                ):
                                                    epsf_df["match_id"] = (
                                                        epsf_df[epsf_x_col]
                                                        .round(2)
                                                        .astype(str)
                                                        + "_"
                                                        + epsf_df[epsf_y_col]
                                                        .round(2)
                                                        .astype(str)
                                                    )

                                                if (
                                                    final_x_col in final_table.columns
                                                    and final_y_col
                                                    in final_table.columns
                                                ):
                                                    final_table["match_id"] = (
                                                        final_table[final_x_col]
                                                        .round(2)
                                                        .astype(str)
                                                        + "_"
                                                        + final_table[final_y_col]
                                                        .round(2)
                                                        .astype(str)
                                                    )
                                                else:
                                                    st.warning(
                                                        f"Coordinate columns not found in final table. Available columns: {final_table.columns.tolist()}"
                                                    )

                                                epsf_cols = {}
                                                epsf_cols["match_id"] = "match_id"
                                                epsf_cols["flux_fit"] = "psf_flux_fit"
                                                epsf_cols["flux_unc"] = "psf_flux_unc"
                                                epsf_cols["instrumental_mag"] = (
                                                    "psf_instrumental_mag"
                                                )

                                                if (
                                                    len(epsf_cols) > 1
                                                    and "match_id" in epsf_df.columns
                                                    and "match_id"
                                                    in final_table.columns
                                                ):
                                                    epsf_subset = epsf_df[
                                                        [
                                                            col
                                                            for col in epsf_cols.keys()
                                                            if col in epsf_df.columns
                                                        ]
                                                    ].rename(columns=epsf_cols)

                                                    final_table = pd.merge(
                                                        final_table,
                                                        epsf_subset,
                                                        on="match_id",
                                                        how="left",
                                                    )

                                                    if (
                                                        "psf_instrumental_mag"
                                                        in final_table.columns
                                                    ):
                                                        final_table["psf_calib_mag"] = (
                                                            final_table[
                                                                "psf_instrumental_mag"
                                                            ]
                                                            + zero_point_value
                                                            + 0.1 * air
                                                        )
                                                        st.success(
                                                            "Added PSF photometry results"
                                                        )

                                                    if (
                                                        "instrumental_mag"
                                                        in final_table.columns
                                                    ):
                                                        if (
                                                            "calib_mag"
                                                            not in final_table.columns
                                                        ):
                                                            final_table[
                                                                "aperture_instrumental_mag"
                                                            ] = final_table[
                                                                "instrumental_mag"
                                                            ]
                                                            final_table[
                                                                "aperture_calib_mag"
                                                            ] = (
                                                                final_table[
                                                                    "instrumental_mag"
                                                                ]
                                                                + zero_point_value
                                                                + 0.1 * air
                                                            )
                                                        else:
                                                            final_table = final_table.rename(
                                                                columns={
                                                                    "instrumental_mag": "aperture_instrumental_mag",
                                                                    "calib_mag": "aperture_calib_mag",
                                                                }
                                                            )
                                                        st.success(
                                                            "Added aperture photometry results"
                                                        )

                                                    final_table.drop(
                                                        "match_id", axis=1, inplace=True
                                                    )

                                            csv_buffer = StringIO()

                                            cols_to_drop = []
                                            for col_name in [
                                                "sky_center.ra",
                                                "sky_center.dec",
                                            ]:
                                                if col_name in final_table.columns:
                                                    cols_to_drop.append(col_name)

                                            if cols_to_drop:
                                                final_table = final_table.drop(
                                                    columns=cols_to_drop
                                                )

                                            if "match_id" in final_table.columns:
                                                final_table.drop(
                                                    "match_id", axis=1, inplace=True
                                                )

                                            final_table["zero_point"] = zero_point_value
                                            final_table["zero_point_std"] = (
                                                zero_point_std
                                            )
                                            final_table["airmass"] = air

                                            st.subheader("Final Photometry Catalog")
                                            st.dataframe(final_table.head(10))

                                            st.success(
                                                f"Catalog includes {len(final_table)} sources."
                                            )

                                            if (
                                                final_table is not None
                                                and "ra" in final_table.columns
                                                and "dec" in final_table.columns
                                            ):
                                                st.subheader(
                                                    "Cross-matching with Astronomical Catalogs"
                                                )
                                                search_radius = (
                                                    2
                                                    * mean_fwhm_pixel
                                                    * pixel_size_arcsec
                                                )
                                                final_table = enhance_catalog_with_crossmatches(
                                                    colibri_api_key,
                                                    final_table,
                                                    matched_table,
                                                    header_to_process,
                                                    pixel_size_arcsec,
                                                    search_radius_arcsec=search_radius,
                                                )
                                            elif final_table is not None:
                                                st.warning(
                                                    "RA/DEC coordinates not available for catalog cross-matching"
                                                )
                                            else:
                                                st.error(
                                                    "Final photometry table is None - cannot perform cross-matching"
                                                )

                                            final_table.to_csv(csv_buffer, index=False)
                                            csv_data = csv_buffer.getvalue()

                                            timestamp_str = (
                                                datetime.now().strftime(
                                                    "%Y%m%d_%H%M%S"
                                                )
                                            )
                                            base_catalog_name = catalog_name
                                            if base_catalog_name.endswith(".csv"):
                                                base_catalog_name = base_catalog_name[
                                                    :-4
                                                ]
                                            filename = f"{base_catalog_name}.csv"

                                            catalog_path = os.path.join(
                                                output_dir, filename
                                            )

                                            with open(catalog_path, "w") as f:
                                                f.write(csv_data)

                                            metadata_filename = (
                                                f"{base_catalog_name}_metadata.txt"
                                            )
                                            metadata_path = os.path.join(
                                                output_dir, metadata_filename
                                            )

                                            with open(metadata_path, "w") as f:
                                                f.write(
                                                    "RAPAS Photometry Analysis Metadata\n"
                                                )
                                                f.write(
                                                    "================================\n\n"
                                                )
                                                f.write(
                                                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                                )
                                                f.write(
                                                    f"Input File: {science_file.name}\n"
                                                )
                                                f.write(f"Catalog File: {filename}\n\n")

                                                f.write(
                                                    f"Zero Point: {zero_point_value:.5f} ± {zero_point_std:.5f}\n"
                                                )
                                                f.write(f"Airmass: {air:.3f}\n")
                                                f.write(
                                                    f"Pixel Scale: {pixel_size_arcsec:.3f} arcsec/pixel\n"
                                                )
                                                f.write(
                                                    f"FWHM (estimated): {mean_fwhm_pixel:.2f} pixels ({seeing:.2f} arcsec)\n\n"
                                                )

                                                f.write("Detection Parameters:\n")
                                                f.write(
                                                    f"  Threshold: {threshold_sigma} sigma\n"
                                                )
                                                f.write(
                                                    f"  Edge Mask: {detection_mask} pixels\n"
                                                )
                                                f.write(
                                                    f"  Gaia Magnitude Range: {gaia_min_mag:.1f} - {gaia_max_mag:.1f}\n"
                                                )

                                            write_to_log(
                                                log_buffer, "Saved catalog metadata"
                                            )

                                        except Exception as e:
                                            st.error(f"Error preparing download: {e}")
                        else:
                            st.error(
                                "Failed to cross-match with Gaia catalog. Check WCS information in image header."
                            )
                except Exception as e:
                    st.error(f"Error during zero point calibration: {str(e)}")
                    st.exception(e)

                ra_center = None
                dec_center = None

                if "CRVAL1" in header_to_process and "CRVAL2" in header_to_process:
                    ra_center = header_to_process["CRVAL1"]
                    dec_center = header_to_process["CRVAL2"]
                elif "RA" in header_to_process and "DEC" in header_to_process:
                    ra_center = header_to_process["RA"]
                    dec_center = header_to_process["DEC"]
                elif "OBJRA" in header_to_process and "OBJDEC" in header_to_process:
                    ra_center = header_to_process["OBJRA"]
                    dec_center = header_to_process["OBJDEC"]

                if ra_center is not None and dec_center is not None:
                    if (
                        "final_phot_table" in st.session_state
                        and st.session_state["final_phot_table"] is not None
                        and not st.session_state["final_phot_table"].empty
                    ):
                        st.subheader("Aladin Catalog Viewer")

                        display_catalog_in_aladin(
                            final_table=final_table,
                            ra_center=ra_center,
                            dec_center=dec_center,
                            fov=1.0,
                            alt_mag_col="aperture_calib_mag",
                            id_cols=["simbad_main_id"],
                        )

                    st.link_button(
                        "ESA Sky Viewer",
                        f"https://sky.esa.int/esasky/?target={ra_center}%20{dec_center}&hips=DSS2+color&fov=0.5&projection=SIN&cooframe=J2000&sci=true&lang=en",
                        help="Open ESA Sky with the same target coordinates",
                    )

                    st.success(f"All Results are stocked in /{output_dir}")
                    provide_download_buttons(output_dir)

                    cleanup_temp_files()

                else:
                    st.warning(
                        "Could not determine coordinates from image header. Cannot display ESASky."
                    )
else:
    st.text("👆 Please upload a science image FITS file to start.", )


if "log_buffer" in st.session_state and st.session_state["log_buffer"] is not None:
    log_buffer = st.session_state["log_buffer"]
    log_filename = f"{st.session_state['base_filename']}.log"
    log_filepath = os.path.join(output_dir, log_filename)

    # Log all sidebar parameters
    write_to_log(log_buffer, "Analysis Parameters", level="INFO")
    write_to_log(log_buffer, f"Seeing: {seeing} arcsec")
    write_to_log(log_buffer, f"Detection Threshold: {threshold_sigma} sigma")
    write_to_log(log_buffer, f"Border Mask: {detection_mask} pixels")

    write_to_log(log_buffer, "Observatory Information", level="INFO")
    write_to_log(
        log_buffer, f"Observatory Name: {st.session_state.observatory_data['name']}"
    )
    write_to_log(
        log_buffer, f"Latitude: {st.session_state.observatory_data['latitude']}°"
    )
    write_to_log(
        log_buffer, f"Longitude: {st.session_state.observatory_data['longitude']}°"
    )
    write_to_log(
        log_buffer, f"Elevation: {st.session_state.observatory_data['elevation']} m"
    )

    write_to_log(log_buffer, "Gaia Parameters", level="INFO")
    write_to_log(log_buffer, f"Gaia Band: {gaia_band}")
    write_to_log(log_buffer, f"Gaia Min Magnitude: {gaia_min_mag}")
    write_to_log(log_buffer, f"Gaia Max Magnitude: {gaia_max_mag}")

    write_to_log(log_buffer, "Calibration Options", level="INFO")
    write_to_log(log_buffer, f"Apply Bias: {calibrate_bias}")
    write_to_log(log_buffer, f"Apply Dark: {calibrate_dark}")
    write_to_log(log_buffer, f"Apply Flat: {calibrate_flat}")

    # Finalize and save the log
    write_to_log(log_buffer, "Processing completed", level="INFO")
    with open(log_filepath, "w") as f:
        f.write(log_buffer.getvalue())
        write_to_log(log_buffer, f"Log saved to {log_filepath}")
