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
import streamlit as st
import streamlit.components.v1 as components
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
from stdpipe import photometry, astrometry, catalogs, pipeline

# Local Application Imports
from tools import (FIGURE_SIZES, URL, GAIA_BAND, extract_coordinates,
                   extract_pixel_scale, get_base_filename, safe_catalog_query,
                   safe_wcs_create, ensure_output_directory, cleanup_temp_files,
                   initialize_log, write_to_log, zip_rpp_results_on_exit)
from __version__ import version

# Conditional Import (already present, just noting its location)
if getattr(sys, "frozen", False):
    import importlib.metadata
    importlib.metadata.distributions = lambda **kwargs: []

warnings.filterwarnings("ignore")


st.set_page_config(page_title="RAPAS Photometry Pipeline", page_icon="🔭",
                   layout="wide")


def initialize_session_state():
    """
    Initialize all session state variables for the application.
    Ensures all required keys have default values.
    """
    # Login/User State
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    # Core Data/Results State
    if "calibrated_header" not in st.session_state:
        st.session_state.calibrated_header = None
    if "final_phot_table" not in st.session_state:
        st.session_state.final_phot_table = None
    if "epsf_model" not in st.session_state:
        st.session_state.epsf_model = None
    if "epsf_photometry_result" not in st.session_state:
        st.session_state.epsf_photometry_result = None

    # Logging and File Handling State
    if "log_buffer" not in st.session_state:
        st.session_state.log_buffer = None
    if "base_filename" not in st.session_state:
        st.session_state.base_filename = "photometry"
    if "science_file_path" not in st.session_state:
        st.session_state.science_file_path = None
    if "output_dir" not in st.session_state:
        # Ensure the default output directory exists when initialized
        st.session_state.output_dir = ensure_output_directory("rpp_results")

    # Analysis Parameters State (consolidated)
    default_analysis_params = {
        "seeing": 3.0,
        "threshold_sigma": 3.0,
        "detection_mask": 25,
        "filter_band": "phot_g_mean_mag",  # Default Gaia band
        "filter_max_mag": 20.0,           # Default Gaia mag limit
        "astrometry_check": False,        # Astrometry refinement toggle
        "calibrate_cosmic_rays": False,   # CRR toggle
        "cr_gain": 1.0,                   # CRR default gain
        "cr_readnoise": 6.5,              # CRR default readnoise
        "cr_sigclip": 4.5,                # CRR default sigclip
        "cr_sigfrac": 0.3,                # CRR default sigfrac
        "cr_objlim": 5.0,                 # CRR default objlim
    }
    if "analysis_parameters" not in st.session_state:
        st.session_state.analysis_parameters = default_analysis_params.copy()
    else:
        # Ensure all keys exist, adding defaults if missing from loaded config
        for key, value in default_analysis_params.items():
            if key not in st.session_state.analysis_parameters:
                st.session_state.analysis_parameters[key] = value

    # Observatory Parameters State
    default_observatory_data = {
        "name": "",
        "latitude": 0.,
        "longitude": 0.,
        "elevation": 0.,
    }
    if "observatory_data" not in st.session_state:
        st.session_state.observatory_data = default_observatory_data.copy()
    else:
        # Ensure all keys exist, adding defaults if missing
        for key, value in default_observatory_data.items():
            if key not in st.session_state.observatory_data:
                st.session_state.observatory_data[key] = value

    # Individual observatory keys for direct widget binding (synced later)
    if "observatory_name" not in st.session_state:
        st.session_state.observatory_name = st.session_state.observatory_data["name"]
    if "observatory_latitude" not in st.session_state:
        st.session_state.observatory_latitude = st.session_state.observatory_data["latitude"]
    if "observatory_longitude" not in st.session_state:
        st.session_state.observatory_longitude = st.session_state.observatory_data["longitude"]
    if "observatory_elevation" not in st.session_state:
        st.session_state.observatory_elevation = st.session_state.observatory_data["elevation"]

    # API Keys State
    if "colibri_api_key" not in st.session_state:
        st.session_state.colibri_api_key = None  # Or load from env var if preferred

    # File Loading State (Track which calibration files are loaded)
    if "files_loaded" not in st.session_state:
        st.session_state.files_loaded = {
            "science_file": None,
        }


# --- Initialize Session State Early ---
initialize_session_state()

# Redirect to login if not authenticated
if not st.session_state.logged_in:
    st.warning("You must log in to access this page.")
    st.switch_page("pages/login.py")

# Add application version to the sidebar
st.sidebar.markdown(f"**App Version:** _{version}_")

# Add logout button at the top right if user is logged in
if st.session_state.logged_in:
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("Logged out successfully.")
        st.switch_page("pages/login.py")

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
    pass


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
    pass


def detection_and_photometry(
    image_data: np.ndarray,
    _science_header: Optional[dict | fits.Header],
    data_not_normalized: Optional[np.ndarray],
    mean_fwhm_pixel: float,
    threshold_sigma: float,
    detection_mask: int,
    filter_band: str,
    gb: str = "Gmag",
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
    # ...existing code...
    bkg, bkg_error = estimate_background(image_data, box_size=128, filter_size=7)
    if bkg is None:
        st.error("Background estimation failed.")
        return None, None, None, None

    # ...existing code...
    fwhm_estimate = fwhm_fit(image_data - bkg.background, mean_fwhm_pixel, mask)

    if fwhm_estimate is None:
        st.warning("FWHM estimation failed, using initial guess.")
        fwhm_estimate = mean_fwhm_pixel # Fallback

    # ...existing code...
    if sources is None or len(sources) == 0:
        st.warning("No sources detected.")
        return None, None, bkg, fwhm_estimate

    # ...existing code...
    try:
        phot_table = aperture_photometry(image_data - bkg.background, apertures)
        # Placeholder for PSF photometry results
        epsf_table = None # perform_psf_photometry would go here
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

    # ...existing code...
    if gaia_table is None or len(gaia_table) == 0:
        st.warning("No Gaia sources found in the field of view.")
        return None

    # ...existing code...
    try:
        # Convert pixel coordinates to SkyCoord
        wcs_obj, wcs_error = safe_wcs_create(_science_header)
        if wcs_obj is None:
            st.error(f"Cannot cross-match: Failed to create WCS object - {wcs_error}")
            return None

        source_coords = wcs_obj.pixel_to_world(_phot_table['xcentroid'], _phot_table['ycentroid'])
        gaia_coords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit=(u.deg, u.deg))

        # Perform cross-matching
        max_separation = 2 * mean_fwhm_pixel * pixel_size_arcsec * u.arcsec
        idx, d2d, _ = source_coords.match_to_catalog_sky(gaia_coords)
        sep_constraint = d2d < max_separation

        matched_phot = _phot_table[sep_constraint]
        matched_gaia = gaia_table[idx[sep_constraint]]

        if len(matched_phot) == 0:
            st.warning("No matches found between detected sources and Gaia catalog.")
            return None

        # Combine matched tables
        matched_phot_df = matched_phot.to_pandas()
        matched_gaia_df = matched_gaia.to_pandas()

        # Ensure unique columns before merging, prefix Gaia columns
        gaia_cols_renamed = {col: f"gaia_{col}" for col in matched_gaia_df.columns if col in matched_phot_df.columns}
        matched_gaia_df.rename(columns=gaia_cols_renamed, inplace=True)

        # Reset index for safe concatenation
        matched_phot_df.reset_index(drop=True, inplace=True)
        matched_gaia_df.reset_index(drop=True, inplace=True)

        combined_table = pd.concat([matched_phot_df, matched_gaia_df], axis=1)

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

    if 'instrumental_mag' not in _matched_table.columns:
        st.error(f"Cannot calculate zero point: 'instrumental_mag' column missing from matched table.")
        return None, None, None

    if filter_band not in _matched_table.columns:
        st.error(f"Cannot calculate zero point: Gaia filter band '{filter_band}' column missing from matched table.")
        return None, None, None

    try:
        # Calculate the difference: Gaia_mag - instrumental_mag
        # This difference represents the zero point (ignoring extinction for now)
        zp_diff = _matched_table[filter_band] - _matched_table['instrumental_mag']

        # Use sigma clipping to get a robust estimate of the zero point
        mean_zp, median_zp, std_zp = sigma_clipped_stats(zp_diff, sigma=3.0)

        zero_point_value = median_zp # Use median as the robust zero point
        zero_point_std = std_zp

        # Apply zero point and extinction correction to the full photometry table
        # Ensure _phot_table is a DataFrame for easier column addition
        if isinstance(_phot_table, Table):
            _phot_table = _phot_table.to_pandas()
        elif not isinstance(_phot_table, pd.DataFrame):
            st.error("Input _phot_table must be an Astropy Table or Pandas DataFrame.")
            return None, None, None

        if 'instrumental_mag' not in _phot_table.columns:
             st.error("'_phot_table' is missing 'instrumental_mag' column for calibration.")
             # Attempt to find a magnitude column if possible, otherwise fail
             mag_cols = [col for col in _phot_table.columns if 'mag' in col.lower()]
             if not mag_cols:
                 return None, None, None
             # Heuristic: use the first found magnitude column
             inst_mag_col_name = mag_cols[0]
             st.warning(f"Using '{inst_mag_col_name}' as instrumental magnitude.")
        else:
            inst_mag_col_name = 'instrumental_mag'

        _phot_table["calib_mag"] = (
            _phot_table[inst_mag_col_name] + zero_point_value + 0.1 * air
        )

        # Update the matched table as well for plotting
        _matched_table["calib_mag"] = (
             _matched_table['instrumental_mag'] + zero_point_value + 0.1 * air
        )

        st.session_state["final_phot_table"] = _phot_table

        # --- Plotting --- #
        fig = create_figure(size="medium", dpi=100)
        ax = fig.add_subplot(111)

        # Calculate residuals
        _matched_table["residual"] = (
            _matched_table[filter_band] - _matched_table["calib_mag"]
        )

        # Create bins for magnitude ranges
        bin_width = 0.5
        min_mag = _matched_table[filter_band].min()
        max_mag = _matched_table[filter_band].max()
        if pd.isna(min_mag) or pd.isna(max_mag):
             st.warning("Could not determine magnitude range for plotting bins.")
             bins = np.array([])
        else:
             bins = np.arange(np.floor(min_mag), np.ceil(max_mag) + bin_width, bin_width)

        # Group data by magnitude bins if bins exist
        if len(bins) > 1:
            grouped = _matched_table.groupby(pd.cut(_matched_table[filter_band], bins))
            bin_centers = [(bin.left + bin.right) / 2 for bin in grouped.groups.keys()]
            bin_means = grouped["calib_mag"].mean().values
            bin_stds = grouped["calib_mag"].std().values
            valid_bins = ~np.isnan(bin_means) & ~np.isnan(bin_stds)
        else:
            valid_bins = np.array([], dtype=bool)
            bin_centers = []
            bin_means = []
            bin_stds = []

        # Plot individual points
        ax.scatter(
            _matched_table[filter_band],
            _matched_table["calib_mag"],
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
                label="Mean ± StdDev (binned)",
                capsize=5,
            )

        # Add a diagonal line for reference (y=x)
        if not pd.isna(min_mag) and not pd.isna(max_mag):
            ideal_mag = np.linspace(min_mag, max_mag, 10)
            ax.plot(ideal_mag, ideal_mag, "k--", alpha=0.7, label="y=x")

        ax.set_xlabel(f"Gaia {filter_band}")
        ax.set_ylabel("Calibrated Instrumental Mag")
        ax.set_title(f"Zero Point Calibration (ZP = {zero_point_value:.2f} ± {zero_point_std:.2f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # Invert y-axis for magnitudes
        ax.invert_yaxis()

        st.success(
            f"Calculated Zero Point: {zero_point_value:.2f} ± {zero_point_std:.2f}"
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
    with st.spinner("Finding sources and performing photometry..."):
        # Assuming data_not_normalized and gb are not strictly needed based on usage
        phot_table_qtable, epsf_table, _, fwhm_estimate = detection_and_photometry(
            image_to_process,
            header,
            None, # data_not_normalized placeholder
            mean_fwhm_pixel,
            threshold_sigma,
            detection_mask,
            filter_band,
            # gb="Gmag" # Default, placeholder
        )

        if phot_table_qtable is None:
            st.error("Source detection or photometry failed. Cannot proceed.")
            return None, None, None
        else:
            st.success(f"Detected {len(phot_table_qtable)} sources.")
            # Add instrumental magnitude calculation if not done in detection_and_photometry
            if 'instrumental_mag' not in phot_table_qtable.colnames:
                 if 'aperture_sum' in phot_table_qtable.colnames:
                      # Basic instrumental mag: -2.5 * log10(flux)
                      # Handle zero or negative flux
                      valid_flux = phot_table_qtable['aperture_sum'] > 0
                      phot_table_qtable['instrumental_mag'] = np.nan
                      phot_table_qtable['instrumental_mag'][valid_flux] = -2.5 * np.log10(phot_table_qtable['aperture_sum'][valid_flux])
                 else:
                      st.warning("Cannot calculate instrumental magnitude: 'aperture_sum' missing.")
                      # Cannot proceed without instrumental magnitudes
                      return None, None, None

    with st.spinner("Cross-matching with Gaia..."):
        # Use the estimated FWHM if available, otherwise the input guess
        fwhm_to_use = fwhm_estimate if fwhm_estimate is not None else mean_fwhm_pixel
        matched_table = cross_match_with_gaia(
            phot_table_qtable,
            header,
            pixel_size_arcsec,
            fwhm_to_use,
            filter_band,
            filter_max_mag,
        )

        if matched_table is None:
            st.warning("Cross-matching with Gaia failed or found no matches. Zero point calibration skipped.")
            # Return the uncalibrated table if matching failed
            final_phot_table = phot_table_qtable.to_pandas() if isinstance(phot_table_qtable, Table) else phot_table_qtable
            st.session_state["final_phot_table"] = final_phot_table
            return None, None, final_phot_table

    st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
    st.dataframe(matched_table.head(10))

    with st.spinner("Calculating zero point..."):
        zero_point_value, zero_point_std, zp_plot = calculate_zero_point(
            phot_table_qtable, matched_table, filter_band, air
        )

        # Retrieve the potentially updated table from session state
        final_phot_table = st.session_state.get("final_phot_table")

        if zero_point_value is not None:
            st.success("Zero point calibration completed.")
            # Optionally add ZP info to the final table metadata or columns if needed
        else:
            st.error("Zero point calculation failed.")
            # Ensure final_phot_table is still set, even if uncalibrated
            if final_phot_table is None:
                 final_phot_table = phot_table_qtable.to_pandas() if isinstance(phot_table_qtable, Table) else phot_table_qtable
                 st.session_state["final_phot_table"] = final_phot_table

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
    Enhance the photometric catalog by cross-matching with multiple astronomical databases.

    Queries GAIA DR3 (using pre-matched calibration stars), Astro-Colibri,
    SIMBAD, SkyBoT (solar system objects), AAVSO VSX (variable stars), and
    VizieR VII/294 (Milliquas quasar catalog) to identify known objects within
    the field of view.

    Parameters
    ----------
    api_key : Optional[str]
        API key for Astro-Colibri service. If None, attempts to read from
        the 'ASTROCOLIBRI_API' environment variable.
    final_table : Optional[pd.DataFrame]
        The main photometry catalog DataFrame, expected to contain 'ra' and 'dec'
        columns in degrees.
    matched_table : Optional[pd.DataFrame]
        DataFrame containing sources already matched with Gaia (used for calibration).
        Used here to flag calibration stars.
    header : Optional[dict | fits.Header]
        FITS header containing observation metadata, used to determine field center,
        observation time (for SkyBoT), and field size.
    pixel_scale_arcsec : Optional[float]
        Pixel scale in arcseconds per pixel. Used with header NAXIS values to
        estimate field size if available.
    search_radius_arcsec : float, optional
        Search radius used for matching sources within the `final_table` to
        catalog results (e.g., Astro-Colibri), by default 60.0 arcseconds.
        Note: Catalog queries (SIMBAD, VizieR, etc.) use the field width.

    Returns
    -------
    Optional[pd.DataFrame]
        The input `final_table` DataFrame enhanced with new columns for each
        catalog queried (e.g., 'simbad_main_id', 'skybot_NAME', 'aavso_Name',
        'qso_name', 'astrocolibri_name'). Also adds a 'catalog_matches' summary
        column listing all catalogs a source was matched in. Returns None or the
        original table if input is invalid.

    Notes
    -----
    - Determines field center and size from the header if possible.
    - Uses `safe_catalog_query` for robust SIMBAD queries.
    - Queries external services (Astro-Colibri, SkyBoT, VizieR) via HTTP requests
      or astroquery, handling potential errors.
    - Matches catalog results back to the `final_table` based on RA/Dec proximity.
    - Updates Streamlit interface with progress messages and summary tables.
    - Logs actions and findings using `write_to_log`.
    """
    # ...existing code...
    if final_table is None or len(final_table) == 0:
        st.warning("No sources in the final table to cross-match with catalogs.")
        return final_table

    # Ensure RA/Dec columns exist
    if 'ra' not in final_table.columns or 'dec' not in final_table.columns:
        st.error("Final table must contain 'ra' and 'dec' columns for cross-matching.")
        return final_table

    # Initialize catalog match columns if they don't exist
    catalog_info = {
        'gaia_calib_star': False, # Flag for stars used in ZP calc
        'astrocolibri_name': None,
        'astrocolibri_type': None,
        'astrocolibri_classification': None,
        'simbad_main_id': None,
        'simbad_otype': None,
        'skybot_NAME': None,
        'skybot_class': None,
        'aavso_Name': None,
        'aavso_Type': None,
        'qso_name': None,
        'qso_redshift': np.nan,
        'catalog_matches': None # Summary column
    }
    for col, default_val in catalog_info.items():
        if col not in final_table.columns:
            # Use appropriate dtype based on default
            dtype = object if default_val is None or isinstance(default_val, str) else type(default_val)
            if dtype == float:
                 final_table[col] = pd.Series(np.nan, index=final_table.index, dtype=float)
            else:
                 final_table[col] = pd.Series(default_val, index=final_table.index, dtype=dtype)

    # Compute field of view (arcmin) ONCE
    # ...existing code...
    if header is not None:
        # ... [logic to find field_center_ra, field_center_dec] ...
        ra_keys = ["CRVAL1", "RA", "OBJRA"]
        dec_keys = ["CRVAL2", "DEC", "OBJDEC"]
        field_center_ra = get_header_value(header, ra_keys)
        field_center_dec = get_header_value(header, dec_keys)

        # Compute field width if possible
        if "NAXIS1" in header and pixel_scale_arcsec:
            field_width_arcmin = (header["NAXIS1"] * pixel_scale_arcsec) / 60.0
        elif "NAXIS2" in header and pixel_scale_arcsec:
             # Fallback using NAXIS2 if NAXIS1 missing
             field_width_arcmin = (header["NAXIS2"] * pixel_scale_arcsec) / 60.0
        else:
             st.warning("Could not determine field width accurately from header.")

    # ...existing code...
    if matched_table is not None and len(matched_table) > 0:
        status_text.write("Adding Gaia calibration matches...")
        # Assuming matched_table contains unique identifiers or coordinates
        # that can be used to match back to final_table.
        # If matched_table has 'ra'/'dec', we can match by coordinates.
        # If it has an index corresponding to final_table, use that.

        # Example: Matching by coordinates (adjust tolerance as needed)
        if 'ra' in matched_table.columns and 'dec' in matched_table.columns:
            calib_coords = SkyCoord(ra=matched_table['ra'].values, dec=matched_table['dec'].values, unit='deg')
            final_coords = SkyCoord(ra=final_table['ra'].values, dec=final_table['dec'].values, unit='deg')
            idx, d2d, _ = final_coords.match_to_catalog_sky(calib_coords)
            # Use a small tolerance, e.g., 1 arcsec, assuming they are the same sources
            is_calib_match = d2d < 1.0 * u.arcsec
            final_table.loc[is_calib_match, 'gaia_calib_star'] = True
            st.write(f"Flagged {is_calib_match.sum()} Gaia calibration stars.")
        else:
            st.warning("Could not identify Gaia calibration stars in the final table.")

    # --- Astro-Colibri Query --- #
    if field_center_ra is not None and field_center_dec is not None:
        # ... [validation for field_center_ra, field_center_dec] ...
        if not (-360 <= field_center_ra <= 360) or not (-90 <= field_center_dec <= 90):
            st.error("Invalid field center coordinates from header.")
            # Decide whether to proceed without Astro-Colibri or stop
        else:
            st.info("Querying Astro-Colibri API...")
            if api_key is None:
                api_key = os.environ.get("ASTROCOLIBRI_API")
                if api_key is None:
                    st.warning("Astro-Colibri API key not provided. Skipping query.")
                else:
                    st.info("Using Astro-Colibri API key from environment variable.")

            if api_key:
                try:
                    # ... [Astro-Colibri API call logic] ...
                    url = "https://astro-colibri.science/api/v1/events/cone_search"
                    headers = {"Content-Type": "application/json"}
                    observation_date = get_header_value(header, ["DATE-OBS", "DATE"])
                    if observation_date:
                        try:
                            base_date = Time(observation_date).datetime
                        except ValueError:
                            st.warning(f"Could not parse observation date '{observation_date}', using current date.")
                            base_date = datetime.utcnow()
                    else:
                        st.warning("Observation date not found in header, using current date.")
                        base_date = datetime.utcnow()

                    date_min = (base_date - timedelta(days=14)).isoformat()
                    date_max = (base_date + timedelta(days=7)).isoformat()

                    body = {
                        "uid": api_key,
                        "filter": None,
                        "time_range": {"max": date_max, "min": date_min},
                        "properties": {
                            "type": "cone",
                            "position": {"ra": field_center_ra, "dec": field_center_dec},
                            "radius": field_width_arcmin / 2.0, # Use field radius in arcmin
                        },
                    }
                    response = requests.post(url, headers=headers, data=json.dumps(body), timeout=30)
                    response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                    events = response.json().get("events", [])
                    if events:
                        # ... [Process Astro-Colibri results and match to final_table] ...
                        sources = {
                            "ra": [], "dec": [], "name": [], "type": [], "classification": []
                        }
                        for event in events:
                            pos = event.get('position', {}).get('coordinates')
                            if pos and len(pos) == 2:
                                sources["ra"].append(pos[0])
                                sources["dec"].append(pos[1])
                                sources["name"].append(event.get('discoverer_internal_name', 'Unknown'))
                                sources["type"].append(event.get('type', 'Unknown'))
                                sources["classification"].append(event.get('classification', 'Unknown'))

                        astrostars = pd.DataFrame(sources)
                        st.success(f"Found {len(astrostars)} Astro-Colibri sources in field.")
                        # st.dataframe(astrostars) # Optional: display raw results

                        # Match to final_table
                        if not astrostars.empty:
                            source_coords = SkyCoord(ra=final_table["ra"].values, dec=final_table["dec"].values, unit="deg")
                            astro_colibri_coords = SkyCoord(ra=astrostars["ra"].values, dec=astrostars["dec"].values, unit="deg")

                            match_radius = search_radius_arcsec * u.arcsec
                            idx, d2d, _ = source_coords.match_to_catalog_sky(astro_colibri_coords)
                            matches = d2d < match_radius

                            matched_indices_final = np.where(matches)[0]
                            matched_indices_astro = idx[matches]

                            final_table.loc[matched_indices_final, 'astrocolibri_name'] = astrostars.iloc[matched_indices_astro]['name'].values
                            final_table.loc[matched_indices_final, 'astrocolibri_type'] = astrostars.iloc[matched_indices_astro]['type'].values
                            final_table.loc[matched_indices_final, 'astrocolibri_classification'] = astrostars.iloc[matched_indices_astro]['classification'].values

                            st.success(f"Matched {len(matched_indices_final)} sources with Astro-Colibri.")
                            write_to_log(st.session_state.get("log_buffer"), f"Matched {len(matched_indices_final)} sources with Astro-Colibri.")
                        else:
                             st.write("No valid Astro-Colibri sources to match.")
                    else:
                        st.write("No Astro-Colibri events found in the specified region and time range.")
                        write_to_log(st.session_state.get("log_buffer"), "No Astro-Colibri events found.")

                except requests.exceptions.RequestException as e:
                    st.error(f"Error querying Astro-Colibri API: {str(e)}")
                    write_to_log(st.session_state.get("log_buffer"), f"Astro-Colibri API Error: {str(e)}", "ERROR")
                except json.JSONDecodeError:
                    st.error("Error decoding Astro-Colibri API response.")
                    write_to_log(st.session_state.get("log_buffer"), "Astro-Colibri JSON Decode Error", "ERROR")
                except Exception as e:
                    st.error(f"An unexpected error occurred during Astro-Colibri query: {str(e)}")
                    write_to_log(st.session_state.get("log_buffer"), f"Astro-Colibri Unexpected Error: {str(e)}", "ERROR")
    else:
        st.warning("Could not extract field center coordinates from header. Skipping Astro-Colibri query.")
        write_to_log(st.session_state.get("log_buffer"), "Skipped Astro-Colibri query due to missing field center.")

    # --- SIMBAD Query --- #
    status_text.write("Querying SIMBAD for object identifications...")
    if field_center_ra is not None and field_center_dec is not None:
        try:
            custom_simbad = Simbad()
            # Add fields relevant for identification
            custom_simbad.add_votable_fields("otype", "main_id", "ids")
            # Remove default flux fields if not needed
            # custom_simbad.remove_votable_fields('flux(U)', 'flux(B)', ...) 

            center_coord = SkyCoord(ra=field_center_ra, dec=field_center_dec, unit="deg")
            simbad_result, error = safe_catalog_query(
                custom_simbad.query_region,
                "SIMBAD query failed",
                center_coord,
                radius=field_width_arcmin * u.arcmin,
            )
            if error:
                st.warning(f"SIMBAD query issue: {error}")
                write_to_log(st.session_state.get("log_buffer"), f"SIMBAD query issue: {error}", "WARNING")
            elif simbad_result is not None and len(simbad_result) > 0:
                st.success(f"Found {len(simbad_result)} SIMBAD objects in field.")
                simbad_df = simbad_result.to_pandas()
                # Match to final_table
                source_coords = SkyCoord(ra=final_table["ra"].values, dec=final_table["dec"].values, unit="deg")
                simbad_coords = SkyCoord(ra=simbad_df["RA"].values, dec=simbad_df["DEC"].values, unit="deg", frame='icrs') # Assuming Simbad returns ICRS

                match_radius = search_radius_arcsec * u.arcsec
                idx, d2d, _ = source_coords.match_to_catalog_sky(simbad_coords)
                matches = d2d < match_radius

                matched_indices_final = np.where(matches)[0]
                matched_indices_simbad = idx[matches]

                final_table.loc[matched_indices_final, 'simbad_main_id'] = simbad_df.iloc[matched_indices_simbad]['MAIN_ID'].values
                final_table.loc[matched_indices_final, 'simbad_otype'] = simbad_df.iloc[matched_indices_simbad]['OTYPE'].values

                st.success(f"Matched {len(matched_indices_final)} sources with SIMBAD.")
                write_to_log(st.session_state.get("log_buffer"), f"Matched {len(matched_indices_final)} sources with SIMBAD.")
            else:
                st.write("No SIMBAD objects found in the field.")
                write_to_log(st.session_state.get("log_buffer"), "No SIMBAD objects found.")
        except Exception as e:
            st.error(f"SIMBAD query execution failed: {str(e)}")
            write_to_log(st.session_state.get("log_buffer"), f"SIMBAD query execution failed: {str(e)}", "ERROR")
    else:
        st.warning("Skipping SIMBAD query due to missing field center.")
        write_to_log(st.session_state.get("log_buffer"), "Skipped SIMBAD query due to missing field center.")

    # --- SkyBoT Query --- #
    status_text.write("Querying SkyBoT for solar system objects...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            obs_date = get_header_value(header, ["DATE-OBS", "DATE"])
            if not obs_date:
                st.warning("Observation date not found for SkyBoT query, using current time.")
                obs_time_obj = Time.now()
            else:
                try:
                    obs_time_obj = Time(obs_date)
                except ValueError:
                    st.warning(f"Could not parse observation date '{obs_date}' for SkyBoT, using current time.")
                    obs_time_obj = Time.now()

            obs_time_iso = obs_time_obj.isot
            # Use a slightly larger radius for cone search than field width? Or stick to field width?
            # Skybot uses degrees for SR
            sr_value_deg = field_width_arcmin / 60.0
            # Ensure minimum radius if field is very small?
            sr_value_deg = max(sr_value_deg, 0.1) # Example: min 6 arcmin radius

            skybot_url = (
                f"http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?"
                f"RA={field_center_ra}&DEC={field_center_dec}&SR={sr_value_deg:.4f}&"
                f"EPOCH={quote(obs_time_iso)}&mime=json"
            )
            st.info(f"Querying SkyBoT: RA={field_center_ra:.4f}, Dec={field_center_dec:.4f}, SR={sr_value_deg:.4f} deg, Epoch={obs_time_iso}")

            try:
                skybot_response = requests.get(skybot_url, timeout=30)
                skybot_response.raise_for_status()
                skybot_data = skybot_response.json()

                if skybot_data and isinstance(skybot_data, list) and len(skybot_data) > 0:
                    st.success(f"Found {len(skybot_data)} potential solar system objects via SkyBoT.")
                    skybot_df = pd.DataFrame(skybot_data)
                    # Convert SkyBoT RA/Dec (often sexagesimal) to degrees if needed
                    # Assuming SkyBoT returns degrees directly in JSON response based on mime=json
                    skybot_df['RA_deg'] = skybot_df['RA'].astype(float)
                    skybot_df['DEC_deg'] = skybot_df['DEC'].astype(float)

                    # Match to final_table
                    source_coords = SkyCoord(ra=final_table["ra"].values, dec=final_table["dec"].values, unit="deg")
                    skybot_coords = SkyCoord(ra=skybot_df["RA_deg"].values, dec=skybot_df["DEC_deg"].values, unit="deg")

                    match_radius = search_radius_arcsec * u.arcsec
                    idx, d2d, _ = source_coords.match_to_catalog_sky(skybot_coords)
                    matches = d2d < match_radius

                    matched_indices_final = np.where(matches)[0]
                    matched_indices_skybot = idx[matches]

                    final_table.loc[matched_indices_final, 'skybot_NAME'] = skybot_df.iloc[matched_indices_skybot]['NAME'].values
                    final_table.loc[matched_indices_final, 'skybot_class'] = skybot_df.iloc[matched_indices_skybot]['CLASS'].values

                    st.success(f"Matched {len(matched_indices_final)} sources with SkyBoT.")
                    write_to_log(st.session_state.get("log_buffer"), f"Matched {len(matched_indices_final)} sources with SkyBoT.")
                else:
                    st.write("No solar system objects found by SkyBoT in the field.")
                    write_to_log(st.session_state.get("log_buffer"), "No SkyBoT objects found.")

            except requests.exceptions.RequestException as req_err:
                st.warning(f"SkyBoT query failed: {req_err}")
                write_to_log(st.session_state.get("log_buffer"), f"SkyBoT query failed: {req_err}", "WARNING")
            except json.JSONDecodeError:
                 st.warning("Failed to decode SkyBoT response.")
                 write_to_log(st.session_state.get("log_buffer"), "SkyBoT JSON Decode Error", "WARNING")
        else:
            st.warning("Could not determine field center for SkyBoT query")
            write_to_log(st.session_state.get("log_buffer"), "Skipped SkyBoT query due to missing field center.")
    except Exception as e:
        st.error(f"Error in SkyBoT processing: {str(e)}")
        write_to_log(st.session_state.get("log_buffer"), f"SkyBoT processing error: {str(e)}", "ERROR")

    # --- AAVSO VSX Query --- #
    st.info("Querying AAVSO VSX for variable stars...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            Vizier.ROW_LIMIT = -1 # Ensure all results are fetched
            # Query VizieR B/vsx/vsx catalog
            vizier_result = Vizier.query_region(
                SkyCoord(ra=field_center_ra, dec=field_center_dec, unit=u.deg),
                radius=field_width_arcmin * u.arcmin,
                catalog=["B/vsx/vsx"],
            )

            if vizier_result and "B/vsx/vsx" in vizier_result.keys() and len(vizier_result["B/vsx/vsx"]) > 0:
                vsx_table = vizier_result["B/vsx/vsx"]
                st.success(f"Found {len(vsx_table)} AAVSO VSX entries in field.")
                vsx_df = vsx_table.to_pandas()

                # Match to final_table
                source_coords = SkyCoord(ra=final_table["ra"].values, dec=final_table["dec"].values, unit="deg")
                # VSX coordinates are often J2000
                vsx_coords = SkyCoord(ra=vsx_df["RAJ2000"].values, dec=vsx_df["DEJ2000"].values, unit="deg")

                match_radius = search_radius_arcsec * u.arcsec
                idx, d2d, _ = source_coords.match_to_catalog_sky(vsx_coords)
                matches = d2d < match_radius

                matched_indices_final = np.where(matches)[0]
                matched_indices_vsx = idx[matches]

                final_table.loc[matched_indices_final, 'aavso_Name'] = vsx_df.iloc[matched_indices_vsx]['Name'].values
                final_table.loc[matched_indices_final, 'aavso_Type'] = vsx_df.iloc[matched_indices_vsx]['Type'].values

                st.success(f"Matched {len(matched_indices_final)} sources with AAVSO VSX.")
                write_to_log(st.session_state.get("log_buffer"), f"Matched {len(matched_indices_final)} sources with AAVSO VSX.")
            else:
                st.write("No AAVSO VSX variable stars found in the field.")
                write_to_log(st.session_state.get("log_buffer"), "No AAVSO VSX objects found.")
        else:
             st.warning("Skipping AAVSO VSX query due to missing field center.")
             write_to_log(st.session_state.get("log_buffer"), "Skipped AAVSO VSX query due to missing field center.")
    except Exception as e:
        st.error(f"Error querying AAVSO VSX via VizieR: {e}")
        write_to_log(st.session_state.get("log_buffer"), f"AAVSO VSX query error: {str(e)}", "ERROR")

    # --- Milliquas Quasar Query --- #
    st.info("Querying Milliquas Catalog (VII/294) for quasars...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            v = Vizier(columns=["RAJ2000", "DEJ2000", "Name", "z", "Rmag"])
            v.ROW_LIMIT = -1
            result_tables = v.query_region(
                SkyCoord(ra=field_center_ra, dec=field_center_dec, unit=(u.deg, u.deg)),
                radius=field_width_arcmin * u.arcmin, # Use radius here
                catalog="VII/294",
            )

            if result_tables and len(result_tables) > 0:
                qso_table = result_tables[0] # Assuming the first table is the result
                st.success(f"Found {len(qso_table)} quasars in field from Milliquas catalog.")
                qso_df = qso_table.to_pandas()

                # Match to final_table
                source_coords = SkyCoord(ra=final_table["ra"].values, dec=final_table["dec"].values, unit="deg")
                qso_coords = SkyCoord(ra=qso_df["RAJ2000"].values, dec=qso_df["DEJ2000"].values, unit="deg")

                match_radius = search_radius_arcsec * u.arcsec
                idx, d2d, _ = source_coords.match_to_catalog_sky(qso_coords)
                matches = d2d < match_radius

                matched_indices_final = np.where(matches)[0]
                matched_indices_qso = idx[matches]

                final_table.loc[matched_indices_final, 'qso_name'] = qso_df.iloc[matched_indices_qso]['Name'].values
                # Ensure redshift 'z' column exists and handle potential non-numeric values
                if 'z' in qso_df.columns:
                     final_table.loc[matched_indices_final, 'qso_redshift'] = pd.to_numeric(qso_df.iloc[matched_indices_qso]['z'].values, errors='coerce')
                else:
                     final_table.loc[matched_indices_final, 'qso_redshift'] = np.nan

                st.success(f"Matched {len(matched_indices_final)} sources with Milliquas quasars.")
                write_to_log(st.session_state.get("log_buffer"), f"Matched {len(matched_indices_final)} sources with Milliquas.")
            else:
                st.write("No quasars found in field from Milliquas catalog.")
                write_to_log(st.session_state.get("log_buffer"), "No Milliquas quasars found.")
        else:
             st.warning("Skipping Milliquas query due to missing field center.")
             write_to_log(st.session_state.get("log_buffer"), "Skipped Milliquas query due to missing field center.")
    except Exception as e:
        st.error(f"Error querying VizieR Milliquas (VII/294): {str(e)}")
        write_to_log(st.session_state.get("log_buffer"), f"Milliquas query error: {str(e)}", "ERROR")

    # --- Final Summary Column --- #
    status_text.write("Generating catalog match summary...")
    final_table["catalog_matches"] = ""

    # Add flags based on which columns have non-null values
    if "gaia_calib_star" in final_table.columns:
        is_calib = final_table["gaia_calib_star"] == True # Explicit check for True
        final_table.loc[is_calib, "catalog_matches"] += "GAIA(calib); "

    if "astrocolibri_name" in final_table.columns:
        has_match = final_table["astrocolibri_name"].notna()
        final_table.loc[has_match, "catalog_matches"] += "AstroColibri; "

    if "simbad_main_id" in final_table.columns:
        has_match = final_table["simbad_main_id"].notna()
        final_table.loc[has_match, "catalog_matches"] += "SIMBAD; "

    if "skybot_NAME" in final_table.columns:
        has_match = final_table["skybot_NAME"].notna()
        final_table.loc[has_match, "catalog_matches"] += "SkyBoT; "

    if "aavso_Name" in final_table.columns:
        has_match = final_table["aavso_Name"].notna()
        final_table.loc[has_match, "catalog_matches"] += "AAVSO; "

    if "qso_name" in final_table.columns:
        has_match = final_table["qso_name"].notna()
        final_table.loc[has_match, "catalog_matches"] += "QSO; "

    # Clean up the summary string
    final_table["catalog_matches"] = final_table["catalog_matches"].str.strip("; ")
    # Replace empty strings with None for clarity
    final_table.loc[final_table["catalog_matches"] == "", "catalog_matches"] = None

    status_text.write("Cross-matching complete.")

    # Display summary of matched objects
    matches_count = final_table["catalog_matches"].notna().sum()
    if matches_count > 0:
        st.subheader(f"Matched Objects Summary ({matches_count} sources)")
        matched_df = final_table[final_table["catalog_matches"].notna()].copy()

        # Define columns to display, ensuring they exist
        display_cols_base = [
            "xcenter", "ycenter", "ra", "dec",
        ]
        # Try to find the best magnitude column available
        mag_col_options = ['calib_mag', 'instrumental_mag', 'aperture_sum']
        mag_col_to_display = next((col for col in mag_col_options if col in matched_df.columns), None)
        if mag_col_to_display:
             display_cols_base.append(mag_col_to_display)

        display_cols_final = display_cols_base + ["catalog_matches"]
        display_cols_final = [col for col in display_cols_final if col in matched_df.columns]

        # Format coordinate columns for display
        for col in ['ra', 'dec', 'xcenter', 'ycenter']:
             if col in display_cols_final:
                  matched_df[col] = matched_df[col].map('{:.4f}'.format)
        if mag_col_to_display and mag_col_to_display in display_cols_final:
             matched_df[mag_col_to_display] = matched_df[mag_col_to_display].map('{:.3f}'.format)

        st.dataframe(matched_df[display_cols_final])
    else:
        st.info("No sources were matched with any external catalogs.")

    # Clean up temporary columns if any were added (e.g., match_id)
    # if "match_id" in final_table.columns:
    #     final_table.drop("match_id", axis=1, inplace=True)

    return final_table


def save_header_to_txt(header: Optional[dict | fits.Header], filename: str) -> Optional[str]:
    """
    Save a FITS header to a formatted text file in the designated output directory.

    Parameters
    ----------
    header : Optional[dict | astropy.io.fits.Header]
        The FITS header object or dictionary to save. If None, returns None.
    filename : str
        The base filename (without extension) for the output text file.
        The final name will be '{filename}.txt'.

    Returns
    -------
    Optional[str]
        The full path to the saved text file if successful, otherwise None.

    Notes
    -----
    - Retrieves the output directory path from `st.session_state['output_dir']`.
    - Formats the header with 'KEY = VALUE' pairs, one per line.
    - Includes a simple text header within the file.
    - Handles potential errors during file writing.
    """
    # ...existing code...
    try:
        with open(output_filename, "w", encoding='utf-8') as f:
            f.write(header_txt)
        return output_filename
    except Exception as e:
        st.error(f"Failed to save header file {output_filename}: {e}")
        return None


def display_catalog_in_aladin(
    final_table: Optional[pd.DataFrame],
    ra_center: Optional[float],
    dec_center: Optional[float],
    fov: float = 0.5, # Default FOV in degrees
    ra_col: str = "ra",
    dec_col: str = "dec",
    mag_col: str = "calib_mag", # Primary magnitude column
    alt_mag_col: str = "instrumental_mag", # Fallback magnitude column
    catalog_col: str = "catalog_matches",
    id_cols: List[str] = [
        "simbad_main_id", "aavso_Name", "qso_name", "skybot_NAME", "astrocolibri_name"
    ],
    fallback_id_prefix: str = "Source",
    survey: str = "CDS/P/DSS2/color",
) -> None:
    """
    Display a catalog DataFrame in an embedded Aladin Lite viewer within Streamlit.

    Generates an interactive HTML component showing a sky survey image centered
    at the given coordinates, overlaying sources from the DataFrame as clickable markers.

    Parameters
    ----------
    final_table : Optional[pd.DataFrame]
        DataFrame containing the catalog data. Must include columns specified by
        `ra_col` and `dec_col`.
    ra_center : Optional[float]
        Right Ascension (degrees) for the center of the Aladin view.
    dec_center : Optional[float]
        Declination (degrees) for the center of the Aladin view.
    fov : float, optional
        Initial field of view in degrees, by default 0.5.
    ra_col : str, optional
        Column name for Right Ascension values, by default "ra".
    dec_col : str, optional
        Column name for Declination values, by default "dec".
    mag_col : str, optional
        Primary column name for source magnitude, by default "calib_mag".
    alt_mag_col : str, optional
        Fallback column name for source magnitude if `mag_col` is absent or NaN,
        by default "instrumental_mag".
    catalog_col : str, optional
        Column name for catalog match information, by default "catalog_matches".
    id_cols : List[str], optional
        List of column names (in order of preference) to use for the source name
        displayed in the popup. Defaults to common catalog ID columns.
    fallback_id_prefix : str, optional
        Prefix used for the source name if no ID is found in `id_cols`, followed
        by the DataFrame index + 1. Default is "Source".
    survey : str, optional
        Initial sky survey image to display (e.g., "CDS/P/DSS2/color").
        Default is "CDS/P/DSS2/color".

    Returns
    -------
    None
        Renders the Aladin Lite component directly in the Streamlit app.

    Notes
    -----
    - Handles missing or invalid input gracefully.
    - Constructs JavaScript code to initialize Aladin Lite and add sources.
    - Source popups display Name, RA, Dec, Magnitude (if available), and Catalog matches.
    - Uses `streamlit.components.v1.html` to embed the viewer.
    """
    # ...existing code...
    if not isinstance(final_table, pd.DataFrame) or final_table.empty:
        st.warning("Input table is empty or not a DataFrame. Cannot display in Aladin.")
        return

    # ...existing code...
    for idx, row in final_table[cols_to_iterate].iterrows():
        ra_val = row[ra_col]
        dec_val = row[dec_col]
        if pd.notna(ra_val) and pd.notna(dec_val):
            try:
                # Ensure coordinates are valid floats
                source = {"ra": float(ra_val), "dec": float(dec_val)}
            except (ValueError, TypeError):
                st.warning(f"Skipping source at index {idx} due to invalid coordinates: RA={ra_val}, Dec={dec_val}")
                continue # Skip this source if coordinates are invalid

            mag_to_use = None
            # Prioritize alt_mag_col if it exists and is valid
            if alt_mag_col in present_optional_cols and pd.notna(row[alt_mag_col]):
                try:
                    mag_to_use = float(row[alt_mag_col])
                except (ValueError, TypeError):
                    mag_to_use = None # Ignore invalid magnitude
            # Fallback to mag_col if alt_mag_col was not used
            elif mag_col in present_optional_cols and pd.notna(row[mag_col]):
                try:
                    mag_to_use = float(row[mag_col])
                except (ValueError, TypeError):
                    mag_to_use = None # Ignore invalid magnitude

            if mag_to_use is not None:
                source["mag"] = mag_to_use

            if catalog_col in present_optional_cols and pd.notna(row[catalog_col]):
                source["catalog"] = str(row[catalog_col])

            # Determine the source name/ID
            source_id = None
            if id_cols:
                for id_col in id_cols:
                    if id_col in present_optional_cols and pd.notna(row[id_col]):
                        source_id = str(row[id_col])
                        break # Use the first valid ID found
            # Fallback name if no ID was found
            if source_id is None:
                source_id = f"{fallback_id_prefix} {idx + 1}"
            source["name"] = source_id

            catalog_sources.append(source)

    # ...existing code...

def provide_download_buttons(folder_path: str) -> None:
    """
    Create a Streamlit download button for a ZIP archive of result files.

    Collects all files within the specified `folder_path` that start with the
    `base_filename` stored in session state, compresses them into a single
    ZIP file in memory, and presents a download button.

    Parameters
    ----------
    folder_path : str
        The path to the directory containing the result files to be zipped.

    Returns
    -------
    None
        Renders a download button and caption directly in the Streamlit app.

    Notes
    -----
    - Relies on `st.session_state['base_filename']` to identify relevant files.
    - Creates a timestamped ZIP filename like '{base_filename}_{YYYYMMDD_HHMMSS}.zip'.
    - Uses `BytesIO` for in-memory ZIP creation.
    - Displays the number of files included in the archive.
    - Handles errors during file listing or ZIP creation.
    """
    # ...existing code...

###################################################################

# ... rest of the Streamlit app code ...
