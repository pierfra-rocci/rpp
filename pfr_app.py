import os
import datetime

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun
from typing import Union, Any, Optional, Dict, Tuple

# Add these imports at the top of your file
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
import requests
from urllib.parse import quote

from astropy.modeling import models, fitting
import streamlit as st
# from streamlit.components.v1 import html  # Add this import for Aladin Lite widget
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

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="RAPAS Photometric Calibration",
    page_icon="🔭",
    layout="wide"
)

# Custom CSS to control plot display size
st.markdown("""
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
""", unsafe_allow_html=True)

# Define standard figure sizes to use throughout the app
FIGURE_SIZES = {
    'small': (6, 5),     # For small plots
    'medium': (8, 6),    # For medium plots
    'large': (10, 8),    # For large plots
    'wide': (12, 6),    # For wide plots
    'stars_grid': (10, 8)  # For grid of stars
}

def solve_with_astrometry_net(image_data, header=None, api_key=None):
    """
    Solve plate using the astrometry.net API.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        The image data to solve
    header : dict, optional
        FITS header with any available metadata to help the solver
    api_key : str, optional
        Astrometry.net API key. If not provided, will look in environment variables.
        
    Returns
    -------
    tuple
        (wcs_object, updated_header, status_message)
    """
    try:
        # Import required libraries
        import tempfile
        import time
        from astroquery.astrometry_net import AstrometryNetClass
        from astropy.io import fits
        import os
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = "cnolwaeoasefkoka"  # Default API key for testing purposes
            if api_key is None:
                return None, header, "No API key provided or found in environment variables"
        
        # Initialize AstrometryNet client
        ast = AstrometryNetClass()
        ast.api_key = api_key
        
        # Extract metadata to help with solving
        solve_kwargs = {}
        
        # If we have coordinates and scale in header, use them to narrow down search
        if header is not None:
            # Extract RA/DEC if available
            ra, dec, _ = extract_coordinates(header)
            if ra is not None and dec is not None:
                solve_kwargs['center_ra'] = ra
                solve_kwargs['center_dec'] = dec
                solve_kwargs['radius'] = 1.0  # Search radius in degrees
                
            # Extract pixel scale if available
            scale, _ = extract_pixel_scale(header)
            if scale > 0:
                # astrometry.net expects scale range in arcsec/pixel
                solve_kwargs['scale_lower'] = scale * 0.8
                solve_kwargs['scale_upper'] = scale * 1.2
        
        # Send the image to astrometry.net
        try_idx = 1
        max_tries = 3
        wcs_header = None
        
        while try_idx <= max_tries and wcs_header is None:
            try:
                file_path = None
                
                # Write image to temporary FITS file
                with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
                    file_path = f.name
                    hdu = fits.PrimaryHDU(data=image_data, header=fits.Header())
                    hdu.writeto(file_path, overwrite=True)
                
                try:
                    # Submit the job to astrometry.net
                    st.info(f"Submitting image to astrometry.net (attempt {try_idx}/{max_tries})...")
                    
                    # First try with a file upload
                    wcs_header = ast.solve_from_image(file_path, **solve_kwargs)
                    
                except Exception as e:
                    st.warning(f"Error with file upload: {str(e)}. Trying with image data directly...")
                    # If file upload fails, try with image data directly
                    wcs_header = ast.solve_from_image(image_data, **solve_kwargs)
                
                # Clean up temporary file
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
            
            except Exception as e:
                try_idx += 1
                st.warning(f"Astrometry.net attempt {try_idx-1} failed: {str(e)}")
                # Wait before retrying
                time.sleep(2)
        
        if wcs_header is None:
            return None, header, f"Astrometry.net failed after {max_tries} attempts"
        
        # Create WCS object from the returned header
        wcs_obj = WCS(wcs_header)
        
        # Update the original header with WCS information
        if header is None:
            header = fits.Header()
            
        # Add WCS keywords to the header
        for key in wcs_header:
            if key in ['SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND']:
                # Skip these structural keywords
                continue
            if key.startswith('HISTORY') or key.startswith('COMMENT'):
                continue
            # Add the WCS keyword to our header
            header[key] = wcs_header[key]
        
        # Also add information that this WCS came from astrometry.net
        header['ASTRSRC'] = 'astrometry.net'
        header['ASTRTRY'] = try_idx
        
        return wcs_obj, header, "Astrometry.net solution successful"
    
    except ImportError as e:
        return None, header, f"Required packages not installed: {str(e)}"
    except Exception as e:
        return None, header, f"Error solving with astrometry.net: {str(e)}"


def ensure_output_directory(directory="rapas_results"):
    """
    Ensure the output directory exists, create it if needed.
    
    Parameters
    ----------
    directory : str
        Name of the directory to create
        
    Returns
    -------
    str
        Path to the output directory
    """
    # Create directory if it doesn't exist
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            return directory
        except Exception as e:
            st.warning(f"Could not create directory '{directory}': {e}")
            return "."  # Fallback to current directory
    return directory

def safe_wcs_create(header):
    """
    Safely create a WCS object from a FITS header with proper error handling.
    
    Parameters
    ----------
    header : dict
        FITS header
        
    Returns
    -------
    tuple
        (wcs_object, None) if successful, (None, error_message) if failed
    """
    if not header:
        return None, "No header provided"
    
    # Check if header contains minimum required WCS keywords
    required_keys = ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2']
    missing_keys = [key for key in required_keys if key not in header]
    
    if missing_keys:
        return None, f"Missing required WCS keywords: {', '.join(missing_keys)}"
    
    try:
        wcs_obj = WCS(header)
        
        # Check if WCS is valid
        if wcs_obj is None:
            return None, "WCS creation returned None"
            
        # Check dimensionality
        if wcs_obj.pixel_n_dim > 2:
            # Reduce to celestial coordinates only
            wcs_obj = wcs_obj.celestial
            
        # Verify WCS has actual transformation
        if not hasattr(wcs_obj, 'wcs'):
            return None, "Created WCS object has no transformation attributes"
        
        return wcs_obj, None
    except Exception as e:
        return None, f"WCS creation error: {str(e)}"

def estimate_background(image_data, box_size=100, filter_size=5):
    """
    Robustly estimate image background with better error handling.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        The science image data
    box_size : int, optional
        The box size for Background2D
    filter_size : int, optional
        The filter size for Background2D
        
    Returns
    -------
    tuple
        (background_2d_object, error_message)
    """
    if image_data is None:
        return None, "No image data provided"
        
    # Validate image dimensionality
    if not isinstance(image_data, np.ndarray):
        return None, f"Image data must be a numpy array, got {type(image_data)}"
        
    if len(image_data.shape) != 2:
        return None, f"Image must be 2D, got shape {image_data.shape}"
        
    # If image is too small, adjust box size
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
            bkg_estimator=bkg_estimator
        )
        
        return bkg, None
    except Exception as e:
        return None, f"Background estimation error: {str(e)}"


def get_header_value(header, keys, default=None):
    """
    Robustly extract a header value trying multiple potential keys.
    
    Parameters
    ----------
    header : dict
        The FITS header dictionary
    keys : list
        List of possible keys to try in order of preference
    default : any
        Default value to return if no keys are found
        
    Returns
    -------
    any
        The header value or default if not found
    """
    if header is None:
        return default
        
    for key in keys:
        if key in header:
            return header[key]
    return default

def extract_pixel_scale(header):
    """
    Extract pixel scale from FITS header using multiple possible keywords.
    
    Parameters
    ----------
    header : dict
        FITS header
        
    Returns
    -------
    tuple
        (pixel_scale_value, source_description)
    """
    if header is None:
        return 1.0, "default (no header)"
    
    # Try direct pixel scale keywords
    for key in ['PIXSIZE', 'PIXSCALE', 'PIXELSCAL']:
        if key in header:
            return header[key], f"from {key}"
    
    # Try WCS CDELT keywords
    for key in ['CDELT2', 'CDELT1']:
        if key in header:
            scale = abs(header[key]) * 3600.0
            return scale, f"from {key}"
    
    # Try calculating from physical pixel size and focal length
    if 'XPIXSZ' in header:
        # Check if we have a focal length
        if 'FOCALLEN' in header:
            focal_length_mm = header['FOCALLEN']
            pixel_size = header['XPIXSZ']
            
            # Check unit of XPIXSZ
            xpixsz_unit = header.get('XPIXSZU', '').strip().lower()
            
            if xpixsz_unit in ['arcsec', 'as']:
                return pixel_size, "from XPIXSZ (in arcsec)"
            elif xpixsz_unit == 'mm':
                scale = (pixel_size * 1000) / focal_length_mm * 206.265
                return scale, "calculated from XPIXSZ (mm) and FOCALLEN"
            else:
                # Assume microns
                scale = pixel_size / focal_length_mm * 206.265
                return scale, "calculated from XPIXSZ (μm) and FOCALLEN"
        else:
            # Assume XPIXSZ is already in arcsec if no FOCALLEN
            return header['XPIXSZ'], "from XPIXSZ (assumed arcsec)"
            
    # Default fallback
    return 1.0, "default fallback value"

def extract_coordinates(header):
    """
    Extract RA and DEC coordinates from FITS header.
    
    Parameters
    ----------
    header : dict
        FITS header dictionary
    
    Returns
    -------
    tuple
        (ra, dec, source_description) or (None, None, error_message)
    """
    if header is None:
        return None, None, "No header available"
    
    # Try different coordinate keyword combinations
    ra_keys = ['RA', 'OBJRA', 'RA---', 'CRVAL1']
    dec_keys = ['DEC', 'OBJDEC', 'DEC---', 'CRVAL2']
    
    ra = get_header_value(header, ra_keys)
    dec = get_header_value(header, dec_keys)
    
    if ra is not None and dec is not None:
        # Determine which keys were used
        ra_source = next((k for k in ra_keys if k in header), 'unknown')
        dec_source = next((k for k in dec_keys if k in header), 'unknown')
        source = f"{ra_source}/{dec_source}"
        
        # Validate coordinate ranges
        try:
            ra_val = float(ra)
            dec_val = float(dec)
            
            # Apply basic validation
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
    Safely execute a catalog query with proper error handling.
    
    Parameters
    ----------
    query_func : callable
        The function to call for the catalog query
    error_msg : str
        Base error message to display
    *args, **kwargs
        Arguments to pass to query_func
        
    Returns
    -------
    tuple
        (result, None) if successful, (None, error_message) if failed
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

# Function to create standardized matplotlib figures
def create_figure(size='medium', dpi=100):
    """Create a matplotlib figure with standardized size"""
    if size in FIGURE_SIZES:
        figsize = FIGURE_SIZES[size]
    else:
        figsize = FIGURE_SIZES['medium']
    return plt.figure(figsize=figsize, dpi=dpi)


def airmass(
    _header: Dict,
    observatory: Optional[Dict] = None,
    return_details: bool = False
) -> Union[float, Tuple[float, Dict]]:
    """
    Calcule la masse d'air pour un objet céleste à partir des données d'en-tête
    et des informations de l'observatoire.

    Cette fonction gère plusieurs formats de coordonnées et effectue des
    vérifications de validité physique des résultats.

    Paramètres
    ----------
    header : Dict
        En-tête FITS ou dictionnaire contenant les informations de l'observation.
        Doit contenir:
        - Coordonnées (RA/DEC ou OBJRA/OBJDEC)
        - Date d'observation (DATE-OBS)
    observatory : Dict, optionnel
        Informations sur l'observatoire. Si non fourni, utilise l'OHP par défaut.
        Format:
        {
            'name': str,           # Nom de l'observatoire
            'latitude': float,     # Latitude en degrés
            'longitude': float,    # Longitude en degrés
            'elevation': float     # Élévation en mètres
        }
    return_details : bool, optionnel
        Si True, retourne des informations supplémentaires sur l'observation.

    Retourne
    -------
    Union[float, Tuple[float, Dict]]
        - Si return_details=False: masse d'air (float)
        - Si return_details=True: (masse d'air, détails_observation)

    Notes
    -----
    La masse d'air est une mesure de la quantité d'atmosphère traversée
    par la lumière d'un objet céleste. Une masse d'air de 1 correspond
    à une observation au zénith.
    """
    # Observatoire par défaut (OHP)
    DEFAULT_OBSERVATORY = {
        'name': 'TJMS',
        'latitude': 48.29166,
        'longitude': 2.43805,
        'elevation': 94.0
    }

    # Utilisation de l'observatoire spécifié ou par défaut
    obs_data = observatory or DEFAULT_OBSERVATORY

    try:
        # Extraction des coordonnées avec gestion de différents formats
        ra = _header.get("RA", _header.get("OBJRA", _header.get("RA---")))
        dec = _header.get("DEC", _header.get("OBJDEC", _header.get("DEC---")))
        obstime_str = _header.get("DATE-OBS", _header.get("DATE"))

        if any(v is None for v in [ra, dec, obstime_str]):
            missing = []
            if ra is None: missing.append("RA")
            if dec is None: missing.append("DEC")
            if obstime_str is None: missing.append("DATE-OBS")
            raise KeyError(f"Missing required header keywords: {', '.join(missing)}")

        # Conversion des coordonnées et création de l'objet SkyCoord
        coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame='icrs')

        # Création de l'objet Time
        obstime = Time(obstime_str)

        # Création de l'objet EarthLocation pour l'observatoire
        location = EarthLocation(
            lat=obs_data['latitude']*u.deg,
            lon=obs_data['longitude']*u.deg,
            height=obs_data['elevation']*u.m
        )

        # Calcul de l'altitude et de la masse d'air
        altaz_frame = AltAz(obstime=obstime, location=location)
        altaz = coord.transform_to(altaz_frame)
        airmass_value = float(altaz.secz)

        # Vérifications physiques
        if airmass_value < 1.0:
            st.warning("Calculated airmass is less than 1 (physically impossible)")
            airmass_value = 1.0
        elif airmass_value > 40.0:
            st.warning("Extremely high airmass (>40), object near horizon")

        # Calcul de la position du Soleil pour vérifier les conditions d'observation
        sun_altaz = get_sun(obstime).transform_to(altaz_frame)
        sun_alt = float(sun_altaz.alt.deg)

        # Création du dictionnaire de détails
        details = {
            'observatory': obs_data['name'],
            'datetime': obstime.iso,
            'target_coords': {
                'ra': coord.ra.to_string(unit=u.hour),
                'dec': coord.dec.to_string(unit=u.deg)
            },
            'altaz': {
                'altitude': round(float(altaz.alt.deg), 2),
                'azimuth': round(float(altaz.az.deg), 2)
            },
            'sun_altitude': round(sun_alt, 2),
            'observation_type': 'night' if sun_alt < -18 else 
                              'twilight' if sun_alt < 0 else 
                              'day'
        }

        # Affichage des informations
        st.write(f"Date & Local Time: {obstime.iso}")
        # ra_deg = round(float(coord.ra.deg), 5)
        # dec_deg = round(float(coord.dec.deg), 5)
        # st.write(f"Target: RA={ra_deg}°, DEC={dec_deg}°")
        st.write(f"Altitude: {details['altaz']['altitude']}°, "
              f"Azimuth: {details['altaz']['azimuth']}°")

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
    Load FITS data from an uploaded file, handling both standard and RGB FITS images.
    
    For RGB/multi-dimensional FITS files, extracts the first image HDU and uses 
    only the first channel if the data is multi-dimensional.
    """
    if file is not None:
        file_content = file.read()
        # Create HDUList explicitly to avoid typing issues
        hdul = fits.open(BytesIO(file_content), mode='readonly')
        try:
            # Start with primary HDU
            data = hdul[0].data
            header = hdul[0].header
            
            # If primary HDU has no data, look for first image HDU
            if data is None:
                for i, hdu in enumerate(hdul[1:], 1):
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        data = hdu.data
                        header = hdu.header
                        st.info(f"Primary HDU has no data. Using data from HDU #{i}.")
                        break
            
            # Handle case where no data was found
            if data is None:
                st.warning("No image data found in the FITS file.")
                return None, None
                
            # Handle different data dimensionality
            if len(data.shape) == 3:
                # Determine if this is likely an RGB image
                if data.shape[0] == 3 or data.shape[0] == 4:  # RGB/RGBA with color as first dimension
                    st.info(f"Detected RGB data with shape {data.shape}. Using first color channel.")
                    data = data[0]
                elif data.shape[2] == 3 or data.shape[2] == 4:  # RGB/RGBA with color as last dimension
                    st.info(f"Detected RGB data with shape {data.shape}. Using first color channel.")
                    data = data[:, :, 0]
                else:  # Some other 3D data structure (e.g., data cube)
                    st.info(f"Detected 3D data with shape {data.shape}. Using first plane.")
                    data = data[0]
            elif len(data.shape) > 3:
                st.warning(f"Data has {len(data.shape)} dimensions. Using first slice only.")
                # Take first slice along all higher dimensions
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


def calibrate_image_streamlit(science_data, science_header, bias_data, dark_data, flat_data,
                           exposure_time_science, exposure_time_dark,
                           apply_bias, apply_dark, apply_flat):
    """Calibrates a science image using bias, dark, and flat frames according to user selections."""
    if not apply_bias and not apply_dark and not apply_flat:
        st.write("Calibration steps are disabled. Returning raw science data.")
        return science_data, science_header

    calibrated_science = science_data.copy()
    steps_applied = []
    
    # Initialize corrected data variables
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
        # Scale dark frame if exposure times are different
        if exposure_time_science != exposure_time_dark:
            dark_scale_factor = exposure_time_science / exposure_time_dark
            scaled_dark = dark_data_corrected * dark_scale_factor
        else:
            scaled_dark = dark_data_corrected
        calibrated_science -= scaled_dark
        steps_applied.append("Soustraction du Dark")

    if apply_flat and flat_data_corrected is not None:
        st.write("Application de la correction du flat field...")
        # Normalize the flat field
        normalized_flat = flat_data_corrected / np.median(flat_data_corrected)
        calibrated_science /= normalized_flat
        steps_applied.append("Correction du Flat Field")

    if not steps_applied:
        st.write("No calibration steps were applied because files are missing or options are disabled.")
        return science_data, science_header

    st.success(f"Calibration steps applied: {', '.join(steps_applied)}")
    return calibrated_science, science_header


@st.cache_data
def make_border_mask(
    image: np.ndarray,
    border: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 50,
    invert: bool = True,
    dtype: np.dtype = bool
) -> np.ndarray:
    """Creates a binary mask for an image excluding one or more borders."""
    # Validation du type d'entrée
    if not isinstance(image, np.ndarray):
        raise TypeError("L'image doit être un numpy.ndarray")

    if image.size == 0:
        raise ValueError("L'image ne peut pas être vide")

    height, width = image.shape[:2]

    # Conversion et validation des bordures
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
            raise ValueError("border doit être un int ou un tuple de 2 ou 4 éléments")
    else:
        raise TypeError("border doit être un int ou un tuple")

    # Validation des dimensions
    if any(b < 0 for b in (top, bottom, left, right)):
        raise ValueError("Les bordures ne peuvent pas être négatives")

    if top + bottom >= height or left + right >= width:
        raise ValueError("Les bordures sont plus grandes que l'image")

    # Création du masque
    mask = np.zeros(image.shape[:2], dtype=dtype)
    mask[top:height-bottom, left:width-right] = True

    return ~mask if invert else mask


@st.cache_data
def fwhm_fit(
    img: np.ndarray,
    fwhm: float,
    pixel_scale: float,
    mask: Optional[np.ndarray] = None,
    std_lo: float = 0.5,
    std_hi: float = 0.5
) -> Optional[float]:
    """
    Calculate the FWHM of an image using marginal sums and 1D Gaussian model fitting.
    
    This function first filters sources based on their flux, keeping those within a range
    defined by std_lo and std_hi around the median. For each filtered source, it extracts
    a sub-image, calculates marginal sums, and fits a 1D Gaussian model to estimate FWHM.
    
    Returns the median estimated FWHM in pixels.
    """
    def compute_fwhm_marginal_sums(image_data, center_row, center_col, box_size):
        """Compute FWHM using marginal sums and Gaussian fitting."""
        half_box = box_size // 2
        
        # Enforce minimum box size for fitting
        if box_size < 5:
            return None  # Box too small for reliable fitting
        
        # Check if box is within image boundaries of the FULL IMAGE
        row_start = center_row - half_box
        row_end = center_row + half_box + 1
        col_start = center_col - half_box
        col_end = center_col + half_box + 1
        
        if row_start < 0 or row_end > image_data.shape[0] or col_start < 0 or col_end > image_data.shape[1]:
            return None  # Box extends beyond image boundaries
        
        # Extract box region from the COMPLETE IMAGE
        box_data = image_data[row_start:row_end, col_start:col_end]
        
        # Skip if box is too small (due to boundary conditions)
        if box_data.shape[0] < 5 or box_data.shape[1] < 5:
            return None
        
        # Calculate marginal sums
        sum_rows = np.sum(box_data, axis=1)
        sum_cols = np.sum(box_data, axis=0)
        
        # Skip if not enough signal (avoid fitting noise)
        if np.max(sum_rows) < 5 * np.median(sum_rows) or np.max(sum_cols) < 5 * np.median(sum_cols):
            return None
        
        # Create axis data for fitting
        row_indices = np.arange(box_data.shape[0])
        col_indices = np.arange(box_data.shape[1])
        
        # Fit Gaussians
        fitter = fitting.LevMarLSQFitter()
        
        # Fit rows - with better initial estimates and error handling
        try:
            # Better initial parameter estimation
            row_max_idx = np.argmax(sum_rows)
            row_max_val = sum_rows[row_max_idx]
            
            model_row = models.Gaussian1D(
                amplitude=row_max_val,
                mean=row_max_idx,
                stddev=box_size/6  # More conservative initial stddev
            )
            
            fitted_row = fitter(model_row, row_indices, sum_rows)
            center_row_fit = fitted_row.mean.value + row_start
            fwhm_row = 2 * np.sqrt(2 * np.log(2)) * fitted_row.stddev.value
        except Exception as e:
            # If row fitting fails, return None
            return None
        
        # Fit columns - with better initial estimates and error handling
        try:
            # Better initial parameter estimation
            col_max_idx = np.argmax(sum_cols)
            col_max_val = sum_cols[col_max_idx]
            
            model_col = models.Gaussian1D(
                amplitude=col_max_val,
                mean=col_max_idx,
                stddev=box_size/6  # More conservative initial stddev
            )
            
            fitted_col = fitter(model_col, col_indices, sum_cols)
            center_col_fit = fitted_col.mean.value + col_start
            fwhm_col = 2 * np.sqrt(2 * np.log(2)) * fitted_col.stddev.value
        except Exception as e:
            # If column fitting fails, return None
            return None
        
        return fwhm_row, fwhm_col, center_row_fit, center_col_fit

    try:
        daofind = DAOStarFinder(fwhm=1.5*fwhm, threshold=6 * np.std(img))
        sources = daofind(img, mask=mask)
        if sources is None:
            st.warning("No sources found !")
            return None

        st.write(f"Number of sources found : {len(sources)}")

        # Filter sources by flux
        flux = sources['flux']
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        mask_flux = (flux > median_flux - std_lo * std_flux) & (flux < median_flux + std_hi * std_flux)
        filtered_sources = sources[mask_flux]

        # Remove sources with NaN flux values
        filtered_sources = filtered_sources[~np.isnan(filtered_sources['flux'])]

        st.write(f"Number of sources after flux filtering: {len(filtered_sources)}")

        if len(filtered_sources) == 0:
            msg = "No valid sources for fitting found after filtering."
            st.error(msg)
            raise ValueError(msg)

        # Define analysis box size (in pixels)
        box_size = int(6 * round(fwhm))
        if box_size % 2 == 0:  # Ensure box_size is odd
            box_size += 1

        fwhm_values = []
        skipped_sources = 0  # Counter for skipped sources

        # Fit model for each filtered source
        for source in filtered_sources:
            try:
                x_cen = int(source['xcentroid'])
                y_cen = int(source['ycentroid'])

                # Calculate FWHM using marginal sums
                fwhm_results = compute_fwhm_marginal_sums(img, y_cen, x_cen, box_size)
                if fwhm_results is None:
                    # Instead of individual warning, just increment counter
                    skipped_sources += 1
                    continue

                fwhm_row, fwhm_col, _, _ = fwhm_results

                # Use average of FWHM_row and FWHM_col as FWHM estimate for this source
                fwhm_source = np.mean([fwhm_row, fwhm_col])
                fwhm_values.append(fwhm_source)

            except Exception as e:
                skipped_sources += 1
                continue

        # Display summary of skipped sources
        if skipped_sources > 0:
            st.write(f"FWHM calculation failed for {skipped_sources} sources out of {len(filtered_sources)} and were skipped.")
            
        if len(fwhm_values) == 0:
            msg = "No valid sources for FWHM fitting after marginal sums adjustment."
            st.error(msg)
            raise ValueError(msg)

        # Convert list to array to filter NaN and infinite values
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
    mask: Optional[np.ndarray] = None
) -> Tuple[Table, Any]:
    """
    Perform PSF photometry using an EPSF model.
    
    Parameters
    ----------
    img : np.ndarray
        Image with sky background subtracted.
    phot_table : astropy.table.Table
        Table containing source positions.
    fwhm : float
        Full Width at Half Maximum used to define fitting size.
    daostarfind : callable
        Star detection function used as "finder" in PSF photometry.
    mask : np.ndarray, optional
        Mask to exclude certain image areas.
        
    Returns
    -------
    Tuple[astropy.table.Table, photutils.epsf.EPSF]
        - phot_epsf : PSF photometry results with EPSF model.
        - epsf : Fitted EPSF model.
    """
    try:
        # Prepare data: convert image to NDData object
        nddata = NDData(data=img)
        # st.write("NDData created successfully.")
    except Exception as e:
        st.error(f"Error creating NDData: {e}")
        raise

    try:
        # Build star table from phot_table
        stars_table = Table()
        stars_table['x'] = phot_table['xcenter']
        stars_table['y'] = phot_table['ycenter']
        st.write("Star positions table prepared.")
    except Exception as e:
        st.error(f"Error preparing star positions table: {e}")
        raise

    try:
        # Define fitting shape (box size for star extraction)
        fit_shape = 2 * round(fwhm) + 1
        st.write(f"Fitting shape: {fit_shape} pixels.")
    except Exception as e:
        st.error(f"Error calculating fitting shape: {e}")
        raise

    try:
        # Extract stars in sub-image
        stars = extract_stars(nddata, stars_table, size=fit_shape)
        st.write(f"{len(stars)} stars extracted.")
    except Exception as e:
        st.error(f"Error extracting stars: {e}")
        raise

    try:
        # Display extracted stars (optional)
        nrows, ncols = 5, 5
        fig_stars, ax_stars = plt.subplots(nrows=nrows, ncols=ncols, figsize=FIGURE_SIZES['stars_grid'], squeeze=False)
        ax_stars = ax_stars.ravel()
        n_disp = min(len(stars), nrows * ncols)
        for i in range(n_disp):
            norm = simple_norm(stars[i].data, 'log', percent=99.0)
            ax_stars[i].imshow(stars[i].data, norm=norm, origin='lower', cmap='viridis')
        plt.tight_layout()
        st.pyplot(fig_stars)
    except Exception as e:
        st.warning(f"Error displaying extracted stars: {e}")

    try:
        # Build and fit EPSF model
        epsf_builder = EPSFBuilder(oversampling=2, maxiters=5, progress_bar=False)
        epsf, _ = epsf_builder(stars)
        st.write("PSF model fitted successfully.")
        st.session_state['epsf_model'] = epsf
    except Exception as e:
        st.error(f"Error fitting PSF model: {e}")
        raise

    try:
        # Display fitted EPSF model
        norm_epsf = simple_norm(epsf.data, 'log', percent=99.)
        fig_epsf_model, ax_epsf_model = plt.subplots(figsize=FIGURE_SIZES['medium'], dpi=100)
        ax_epsf_model.imshow(epsf.data, norm=norm_epsf, origin='lower', cmap='viridis', interpolation='nearest')
        # plt.colorbar(ax=ax_epsf_model)
        ax_epsf_model.set_title("Fitted PSF Model")
        st.pyplot(fig_epsf_model)
    except Exception as e:
        st.warning(f"Error displaying PSF model: {e}")

    try:
        # Perform PSF photometry with EPSF model
        psfphot = IterativePSFPhotometry(
            epsf,
            fit_shape,
            finder=daostarfind,
            aperture_radius=fit_shape / 2,
            maxiters=3,
            mode='new',
            progress_bar=False
        )
        # Specify source positions
        psfphot.x = phot_table['xcenter']
        psfphot.y = phot_table['ycenter']
    except Exception as e:
        st.error(f"Error configuring PSF photometry: {e}")
        raise

    try:
        # Run PSF photometry with provided mask
        phot_epsf_result = psfphot(img, mask=mask)
        st.session_state['epsf_photometry_result'] = phot_epsf_result
        st.write("PSF photometry completed successfully.")
    except Exception as e:
        st.error(f"Error executing PSF photometry: {e}")
        raise

    return phot_epsf_result, epsf


@st.cache_data
def find_sources_and_photometry_streamlit(image_data, _science_header, mean_fwhm_pixel, threshold_sigma, detection_mask):
    """
    Find sources and perform photometry on astronomical images.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Science image data
    _science_header : dict
        Header information from FITS file (underscore prevents caching issues)
    mean_fwhm_pixel : float
        Estimated FWHM in pixels
    threshold_sigma : float
        Detection threshold in sigma
    detection_mask : int
        Border size to mask during detection
        
    Returns
    -------
    tuple
        (phot_table, epsf_table, daofind, bkg)
    """
    # Initialize daofind to None
    daofind = None
    
    # Get pixel scale from header
    try:
        w, wcs_error = safe_wcs_create(_science_header)
        if w is None:
            st.error(f"Error creating WCS: {wcs_error}")
            return None, None, daofind, None
    except Exception as e:
        st.error(f"Error creating WCS: {e}")
        return None, None, daofind, None
    
    pixel_scale = _science_header.get('PIXSCALE', _science_header.get('PIXSIZE', _science_header.get('PIXELSCAL', 1.0)))
    
    # Estimate background and noise
    bkg, bkg_error = estimate_background(image_data, box_size=100, filter_size=5)
    if bkg is None:
        st.error(f"Error estimating background: {bkg_error}")
        return None, None, daofind, None
    
    # Create detection mask
    mask = make_border_mask(image_data, border=detection_mask)
    
    # Calculate total error
    total_error = np.sqrt(bkg.background_rms**2 + bkg.background_median)/np.sqrt(bkg.background_median)
    
    st.write("Estimating FWHM...")
    fwhm_estimate = fwhm_fit(image_data - bkg.background, mean_fwhm_pixel, pixel_scale, mask)
    
    if fwhm_estimate is None:
        st.warning("Failed to estimate FWHM. Using the initial estimate.")
        fwhm_estimate = mean_fwhm_pixel
    
    # Source detection using DAOStarFinder
    daofind = DAOStarFinder(fwhm=1.5*fwhm_estimate, threshold=threshold_sigma * np.std(image_data - bkg.background))
    sources = daofind(image_data - bkg.background, mask=mask)
    
    if sources is None or len(sources) == 0:
        st.warning("No sources found!")
        return None, None, daofind, bkg
    
    # Aperture photometry
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=1.5*fwhm_estimate)
    
    # Perform aperture photometry
    try:
        # Check if WCS exists and has the right dimensionality
        wcs_obj = None
        if 'CTYPE1' in _science_header:
            try:
                wcs_obj = WCS(_science_header)
                # Check if WCS has more than 2 dimensions, and if so, reduce it
                if wcs_obj.pixel_n_dim > 2:
                    wcs_obj = wcs_obj.celestial
                    st.info("Reduced WCS to 2D celestial coordinates for photometry")
            except Exception as e:
                st.warning(f"Error creating WCS object: {e}")
                wcs_obj = None
                
        phot_table = aperture_photometry(
            image_data - bkg.background,
            apertures,
            error=total_error,
            wcs=wcs_obj
        )
        
        # Add source positions to table
        phot_table['xcenter'] = sources['xcentroid']
        phot_table['ycenter'] = sources['ycentroid']
        
        # Calculate instrumental magnitudes for aperture photometry
        instrumental_mags = -2.5 * np.log10(phot_table['aperture_sum'])
        phot_table['instrumental_mag'] = instrumental_mags
        
        # Perform PSF/EPSF photometry
        try:
            epsf_table, _ = perform_epsf_photometry(
                image_data - bkg.background, 
                phot_table, 
                fwhm_estimate, 
                daofind, 
                mask
            )
            
            # Calculate instrumental magnitudes for PSF photometry
            epsf_instrumental_mags = -2.5 * np.log10(epsf_table['flux_fit'])
            epsf_table['instrumental_mag'] = epsf_instrumental_mags
        except Exception as e:
            st.error(f"Error performing EPSF photometry: {e}")
            epsf_table = None
        
        # Keep only valid sources for aperture photometry
        valid_sources = (phot_table['aperture_sum'] > 0) & np.isfinite(phot_table['instrumental_mag'])
        phot_table = phot_table[valid_sources]
        
        # Keep only valid sources for PSF photometry if available
        if epsf_table is not None:
            epsf_valid_sources = (epsf_table['flux_fit'] > 0) & np.isfinite(epsf_table['instrumental_mag'])

            epsf_table = epsf_table[epsf_valid_sources]

        # Add RA and Dec if WCS is available
        try:
            if wcs_obj is not None:
                # Process phot_table
                ra, dec = wcs_obj.pixel_to_world_values(phot_table['xcenter'], phot_table['ycenter'])
                phot_table['ra'] = ra * u.deg
                phot_table['dec'] = dec * u.deg
                
                # Process epsf_table if available
                if epsf_table is not None:
                    try:
                        epsf_ra, epsf_dec = wcs_obj.pixel_to_world_values(epsf_table['x_fit'], epsf_table['y_fit'])
                        epsf_table['ra'] = epsf_ra * u.deg
                        epsf_table['dec'] = epsf_dec * u.deg
                    except Exception as e:
                        st.warning(f"Could not add coordinates to EPSF table: {e}")
            else:
                # If no WCS but we have RA/DEC in header, try to create approximate coordinates
                if all(key in _science_header for key in ['RA', 'DEC', 'NAXIS1', 'NAXIS2']):
                    st.info("Using simple linear approximation for RA/DEC coordinates")
                    center_ra = _science_header['RA']
                    center_dec = _science_header['DEC']
                    width = _science_header['NAXIS1']
                    height = _science_header['NAXIS2']
                    
                    # Estimate pixel scale if not available
                    pix_scale = pixel_scale or 1.0  # arcsec/pixel
                    
                    # Create simple WCS transformation (approximate)
                    center_x = width / 2
                    center_y = height / 2
                    
                    # Add coordinates to tables
                    for i, row in enumerate(phot_table):
                        x = row['xcenter']
                        y = row['ycenter']
                        dx = (x - center_x) * pix_scale / 3600.0  # convert to degrees
                        dy = (y - center_y) * pix_scale / 3600.0  # convert to degrees
                        phot_table[i]['ra'] = center_ra + dx / np.cos(np.radians(center_dec))
                        phot_table[i]['dec'] = center_dec + dy
        except Exception as e:
            st.warning(f"WCS transformation failed: {e}. RA and Dec not added to tables.")
            
        st.success(f"Found {len(phot_table)} sources and performed photometry.")
        return phot_table, epsf_table, daofind, bkg
    except Exception as e:
        st.error(f"Error performing aperture photometry: {e}")
        return None, None, daofind, bkg


@st.cache_data
def cross_match_with_gaia_streamlit(_phot_table, _science_header, pixel_size_arcsec, mean_fwhm_pixel, gaia_band, gaia_min_mag, gaia_max_mag):
    """
    Cross-matches sources with Gaia catalog.
    
    Parameters
    ----------
    _phot_table : astropy.table.Table
        Table containing source positions (underscore prevents caching issues)
    _science_header : dict
        FITS header with WCS information (underscore prevents caching issues)
    pixel_size_arcsec : float
        Pixel scale in arcseconds per pixel
    mean_fwhm_pixel : float
        FWHM in pixels
    gaia_band : str
        Gaia magnitude band to use (e.g., 'phot_g_mean_mag')
    gaia_min_mag : float
        Minimum magnitude for filtering Gaia sources
    gaia_max_mag : float
        Maximum magnitude for filtering Gaia sources
        
    Returns
    -------
    pandas.DataFrame
        Matched sources table
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

    # Convert pixel positions to sky coordinates
    try:
        source_positions_pixel = np.transpose((_phot_table['xcenter'], _phot_table['ycenter']))
        source_positions_sky = w.pixel_to_world(source_positions_pixel[:,0], source_positions_pixel[:,1])
    except Exception as e:
        st.error(f"Error converting pixel positions to sky coordinates: {e}")
        return None

    # Query Gaia
    try:
        image_center_ra_dec = w.pixel_to_world(_science_header['NAXIS1']//2, _science_header['NAXIS2']//2)
        # Calculate search radius as half the image diagonal
        gaia_search_radius_arcsec = max(_science_header['NAXIS1'], _science_header['NAXIS2']) * pixel_size_arcsec / 2.0
        radius_query = gaia_search_radius_arcsec * u.arcsec

        st.write(f"Querying Gaia DR3 in a radius of {round(radius_query.value/60.,2)} arcmin.")
        job = Gaia.cone_search(image_center_ra_dec, radius=radius_query)
        gaia_table = job.get_results()
    except Exception as e:
        st.error(f"Error querying Gaia: {e}")
        return None

    if gaia_table is None or len(gaia_table) == 0:
        st.warning("No Gaia sources found within search radius.")
        return None

    # Apply filters to Gaia catalog
    try:
        # Apply magnitude filter
        mag_filter = (gaia_table[gaia_band] < gaia_max_mag) & (gaia_table[gaia_band] > gaia_min_mag)
        
        # Apply variable star filter if column exists
        if "phot_variable_flag" in gaia_table.colnames:
            var_filter = gaia_table["phot_variable_flag"] != "VARIABLE"
            combined_filter = mag_filter & var_filter
        else:
            combined_filter = mag_filter
            
        gaia_table_filtered = gaia_table[combined_filter]
        
        if len(gaia_table_filtered) == 0:
            st.warning(f"No Gaia sources found within magnitude range {gaia_min_mag} < {gaia_band} < {gaia_max_mag}.")
            return None
            
        st.write(f"Filtered Gaia catalog to {len(gaia_table_filtered)} sources.")
    except Exception as e:
        st.error(f"Error filtering Gaia catalog: {e}")
        return None

    # Cross-match
    try:
        gaia_skycoords = SkyCoord(ra=gaia_table_filtered['ra'], dec=gaia_table_filtered['dec'], unit='deg')
        idx, d2d, _ = source_positions_sky.match_to_catalog_sky(gaia_skycoords)
        
        # Set maximum separation constraint based on FWHM and pixel scale
        max_sep_constraint = 2 * mean_fwhm_pixel * pixel_size_arcsec * u.arcsec
        gaia_matches = (d2d <= max_sep_constraint)

        matched_indices_gaia = idx[gaia_matches]
        matched_indices_phot = np.where(gaia_matches)[0]

        if len(matched_indices_gaia) == 0:
            st.warning("No Gaia matches found within the separation constraint.")
            return None
            
        # Extract matched sources
        matched_table_qtable = _phot_table[matched_indices_phot]
        
        # Convert to pandas DataFrame
        matched_table = matched_table_qtable.to_pandas()
        matched_table['gaia_index'] = matched_indices_gaia
        matched_table['gaia_separation_arcsec'] = d2d[gaia_matches].arcsec
        matched_table[gaia_band] = gaia_table_filtered[gaia_band][matched_indices_gaia]
        
        # Filter invalid magnitudes
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
    Calculates photometric zero point using matched sources.
    
    Parameters
    ----------
    _phot_table : astropy.table.Table or pandas.DataFrame
        Photometry results table (underscore prevents caching issues)
    _matched_table : pandas.DataFrame
        Table of Gaia cross-matched sources (underscore prevents caching issues)
    gaia_band : str
        Gaia magnitude band used
    air : float
        Airmass value for extinction correction
        
    Returns
    -------
    tuple
        (zero_point_value, zero_point_std, matplotlib_figure)
    """
    st.write("Calculating zero point...")

    if _matched_table is None or len(_matched_table) == 0:
        st.warning("No matched sources to calculate zero point.")
        return None, None, None

    try:
        # Calculate zero points
        zero_points = _matched_table[gaia_band] - _matched_table['instrumental_mag']
        _matched_table['zero_point'] = zero_points
        _matched_table['zero_point_error'] = np.std(zero_points)
        
        # Apply sigma clipping to remove outliers
        clipped_zero_points = sigma_clip(zero_points, sigma=3)
        zero_point_value = np.mean(clipped_zero_points)
        zero_point_std = np.std(clipped_zero_points)
        
        # Calculate calibrated magnitude for matched sources
        _matched_table['calib_mag'] = _matched_table['instrumental_mag'] + zero_point_value + 0.1*air
        
        # Convert _phot_table to DataFrame if needed
        if not isinstance(_phot_table, pd.DataFrame):
            _phot_table = _phot_table.to_pandas()
            
        # Calculate calibrated magnitudes for all sources
        _phot_table['calib_mag'] = _phot_table['instrumental_mag'] + zero_point_value + 0.1*air
        
        # Calculate errors
        # if 'aperture_sum_err' in _phot_table.columns and 'aperture_sum' in _phot_table.columns:
        #     _phot_table['calib_mag_err'] = (2.5/np.log(10) * _phot_table['aperture_sum_err'] / 
        #                                   _phot_table['aperture_sum']) + zero_point_std
        
        # Store results in session state
        st.session_state['final_phot_table'] = _phot_table
        
        # Create plot
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'], dpi=100)
        ax.scatter(_matched_table[gaia_band], _matched_table['calib_mag'], alpha=0.75, label='Matched sources')
    
        ax.set_xlabel(f"Gaia {gaia_band}")
        ax.set_ylabel("Calibrated magnitude")
        ax.set_title("Gaia magnitude vs Calibrated magnitude")
        ax.legend()
        ax.grid(True, alpha=0.5)
                
        st.success(f"Calculated Zero Point: {zero_point_value:.3f} ± {zero_point_std:.3f}")
        
        # Save plot but handle exceptions
        try:
            zero_point_plot_path = os.path.join(output_dir, "zero_point_plot.png")
            plt.savefig(zero_point_plot_path)
            st.write(f"Zero point plot saved as '{zero_point_plot_path}'")
        except Exception as e:
            st.warning(f"Could not save plot to file: {e}")
            
        return zero_point_value, zero_point_std, fig
    except Exception as e:
        st.error(f"Error calculating zero point: {e}")
        return None, None, None


# Main workflow function for zero point calibration
def run_zero_point_calibration(header, pixel_size_arcsec, mean_fwhm_pixel, 
                               threshold_sigma, detection_mask, gaia_band, gaia_min_mag, gaia_max_mag, air):
    """
    Runs the complete zero point calibration workflow.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Image data to process
    header : dict
        FITS header
    pixel_size_arcsec : float
        Pixel scale in arcseconds
    mean_fwhm_pixel : float
        FWHM estimate in pixels
    threshold_sigma : float
        Detection threshold
    detection_mask : int
        Border mask size
    gaia_band : str
        Gaia magnitude band
    gaia_min_mag, gaia_max_mag : float
        Magnitude limits for Gaia sources
    air : float
        Airmass value
        
    Returns
    -------
    tuple
        (zero_point_value, zero_point_std, phot_table)
    """
    with st.spinner("Finding sources and performing photometry..."):
        phot_table_qtable, epsf_table, daofind, bkg = find_sources_and_photometry_streamlit(
            image_to_process, header_to_process, mean_fwhm_pixel, threshold_sigma, detection_mask
        )
        
        if phot_table_qtable is None:
            st.error("Failed to perform photometry - no sources found")
            return None, None, None
            
    with st.spinner("Cross-matching with Gaia..."):
        matched_table = cross_match_with_gaia_streamlit(
            phot_table_qtable, header, pixel_size_arcsec, mean_fwhm_pixel,
            gaia_band, gaia_min_mag, gaia_max_mag
        )
        
        if matched_table is None:
            st.error("Failed to cross-match with Gaia")
            return None, None, phot_table_qtable
    
    st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
    st.dataframe(matched_table.head(10))
    
    with st.spinner("Calculating zero point..."):
        zero_point_value, zero_point_std, zp_plot = calculate_zero_point_streamlit(
            phot_table_qtable, matched_table, gaia_band, air  # Use phot_table_qtable, not phot_table
        )
        
        if zero_point_value is not None:
            st.pyplot(zp_plot)
            write_to_log(log_buffer, f"Zero point: {zero_point_value:.3f} ± {zero_point_std:.3f}")
            write_to_log(log_buffer, f"Airmass: {air:.2f}")
            
            # Save the zero point plot with timestamp
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            final_table = st.session_state['final_phot_table']
                
            try:
                # Add PSF photometry results if available
                if 'epsf_photometry_result' in st.session_state and epsf_table is not None:
                    # Convert tables to pandas if needed
                    epsf_df = epsf_table.to_pandas() if not isinstance(epsf_table, pd.DataFrame) else epsf_table
                    
                    # Add a unique ID for matching (based on x,y position)
                    epsf_df['match_id'] = epsf_df['xcenter'].round(2).astype(str) + "_" + epsf_df['ycenter'].round(2).astype(str)
                    final_table['match_id'] = final_table['xcenter'].round(2).astype(str) + "_" + final_table['ycenter'].round(2).astype(str)
                    
                    # Select just the columns we need from EPSF results
                    epsf_cols = {
                        'match_id': 'match_id',
                        'flux_fit': 'epsf_flux_fit', 
                        'flux_unc': 'epsf_flux_unc',
                        'instrumental_mag': 'epsf_instrumental_mag'
                    }
                    
                    # Only continue if we have the necessary columns
                    if len(epsf_cols) > 1 and 'match_id' in epsf_df.columns and 'match_id' in final_table.columns:
                        # Select columns that actually exist
                        epsf_subset = epsf_df[[col for col in epsf_cols.keys() if col in epsf_df.columns]].rename(columns=epsf_cols)
                    
                    # Merge the EPSF results with the final table
                    final_table = pd.merge(final_table, epsf_subset, on='match_id', how='left')
                    
                    # Calculate calibrated magnitudes for EPSF photometry
                    if 'epsf_instrumental_mag' in final_table.columns:
                        final_table['epsf_calib_mag'] = final_table['epsf_instrumental_mag'] + zero_point_value + 0.1*air
                    
                    # Remove temporary match column
                    final_table.drop('match_id', axis=1, inplace=True)
                    
                    st.success("Added EPSF photometry results to the catalog")
                
                # Create a StringIO buffer for CSV data
                csv_buffer = StringIO()
                
                # Check and remove problematic columns if they exist
                cols_to_drop = []
                for col_name in ['sky_center.ra', 'sky_center.dec']:
                    if col_name in final_table.columns:
                        cols_to_drop.append(col_name)
                
                if cols_to_drop:
                    final_table = final_table.drop(columns=cols_to_drop)
                
                # Before writing to CSV, ensure match_id column is removed
                if 'match_id' in final_table.columns:
                    final_table.drop('match_id', axis=1, inplace=True)
                
                # Add zero point information to the table
                final_table['zero_point'] = zero_point_value
                final_table['zero_point_std'] = zero_point_std
                final_table['airmass'] = air
                
                # Create a catalog summary in the UI
                st.subheader("Final Photometry Catalog")
                st.dataframe(final_table.head(10))
                
                # Write the filtered DataFrame to the buffer
                final_table.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()
                
                # Ensure filename has .csv extension
                filename = catalog_name if catalog_name.endswith('.csv') else f"{catalog_name}.csv"
                catalog_path = os.path.join(output_dir, filename)

                # Provide download button with better formatting
                download_link = get_download_link(
                    csv_data, 
                    filename, 
                    link_text="Download Photometry Catalog"
                )
                st.markdown(download_link, unsafe_allow_html=True)

                # Also save locally in the output directory
                with open(catalog_path, 'w') as f:
                    f.write(csv_data)
                st.success(f"Catalog saved to {catalog_path}")
                
                # Also create a metadata file with analysis parameters
                metadata_filename = f"{base_catalog_name}_metadata.txt"
                metadata_path = os.path.join(output_dir, metadata_filename)

                with open(metadata_path, 'w') as f:
                    f.write("RAPAS Photometry Analysis Metadata\n")
                    f.write("================================\n\n")
                    f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Input File: {science_file.name}\n")
                    f.write(f"Catalog File: {filename}\n\n")
                    
                    f.write(f"Zero Point: {zero_point_value:.5f} ± {zero_point_std:.5f}\n")
                    f.write(f"Airmass: {air:.3f}\n")
                    f.write(f"Pixel Scale: {pixel_size_arcsec:.3f} arcsec/pixel\n")
                    f.write(f"FWHM (estimated): {mean_fwhm_pixel:.2f} pixels ({seeing:.2f} arcsec)\n\n")
                    
                    f.write("Detection Parameters:\n")
                    f.write(f"  Threshold: {threshold_sigma} sigma\n")
                    f.write(f"  Edge Mask: {detection_mask} pixels\n")
                    f.write(f"  Gaia Magnitude Range: {gaia_min_mag:.1f} - {gaia_max_mag:.1f}\n")

                write_to_log(log_buffer, f"Saved catalog metadata to {metadata_path}")
                st.success(f"Analysis metadata saved to {metadata_path}")
                
            except Exception as e:
                st.error(f"Error preparing download: {e}")
                st.exception(e)
        
        return zero_point_value, zero_point_std, final_table

def enhance_catalog_with_crossmatches(final_table, matched_table, header, pixel_scale_arcsec, search_radius_arcsec=6.0):
    """
    Enhance the catalog with cross-matches from GAIA DR3, SIMBAD, SkyBoT and AAVSO
    
    Parameters
    ----------
    final_table : pandas.DataFrame
        Final photometry catalog with RA/DEC coordinates
    matched_table : pandas.DataFrame
        Table of already matched Gaia sources (used for calibration)
    header : dict
        FITS header with observation information
    pixel_scale_arcsec : float
        Pixel scale in arcseconds per pixel
    search_radius_arcsec : float, optional
        Search radius for cross-matching in arcseconds
        
    Returns
    -------
    pandas.DataFrame
        Input dataframe with added catalog information
    """
    if final_table is None or len(final_table) == 0:
        st.warning("No sources to cross-match with catalogs.")
        return final_table
    
    if 'ra' not in final_table.columns or 'dec' not in final_table.columns:
        st.warning("RA/DEC columns missing from catalog. Cannot cross-match.")
        return final_table
    
    # Add progress tracking
    status_text = st.empty()
    status_text.write("Starting cross-match process...")
    
    # 1. First, add the GAIA calibration matches we already have
    if matched_table is not None and len(matched_table) > 0:
        status_text.write("Adding Gaia calibration matches...")
        
        # Create a unique ID for matching based on x,y coordinates
        if 'xcenter' in final_table.columns and 'ycenter' in final_table.columns:
            final_table['match_id'] = final_table['xcenter'].round(2).astype(str) + "_" + final_table['ycenter'].round(2).astype(str)
        
        if 'xcenter' in matched_table.columns and 'ycenter' in matched_table.columns:
            matched_table['match_id'] = matched_table['xcenter'].round(2).astype(str) + "_" + matched_table['ycenter'].round(2).astype(str)
            
            # Get Gaia columns to add
            gaia_cols = [col for col in matched_table.columns if any(x in col for x in ['gaia', 'phot_'])]
            gaia_cols.append('match_id')
            
            # Select Gaia data subset
            gaia_subset = matched_table[gaia_cols].copy()
            
            # Rename columns to be clearer
            rename_dict = {}
            for col in gaia_subset.columns:
                if col != 'match_id' and not col.startswith('gaia_'):
                    rename_dict[col] = f'gaia_{col}'
            
            if rename_dict:
                gaia_subset = gaia_subset.rename(columns=rename_dict)
            
            # Merge with final table
            final_table = pd.merge(final_table, gaia_subset, on='match_id', how='left')
            
            # Add a flag for calibration stars
            final_table['gaia_calib_star'] = final_table['match_id'].isin(matched_table['match_id'])
            
            st.success(f"Added {len(matched_table)} Gaia calibration stars to catalog")
    
    # 2. Cross-match with SIMBAD - Fixed version
    status_text.write("Querying SIMBAD for object identifications...")
    try:
        # Extract center coordinates and field size
        field_center_ra = None
        field_center_dec = None
        
        # Try different header keywords for coordinates
        if 'CRVAL1' in header and 'CRVAL2' in header:
            field_center_ra = float(header['CRVAL1'])
            field_center_dec = float(header['CRVAL2'])
        elif 'RA' in header and 'DEC' in header:
            field_center_ra = float(header['RA'])
            field_center_dec = float(header['DEC'])
        elif 'OBJRA' in header and 'OBJDEC' in header:
            field_center_ra = float(header['OBJRA'])
            field_center_dec = float(header['OBJDEC'])
        
        if field_center_ra is not None and field_center_dec is not None:
            # Validate coordinates are within reasonable range
            if not (-360 <= field_center_ra <= 360) or not (-90 <= field_center_dec <= 90):
                st.warning(f"Invalid coordinates: RA={field_center_ra}, DEC={field_center_dec}")
            else:
                # Calculate field width in arcmin based on image dimensions and pixel scale
                if 'NAXIS1' in header and 'NAXIS2' in header:
                    field_width_arcmin = max(header.get('NAXIS1', 1000), header.get('NAXIS2', 1000)) * pixel_scale_arcsec / 60.0
                else:
                    field_width_arcmin = 30.0 # Default to 30 arcmin
                    
                # Configure Simbad
                custom_simbad = Simbad()
                custom_simbad.add_votable_fields('otype', 'main_id', 'ids')
                
                # Query SIMBAD in a cone around field center
                st.info(f"Querying SIMBAD at RA={field_center_ra}, DEC={field_center_dec}")
                
                try:
                    # Create a SkyCoord object for the query
                    center_coord = SkyCoord(ra=field_center_ra, dec=field_center_dec, unit='deg')
                    simbad_result, error = safe_catalog_query(
                        custom_simbad.query_region,
                        "SIMBAD query failed",
                        center_coord,
                        radius=field_width_arcmin * u.arcmin
                    )
                    if error:
                        st.warning(error)
                    else:
                        # Process simbad_result
                        if simbad_result is not None and len(simbad_result) > 0:
                            # Initialize columns if they don't exist
                            final_table['simbad_name'] = None
                            final_table['simbad_type'] = None
                            final_table['simbad_ids'] = None
                            
                            # Convert catalog positions to SkyCoord objects with explicit unit
                            source_coords = SkyCoord(ra=final_table['ra'].values, dec=final_table['dec'].values, unit='deg')
                            
                            # Check if RA and DEC columns exist and create SkyCoord
                            if all(col in simbad_result.colnames for col in ['ra', 'dec']):  # Changed from 'RA', 'DEC' to lowercase
                                # Get SIMBAD coordinates with proper units
                                try:
                                    simbad_coords = SkyCoord(
                                        ra=simbad_result['ra'],  # Changed from 'RA' to 'ra'
                                        dec=simbad_result['dec'],  # Changed from 'DEC' to 'dec'
                                        unit=(u.hourangle, u.deg)
                                    )
                                    
                                    # Match coordinates
                                    idx, d2d, _ = source_coords.match_to_catalog_sky(simbad_coords)
                                    matches = d2d < (search_radius_arcsec * u.arcsec)
                                    
                                    # Add matches to table
                                    for i, (match, match_idx) in enumerate(zip(matches, idx)):
                                        if match:
                                            final_table.loc[i, 'simbad_name'] = simbad_result['main_id'][match_idx]  # Changed from 'MAIN_ID'
                                            final_table.loc[i, 'simbad_type'] = simbad_result['otype'][match_idx]    # Changed from 'OTYPE'
                                            if 'ids' in simbad_result.colnames:  # Changed from 'IDS'
                                                final_table.loc[i, 'simbad_ids'] = simbad_result['ids'][match_idx]
                                    
                                    st.success(f"Found {sum(matches)} SIMBAD objects in field.")
                                except Exception as e:
                                    st.error(f"Error creating SkyCoord objects from SIMBAD data: {str(e)}")
                                    st.write(f"Available SIMBAD columns: {simbad_result.colnames}")
                            else:
                                available_cols = ', '.join(simbad_result.colnames)
                                st.error(f"SIMBAD result missing required columns. Available columns: {available_cols}")
                        else:
                            st.write("No SIMBAD objects found in the field.")
                except Exception as e:
                    st.error(f"SIMBAD query execution failed: {str(e)}")
        else:
            st.warning("Could not extract field center coordinates from header")
    except Exception as e:
        st.error(f"Error in SIMBAD processing: {str(e)}")
    
    # 3. Cross-match with SkyBoT for solar system objects - Fixed version
    status_text.write("Querying SkyBoT for solar system objects...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            # Get observation date
            if 'DATE-OBS' in header:
                obs_date = header['DATE-OBS']
            elif 'DATE' in header:
                obs_date = header['DATE']
            else:
                obs_date = Time.now().isot
                
            # Format date for SkyBoT
            obs_time = Time(obs_date).isot
            
            # Build SkyBoT URL
            sr_value = min(field_width_arcmin/60.0, 1.0)  # Limit to 1 degree max
            skybot_url = (
                f"http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?"
                f"RA={field_center_ra}&DEC={field_center_dec}&SR={sr_value}&"
                f"EPOCH={quote(obs_time)}&mime=json"
            )
            
            st.info("Querying SkyBoT for solar system objects...")
            
            # Query SkyBoT with better error handling
            try:
                # Initialize columns
                final_table['skybot_name'] = None
                final_table['skybot_type'] = None
                final_table['skybot_mag'] = None
                
                # Make request with extended timeout
                response = requests.get(skybot_url, timeout=15)
                
                if response.status_code == 200:
                    response_text = response.text.strip()
                    
                    # Check if we got a valid JSON response
                    if response_text.startswith('{') or response_text.startswith('['):
                        try:
                            skybot_result = response.json()
                            
                            if 'data' in skybot_result and skybot_result['data']:
                                skybot_coords = SkyCoord(
                                    ra=[float(obj['RA']) for obj in skybot_result['data']], 
                                    dec=[float(obj['DEC']) for obj in skybot_result['data']], 
                                    unit=u.deg
                                )
                                
                                # Convert our catalog positions to SkyCoord
                                source_coords = SkyCoord(ra=final_table['ra'].values, dec=final_table['dec'].values, 
                                                         unit=u.deg)
                                
                                # Find best matches
                                idx, d2d, _ = source_coords.match_to_catalog_sky(skybot_coords)
                                matches = d2d < (search_radius_arcsec * u.arcsec)
                                
                                for i, (match, match_idx) in enumerate(zip(matches, idx)):
                                    if match:
                                        obj = skybot_result['data'][match_idx]
                                        final_table.loc[i, 'skybot_name'] = obj['NAME']
                                        final_table.loc[i, 'skybot_type'] = obj['OBJECT_TYPE']
                                        if 'MAGV' in obj:
                                            final_table.loc[i, 'skybot_mag'] = obj['MAGV']
                                
                                st.success(f"Found {sum(matches)} solar system objects in field.")
                            else:
                                st.write("No solar system objects found in the field.")
                        except ValueError as e:
                            st.write(f"No solar system objects found (no valid JSON data returned). {str(e)}")
                    else:
                        st.write("No solar system objects found in the field.")
                else:
                    st.warning(f"SkyBoT query failed with status code {response.status_code}")
                    
            except requests.exceptions.RequestException as req_err:
                st.warning(f"Request to SkyBoT failed: {req_err}")
        else:
            st.warning("Could not determine field center for SkyBoT query")
    except Exception as e:
        st.error(f"Error in SkyBoT processing: {str(e)}")
    
    
    # 4. Cross-match with AAVSO VSX (Variable stars)
    st.info("Querying AAVSO VSX for variable stars...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            # Use VizieR to access the VSX catalog (B/vsx/vsx)
            Vizier.ROW_LIMIT = -1  # No row limit
            vizier_result = Vizier.query_region(
                SkyCoord(ra=field_center_ra, dec=field_center_dec, unit=u.deg),
                radius=field_width_arcmin * u.arcmin,
                catalog=['B/vsx/vsx']
            )
            
            if vizier_result and 'B/vsx/vsx' in vizier_result.keys() and len(vizier_result['B/vsx/vsx']) > 0:
                vsx_table = vizier_result['B/vsx/vsx']
                
                # Create SkyCoord objects
                vsx_coords = SkyCoord(ra=vsx_table['RAJ2000'], dec=vsx_table['DEJ2000'], unit=u.deg)
                source_coords = SkyCoord(ra=final_table['ra'].values, dec=final_table['dec'].values, 
                                         unit=u.deg)
                
                # Find best matches
                idx, d2d, _ = source_coords.match_to_catalog_sky(vsx_coords)
                matches = d2d < (search_radius_arcsec * u.arcsec)
                
                # Add AAVSO information to matched sources
                final_table['aavso_name'] = None
                final_table['aavso_type'] = None
                final_table['aavso_period'] = None
                
                for i, (match, match_idx) in enumerate(zip(matches, idx)):
                    if match:
                        final_table.loc[i, 'aavso_name'] = vsx_table['Name'][match_idx]
                        final_table.loc[i, 'aavso_type'] = vsx_table['Type'][match_idx]
                        if 'Period' in vsx_table.colnames:
                            final_table.loc[i, 'aavso_period'] = vsx_table['Period'][match_idx]
                
                st.success(f"Found {sum(matches)} variable stars in field.")
            else:
                st.write("No variable stars found in the field.")
    except Exception as e:
        st.error(f"Error querying AAVSO VSX: {e}")
    
    status_text.write("Cross-matching complete!")
    
    # Create a readable summary of matches
    final_table['catalog_matches'] = ''
    
    # Add matches to summary
    if 'gaia_calib_star' in final_table.columns:
        is_calib = final_table['gaia_calib_star']
        final_table.loc[is_calib, 'catalog_matches'] += 'GAIA (calib); '
    
    if 'simbad_name' in final_table.columns:
        has_simbad = final_table['simbad_name'].notna()
        final_table.loc[has_simbad, 'catalog_matches'] += 'SIMBAD; '
    
    if 'skybot_name' in final_table.columns:
        has_skybot = final_table['skybot_name'].notna()
        final_table.loc[has_skybot, 'catalog_matches'] += 'SkyBoT; '
    
    if 'aavso_name' in final_table.columns:
        has_aavso = final_table['aavso_name'].notna()
        final_table.loc[has_aavso, 'catalog_matches'] += 'AAVSO; '
    
    # Remove trailing separators and empty entries
    final_table['catalog_matches'] = final_table['catalog_matches'].str.rstrip('; ')
    final_table.loc[final_table['catalog_matches'] == '', 'catalog_matches'] = None
    
    # Display matches summary
    matches_count = final_table['catalog_matches'].notna().sum()
    if matches_count > 0:
        st.subheader(f"Matched Objects Summary ({matches_count} sources)")
        matched_df = final_table[final_table['catalog_matches'].notna()].copy()
        
        # Select columns to display
        display_cols = ['xcenter', 'ycenter', 'ra', 'dec', 'aperture_calib_mag', 'catalog_matches']
        display_cols = [col for col in display_cols if col in matched_df.columns]
        
        st.dataframe(matched_df[display_cols])
    
    # Remove temporary match_id column if it exists
    if 'match_id' in final_table.columns:
        final_table.drop('match_id', axis=1, inplace=True)
        
    return final_table


def get_download_link(data, filename, link_text="Download"):
    """
    Generate a download link for data with improved styling and error handling.
    
    Parameters
    ----------
    data : str
        String data to be downloaded
    filename : str
        Name of the file to be downloaded
    link_text : str, optional
        Text to display on the download button
        
    Returns
    -------
    str
        HTML for the download link
    """
    if not data:
        return '<div class="error-text">No data available to download</div>'
        
    try:
        import base64
        
        # Ensure we have string data
        if isinstance(data, bytes):
            b64 = base64.b64encode(data).decode()
        else:
            b64 = base64.b64encode(data.encode()).decode()
        
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-button">{link_text}</a>'
        
        button_style = """
        <style>
        .download-button {
            display: inline-block;
            padding: 0.8em 1.4em;
            background-color: #4CAF50;  /* Green background */
            color: white;  /* White text */
            text-align: center;
            text-decoration: none;
            font-size: 16px;
            font-weight: 500;
            border-radius: 4px;
            border: none;
            cursor: pointer;
            margin-top: 15px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .download-button:hover {
            background-color: #45a049;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transform: translateY(-1px);
        }
        .error-text {
            color: red;
            font-style: italic;
        }
        </style>
        """
        
        return button_style + href
    except Exception as e:
        return f'<div class="error-text">Error creating download link: {str(e)}</div>'


def save_header_to_txt(header, filename):
    """
    Save FITS header to a text file.

    Parameters
    ----------
    header : dict
        FITS header dictionary
    filename : str
        Base filename to use (without extension)

    Returns
    -------
    str
        Full path to the saved file
    """
    if header is None:
        return None
        
    # Create a nicely formatted text representation of the header
    header_txt = "FITS Header\n"
    header_txt += "==========\n\n"
    
    # Format each key-value pair
    for key, value in header.items():
        header_txt += f"{key:8} = {value}\n"
    
    # Use the output directory
    output_dir = st.session_state.get('output_dir', '.')
    output_filename = os.path.join(output_dir, f"{filename}.txt")
    
    with open(output_filename, 'w') as f:
        f.write(header_txt)
        
    return output_filename


# ------------------------------------------------------------------------------

# Main Script Execution
# ------------------------------------------------------------------------------

def initialize_session_state():
    """Initialize all session state variables in one place to improve organization."""
    # Image data storage
    if 'calibrated_data' not in st.session_state:
        st.session_state['calibrated_data'] = None
    if 'calibrated_header' not in st.session_state:
        st.session_state['calibrated_header'] = None
    if 'final_phot_table' not in st.session_state:
        st.session_state['final_phot_table'] = None
    if 'epsf_model' not in st.session_state:
        st.session_state['epsf_model'] = None
    if 'epsf_photometry_result' not in st.session_state:
        st.session_state['epsf_photometry_result'] = None
    
    # Log handling
    if 'log_buffer' not in st.session_state:
        st.session_state['log_buffer'] = None
    if 'base_filename' not in st.session_state:
        st.session_state['base_filename'] = "photometry"
    
    # Coordinate storage
    if 'manual_ra' not in st.session_state:
        st.session_state['manual_ra'] = ""
    if 'manual_dec' not in st.session_state:
        st.session_state['manual_dec'] = ""
    if 'valid_ra' not in st.session_state:
        st.session_state['valid_ra'] = None
    if 'valid_dec' not in st.session_state:
        st.session_state['valid_dec'] = None
    
    # Analysis parameters
    if 'analysis_parameters' not in st.session_state:
        st.session_state['analysis_parameters'] = {
            'seeing': 3.5,
            'threshold_sigma': 3.0,
            'detection_mask': 50,
            'gaia_band': "phot_g_mean_mag",
            'gaia_min_mag': 11.0,
            'gaia_max_mag': 19.0,
            'calibrate_bias': False,
            'calibrate_dark': False,
            'calibrate_flat': False
        }
    
    # File tracking
    if 'files_loaded' not in st.session_state:
        st.session_state['files_loaded'] = {
            'science_file': None,
            'bias_file': None,
            'dark_file': None,
            'flat_file': None
        }


def get_base_filename(file_obj):
    """
    Extract the base filename from an uploaded file object without extension.
    
    Parameters
    ----------
    file_obj : UploadedFile
        The uploaded file object
    
    Returns
    -------
    str
        Base filename without extension
    """
    if file_obj is None:
        return "photometry"
        
    # Get original filename
    original_name = file_obj.name
    
    # Remove extension(s) from filename
    base_name = os.path.splitext(original_name)[0]
    
    # Remove any additional extensions (e.g. for .fits.fz files)
    base_name = os.path.splitext(base_name)[0]
    
    return base_name


def initialize_log(base_filename):
    """
    Initialize a log file for the current processing session.
    
    Parameters
    ----------
    base_filename : str
        Base name for the log file
    
    Returns
    -------
    StringIO
        Log buffer object
    """
    log_buffer = StringIO()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_buffer.write("Photometric Factory for RAPAS Log\n")
    log_buffer.write("===============================\n")
    log_buffer.write(f"Processing started: {timestamp}\n")
    log_buffer.write(f"Input file: {base_filename}\n\n")
    
    return log_buffer


def write_to_log(log_buffer, message, level="INFO"):
    """
    Write a message to the log buffer with timestamp and level.
    
    Parameters
    ----------
    log_buffer : StringIO
        The log buffer to write to
    message : str
        The message to log
    level : str, optional
        Log level (INFO, WARNING, ERROR)
    """
    if log_buffer is None:
        return
        
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    log_buffer.write(f"[{timestamp}] {level}: {message}\n")


initialize_session_state()

st.title("_Photometric Factory for RAPAS_")

# Photometry parameters in sidebar
with st.sidebar:
    st.sidebar.header("Upload FITS Files")
    
    # Store uploaded files in session state
    bias_file = st.file_uploader("Master Bias (optional)", type=['fits', 'fit', 'fts'], 
                                key="bias_uploader")
    if bias_file is not None:
        st.session_state.files_loaded['bias_file'] = bias_file
    
    dark_file = st.file_uploader("Master Dark (optional)", type=['fits', 'fit', 'fts'],
                                key="dark_uploader")
    if dark_file is not None:
        st.session_state.files_loaded['dark_file'] = dark_file
    
    flat_file = st.file_uploader("Master Flat (optional)", type=['fits', 'fit', 'fts'],
                                key="flat_uploader")
    if flat_file is not None:
        st.session_state.files_loaded['flat_file'] = flat_file
    
    science_file = st.file_uploader("Science Image (required)", type=['fits', 'fit', 'fts'], 
                                   key="science_uploader")
    if science_file is not None:
        st.session_state.files_loaded['science_file'] = science_file
        
        # Get base filename for logs and output
        base_filename = get_base_filename(science_file)
        st.session_state['base_filename'] = base_filename
        
        # Initialize log
        st.session_state['log_buffer'] = initialize_log(science_file.name)
    
     # Also move calibration options to sidebar
    st.header("Calibration Options")
    calibrate_bias = st.checkbox("Apply Bias", value=False,
                              help="Subtract bias frame from science image")
    calibrate_dark = st.checkbox("Apply Dark", value=False,
                              help="Subtract dark frame from science image")
    calibrate_flat = st.checkbox("Apply Flat Field", value=False,
                              help="Divide science image by flat field")
    
    st.header("Analysis Parameters")
    seeing = st.number_input("Seeing (arcsec)", value=3.5, 
                     help="Estimate of the atmospheric seeing in arcseconds")
    threshold_sigma = st.number_input("Detection Threshold (σ)", value=3.0,
                              help="Source detection threshold in sigma above background")
    detection_mask = st.slider("Border Mask (pixels)", 25, 200, 50, 25,
                           help="Size of border to exclude from source detection")
    
    st.header("Gaia Parameters")
    gaia_band = st.selectbox("Gaia Band", 
                          ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
                          index=0,
                          help="Gaia magnitude band to use for calibration")
    gaia_min_mag = st.slider("Gaia Min Magnitude", 7.0, 12.0, 10.0, 0.5,
                          help="Minimum magnitude for Gaia sources")
    gaia_max_mag = st.slider("Gaia Max Magnitude", 16.0, 20.0, 19.0, 0.5,
                          help="Maximum magnitude for Gaia sources")
    
    # Move catalog name to sidebar as well
    st.header("Output Options")
    default_catalog_name = f"{st.session_state['base_filename']}_phot.csv"
    catalog_name = st.text_input("Output Catalog Filename", default_catalog_name)

    st.link_button("GAIA Archive", "https://gea.esac.esa.int/archive/")
    st.link_button("Simbad", "http://simbad.u-strasbg.fr/simbad/")
    st.link_button("VizieR", "http://vizier.u-strasbg.fr/viz-bin/VizieR")
    st.link_button("NED", "https://ned.ipac.caltech.edu/")
    st.link_button("ADS", "https://ui.adsabs.harvard.edu/")

# Make sure the output directory exists
output_dir = ensure_output_directory("rapas_results")
st.session_state['output_dir'] = output_dir

# Main processing logic
if science_file is not None:
    science_data, science_header = load_fits_data(science_file)
    bias_data, _ = load_fits_data(bias_file)
    dark_data, dark_header = load_fits_data(dark_file)
    flat_data, _ = load_fits_data(flat_file)

    # Check if we have valid WCS in the header
    wcs_obj, wcs_error = safe_wcs_create(science_header)
    if wcs_obj is None:
        st.warning(f"No valid WCS found in the FITS header: {wcs_error}")
        
        # Ask user if they want to solve with astrometry.net
        use_astrometry = st.checkbox(
            "Attempt plate solving with astrometry.net?", 
            value=True,
            help="Uses the online astrometry.net service to determine WCS coordinates"
        )
        
        if use_astrometry:
            # Let user input their API key
            api_key = st.text_input(
                "Astrometry.net API Key", 
                value="", 
                type="password",
                help="Get your key at https://nova.astrometry.net/api_help"
            )
            
            if api_key:
                # Try to solve with astrometry.net
                with st.spinner("Running plate solve with astrometry.net (this may take a while)..."):
                    wcs_obj, science_header, astrometry_msg = solve_with_astrometry_net(
                        science_data, science_header, api_key
                    )
                    
                    # Get log buffer from session state
                    log_buffer = st.session_state['log_buffer']
                    
                    if wcs_obj is not None:
                        st.success("Astrometry.net plate solving successful!")
                        write_to_log(log_buffer, f"Solved plate with astrometry.net: {astrometry_msg}")
                        
                        # Save the updated header with WCS information to a file
                        wcs_header_filename = f"{st.session_state['base_filename']}_wcs_header"
                        wcs_header_file_path = save_header_to_txt(science_header, wcs_header_filename)
                        if wcs_header_file_path:
                            st.info(f"Updated WCS header saved to {wcs_header_file_path}")
                    else:
                        st.error(f"Astrometry.net plate solving failed: {astrometry_msg}")
                        write_to_log(log_buffer, f"Failed to solve plate: {astrometry_msg}", level="ERROR")
            else:
                st.info("Please enter your astrometry.net API key to proceed with plate solving.")
    else:
        st.success("Valid WCS found in the FITS header.")
        # Get the log buffer from session state
        log_buffer = st.session_state['log_buffer']
        write_to_log(log_buffer, "Valid WCS found in header")

    # Save header to text file
    if science_header is not None:
        # Get the log buffer from session state first
        log_buffer = st.session_state['log_buffer']
        
        # Format current time for filename
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save the header to a text file with timestamp
        header_filename = f"{st.session_state['base_filename']}_header"
        header_file_path = os.path.join(output_dir, f"{header_filename}.txt")
        
        header_file = save_header_to_txt(science_header, header_filename)
        if header_file:
            write_to_log(log_buffer, f"Saved header to {header_file}")
            st.write(f"FITS header saved to {header_file}")

    # Log file loading
    log_buffer = st.session_state['log_buffer']
    write_to_log(log_buffer, f"Loaded science image: {science_file.name}")
    if bias_file:
        write_to_log(log_buffer, f"Loaded bias frame: {bias_file.name}")
    if dark_file:
        write_to_log(log_buffer, f"Loaded dark frame: {dark_file.name}")
    if flat_file:
        write_to_log(log_buffer, f"Loaded flat field: {flat_file.name}")
    
    # If headers are None, create empty dictionaries to avoid errors
    if science_header is None:
        science_header = {}
    if dark_header is None:
        dark_header = {}

    st.header("Science Image", anchor="science-image")
    
    # Show the image with zscale stretching
    if science_data is not None:
        try:
            # First create the plot
            fig_preview, ax_preview = plt.subplots(figsize=FIGURE_SIZES['medium'])
            
            # Try ZScaleInterval first, fall back to simple normalization if that fails
            try:
                norm = ImageNormalize(science_data, interval=ZScaleInterval())
                im = ax_preview.imshow(science_data, norm=norm, origin='lower', cmap="viridis")
            except Exception as norm_error:
                st.warning(f"ZScale normalization failed: {norm_error}. Using simple normalization.")
                # Fall back to simple normalization
                vmin, vmax = np.percentile(science_data, [1, 99])  # Use percentiles to avoid outliers
                im = ax_preview.imshow(science_data, vmin=vmin, vmax=vmax, origin='lower', cmap="viridis")
            
            fig_preview.colorbar(im, ax=ax_preview, label='Pixel Value')
            ax_preview.set_title(f"{science_file.name}")
            ax_preview.axis('off')
            
            # Display the plot first (before saving)
            st.pyplot(fig_preview)
            
            # Save after displaying
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{st.session_state['base_filename']}_image.png"
            image_path = os.path.join(output_dir, image_filename)
            
            # Saving as separate step
            try:
                fig_preview.savefig(image_path, dpi=150, bbox_inches='tight')
                write_to_log(log_buffer, f"Saved image plot to {image_path}")
                st.write(f"Image saved as {image_path}")
            except Exception as save_error:
                write_to_log(log_buffer, f"Failed to save image plot: {str(save_error)}", level="ERROR")
                st.error(f"Error saving image: {str(save_error)}")
            
            # Clear the figure to avoid memory issues
            plt.close(fig_preview)
            
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            st.exception(e)  # Show full traceback for debugging
    else:
        st.error("No image data available to display. Check if the file was loaded correctly.")

    # Show header information
    with st.expander("Science Image Header"):
        if science_header:
            st.text(repr(science_header))
        else:
            st.warning("No header information available for science image.")

    # Show basic statistics about the image
    st.subheader("Science Image Statistics")
    if science_data is not None:
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("Mean", f"{np.mean(science_data):.3f}")
        stats_col2.metric("Median", f"{np.median(science_data):.3f}")
        stats_col3.metric("Std Dev", f"{np.std(science_data):.3f}")

        # Get pixel scale and calculate estimated FWHM
        pixel_size_arcsec, pixel_scale_source = extract_pixel_scale(science_header)
        st.metric("Pixel Scale (arcsec/pixel)", f"{pixel_size_arcsec:.2f} (estimated from header)")
        write_to_log(log_buffer, f"Pixel scale: {pixel_size_arcsec:.2f} arcsec/pixel ({pixel_scale_source})")

        mean_fwhm_pixel = seeing / pixel_size_arcsec
        st.metric("Mean FWHM (pixels)", f"{mean_fwhm_pixel:.2f} (estimated from seeing)")
        write_to_log(log_buffer, f"Estimated FWHM: {seeing:.2f} arcsec ({mean_fwhm_pixel:.2f} pixels)")
        # Check if RA/DEC are missing from header
       # Check if RA/DEC are missing from header
        ra_val, dec_val, coord_source = extract_coordinates(science_header)
        if ra_val is not None and dec_val is not None:
            st.write(f"Target: RA={ra_val}°, DEC={dec_val}° (from header)")
            write_to_log(log_buffer, f"Target coordinates: RA={ra_val}°, DEC={dec_val}° ({coord_source})")
            ra_missing = dec_missing = False
        else:
            st.warning(f"Coordinate issue: {coord_source}")
            write_to_log(log_buffer, f"Coordinate issue: {coord_source}", level="WARNING")
            ra_missing = dec_missing = True

        # Initialize session state for manual coordinates if needed
        if 'manual_ra' not in st.session_state:
            st.session_state['manual_ra'] = ""
        if 'manual_dec' not in st.session_state:
            st.session_state['manual_dec'] = ""
            
        # If coordinates are missing, show input fields
        if ra_missing or dec_missing:
            st.warning("Target coordinates (RA/DEC) not found in FITS header. Please enter them manually:")
            
            coord_col1, coord_col2 = st.columns(2)
            
            with coord_col1:
                # Default to empty or center of image if available
                default_ra = st.session_state['manual_ra']
                if not default_ra and 'NAXIS1' in science_header and 'CRPIX1' in science_header and 'CD1_1' in science_header:
                    # Try to estimate center RA from WCS if partial WCS exists
                    default_ra = str(science_header.get('CRVAL1', ""))
                    
                manual_ra = st.text_input(
                    "Right Ascension (degrees)", 
                    value=default_ra,
                    help="Enter RA in decimal degrees (0-360)",
                    key="ra_input"
                )
                # Store in session state
                st.session_state['manual_ra'] = manual_ra
            
            with coord_col2:
                default_dec = st.session_state['manual_dec']
                if not default_dec and 'NAXIS2' in science_header and 'CRPIX2' in science_header and 'CD2_2' in science_header:
                    # Try to estimate center DEC from WCS if partial WCS exists
                    default_dec = str(science_header.get('CRVAL2', ""))
                    
                manual_dec = st.text_input(
                    "Declination (degrees)", 
                    value=default_dec,
                    help="Enter DEC in decimal degrees (-90 to +90)",
                    key="dec_input"
                )
                # Store in session state
                st.session_state['manual_dec'] = manual_dec
            
            # Add the manual coordinates to the header if provided
            if manual_ra and manual_dec:
                try:
                    # Validate input is numeric
                    ra_val = float(manual_ra)
                    dec_val = float(manual_dec)
                    
                    # Validate ranges
                    if not (0 <= ra_val < 360):
                        st.error("RA must be between 0 and 360 degrees")
                    elif not (-90 <= dec_val <= 90):
                        st.error("DEC must be between -90 and +90 degrees")
                    else:
                        # Add to header and session state for persistence
                        science_header['RA'] = ra_val
                        science_header['DEC'] = dec_val
                        
                        # Store the validated coordinates in a more permanent session state
                        st.session_state['valid_ra'] = ra_val
                        st.session_state['valid_dec'] = dec_val
                        
                        st.success(f"Using manual coordinates: RA={ra_val}°, DEC={dec_val}°")
                except ValueError:
                    st.error("RA and DEC must be valid numbers")
            else:
                st.warning("Please enter both RA and DEC coordinates")
        else:
            # If header has coordinates, store them in session state
            for ra_key in ['RA', 'OBJRA', 'RA---', 'CRVAL1']:
                if ra_key in science_header:
                    st.session_state['valid_ra'] = science_header[ra_key]
                    break
                    
            for dec_key in ['DEC', 'OBJDEC', 'DEC---', 'CRVAL2']:
                if dec_key in science_header:
                    st.session_state['valid_dec'] = science_header[dec_key]
                    break

        # Ensure the header has the coordinates from session state before calculating airmass
        if 'valid_ra' in st.session_state and 'valid_dec' in st.session_state:
            science_header['RA'] = st.session_state['valid_ra']
            science_header['DEC'] = st.session_state['valid_dec']

        # Calculate airmass
        try:
            air = airmass(science_header)
            st.write(f"Airmass: {air:.2f}")
        except Exception as e:
            st.warning(f"Error calculating airmass: {e}")
            air = 1.0
            st.write(f"Using default airmass: {air:.2f}")

        # Image calibration button
        calibration_disabled = not (calibrate_bias or calibrate_dark or calibrate_flat)
        exposure_time_science = science_header.get('EXPOSURE', science_header.get('EXPTIME', 1.0))
        exposure_time_dark = dark_header.get('EXPOSURE', dark_header.get('EXPTIME', exposure_time_science))
        
        if st.button("Run Image Calibration", disabled=calibration_disabled):
            with st.spinner("Calibrating image..."):
                try:
                    calibrated_data, calibrated_header = calibrate_image_streamlit(
                        science_data, science_header, bias_data, dark_data, flat_data,
                        exposure_time_science, exposure_time_dark,
                        calibrate_bias, calibrate_dark, calibrate_flat
                    )
                    st.session_state['calibrated_data'] = calibrated_data
                    st.session_state['calibrated_header'] = calibrated_header
                    
                    # Show calibrated image
                    if calibrated_data is not None:
                        st.header("Calibrated Science Image")
                        norm_calibrated = ImageNormalize(calibrated_data, interval=ZScaleInterval())
                        fig_calibrated, ax_calibrated = plt.subplots(figsize=FIGURE_SIZES['medium'], dpi=100)
                        im_calibrated = ax_calibrated.imshow(calibrated_data, norm=norm_calibrated,
                                                             origin='lower', cmap="viridis")
                        fig_calibrated.colorbar(im_calibrated, ax=ax_calibrated, label='pixel value')
                        ax_calibrated.set_title("Calibrated Image (zscale)")
                        ax_calibrated.axis('off')
                        st.pyplot(fig_calibrated)
                except Exception as e:
                    st.error(f"Error during calibration: {e}")

        # Zero point calibration button
        zero_point_button_disabled = science_file is None
        if st.button("Run Zero Point Calibration", disabled=zero_point_button_disabled, key="run_zp"):
            # Determine which data to use
            image_to_process = science_data  # Default to raw science data
            header_to_process = science_header
            
            # Use calibrated data if available
            if st.session_state['calibrated_data'] is not None:
                image_to_process = st.session_state['calibrated_data']
                header_to_process = st.session_state['calibrated_header']
            
            st.write("Doing astrometry refinement with GAIA DR3...")
            # Perform astrometry refinement with Gaia DR3 TODO

            if image_to_process is not None:
                try:
                    with st.spinner("Background Extraction, Find Sources and Perform Photometry..."):
                        phot_table_qtable, epsf_table, daofind, bkg = find_sources_and_photometry_streamlit(
                            image_to_process, header_to_process, mean_fwhm_pixel, threshold_sigma, detection_mask
                        )
                        
                        # Create a deep copy to avoid modifying the cached data
                        if phot_table_qtable is not None:
                            phot_table_df = phot_table_qtable.to_pandas().copy(deep=True)
                        else:
                            st.error("No sources detected in the image.")
                            phot_table_df = None

                    if phot_table_df is not None:
                        with st.spinner("Cross-matching with Gaia..."):
                            matched_table = cross_match_with_gaia_streamlit(
                                phot_table_qtable, header_to_process, pixel_size_arcsec, 
                                mean_fwhm_pixel, gaia_band, gaia_min_mag, gaia_max_mag
                            )

                        if matched_table is not None:
                            st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
                            st.dataframe(matched_table.head(10))

                            with st.spinner("Calculating zero point..."):
                                zero_point_value, zero_point_std, zp_plot = calculate_zero_point_streamlit(
                                    phot_table_qtable, matched_table, gaia_band, air
                                )
                                
                                if zero_point_value is not None:
                                    st.pyplot(zp_plot)
                                    write_to_log(log_buffer, f"Zero point: {zero_point_value:.3f} ± {zero_point_std:.3f}")
                                    write_to_log(log_buffer, f"Airmass: {air:.2f}")
                                    
                                    # Prepare for download if final_phot_table exists in session state
                                    if 'final_phot_table' in st.session_state:
                                        final_table = st.session_state['final_phot_table']
                                        
                                        try:
                                            # Add EPSF photometry results if available
                                            if 'epsf_photometry_result' in st.session_state and epsf_table is not None:
                                                # Convert tables to pandas if needed
                                                epsf_df = epsf_table.to_pandas() if not isinstance(epsf_table, pd.DataFrame) else epsf_table
                                                
                                                # Check what column names are used for coordinates
                                                epsf_x_col = 'x_fit' if 'x_fit' in epsf_df.columns else 'xcenter'
                                                epsf_y_col = 'y_fit' if 'y_fit' in epsf_df.columns else 'ycenter'
                                                
                                                # Find coordinate columns in final table
                                                final_x_col = 'xcenter' if 'xcenter' in final_table.columns else 'x_0'
                                                final_y_col = 'ycenter' if 'ycenter' in final_table.columns else 'y_0'
                                                
                                                # Add a unique ID for matching (based on x,y position)
                                                if epsf_x_col in epsf_df.columns and epsf_y_col in epsf_df.columns:
                                                    epsf_df['match_id'] = epsf_df[epsf_x_col].round(2).astype(str) + "_" + epsf_df[epsf_y_col].round(2).astype(str)
                                                    
                                                if final_x_col in final_table.columns and final_y_col in final_table.columns:
                                                    final_table['match_id'] = final_table[final_x_col].round(2).astype(str) + "_" + final_table[final_y_col].round(2).astype(str)
                                                else:
                                                    st.warning(f"Coordinate columns not found in final table. Available columns: {final_table.columns.tolist()}")
                                                    
                                                # Select just the columns we need from EPSF results
                                                epsf_cols = {}
                                                epsf_cols['match_id'] = 'match_id'
                                                epsf_cols['flux_fit'] = 'epsf_flux_fit'
                                                epsf_cols['flux_unc'] = 'epsf_flux_unc'
                                                epsf_cols['instrumental_mag'] = 'epsf_instrumental_mag'
                                                
                                                # Only continue if we have the necessary columns
                                                if len(epsf_cols) > 1 and 'match_id' in epsf_df.columns and 'match_id' in final_table.columns:
                                                    # Select columns that actually exist
                                                    epsf_subset = epsf_df[[col for col in epsf_cols.keys() if col in epsf_df.columns]].rename(columns=epsf_cols)
                                                    
                                                    # Merge the EPSF results with the final table
                                                    final_table = pd.merge(final_table, epsf_subset, on='match_id', how='left')
                                                    
                                                    # Calculate calibrated magnitudes for EPSF photometry
                                                    if 'epsf_instrumental_mag' in final_table.columns:
                                                        final_table['psf_calib_mag'] = final_table['epsf_instrumental_mag'] + zero_point_value + 0.1*air
                                                        st.success("Added EPSF photometry results to the catalog")
                                                    
                                                    # Ensure we have aperture photometry in the final table too
                                                    if 'instrumental_mag' in final_table.columns:
                                                        if 'calib_mag' not in final_table.columns:
                                                            final_table['aperture_instrumental_mag'] = final_table['instrumental_mag']
                                                            final_table['aperture_calib_mag'] = final_table['instrumental_mag'] + zero_point_value + 0.1*air
                                                        else:
                                                            final_table = final_table.rename(columns={
                                                                'instrumental_mag': 'aperture_instrumental_mag',
                                                                'calib_mag': 'aperture_calib_mag'
                                                            })
                                                        st.success("Added aperture photometry results to the catalog")
                                                    
                                                    # Remove temporary match column
                                                    final_table.drop('match_id', axis=1, inplace=True)
                                            
                                            # Create a StringIO buffer for CSV data
                                            csv_buffer = StringIO()
                                            
                                            # Check and remove problematic columns if they exist
                                            cols_to_drop = []
                                            for col_name in ['sky_center.ra', 'sky_center.dec']:
                                                if col_name in final_table.columns:
                                                    cols_to_drop.append(col_name)
                                            
                                            if cols_to_drop:
                                                final_table = final_table.drop(columns=cols_to_drop)
                                            
                                            # Before writing to CSV, ensure match_id column is removed
                                            if 'match_id' in final_table.columns:
                                                final_table.drop('match_id', axis=1, inplace=True)
                                            
                                            # Add zero point information to the table
                                            final_table['zero_point'] = zero_point_value
                                            final_table['zero_point_std'] = zero_point_std
                                            final_table['airmass'] = air
                                            
                                            # Create a catalog summary in the UI
                                            st.subheader("Final Photometry Catalog")
                                            st.dataframe(final_table.head(10))
                                            
                                            # Show which columns are included
                                            st.success(f"Catalog includes {len(final_table)} sources.")
                                            
                                            # Add cross-matching functionality
                                            if 'ra' in final_table.columns and 'dec' in final_table.columns:
                                                st.subheader("Cross-matching with Astronomical Catalogs")
                                                # Calculate search radius based on FWHM
                                                search_radius = 2 * mean_fwhm_pixel * pixel_size_arcsec
                                                final_table = enhance_catalog_with_crossmatches(
                                                    final_table,
                                                    matched_table,  # Pass our Gaia-matched calibration table 
                                                    header_to_process,
                                                    pixel_size_arcsec, 
                                                    search_radius_arcsec=search_radius
                                                )
                                            else:
                                                st.warning("RA/DEC coordinates not available for catalog cross-matching")

                                            # Write the filtered DataFrame to the buffer
                                            final_table.to_csv(csv_buffer, index=False)
                                            csv_data = csv_buffer.getvalue()

                                            # Add timestamp to catalog filename
                                            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                            base_catalog_name = catalog_name
                                            if base_catalog_name.endswith('.csv'):
                                                base_catalog_name = base_catalog_name[:-4]
                                            filename = f"{base_catalog_name}.csv"

                                            # Save to rapas_results directory
                                            catalog_path = os.path.join(output_dir, filename)

                                            # Provide download button with better formatting
                                            download_link = get_download_link(
                                                csv_data, 
                                                filename, 
                                                link_text="Download Photometry Catalog"
                                            )
                                            st.markdown(download_link, unsafe_allow_html=True)

                                            # Also save locally in the output directory
                                            with open(catalog_path, 'w') as f:
                                                f.write(csv_data)
                                            st.write(f"Catalog saved to {catalog_path}")
                                            
                                        except Exception as e:
                                            st.error(f"Error preparing download: {e}")
                        else:
                            st.error("Failed to cross-match with Gaia catalog. Check WCS information in image header.")
                except Exception as e:
                    st.error(f"Error during zero point calibration: {str(e)}")
                    st.exception(e)  # This will show the full traceback for debugging

                # Display PanSTARRS color view with detected sources
                st.subheader("DSS2 Color View")

                # Extract RA/DEC from header or WCS
                ra_center = None
                dec_center = None

                if 'CRVAL1' in header_to_process and 'CRVAL2' in header_to_process:
                    ra_center = header_to_process['CRVAL1']
                    dec_center = header_to_process['CRVAL2']
                elif 'RA' in header_to_process and 'DEC' in header_to_process:
                    ra_center = header_to_process['RA']
                    dec_center = header_to_process['DEC']
                elif 'OBJRA' in header_to_process and 'OBJDEC' in header_to_process:
                    ra_center = header_to_process['OBJRA']
                    dec_center = header_to_process['OBJDEC']

                if ra_center is not None and dec_center is not None:
                    st.write(f"Creating Aladin view centered at RA={ra_center}, DEC={dec_center}")
    
                    # Create a direct URL to Aladin Lite with pre-configured parameters
                    aladin_url = (
                        "https://aladin.u-strasbg.fr/AladinLite/?" +
                        f"target={ra_center}%20{dec_center}" +
                        "&fov=0.5" +
                        "&survey=CDS/P/DSS2/color"
                    )
                    
                    # Create an iframe to embed Aladin Lite
                    iframe_html = f"""
                    <iframe 
                        src="{aladin_url}" 
                        width="100%" 
                        height="550px" 
                        style="border: 1px solid #ddd; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"
                        allowfullscreen>
                    </iframe>
                    """
                    
                    # Display the iframe
                    st.markdown(iframe_html, unsafe_allow_html=True)
                    
                    # Add instructions for manual catalog overlay
                    with st.expander("How to overlay Gaia DR3 catalog"):
                        st.markdown("""
                        1. Click the "layers" icon in the upper right of the Aladin window
                        2. Select "Catalog → VizieR"
                        3. Search for "I/355" (Gaia DR3)
                        4. Click on "I/355/gaiadr3" to add the Gaia DR3 catalog
                        """)
                    
                    # Add ESA Sky button with target coordinates
                    st.link_button(
                        "ESA Sky", 
                        f"https://sky.esa.int/esasky/?target={ra_center}%20{dec_center}&hips=DSS2+color&fov=1&projection=SIN&cooframe=J2000&sci=true&lang=en",
                        help="Open ESA Sky with the same target coordinates"
                    )
                else:
                    st.warning("Could not determine coordinates from image header. Cannot display PanSTARRS view.")
else:
    st.write("👆 Please upload a science image FITS file to start.")


# Save log file only if we have a log buffer
if 'log_buffer' in st.session_state and st.session_state['log_buffer'] is not None:
    log_buffer = st.session_state['log_buffer']
    # Remove timestamp from log filename
    log_filename = f"{st.session_state['base_filename']}.log"
    log_filepath = os.path.join(output_dir, log_filename)

    with open(log_filepath, 'w') as f:
        # Add final timestamp
        write_to_log(log_buffer, "Processing completed", level="INFO")
        f.write(log_buffer.getvalue())
        write_to_log(log_buffer, f"Log saved to {log_filepath}")
    
    # Just show a small notification
    st.write(f"Processing log saved to {log_filepath}")