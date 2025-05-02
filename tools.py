# Standard Library Imports
import os
import json
import zipfile
from io import StringIO
from datetime import datetime

# Third-Party Imports
import requests
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import streamlit as st  # Keep if directly used, otherwise remove if only used in app.py

# Constants
FIGURE_SIZES = {
    "small": (6, 5),  # For small plots
    "medium": (8, 6),  # For medium plots
    "large": (10, 8),  # For large plots
    "wide": (12, 6),  # For wide plots
    "stars_grid": (10, 8),  # For grid of stars
}

URL = "https://astro-colibri.science/"

GAIA_BAND = [
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "u_jkc_mag",
    "v_jkc_mag",
    "b_jkc_mag",
    "r_jkc_mag",
    "u_sdss_mag",
    "g_sdss_mag",
    "r_sdss_mag",
    "i_sdss_mag",
    "z_sdss_mag",
]


def get_json(url: str):
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


def ensure_output_directory(directory="rpp_results"):
    """
    Create an output directory if it doesn't exist.

    Parameters
    ----------
    directory : str, optional
        Path to the directory to create. Default is "rpp_results".

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
        except Exception:
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
        header.remove("XPIXELSZ")
        header.remove("YPIXELSZ")
        header.remove("CDELTM1")
        header.remove("CDELTM2")
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


def create_figure(size="medium", dpi=120):
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


def cleanup_temp_files():
    """
    Remove temporary files created during processing.

    Cleans up:
    1. The temporary files created when uploading the Image
    2. The solved files created during plate solving
    """
    # Clean up temp science file
    if (
        "science_file_path" in st.session_state
        and st.session_state["science_file_path"]
    ):
        try:
            temp_file = st.session_state["science_file_path"]
            if os.path.exists(temp_file):
                base_dir = os.path.dirname(temp_file)
                temp_dir_files = [
                    f
                    for f in os.listdir(base_dir)
                    if os.path.isfile(os.path.join(base_dir, f))
                    and f.lower().endswith((".fits", ".fit", ".fts"))
                ]
                for file in temp_dir_files:
                    try:
                        os.remove(os.path.join(base_dir, file))
                    except Exception as e:
                        st.warning(f"Could not remove {file}: {str(e)}")
        except Exception as e:
            st.warning(f"Could not remove temporary files: {str(e)}")


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

    log_buffer.write("RAPAS Photometry Pipeline Log\n")
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


def zip_rpp_results_on_exit(science_file):
    """Compresses files in the 'rpp_results' directory into a timestamped ZIP archive.

    Takes all files in the 'rpp_results' directory that share the same base filename as the input
    science file and compresses them into a single ZIP archive with a timestamp in the filename.
    The original files are deleted after successful compression.

    Args:
        science_file (str): Path to the science file whose base filename will be used to identify
                           related files for compression.

    Returns:
        None

    Notes:
        - The ZIP archive is created in the 'rpp_results' directory
        - Files are compressed using DEFLATE algorithm
        - Original files are removed after successful compression
        - If 'rpp_results' directory doesn't exist or no matching files found, function returns silently
        - Archive filename format: {base_filename}_{YYYYMMDD_HHMMSS}.zip
    """
    output_dir = os.path.join(os.getcwd(), "rpp_results")
    if not os.path.exists(output_dir):
        return
    base_name = get_base_filename(science_file)
    files = [
        f
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f)) and f.startswith(base_name)
    ]
    if not files:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{base_name}_{timestamp}.zip"
    zip_path = os.path.join(output_dir, zip_filename)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            file_path = os.path.join(output_dir, file)
            zipf.write(file_path, arcname=file)
    # Remove the original files after zipping
    for file in files:
        try:
            os.remove(os.path.join(output_dir, file))
        except Exception:
            pass
