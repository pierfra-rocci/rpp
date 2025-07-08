# Standard Library Imports
import os
import json
import zipfile
from io import StringIO
from datetime import datetime
from astropy.io import fits

# Third-Party Imports
import requests
import numpy as np
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import streamlit as st  # Keep if directly used, otherwise remove

# Constants
FIGURE_SIZES = {
    "small": (6, 5),  # For small plots
    "medium": (8, 6),  # For medium plots
    "large": (10, 8),  # For large plots
    "wide": (12, 6),  # For wide plots
    "stars_grid": (10, 8),  # For grid of stars
}

URL = "https://astro-colibri.science/"

GAIA_BANDS = [
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
    "u_jkc_mag",
    "v_jkc_mag",
    "b_jkc_mag",
    "r_jkc_mag",
    "i_jkc_mag",
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
        The URL to fetch JSON data from. Must start with 'http' or 'https'.

    Returns
    -------
    dict or str
        - Parsed JSON data as a Python dictionary if the request is successful
          and the response contains valid JSON.
        - A JSON-formatted string describing the error if the URL is invalid,
          the request fails (network error, timeout, non-2xx status),
          the response is empty, or the response is not valid JSON.
          Example error format: '{"error": "type", "message": "details"}'

    Notes
    -----
    - Handles network errors (requests.exceptions.RequestException) and JSON
      parsing errors (json.decoder.JSONDecodeError).
    - Validates the URL format.
    - Raises HTTPError for bad responses (4xx or 5xx).
    """
    if not url.startswith("http"):
        return json.dumps({"error": "invalid URL"})
    try:
        req = requests.get(url)
        req.raise_for_status()
        if not req.content:
            return json.dumps({"error": "empty response"})
        return req.json()
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": "request exception", "message": str(e)})
    except json.decoder.JSONDecodeError as e:
        return json.dumps({"error": "invalid json", "message": str(e)})


def ensure_output_directory(directory="rpp_results"):
    """
    Ensure the specified output directory exists, creating it if necessary.

    Parameters
    ----------
    directory : str, optional
        The path to the directory to ensure exists. Defaults to "rpp_results".

    Returns
    -------
    str
        The absolute path to the created or existing directory. If directory
        creation fails due to an exception (e.g., permission error), it
        returns the path to the current working directory (".") and may
        display a warning using Streamlit if available in the context.

    Notes
    -----
    - Uses os.makedirs to create parent directories as needed.
    - Catches potential exceptions during directory creation.
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
    Safely create an Astropy WCS object from a FITS header.

    Validates the presence of required WCS keywords, handles potential errors
    during WCS object creation, and attempts to simplify higher-dimensional WCS
    objects to celestial coordinates. Also attempts to remove potentially
    problematic non-standard keywords before WCS creation.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict-like
        The FITS header containing WCS information.

    Returns
    -------
    tuple (astropy.wcs.WCS | None, str | None)
        - (wcs_object, None): If successful, returns the created WCS object
          and None for the error message.
        - (None, error_message): If failed, returns None for the WCS object
          and a string describing the error (e.g., missing keywords,
          creation error, invalid object).

    Notes
    -----
    - Required WCS keywords checked: CTYPE1, CTYPE2, CRVAL1, CRVAL2,
      CRPIX1, CRPIX2.
    - Attempts to remove 'XPIXELSZ', 'YPIXELSZ', 'CDELTM1', 'CDELTM2' if
      present.
    - If the initial WCS object has more than 2 pixel dimensions, it attempts
      to extract the celestial WCS using `wcs_obj.celestial`.
    - Validates that the final WCS object has transformation attributes
      (`.wcs`).
    """
    if not header:
        return None, "No header provided"

    # Create a copy of the header to avoid modifying the original
    try:
        working_header = header.copy()
    except AttributeError:
        # Handle case where header is a regular dict
        working_header = dict(header)

    # Remove problematic keywords that can interfere with WCS
    problematic_keywords = ['XPIXELSZ', 'YPIXELSZ', 'CDELTM1', 'CDELTM2', 
                          'PIXSCALE', 'SCALE', 'XORGSUBF', 'YORGSUBF']
    
    removed_keywords = []
    for keyword in problematic_keywords:
        if keyword in working_header:
            try:
                del working_header[keyword]
                removed_keywords.append(keyword)
            except KeyError:
                pass
    
    if removed_keywords and hasattr(st, 'info'):
        st.info("Removed problematic WCS keywords")  # {', '.join(removed_keywords)}")

    required_keys = ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2",
                     "CRPIX1", "CRPIX2"]
    missing_keys = [key for key in required_keys if key not in working_header]

    if missing_keys:
        return None, f"Missing required WCS keywords: {', '.join(missing_keys)}"

    try:
        wcs_obj = WCS(working_header)

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

    Iterates through a list of potential keywords and returns the value of the
    first one found in the header. Useful for handling variations in FITS
    keyword conventions.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict-like or None
        The FITS header object or dictionary to search within. If None, the
        default value is returned immediately.
    keys : list[str]
        A list of header keyword strings to try, in order of preference.
    default : any, optional
        The value to return if none of the specified keys are found in the
        header. Defaults to None.

    Returns
    -------
    any
        The value associated with the first key found in the header, or the
        `default` value if none of the keys are present or the header is None.

    Examples
    --------
    >>> from astropy.io import fits
    >>> hdr = fits.Header([('EXPTIME', 120.0), ('INSTRUME', 'CCD')])
    >>> exposure = get_header_value(hdr, ['EXPTIME', 'EXPOSURE', 'EXP'], 0.0)
    >>> print(exposure)
    120.0
    >>> filter_name = get_header_value(hdr, ['FILTER'], 'Unknown')
    >>> print(filter_name)
    Unknown
    """
    if header is None:
        return default

    for key in keys:
        if key in header:
            return header[key]
    return default


def fix_header(header):
    """
    Fix common issues in FITS headers based on stdweb processing approach.
    
    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header object to fix
        
    Returns
    -------
    astropy.io.fits.Header
        Fixed header object
    """
    try:
        # Create a copy to avoid modifying the original
        fixed_header = header.copy()
        
        # Define problematic keywords to remove
        problematic_keywords = [
            'XPIXELSZ', 'YPIXELSZ',      # Pixel size keywords that can conflict with WCS
            'CDELTM1', 'CDELTM2',        # Alternative delta keywords
            'PIXSCALE',                   # Non-standard pixel scale
            'SCALE',                      # Generic scale keyword
            'XORGSUBF', 'YORGSUBF',      # Origin subframe keywords
        ]
        
        # Remove problematic keywords
        removed_keywords = []
        for keyword in problematic_keywords:
            if keyword in fixed_header:
                try:
                    del fixed_header[keyword]
                    removed_keywords.append(keyword)
                except KeyError:
                    pass
        
        if removed_keywords:
            st.info(f"Removed problematic keywords: {', '.join(removed_keywords)}")
        
        # Define WCS keywords to check for problems
        wcs_keywords = {
            'CD': ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'],
            'PC': ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'],
            'CDELT': ['CDELT1', 'CDELT2'],
            'CRPIX': ['CRPIX1', 'CRPIX2'],
            'CRVAL': ['CRVAL1', 'CRVAL2'],
            'CTYPE': ['CTYPE1', 'CTYPE2']
        }
        
        # Check for problematic values and remove entire WCS if found
        remove_all_wcs = False
        
        # Check CD matrix for singularity
        cd_keys = wcs_keywords['CD']
        if all(key in fixed_header for key in cd_keys):
            try:
                cd11 = float(fixed_header['CD1_1'])
                cd12 = float(fixed_header['CD1_2'])
                cd21 = float(fixed_header['CD2_1'])
                cd22 = float(fixed_header['CD2_2'])
                
                # Calculate determinant to check for singularity
                determinant = cd11 * cd22 - cd12 * cd21
                
                # Check for singular matrix or invalid values
                if (abs(determinant) < 1e-15 or 
                    any(not np.isfinite(val) for val in [cd11, cd12, cd21, cd22]) or
                        any(val == 0 for val in [cd11, cd22])):
                    st.warning("Detected problematic CD matrix - removing all WCS keywords")
                    remove_all_wcs = True
                    
            except (ValueError, TypeError):
                st.warning("Invalid CD matrix values - removing all WCS keywords")
                remove_all_wcs = True
        
        # Check for obviously fake coordinate values
        if 'CRVAL1' in fixed_header and 'CRVAL2' in fixed_header:
            try:
                ra = float(fixed_header['CRVAL1'])
                dec = float(fixed_header['CRVAL2'])
                
                # Check for clearly fake values (exactly 0,0 or out of range)
                if ((ra == 0.0 and dec == 0.0) or
                        not (0 <= ra < 360) or
                        not (-90 <= dec <= 90) or
                        not np.isfinite(ra) or
                        not np.isfinite(dec)):
                    st.warning(f"Detected fake coordinates (RA={ra}, DEC={dec}) - removing all WCS keywords")
                    remove_all_wcs = True
                    
            except (ValueError, TypeError):
                st.warning("Invalid coordinate values - removing all WCS keywords")
                remove_all_wcs = True
        
        # Check for invalid pixel reference points
        if 'CRPIX1' in fixed_header and 'CRPIX2' in fixed_header:
            try:
                crpix1 = float(fixed_header['CRPIX1'])
                crpix2 = float(fixed_header['CRPIX2'])
                
                # Check for obviously wrong values
                if (not np.isfinite(crpix1) or not np.isfinite(crpix2) or
                    crpix1 <= 0 or crpix2 <= 0):
                    st.warning("Detected invalid CRPIX values - removing all WCS keywords")
                    remove_all_wcs = True
                    
            except (ValueError, TypeError):
                remove_all_wcs = True
        
        # Check for invalid CTYPE values
        for ctype_key in ['CTYPE1', 'CTYPE2']:
            if ctype_key in fixed_header:
                ctype_val = str(fixed_header[ctype_key]).strip()
                # If CTYPE is empty or contains obvious placeholder text
                if (not ctype_val or 
                    ctype_val.lower() in ['', 'none', 'null', 'undefined'] or
                        len(ctype_val) < 3):
                    st.warning(f"Detected invalid {ctype_key} value - removing all WCS keywords")
                    remove_all_wcs = True
        
        # Remove all WCS keywords if problems detected
        if remove_all_wcs:
            st.info("Removing all WCS keywords due to detected problems")
            
            # Remove all WCS-related keywords
            keys_to_remove = []
            for key in list(fixed_header.keys()):
                if (key.startswith('CD') or 
                    key.startswith('PC') or 
                    key.startswith('CDELT') or 
                    key.startswith('CRPIX') or 
                    key.startswith('CRVAL') or 
                    key.startswith('CTYPE') or
                    key.startswith('CROTA') or
                    key.startswith('PV') or
                    key.startswith('PROJP') or
                    key in ['EQUINOX', 'EPOCH', 'RADESYS', 'LONPOLE', 'LATPOLE']):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                try:
                    del fixed_header[key]
                except KeyError:
                    pass
            
            st.info(f"Removed {len(keys_to_remove)} WCS keywords: {', '.join(keys_to_remove[:10])}{'...' if len(keys_to_remove) > 10 else ''}")
            
            # Add minimal default WCS if image dimensions are available
            if 'NAXIS1' in fixed_header and 'NAXIS2' in fixed_header:
                try:
                    naxis1 = int(fixed_header['NAXIS1'])
                    naxis2 = int(fixed_header['NAXIS2'])
                    
                    fixed_header['CTYPE1'] = 'RA---TAN'
                    fixed_header['CTYPE2'] = 'DEC--TAN'
                    fixed_header['CRPIX1'] = naxis1 / 2.0
                    fixed_header['CRPIX2'] = naxis2 / 2.0
                    fixed_header['CRVAL1'] = 0.0  # Will need manual input
                    fixed_header['CRVAL2'] = 0.0  # Will need manual input
                    fixed_header['EQUINOX'] = 2000.0
                    fixed_header['RADESYS'] = 'ICRS'
                    
                    st.info("Added minimal default WCS (1 arcsec/pixel) - coordinates will need manual input")
                    
                except (ValueError, TypeError):
                    st.warning("Could not add default WCS due to invalid NAXIS values")
        
        else:
            # If WCS seems valid, apply minor fixes
            
            # Fix CTYPE formatting
            if 'CTYPE1' in fixed_header:
                ctype1 = str(fixed_header['CTYPE1']).strip()
                if ctype1 and not ctype1.endswith('---'):
                    if 'RA' in ctype1.upper():
                        fixed_header['CTYPE1'] = 'RA---TAN'
                    elif 'GLON' in ctype1.upper():
                        fixed_header['CTYPE1'] = 'GLON-TAN'
            
            if 'CTYPE2' in fixed_header:
                ctype2 = str(fixed_header['CTYPE2']).strip()
                if ctype2 and not ctype2.endswith('---'):
                    if 'DEC' in ctype2.upper():
                        fixed_header['CTYPE2'] = 'DEC--TAN'
                    elif 'GLAT' in ctype2.upper():
                        fixed_header['CTYPE2'] = 'GLAT-TAN'
            
            # Fix missing EQUINOX/EPOCH
            if 'EQUINOX' not in fixed_header and 'EPOCH' in fixed_header:
                fixed_header['EQUINOX'] = fixed_header['EPOCH']
            elif 'EQUINOX' not in fixed_header:
                fixed_header['EQUINOX'] = 2000.0
                
            # Fix RADESYS if missing
            if 'RADESYS' not in fixed_header:
                if fixed_header.get('EQUINOX', 2000.0) == 2000.0:
                    fixed_header['RADESYS'] = 'ICRS'
                else:
                    fixed_header['RADESYS'] = 'FK5'
            
            # Remove PC matrix if CD matrix is present (they conflict)
            if all(key in fixed_header for key in ['CD1_1', 'CD2_2']):
                pc_keys = ['PC1_1', 'PC1_2', 'PC2_1', 'PC2_2']
                removed_pc = []
                for key in pc_keys:
                    if key in fixed_header:
                        del fixed_header[key]
                        removed_pc.append(key)
                if removed_pc:
                    st.info(f"Removed conflicting PC matrix keywords: {', '.join(removed_pc)}")
        
        return fixed_header
        
    except Exception as e:
        # If fixing fails completely, return original header
        st.warning(f"Header fixing failed: {str(e)}")
        return header


def safe_catalog_query(query_func, error_msg, *args, **kwargs):
    """
    Execute an astronomical catalog query function with robust error handling.

    Wraps a callable (e.g., a function from astroquery) to catch common
    exceptions like network errors, timeouts, and value errors that can occur
    during queries to online astronomical databases.

    Parameters
    ----------
    query_func : callable
        The function to call for performing the catalog query.
    error_msg : str
        A base error message string to prepend to any specific exception
        message caught during the query execution.
    *args
        Positional arguments to pass directly to `query_func`.
    **kwargs
        Keyword arguments to pass directly to `query_func`.

    Returns
    -------
    tuple (any | None, str | None)
        - (result, None): If `query_func` executes successfully, returns its
          result and None for the error message. The type of `result` depends
          on `query_func`.
        - (None, error_message): If an exception occurs during execution,
          returns None for the result and a formatted string describing the
          error (e.g., "Failed to query SIMBAD: Network error - details").

    Examples
    --------
    >>> from astroquery.simbad import Simbad
    >>> # Assuming Simbad.query_object raises a Timeout error
    >>> result, error = safe_catalog_query(
    ...     Simbad.query_object,
    ...     "Failed to query SIMBAD",
    ...     "M31"
    ... )
    >>> if error:
    ...     # Output: Query failed: Failed to query SIMBAD: Query timed out
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
    Create a matplotlib Figure object with a predefined or default size and DPI.

    Uses a dictionary `FIGURE_SIZES` to map descriptive size names ('small',
    'medium', 'large', 'wide', 'stars_grid') to (width, height) tuples in
    inches.

    Parameters
    ----------
    size : str, optional
        A key corresponding to a predefined figure size in the `FIGURE_SIZES`
        global dictionary. Allowed values are 'small', 'medium', 'large',
        'wide', 'stars_grid'. If an invalid key is provided, it defaults to
        'medium'. Defaults to "medium".
    dpi : int, optional
        The resolution of the figure in dots per inch. Defaults to 120.

    Returns
    -------
    matplotlib.figure.Figure
        A matplotlib Figure object initialized with the specified or default
        figsize and dpi.

    Notes
    -----
    - Relies on the global `FIGURE_SIZES` dictionary for size definitions.
    """
    if size in FIGURE_SIZES:
        figsize = FIGURE_SIZES[size]
    else:
        figsize = FIGURE_SIZES["medium"]
    return plt.figure(figsize=figsize, dpi=dpi)


def extract_coordinates(header):
    """
    Extract celestial coordinates (RA, Dec) from a FITS header.

    Attempts to find Right Ascension (RA) and Declination (Dec) values by
    checking a predefined list of common FITS keywords. Validates that the
    extracted values are numeric and fall within reasonable astronomical
    ranges.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict-like or None
        The FITS header object or dictionary to search for coordinate keywords.
        If None, returns (None, None, "No header available").

    Returns
    -------
    tuple (float | None, float | None, str)
        - (ra, dec, source_description): If coordinates are found and valid,
          returns RA (degrees), Dec (degrees), and a string indicating the
          keywords used (e.g., "RA/DEC", "OBJRA/OBJDEC").
        - (None, None, error_message): If coordinates are not found, are
          non-numeric, or fall outside valid ranges (RA: -360 to 360,
          Dec: -90 to 90), returns None for RA and Dec, and a string
          describing the issue (e.g., "Coordinates not found in header",
          "Invalid RA value: ...", "Non-numeric coordinates: ...").

    Notes
    -----
    - RA keywords checked (in order): 'RA', 'OBJRA', 'RA---', 'CRVAL1'.
    - Dec keywords checked (in order): 'DEC', 'OBJDEC', 'DEC---', 'CRVAL2'.
    - Validation ranges: RA [-360, 360], Dec [-90, 90].
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
    Extract or calculate the pixel scale (in arcseconds per pixel) from a
    FITS header.

    Tries multiple methods in order of preference:
    1. Looks for direct pixel scale keywords ('PIXSIZE', 'PIXSCALE',
       'PIXELSCAL').
    2. Uses CD matrix elements (CD1_1, CD2_2) converted from deg to arcsec.
    3. Calculates from pixel size ('XPIXSZ') and focal length ('FOCALLEN'),
       using the simple formula: 206 × pixel_size_microns / focal_length_mm.
    4. If none of the above succeed, returns a default value.

    Parameters
    ----------
    header : astropy.io.fits.Header or dict-like or None
        The FITS header object or dictionary containing metadata. If None,
        returns (1.0, "default (no header)").

    Returns
    -------
    tuple (float, str)
        - (pixel_scale_value, source_description): Returns the determined pixel
          scale in arcseconds per pixel and a string describing how it was
          obtained.
        - The default pixel scale value returned if no information is found is
          1.0 arcsec/pixel.

    Notes
    -----
    - Uses the simple formula: pixel_scale = 206 × pixel_size_microns / focal_length_mm
    - Assumes XPIXSZ is in micrometers unless XPIXSZU specifies otherwise.
    """
    if header is None:
        return 1.0, "default (no header)"

    # Method 1: Direct pixel scale keywords (these should be in arcsec/pixel)
    for key in ["PIXSIZE", "PIXSCALE", "PIXELSCAL", "SECPIX"]:
        if key in header:
            value = float(header[key])
            # Sanity check: pixel scale should be reasonable (0.01 to 100 arcsec/pixel)
            if 0.01 <= value <= 100:
                return value, f"from {key}"

    # Method 2: CD matrix elements (most common after plate solving)
    if "CD1_1" in header and "CD2_2" in header:
        cd1_1 = abs(float(header["CD1_1"]))
        cd2_2 = abs(float(header["CD2_2"]))
        
        # Take average of both diagonal elements and convert from degrees to arcsec
        avg_cd = (cd1_1 + cd2_2) / 2.0
        scale = avg_cd * 3600.0
        
        # Sanity check: CD matrix values should result in reasonable pixel scales
        if 0.01 <= scale <= 100:
            return scale, f"from CD matrix (CD1_1={cd1_1:.2e}, CD2_2={cd2_2:.2e})"

    # Method 3: Calculate from pixel size and focal length using simple formula
    focal_length = None
    pixel_size = None
    
    # Check for focal length keywords
    for focal_key in ['FOCALLEN', 'FOCAL', 'FOCLEN', 'FL']:
        if focal_key in header:
            focal_length = float(header[focal_key])
            break
    
    # Check for pixel size keywords
    for pixel_key in ['PIXELSIZE', 'PIXSIZE', 'XPIXSZ', 'YPIXSZ', 'PIXELMICRONS']:
        if pixel_key in header:
            pixel_size = float(header[pixel_key])
            break
    
    if focal_length is not None and pixel_size is not None and focal_length > 0:
        xpixsz_unit = header.get("XPIXSZU", "").strip().lower()
        
        # Handle different units
        if xpixsz_unit == "mm":
            # Convert mm to microns
            pixel_size_microns = pixel_size * 1000.0
            unit_desc = "mm"
        elif xpixsz_unit in ["um", "μm", "micron", "microns"]:
            pixel_size_microns = pixel_size
            unit_desc = "μm"
        else:
            # Default assumption: micrometers
            pixel_size_microns = pixel_size
            unit_desc = "μm (assumed)"
        
        # Use the simple formula: pixel_scale = 206 × pixel_size_microns / focal_length_mm
        scale = 206.0 * pixel_size_microns / focal_length
        
        if 0.01 <= scale <= 100:
            return scale, f"calculated: 206 × {pixel_size_microns} {unit_desc} / {focal_length} mm"

    # Method 4: Default fallback
    return 0.0, "default fallback value"


def get_base_filename(file_obj):
    """
    Extract the base filename (without extension) from a file object.

    Handles common cases like single extensions (.fits) and double extensions
    (.fits.fz, .tar.gz).

    Parameters
    ----------
    file_obj : file-like object or None
        An object representing the uploaded file, typically expected to have a
        `.name` attribute (like Streamlit's `UploadedFile`). If None, returns
        a default filename "photometry".

    Returns
    -------
    str
        The base filename derived from `file_obj.name` by removing the
        extension(s). Returns "photometry" if `file_obj` is None.

    Examples
    --------
    >>> class MockFile: name = "image.fits.fz"
    >>> get_base_filename(MockFile())
    'image'
    >>> class MockFile: name = "catalog.csv"
    >>> get_base_filename(MockFile())
    'catalog'
    >>> get_base_filename(None)
    'photometry'
    """
    if file_obj is None:
        return "photometry"

    original_name = file_obj.name
    base_name = os.path.splitext(original_name)[0]
    base_name = os.path.splitext(base_name)[0]

    return base_name


def cleanup_temp_files():
    """
    Remove temporary FITS files potentially created during processing.

    Specifically targets files stored in the directory indicated by
    `st.session_state['science_file_path']`. It attempts to remove all files
    ending with '.fits', '.fit', or '.fts' (case-insensitive) within that
    directory.

    Notes
    -----
    - Relies on `st.session_state['science_file_path']` being set and pointing
      to a valid temporary file path whose directory contains the files to be
      cleaned.
    - Uses `st.warning` to report errors if removal fails for individual files
      or if the base directory cannot be accessed.
    - This function has side effects (deleting files) and depends on
      Streamlit's session state. It does not return any value.
    """
    if (
        "science_file_path" in st.session_state
        and st.session_state["science_file_path"]
    ):
        try:
            temp_file = st.session_state["science_file_path"]
            if os.path.exists(temp_file):
                base_dir = os.path.dirname(temp_file)
                if os.path.isdir(base_dir):
                    temp_dir_files = [
                        f
                        for f in os.listdir(base_dir)
                        if os.path.isfile(os.path.join(base_dir, f))
                        and f.lower().endswith((".fits", ".fit", ".fts",".ssf",".log"))
                    ]
                    for file in temp_dir_files:
                        try:
                            os.remove(os.path.join(base_dir, file))
                        except Exception as e:
                            st.warning(f"Could not remove {file}: {str(e)}")
                else:
                    st.warning(
                        f"Temporary path {base_dir} is not a directory."
                    )
        except Exception as e:
            st.warning(f"Could not remove temporary files: {str(e)}")


def initialize_log(base_filename):
    """
    Initialize and return an in-memory text buffer (StringIO) for logging.

    Creates a StringIO object and writes a standard header including a
    timestamp and the provided base filename, suitable for logging processing
    steps.

    Parameters
    ----------
    base_filename : str
        The base name of the input file being processed, used in the log
        header.

    Returns
    -------
    io.StringIO
        An initialized StringIO buffer containing the log header.

    Notes
    -----
    - The log header format includes the title "RAPAS Photometry Pipeline Log",
      a separator line, the start timestamp, and the input filename.
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
    Write a formatted message to the provided log buffer.

    Prepends the message with a timestamp ([HH:MM:SS]) and the specified
    log level (e.g., INFO, WARNING, ERROR).

    Parameters
    ----------
    log_buffer : io.StringIO or None
        The StringIO buffer object to write the log message to. If None,
        the function does nothing.
    message : str
        The log message content.
    level : str, optional
        The severity level of the log message (e.g., "INFO", "WARNING",
        "ERROR"). Defaults to "INFO".

    Returns
    -------
    None
        This function modifies the `log_buffer` in place and returns None.

    Notes
    -----
    - Format: "[HH:MM:SS] LEVEL: Message\n"
    """
    if log_buffer is None:
        return

    timestamp = datetime.now().strftime("%H:%M:%S")
    log_buffer.write(f"[{timestamp}] {level.upper()}: {message}\n")


def zip_rpp_results_on_exit(science_file_obj, outputdir):
    """Compresses analysis result files into a timestamped ZIP archive.
    """
    output_dir = outputdir
    if not os.path.exists(output_dir):
        return
    base_name = get_base_filename(science_file_obj)
    files = [
        f
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
        and f.startswith(base_name)
        and not f.lower().endswith(".zip")
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
    # Do not remove .zip files, only remove the files that were zipped (non-zip)
    for file in files:
        try:
            os.remove(os.path.join(output_dir, file))
        except Exception as e:
            print(f"Warning: Could not remove file {file} after zipping: {e}")


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


def save_header_to_fits(header, filename):
    """
    Save a FITS header to a FITS file with an empty primary HDU.

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
    The function creates a minimal FITS file with the header information
    preserved in the primary HDU, and saves it to the current output directory
    (retrieved from session state).
    """
    if header is None:
        return None

    try:
        # Create a minimal primary HDU with the header
        primary_hdu = fits.PrimaryHDU(header=header)
        
        # Create HDU list
        hdul = fits.HDUList([primary_hdu])
        
        # Get output directory from session state
        output_dir = st.session_state.get("output_dir", ".")
        output_filename = os.path.join(output_dir, f"{filename}.fits")
        
        # Write FITS file
        hdul.writeto(output_filename, overwrite=True)
        hdul.close()
        
        return output_filename
        
    except Exception as e:
        if hasattr(st, 'warning'):
            st.warning(f"Failed to save header as FITS file: {str(e)}")
        return None


def save_catalog_files(final_table, catalog_name, output_dir):
    """
    Save the final photometry table as both VOTable (XML) and CSV files.

    Args:
        final_table (pd.DataFrame): The photometry results table.
        catalog_name (str): The base name for the catalog files.
        output_dir (str): The directory to save the files in.

    Returns:
        None
    """
    # Add safety check before VOTable creation
    if final_table is not None and len(final_table) > 0:
        try:
            # Clean the DataFrame before conversion to handle problematic columns
            df_for_votable = final_table.copy()
            
            # Remove or fix columns that might cause issues with astropy Table conversion
            problematic_columns = []
            for col in df_for_votable.columns:
                # Check for columns with mixed types or object arrays that might cause issues
                if df_for_votable[col].dtype == 'object':
                    # Try to convert to string, handling None/NaN values
                    try:
                        df_for_votable[col] = df_for_votable[col].astype(str)
                        df_for_votable[col] = df_for_votable[col].replace('nan', '')
                        df_for_votable[col] = df_for_votable[col].replace('None', '')
                    except:
                        problematic_columns.append(col)
            
            # Remove columns that still cause issues
            if problematic_columns:
                df_for_votable = df_for_votable.drop(columns=problematic_columns)
                st.warning(f"Removed problematic columns from VOTable: {problematic_columns}")
            
            # Convert pandas DataFrame to astropy Table
            astropy_table = Table.from_pandas(df_for_votable)
            
            # Create VOTable
            votable = from_table(astropy_table)
            
            # Define base_catalog_name here to ensure it's available for both success and fallback
            base_catalog_name = catalog_name
            if base_catalog_name.endswith(".csv"):
                base_catalog_name = base_catalog_name[:-4]
            filename = f"{base_catalog_name}.xml"  # VOTable extension

            catalog_path = os.path.join(output_dir, filename)

            # Write VOTable to file
            writeto(votable, catalog_path)
                
            st.success(f"Catalog saved successfully")
            
            # Also create CSV buffer for backward compatibility if needed
            csv_buffer = StringIO()
            final_table.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            # Save CSV file using catalog_name as file name
            csv_file_path = os.path.join(output_dir, catalog_name)
            with open(csv_file_path, "w", encoding="utf-8") as f:
                f.write(csv_data)
            st.success(f"CSV catalog saved as {catalog_name}")
            
        except Exception as e:
            st.error(f"Error preparing VOTable download: {e}")
            st.error("final_table status:")
            st.error(f"  Type: {type(final_table)}")
            st.error(f"  Length: {len(final_table) if final_table is not None else 'None'}")
            if final_table is not None:
                st.error(f"  Columns: {list(final_table.columns)}")
    else:
        st.error("Cannot create VOTable: final_table is None or empty")
        if final_table is None:
            st.error("final_table is None")
        else:
            st.error(f"final_table length: {len(final_table)}")