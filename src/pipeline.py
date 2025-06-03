import os
import subprocess
import json
import pathlib

import streamlit as st

from datetime import datetime, timedelta
from urllib.parse import quote
import requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astroscrappy
from astropy.table import Table, join
from astropy.wcs import WCS
from astropy.stats import sigma_clip, SigmaClip
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
from astropy.modeling import models, fitting
from astropy.nddata import NDData
from astropy.visualization import (ZScaleInterval, simple_norm)

import astropy.units as u
from astropy.io import fits
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from photutils.utils import calc_total_error
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, SExtractorBackground
from photutils.psf import EPSFBuilder, extract_stars, IterativePSFPhotometry
from stdpipe import photometry, astrometry, catalogs, pipeline

from src.tools import (FIGURE_SIZES, URL, safe_catalog_query,
                       safe_wcs_create, ensure_output_directory,
                       write_to_log)

from typing import Union, Any, Optional, Dict, Tuple


def solve_with_siril(file_path):
    """
    Solve astrometric plate using Siril through a PowerShell or Bash script.
    This function sends an image file path to a platform-appropriate script that uses Siril
    to determine accurate WCS (World Coordinate System) information for the image.
    It then reads the resulting solved file to extract the WCS data.

    file_path : str
        Path to the image file that needs astrometric solving
    Returns
    -------
    - wcs_object: astropy.wcs.WCS object containing the WCS solution
    - updated_header: Header with WCS keywords from the solved image

    The function expects a PowerShell script named 'plate_solve.ps1' (Windows)
    or a Bash script named 'plate_solve.sh' (Linux/macOS) to be available
    in the current or parent directory. The solved image will be saved with '_solved' appended
    to the original filename.
    """
    import platform
    try:
        if os.path.exists(file_path):
            # Find the absolute path to the plate solving script
            script_dir = pathlib.Path(__file__).parent.resolve()
            system = platform.system()
            if system == "Windows":
                script_name = "plate_solve.ps1"
                command_base = ["powershell.exe", "-ExecutionPolicy", "Bypass",
                                "-File"]
            else:
                script_name = "plate_solve.sh"
                command_base = ["bash"]
            script_path = script_dir / script_name
            if not script_path.exists():
                # Try parent directory if not found in current
                script_path = script_dir.parent / script_name
            if not script_path.exists():
                st.error(f"Could not find {script_name} at {script_path}")
                return None

            if system == "Windows":
                command = command_base + [str(script_path), "-filepath",
                                          str(file_path)]
            else:
                command = command_base + [str(script_path), str(file_path)]
            subprocess.run(command, check=True)
        else:
            st.warning(f"File {file_path} does not exist.")

    except Exception as e:
        st.error(f"Error solving with : {str(e)}")
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
    Detect and remove cosmic rays from an astronomical image using
    astroscrappy.

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
            raise ValueError("border must be an int")
    else:
        raise TypeError("border must be an int")

    if any(b < 0 for b in (top, bottom, left, right)):
        raise ValueError("Borders cannot be negative")

    if top + bottom >= height or left + right >= width:
        raise ValueError("Borders are larger than the image")

    mask = np.zeros(image.shape[:2], dtype=dtype)
    mask[top: height - bottom, left: width - right] = True

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
            # Create a figure with two subplots side by side for background/RMS
            fig_bkg, (ax1, ax2) = plt.subplots(1, 2,
                                               figsize=FIGURE_SIZES["wide"])
            
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
            username = st.session_state.get("username", "anonymous")
            output_dir = ensure_output_directory(f"{username}_rpp_results")
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
        Information about the observatory. If not provided, uses the default.
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

    # Check if airmass already exists in header
    airmass_keywords = ['AIRMASS', 'SECZ', 'AIRMASS_START', 'AIRMASS_END']
    for keyword in airmass_keywords:
        if keyword in _header and _header[keyword] is not None:
            try:
                existing_airmass = float(_header[keyword])
                # Validate the existing airmass value
                if 1.0 <= existing_airmass <= 30.0:
                    st.write("Using existing airmass from header...")
                    if return_details:
                        return existing_airmass, {"airmass_source": f"header_{keyword}"}
                    return existing_airmass
                else:
                    st.warning(f"Invalid airmass value in header ({existing_airmass}), calculating from coordinates")
                    break
            except (ValueError, TypeError):
                st.warning(f"Could not parse airmass value from header keyword {keyword}")
                continue

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
        
        # Check for various date keywords in order of preference
        date_keywords = ['DATE-OBS', 'DATE', 'DATE_OBS', 'DATEOBS', 'MJD-OBS']
        obstime_str = None
        
        for keyword in date_keywords:
            if keyword in _header and _header[keyword] is not None:
                obstime_str = _header[keyword]
                break

        if any(v is None for v in [ra, dec, obstime_str]):
            missing = []
            if ra is None:
                missing.append("RA/OBJRA/CRVAL1")
            if dec is None:
                missing.append("DEC/OBJDEC/CRVAL2")
            if obstime_str is None:
                missing.append("DATE-OBS/DATE")
            raise KeyError(f"Missing required header keywords: {', '.join(missing)}")

        coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame="icrs")
        
        # Try to parse the observation time with different formats
        try:
            # First, try direct parsing
            obstime = Time(obstime_str)
    
        except Exception:
            # If that fails, try different parsing strategies
            try:
                # Handle common FITS date formats
                if isinstance(obstime_str, str):
                    # Remove any timezone info and try to parse
                    clean_date = obstime_str.replace('Z', '').replace('T', ' ')
                    # Try ISO format
                    obstime = Time(clean_date, format='iso')
                else:
                    # If it's not a string, convert it
                    obstime = Time(str(obstime_str))
            except Exception:
                # Last resort: try as MJD if it's a number
                try:
                    mjd_value = float(obstime_str)
                    obstime = Time(mjd_value, format='mjd')
                except (ValueError, TypeError):
                    raise ValueError(f"Could not parse observation time '{obstime_str}' in any recognized format")
        
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

        st.write(f"Date & UTC-Time: {obstime.iso}")
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


def fwhm_fit(
    _img: np.ndarray,
    fwhm: float,
    mask: Optional[np.ndarray] = None,
    std_lo: float = 1.0,
    std_hi: float = 1.0,
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

        # Calculate relative flux in the box
        box_flux = np.sum(box_data)
        total_flux = np.sum(image_data)
        relative_flux = box_flux / total_flux if total_flux != 0 else np.nan

        return fwhm_row, fwhm_col, center_row_fit, center_col_fit, relative_flux

    try:
        peak = 0.90 * np.nanmax(_img)
        daofind = DAOStarFinder(
            fwhm=1.5 * fwhm, threshold=5 * np.std(_img), peakmax=peak
        )
        sources = daofind(_img, mask=mask)
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

        st.write(f"Sources after flux filtering: {len(filtered_sources)}")

        if len(filtered_sources) == 0:
            msg = "No valid sources for fitting found after filtering."
            st.error(msg)
            raise ValueError(msg)

        box_size = int(6 * round(fwhm))
        if box_size % 2 == 0:
            box_size += 1

        fwhm_values = []
        relative_fluxes = []
        skipped_sources = 0

        for source in filtered_sources:
            try:
                x_cen = int(source["xcentroid"])
                y_cen = int(source["ycentroid"])

                fwhm_results = compute_fwhm_marginal_sums(_img, y_cen, x_cen, box_size)
                if fwhm_results is None:
                    skipped_sources += 1
                    continue

                fwhm_row, fwhm_col, _, _, relative_flux = fwhm_results

                fwhm_source = np.mean([fwhm_row, fwhm_col])
                fwhm_values.append(fwhm_source)
                relative_fluxes.append(relative_flux)

            except Exception:
                skipped_sources += 1
                continue

        if skipped_sources > 0:
            st.write(
                f"FWHM failed for {skipped_sources} sources out of {len(filtered_sources)}."
            )

        if len(fwhm_values) == 0:
            msg = "No valid sources for FWHM fitting after marginal sums adjustment."
            st.error(msg)
            raise ValueError(msg)

        fwhm_values_arr = np.array(fwhm_values)
        relative_fluxes_arr = np.array(relative_fluxes)
        valid = ~np.isnan(fwhm_values_arr) & ~np.isinf(fwhm_values_arr) & \
                ~np.isnan(relative_fluxes_arr) & ~np.isinf(relative_fluxes_arr)
        if not np.any(valid):
            msg = "All FWHM or relative flux values are NaN or infinite after marginal sums calculation."
            st.error(msg)
            raise ValueError(msg)

        # Plot scatter of FWHM vs relative flux
        fig_fwhm, ax_fwhm = plt.subplots(figsize=FIGURE_SIZES["medium"])
        ax_fwhm.scatter(fwhm_values_arr[valid], relative_fluxes_arr[valid],
                        color='skyblue', edgecolor='black', alpha=0.7,
                        s=18)
        ax_fwhm.set_xlabel('FWHM (pixels)')
        ax_fwhm.set_ylabel('Relative Flux in Box')
        ax_fwhm.set_title('FWHM vs Relative Flux')
        ax_fwhm.grid(True, alpha=0.3)

        # Add median FWHM line in red
        median_fwhm = np.median(fwhm_values_arr[valid])
        ax_fwhm.axvline(median_fwhm, color='red', linestyle='--',
                        linewidth=2, label=f"Median FWHM = {median_fwhm:.2f}")
        ax_fwhm.legend()

        st.pyplot(fig_fwhm)
        
        # Save the FWHM scatter figure
        try:
            base_filename = st.session_state.get("base_filename", "photometry")
            username = st.session_state.get("username", "anonymous")
            output_dir = ensure_output_directory(f"{username}_rpp_results")
            fwhm_filename = f"{base_filename}_fwhm.png"
            fwhm_filepath = os.path.join(output_dir, fwhm_filename)
            
            fig_fwhm.savefig(fwhm_filepath, dpi=150, bbox_inches="tight")
            
            # Write to log if available
            log_buffer = st.session_state.get("log_buffer")
            if log_buffer is not None:
                write_to_log(log_buffer,
                             f"FWHM scatter plot saved to {fwhm_filename}")
        except Exception as e:
            st.warning(f"Error saving FWHM scatter plot: {str(e)}")

        mean_fwhm = np.median(fwhm_values_arr[valid])
        st.success(f"FWHM based on Gaussian model: {round(mean_fwhm, 2)} pixels")

        return round(mean_fwhm, 2)
    except ValueError as e:
        raise e
    except Exception as e:
        st.error(f"Unexpected error in fwhm_fit: {e}")
        raise ValueError(f"Unexpected error in fwhm_fit: {e}")


def perform_psf_photometry(
    img: np.ndarray,
    photo_table: Table,
    fwhm: float,
    daostarfind: Any,
    mask: Optional[np.ndarray] = None,
    error=None
) -> Tuple[Table, Any]:
    """
    Perform PSF (Point Spread Function) photometry using an empirically-constructed PSF model.

    This function builds an empirical PSF (EPSF) from bright stars in the image
    and then uses this model to perform PSF photometry on all detected sources.

    Parameters
    ----------
    img : numpy.ndarray
        Image with sky background subtracted
    photo_table : astropy.table.Table
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

    # Filter photo_table to select only the best stars for PSF model
    try:
        st.write("Filtering stars for PSF model construction...")
        
        # Ensure all arrays are numpy arrays and handle NaN values first
        flux = np.asarray(photo_table["flux"])
        roundness1 = np.asarray(photo_table["roundness1"]) 
        sharpness = np.asarray(photo_table["sharpness"])
        xcentroid = np.asarray(photo_table["xcentroid"])
        ycentroid = np.asarray(photo_table["ycentroid"])
        
        # Get flux statistics with NaN handling
        flux_finite = flux[np.isfinite(flux)]
        if len(flux_finite) == 0:
            raise ValueError("No sources with finite flux values found")
            
        flux_median = np.median(flux_finite)
        flux_std = np.std(flux_finite)
        
        # Define flux filtering criteria
        flux_min = flux_median - 3*flux_std
        flux_max = flux_median + 3*flux_std
        
        # Create individual boolean masks with explicit NaN handling
        valid_flux = np.isfinite(flux)
        valid_roundness = np.isfinite(roundness1)
        valid_sharpness = np.isfinite(sharpness)
        valid_xcentroid = np.isfinite(xcentroid)
        valid_ycentroid = np.isfinite(ycentroid)
        
        # Combine validity checks
        valid_all = (valid_flux & valid_roundness & valid_sharpness &
                     valid_xcentroid & valid_ycentroid)
        
        if not np.any(valid_all):
            raise ValueError("No sources with all valid parameters found")
        
        # Apply criteria only to valid sources
        flux_criteria = (flux >= flux_min) & (flux <= flux_max)
        roundness_criteria = np.abs(roundness1) < 0.25
        sharpness_criteria = np.abs(sharpness) < 0.6
        
        # Edge criteria
        edge_buffer = 2 * fwhm
        edge_criteria = (
            (xcentroid > edge_buffer) &
            (xcentroid < img.shape[1] - edge_buffer) &
            (ycentroid > edge_buffer) &
            (ycentroid < img.shape[0] - edge_buffer)
        )
        
        # Combine all criteria with validity checks
        good_stars_mask = (
            valid_all &
            flux_criteria & 
            roundness_criteria & 
            sharpness_criteria & 
            edge_criteria
        )
        
        # Ensure we have a 1D boolean array
        good_stars_mask = np.asarray(good_stars_mask, dtype=bool)
        
        # Apply filters
        filtered_photo_table = photo_table[good_stars_mask]
        
        st.write(f"Original sources: {len(photo_table)}")
        st.write(f"Filtered sources for PSF model: {len(filtered_photo_table)}")
        st.write(f"Flux range for PSF stars: {flux_min:.1f} - {flux_max:.1f}")
        
        # Check if we have enough stars for PSF construction
        if len(filtered_photo_table) < 10:
            st.warning(f"Only {len(filtered_photo_table)} stars available for PSF model. Relaxing criteria...")
            
            # Relax criteria with the same explicit approach
            roundness_criteria_relaxed = np.abs(roundness1) < 0.4
            sharpness_criteria_relaxed = np.abs(sharpness) < 1.0
            flux_criteria_relaxed = (
                (flux >= flux_median - 2*flux_std) & 
                (flux <= flux_median + 2*flux_std)
            )
            
            good_stars_mask = (
                valid_all &
                flux_criteria_relaxed & 
                roundness_criteria_relaxed & 
                sharpness_criteria_relaxed & 
                edge_criteria
            )
            
            # Ensure we have a 1D boolean array
            good_stars_mask = np.asarray(good_stars_mask, dtype=bool)
            filtered_photo_table = photo_table[good_stars_mask]
            
            st.write(f"After relaxing criteria: {len(filtered_photo_table)} stars")
            
        if len(filtered_photo_table) < 5:
            raise ValueError("Too few good stars for PSF model construction. Need at least 5 stars.")
            
    except Exception as e:
        st.error(f"Error filtering stars for PSF model: {e}")
        raise

    try:
        stars_table = Table()
        stars_table["x"] = filtered_photo_table["xcentroid"]
        stars_table["y"] = filtered_photo_table["ycentroid"]
        st.write("Star positions table prepared from filtered sources.")
    except Exception as e:
        st.error(f"Error preparing star positions table: {e}")
        raise

    try:
        fit_shape = 2 * round(fwhm) + 1
        st.write(f"Fitting shape: {fit_shape} pixels.")
    except Exception as e:
        st.error(f"Error calculating fitting shape: {e}")
        raise

    try:
        stars = extract_stars(nddata, stars_table, size=fit_shape)
        st.write(f"{len(stars)} stars extracted for PSF model.")
    except Exception as e:
        st.error(f"Error extracting stars: {e}")
        raise

    try:
        epsf_builder = EPSFBuilder(oversampling=3, maxiters=5, progress_bar=True)
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
            hdu.header["OVERSAMP"] = (3, "Oversampling factor")
            hdu.header["NSTARS"] = (len(filtered_photo_table), "Number of stars used for PSF model")

            psf_filename = (
                f"{st.session_state.get('base_filename', 'psf_model')}_psf.fits"
            )
            username = st.session_state.get("username", "anonymous")
            psf_filepath = os.path.join(
                ensure_output_directory(f"{username}_rpp_results"), psf_filename
            )

            hdu.writeto(psf_filepath, overwrite=True)
            st.write("PSF model saved as FITS file")

            norm_epsf = simple_norm(epsf.data, "log", percent=99.0)
            fig_epsf_model, ax_epsf_model = plt.subplots(
                figsize=FIGURE_SIZES["medium"], dpi=120
            )
            ax_epsf_model.imshow(
                epsf.data,
                norm=norm_epsf,
                origin="lower",
                cmap="viridis",
                interpolation="nearest",
            )
            ax_epsf_model.set_title(f"Fitted PSF Model ({len(filtered_photo_table)} stars)")
            st.pyplot(fig_epsf_model)
        except Exception as e:
            st.warning(f"Error working with PSF model: {e}")
    else:
        st.warning("EPSF data is empty or invalid. Cannot display the PSF model.")
        if not callable(daostarfind):
            raise ValueError(
                "The 'finder' parameter must be a callable star finder, such as DAOStarFinder."
            )

    # Use the original photo_table for the actual PSF photometry (not just the filtered subset)
    psfphot = IterativePSFPhotometry(
        psf_model=epsf,
        fit_shape=fit_shape,
        finder=daostarfind,
        aperture_radius=fit_shape / 2,
        maxiters=3,
        mode="new",
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

    # Use original photo_table positions for PSF photometry
    psfphot.x = photo_table["xcentroid"]
    psfphot.y = photo_table["ycentroid"]

    try:
        st.write("Performing PSF photometry on all sources...")
        phot_epsf_result = psfphot(img, mask=mask, error=error)
        st.session_state["epsf_photometry_result"] = phot_epsf_result
        st.write("PSF photometry completed successfully.")
    except Exception as e:
        st.error(f"Error executing PSF photometry: {e}")
        raise

    return phot_epsf_result, epsf


def refine_astrometry_with_stdpipe(
    image_data: np.ndarray,
    science_header: dict,
    wcs: WCS,
    fwhm_estimate: float,
    pixel_scale: float,
    filter_band: str
) -> Optional[WCS]:
    """
    Perform astrometry refinement using stdpipe and GAIA DR3 catalog.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        The 2D image array
    science_header : dict
        FITS header with WCS information
    wcs : astropy.wcs.WCS
        Initial WCS object
    fwhm_estimate : float
        FWHM estimate in pixels
    pixel_scale : float
        Pixel scale in arcseconds per pixel
    filter_band : str
        Gaia magnitude band to use for catalog matching
        
    Returns
    -------
    astropy.wcs.WCS or None
        Refined WCS object if successful, None otherwise
    """
    try:
        st.write("Doing astrometry refinement using Stdpipe and Astropy...")
        
        # Get objects using stdpipe
        obj = photometry.get_objects_sep(
            image_data,
            header=science_header,
            thresh=3.0,
            sn=5,
            aper=1.5 * fwhm_estimate,
            mask=None,
            use_mask_large=True,
            get_segmentation=False,
            subtract_bg=True
        )
        
        # Get frame center
        ra0, dec0, sr0 = astrometry.get_frame_center(
            header=science_header,
            wcs=wcs,
            width=image_data.shape[1],
            height=image_data.shape[0]
        )
        
        # Map filter band to GAIA band names
        gaia_band_mapping = {
            "phot_bp_mean_mag": "BPmag",
            "phot_rp_mean_mag": "RPmag",
            "phot_g_mean_mag": "Gmag"
        }
        
        gb = gaia_band_mapping.get(filter_band, "Gmag")
        
        # Get catalog from GAIA
        cat = catalogs.get_cat_vizier(
            ra0, dec0, sr0, "gaiaedr3",
            filters={gb: "<20"}
        )
        
        # Perform astrometry refinement
        wcs_result = pipeline.refine_astrometry(
            obj,
            cat,
            1.5 * fwhm_estimate * pixel_scale / 3600,
            wcs=wcs,
            order=2,
            cat_col_mag=gb,
            cat_col_mag_err=None,
            n_iter=3,
            min_matches=5,
            use_photometry=True,
            verbose=True,
        )
        
        # Extract WCS from result
        if isinstance(wcs_result, tuple):
            refined_wcs = wcs_result[0]
        else:
            refined_wcs = wcs_result
            
        if refined_wcs:
            st.info("Refined WCS successfully.")
            
            # Clear old WCS keywords and update with new ones
            astrometry.clear_wcs(
                science_header,
                remove_comments=True,
                remove_underscored=True,
                remove_history=True,
            )
            science_header.update(refined_wcs.to_header(relax=True))
            
            return refined_wcs
        else:
            st.warning("WCS refinement failed.")
            return None
            
    except Exception as e:
        st.warning(f"Skipping WCS refinement: {str(e)}")
        return None


def detection_and_photometry(
    image_data,
    _science_header,
    mean_fwhm_pixel,
    threshold_sigma,
    detection_mask,
    filter_band
):
    """
    Perform a complete photometry workflow on an astronomical image.

    This function executes the following steps:
      1. Estimate the sky background and its RMS.
      2. Detect sources using DAOStarFinder.
      3. Optionally refine the WCS using GAIA DR3 and stdpipe.
      4. Perform aperture photometry on detected sources.
      5. Perform PSF photometry using an empirical PSF model.

    Parameters
    ----------
    image_data : numpy.ndarray
        2D array of the image data.
    _science_header : dict or astropy.io.fits.Header
        FITS header or dictionary with image metadata.
    mean_fwhm_pixel : float
        Estimated FWHM in pixels, used for aperture and PSF sizing.
    threshold_sigma : float
        Detection threshold in sigma above background.
    detection_mask : int
        Border size in pixels to mask during detection.
    filter_band : str
        Photometric band for catalog matching and calibration.
    gb : str, optional
        Gaia band name for catalog queries (default: "Gmag").

    Returns
    -------
    tuple
        (phot_table, epsf_table, daofind, bkg), where:
        phot_table : astropy.table.Table
            Results of aperture photometry.
        epsf_table : astropy.table.Table or None
            Results of PSF photometry, or None if PSF fitting failed.
        daofind : photutils.detection.DAOStarFinder
            DAOStarFinder object used for source detection.
        bkg : photutils.background.Background2D
            Background2D object containing the background model.

    Notes
    -----
    - WCS refinement uses stdpipe and GAIA DR3 if enabled in session state.
    - Adds RA/Dec coordinates to photometry tables if WCS is available.
    - Computes instrumental magnitudes for both aperture and PSF photometry.
    - Displays progress and results in the Streamlit interface.
    """
    daofind = None

    try:
        w, wcs_error = safe_wcs_create(_science_header)
        if w is None:
            st.error(f"Error creating WCS: {wcs_error}")
            return None, None, daofind, None, None
    except Exception as e:
        st.error(f"Error creating WCS: {e}")
        return None, None, daofind, None, None

    pixel_scale = _science_header.get(
        "PIXSCALE",
        _science_header.get("PIXSIZE", _science_header.get("PIXELSCAL", 1.0)),
    )

    bkg, bkg_error = estimate_background(image_data, box_size=128, filter_size=7)
    if bkg is None:
        st.error(f"Error estimating background: {bkg_error}")
        return None, None, daofind, None, None

    mask = make_border_mask(image_data, border=detection_mask)
    image_sub = image_data - bkg.background

    bkg_error = np.full_like(image_sub, bkg.background_rms)

    exposure_time = 1.0
    if (np.max(image_data) - np.min(image_data)) > 10:
        if _science_header["EXPTIME"]:
            exposure_time = _science_header["EXPTIME"]
        elif _science_header["EXPOSURE"]:
            exposure_time = _science_header["EXPOSURE"]

    effective_gain = 2.5/np.std(image_data) * exposure_time

    total_error = calc_total_error(image_sub, bkg_error, effective_gain)

    st.write("Estimating FWHM...")
    fwhm_estimate = fwhm_fit(image_sub, mean_fwhm_pixel, mask)

    if fwhm_estimate is None:
        st.warning("Failed to estimate FWHM. Using the initial estimate.")
        fwhm_estimate = mean_fwhm_pixel

    peak_max = 0.99 * np.max(image_sub)
    daofind = DAOStarFinder(
        fwhm=1.5 * fwhm_estimate,
        threshold=threshold_sigma * np.std(image_sub),
        peakmax=peak_max,
    )

    sources = daofind(image_sub,
                      mask=mask)

    if sources is None or len(sources) == 0:
        st.warning("No sources found!")
        return None, None, daofind, bkg, None

    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))

    # Check Astrometry option before refinement
    if hasattr(st, "session_state") and st.session_state.get("astrometry_check", False):
        refined_wcs = refine_astrometry_with_stdpipe(
            image_data=image_data,
            science_header=_science_header,
            wcs=w,
            fwhm_estimate=fwhm_estimate,
            pixel_scale=pixel_scale,
            filter_band=filter_band
        )

        # Validate WCS after refinement
        if refined_wcs:
            # Test a few source positions to ensure coordinates make sense
            test_coords = positions[:min(5, len(positions))]
            if not validate_wcs_orientation(_science_header, _science_header, test_coords):
                st.warning("WCS refinement may have introduced coordinate issues")
            w = refined_wcs
    else:
        st.info("Refine Astrometry is disabled. Skipping astrometry refinement.")

    # Create multiple circular apertures with different radii
    aperture_radii = [1.5, 2.0, 2.5, 3.0]
    apertures = [CircularAperture(positions, r=radius * fwhm_estimate) 
                 for radius in aperture_radii]
    
    # Create circular annulus apertures for background estimation
    from photutils.aperture import CircularAnnulus
    annulus_apertures = []
    for radius in aperture_radii:
        r_in = 2.0 * radius * fwhm_estimate
        r_out = 2.5 * radius * fwhm_estimate
        annulus_apertures.append(CircularAnnulus(positions, r_in=r_in, r_out=r_out))

    try:
        wcs_obj = None
        if w is not None:
            try:
                # Use the refined WCS object instead of recreating from header
                wcs_obj = w
                if wcs_obj.pixel_n_dim > 2:
                    wcs_obj = wcs_obj.celestial
                    st.info("Reduced WCS to 2D celestial coordinates for photometry")
            except Exception as e:
                st.warning(f"Error creating WCS object: {e}")
                wcs_obj = None
        elif "CTYPE1" in _science_header:
            try:
                # Fallback to header WCS if no refined WCS available
                wcs_obj = WCS(_science_header)
                if wcs_obj.pixel_n_dim > 2:
                    wcs_obj = wcs_obj.celestial
                    st.info("Reduced WCS to 2D celestial coordinates for photometry")
            except Exception as e:
                st.warning(f"Error creating WCS object: {e}")
                wcs_obj = None

        # Perform photometry for all apertures
        phot_tables = []
        
        for i, (aperture, annulus) in enumerate(zip(apertures, annulus_apertures)):
            # Aperture photometry
            phot_result = aperture_photometry(
                image_sub, aperture, error=total_error, wcs=wcs_obj)
            
            # Background estimation from annulus
            bkg_result = aperture_photometry(
                image_sub, annulus, error=total_error, wcs=wcs_obj)
            
            # Add radius information to column names
            radius_suffix = f"_r{aperture_radii[i]:.1f}"
            
            # Rename aperture columns
            if "aperture_sum" in phot_result.colnames:
                phot_result.rename_column("aperture_sum", f"aperture_sum{radius_suffix}")
            if "aperture_sum_err" in phot_result.colnames:
                phot_result.rename_column("aperture_sum_err", f"aperture_sum_err{radius_suffix}")
            
            # Rename annulus columns and calculate background per pixel
            if "aperture_sum" in bkg_result.colnames:
                annulus_area = annulus.area
                bkg_per_pixel = bkg_result["aperture_sum"] / annulus_area
                aperture_area = aperture.area
                total_bkg = bkg_per_pixel * aperture_area
                
                # Store background-corrected flux
                if f"aperture_sum{radius_suffix}" in phot_result.colnames:
                    phot_result[f"aperture_sum_bkg_corr{radius_suffix}"] = (
                        phot_result[f"aperture_sum{radius_suffix}"] - total_bkg
                    )
                
                # Store background information
                phot_result[f"background{radius_suffix}"] = total_bkg
                phot_result[f"background_per_pixel{radius_suffix}"] = bkg_per_pixel
            
            phot_tables.append(phot_result)
        
        # Combine all photometry results
        phot_table = phot_tables[0]
        for i in range(1, len(phot_tables)):
            # Add columns from additional apertures
            for col in phot_tables[i].colnames:
                if col not in phot_table.colnames:
                    phot_table[col] = phot_tables[i][col]

        phot_table["xcenter"] = sources["xcentroid"]
        phot_table["ycenter"] = sources["ycentroid"]

        # Calculate SNR and magnitudes for each aperture
        for i, radius in enumerate(aperture_radii):
            radius_suffix = f"_r{radius:.1f}"
            aperture_sum_col = f"aperture_sum{radius_suffix}"
            aperture_err_col = f"aperture_sum_err{radius_suffix}"
            bkg_corr_col = f"aperture_sum_bkg_corr{radius_suffix}"
            
            if aperture_sum_col in phot_table.colnames and aperture_err_col in phot_table.colnames:
                # SNR for raw aperture sum
                phot_table[f"snr{radius_suffix}"] = np.round(
                    phot_table[aperture_sum_col] / phot_table[aperture_err_col]
                )
                m_err = 1.0857 / phot_table[f"snr{radius_suffix}"]
                phot_table[f"aperture_mag_err{radius_suffix}"] = m_err
                
                # Instrumental magnitude for raw aperture sum
                instrumental_mags = -2.5 * np.log10(phot_table[aperture_sum_col] / exposure_time)
                phot_table[f"instrumental_mag{radius_suffix}"] = instrumental_mags
                
                # If background-corrected flux is available, calculate its magnitude too
                if bkg_corr_col in phot_table.colnames:
                    # Handle negative or zero background-corrected fluxes
                    valid_flux = phot_table[bkg_corr_col] > 0
                    phot_table[f"instrumental_mag_bkg_corr{radius_suffix}"] = np.nan
                    phot_table[f"instrumental_mag_bkg_corr{radius_suffix}"][valid_flux] = (
                        -2.5 * np.log10(phot_table[bkg_corr_col][valid_flux] / exposure_time)
                    )
            else:
                phot_table[f"snr{radius_suffix}"] = np.nan
                phot_table[f"aperture_mag_err{radius_suffix}"] = np.nan
                phot_table[f"instrumental_mag{radius_suffix}"] = np.nan

        # Keep the original columns for backward compatibility (using 1.5*FWHM aperture)
        if "aperture_sum_r1.5" in phot_table.colnames:
            phot_table["aperture_sum"] = phot_table["aperture_sum_r1.5"]
            phot_table["aperture_sum_err"] = phot_table["aperture_sum_err_r1.5"]
            phot_table["snr"] = phot_table["snr_r1.5"]
            phot_table["aperture_mag_err"] = phot_table["aperture_mag_err_r1.5"]
            phot_table["instrumental_mag"] = phot_table["instrumental_mag_r1.5"]

        try:
            epsf_table, _ = perform_psf_photometry(
                image_sub, sources, fwhm_estimate, daofind, mask, total_error
            )
            
            epsf_table["snr"] = np.round(epsf_table["flux_fit"] / np.sqrt(epsf_table["flux_err"]))
            m_err = 1.0857 / epsf_table["snr"]
            epsf_table['psf_mag_err'] = m_err

            epsf_instrumental_mags = -2.5 * np.log10(
                epsf_table["flux_fit"] / exposure_time
            )
            epsf_table["instrumental_mag"] = epsf_instrumental_mags
        except Exception as e:
            st.error(f"Error performing EPSF photometry: {e}")
            epsf_table = None

        valid_sources = (phot_table["aperture_sum"] > 0) & np.isfinite(
            phot_table["instrumental_mag"]
        )
        phot_table = phot_table[valid_sources]

        if epsf_table is not None:
            epsf_valid_sources = np.isfinite(epsf_table["instrumental_mag"])
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

        st.write(f"Found {len(phot_table)} sources and performed photometry.")
        return phot_table, epsf_table, daofind, bkg, wcs_obj
    except Exception as e:
        st.error(f"Error performing aperture photometry: {e}")
        return None, None, daofind, bkg, wcs_obj


def cross_match_with_gaia(
    _phot_table,
    _science_header,
    pixel_size_arcsec,
    mean_fwhm_pixel,
    filter_band,
    filter_max_mag,
    refined_wcs=None
):
    """
    Cross-match detected sources with the GAIA DR3 star catalog.
    This function queries the GAIA catalog for a region matching the image field of view,
    applies filtering based on magnitude range, and matches GAIA sources to the detected sources.
    It also applies quality filters including variability, color index, and astrometric quality
    to ensure reliable photometric calibration stars.

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
    filter_band : str
        GAIA magnitude band to use for filtering (e.g., 'phot_g_mean_mag', 'phot_bp_mean_mag', 
        'phot_rp_mean_mag' or other synthetic photometry bands)
    filter_max_mag : float
        Maximum magnitude for GAIA source filtering

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing matched sources with both measured and GAIA catalog data,
        or None if the cross-match failed or found no matches

    Notes
    -----
    - The maximum separation for matching is set to twice the FWHM in arcseconds
    - Applies multiple quality filters to ensure reliable calibration:
      - Excludes variable stars (phot_variable_flag != "VARIABLE")
      - Limits color index range (abs(bp_rp) < 1.5)
      - Ensures good astrometric solutions (ruwe <= 1.5)
    - For non-standard Gaia bands, queries synthetic photometry from gaiadr3.synthetic_photometry_gspc
    - Progress and status updates are displayed in the Streamlit interface
    - The underscore prefix in parameter names prevents issues with caching when the function is called repeatedly
    """
    st.write("Cross-matching with Gaia DR3...")

    if _science_header is None:
        st.warning("No header information available. Cannot cross-match with Gaia.")
        return None

    try:
        # Use refined WCS if available, otherwise fallback to header WCS
        if refined_wcs is not None:
            w = refined_wcs
            st.info("Using refined WCS for Gaia cross-matching")
        else:
            w = WCS(_science_header)
            st.info("Using header WCS for Gaia cross-matching")
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
        # Validate RA/DEC coordinates before using them
        if "RA" not in _science_header or "DEC" not in _science_header:
            st.error("Missing RA/DEC coordinates in header")
            return None

        image_center_ra_dec = [_science_header["RA"],
                               _science_header["DEC"]]

        # Validate coordinate values
        if not (0 <= image_center_ra_dec[0] <= 360) or not (-90 <= image_center_ra_dec[1] <= 90):
            st.error(f"Invalid coordinates: RA={image_center_ra_dec[0]}, DEC={image_center_ra_dec[1]}")
            return None

        # Calculate search radius (divided by 1.5 to avoid field edge effects)
        gaia_search_radius_arcsec = (
            max(_science_header["NAXIS1"], _science_header["NAXIS2"])
            * pixel_size_arcsec
            / 1.5
        )
        radius_query = gaia_search_radius_arcsec * u.arcsec

        st.write(
            f"Querying Gaia in a radius of {round(radius_query.value / 60., 2)} arcmin."
        )

        # Set Gaia data release
        Gaia.MAIN_GAIA_TABLE = 'gaiadr3.gaia_source'

        # Create a SkyCoord object for more reliable coordinate handling
        center_coord = SkyCoord(ra=image_center_ra_dec[0],
                                dec=image_center_ra_dec[1], unit="deg")

        try:
            # Use the SkyCoord object for cone search
            job = Gaia.cone_search(center_coord, radius=radius_query)
            gaia_table = job.get_results()

            st.info(f"Retrieved {len(gaia_table) if gaia_table is not None else 0} sources from Gaia")

            # Different query strategies based on filter band
            if (filter_band not in ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"] and
                    gaia_table is not None and len(gaia_table) > 0):

                # Create a comma-separated list of source_ids (limit to 1000)
                max_sources = min(len(gaia_table), 1000)
                source_ids = list(gaia_table['source_id'][:max_sources])
                source_ids_str = ','.join(str(id) for id in source_ids)
                # Query synthetic photometry just for these specific sources
                synth_query = f"""
                SELECT source_id, c_star, u_jkc_mag, v_jkc_mag, b_jkc_mag,
                r_jkc_mag, i_jkc_mag, u_sdss_mag, g_sdss_mag, r_sdss_mag,
                i_sdss_mag, z_sdss_mag
                FROM gaiadr3.synthetic_photometry_gspc
                WHERE source_id IN ({source_ids_str})
                """
                synth_job = Gaia.launch_job(query=synth_query)
                synth_table = synth_job.get_results()

                # Join the two tables
                if synth_table is not None and len(synth_table) > 0:
                    st.info(f"Retrieved {len(synth_table)} synthetic photometry entries")
                    gaia_table = join(gaia_table, synth_table, keys='source_id',
                                      join_type='right')

        except Exception as cone_error:
            st.warning(f"Gaia query failed: {cone_error}")
            return None
    except KeyError as ke:
        st.error(f"Missing header keyword: {ke}")
        return None
    except Exception as e:
        st.error(f"Error querying Gaia: {e}")
        return None

    st.write(gaia_table)

    if gaia_table is None or len(gaia_table) == 0:
        st.warning("No Gaia sources found within search radius.")
        return None

    try:
        mag_filter = (gaia_table[filter_band] < filter_max_mag)
        var_filter = gaia_table["phot_variable_flag"] != "VARIABLE"
        color_index_filter = (gaia_table["bp_rp"] > -1) & (gaia_table["bp_rp"] < 2)
        astrometric_filter = gaia_table["ruwe"] < 1.6

        if synth_table is None:
            combined_filter = (mag_filter &
                        var_filter &
                        color_index_filter &
                        astrometric_filter)
        else:
            combined_filter = (mag_filter &
                        var_filter &
                        astrometric_filter)

        gaia_table_filtered = gaia_table[combined_filter]

        if len(gaia_table_filtered) == 0:
            st.warning(
                f"No Gaia sources found within magnitude range {filter_band} < {filter_max_mag}."
            )
            return None

        st.write(f"Filtered Gaia catalog to {len(gaia_table_filtered)} sources.")
    except Exception as e:
        st.error(f"Error filtering Gaia catalog: {e}")
        return None

    try:
        gaia_skycoords = SkyCoord(
            ra=gaia_table_filtered["ra"], dec=gaia_table_filtered["dec"],
            unit="deg"
        )
        idx, d2d, _ = source_positions_sky.match_to_catalog_sky(gaia_skycoords)

        max_sep_constraint = 2 * mean_fwhm_pixel * pixel_size_arcsec * u.arcsec
        gaia_matches = d2d < max_sep_constraint

        matched_indices_gaia = idx[gaia_matches]
        matched_indices_phot = np.where(gaia_matches)[0]

        if len(matched_indices_gaia) == 0:
            st.warning("No Gaia matches found within the separation constraint.")
            return None

        matched_table_qtable = _phot_table[matched_indices_phot]

        matched_table = matched_table_qtable.to_pandas()
        matched_table["gaia_index"] = matched_indices_gaia
        matched_table["gaia_separation_arcsec"] = d2d[gaia_matches].arcsec

        # Add the filter_band column from the filtered Gaia table
        matched_table[filter_band] = gaia_table_filtered[filter_band][matched_indices_gaia]

        valid_gaia_mags = np.isfinite(matched_table[filter_band])
        matched_table = matched_table[valid_gaia_mags]

        # Remove sources with SNR < 1 before zero point calculation
        if "snr" in matched_table.columns:
            matched_table = matched_table[matched_table["snr"] >= 1]

        st.success(f"Found {len(matched_table)} Gaia matches after filtering.")
        return matched_table
    except Exception as e:
        st.error(f"Error during cross-matching: {e}")
        return None


def calculate_zero_point(_phot_table, _matched_table, filter_band, air):
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
    filter_band : str
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
    if _matched_table is None or len(_matched_table) == 0:
        st.warning("No matched sources to calculate zero point.")
        return None, None, None

    try:
        valid = np.isfinite(_matched_table["instrumental_mag"]) & np.isfinite(_matched_table[filter_band])

        zero_points = _matched_table[filter_band][valid] - _matched_table["instrumental_mag"][valid]
        _matched_table["zero_point"] = zero_points
        _matched_table["zero_point_error"] = np.std(zero_points)

        clipped_zero_points = sigma_clip(zero_points, sigma=3,
                                         cenfunc="mean", masked=False)

        zero_point_value = np.median(clipped_zero_points)
        zero_point_std = np.std(clipped_zero_points)

        if np.ma.is_masked(zero_point_value) or np.isnan(zero_point_value):
            zero_point_value = float('nan')
        if np.ma.is_masked(zero_point_std) or np.isnan(zero_point_std):
            zero_point_std = float('nan')

        _matched_table["calib_mag"] = (
            _matched_table["instrumental_mag"] + zero_point_value + 0.09 * air
        )

        if not isinstance(_phot_table, pd.DataFrame):
            _phot_table = _phot_table.to_pandas()

        # Apply calibration to all aperture radii
        aperture_radii = [1.5, 2.0, 2.5, 3.0]
        
        # Remove old single-aperture columns if they exist
        old_columns = ["aperture_mag", "aperture_instrumental_mag", "aperture_mag_err"]
        for col in old_columns:
            if col in _phot_table.columns:
                _phot_table.drop(columns=[col], inplace=True)
        
        # Add calibrated magnitudes for all aperture radii
        for radius in aperture_radii:
            radius_suffix = f"_r{radius:.1f}"
            instrumental_col = f"instrumental_mag{radius_suffix}"
            aperture_mag_col = f"aperture_mag{radius_suffix}"
            
            if instrumental_col in _phot_table.columns:
                _phot_table[aperture_mag_col] = (
                    _phot_table[instrumental_col] + zero_point_value + 0.09 * air
                )

        # Also apply to matched table for all aperture radii
        for radius in aperture_radii:
            radius_suffix = f"_r{radius:.1f}"
            instrumental_col = f"instrumental_mag{radius_suffix}"
            aperture_mag_col = f"aperture_mag{radius_suffix}"
            
            if instrumental_col in _matched_table.columns:
                _matched_table[aperture_mag_col] = (
                    _matched_table[instrumental_col] + zero_point_value + 0.09 * air
                )

        # Keep the legacy "calib_mag" column using the 1.5*FWHM aperture for backward compatibility
        if "instrumental_mag_r1.5" in _phot_table.columns:
            _phot_table["calib_mag"] = (
                _phot_table["instrumental_mag_r1.5"] + zero_point_value + 0.09 * air
            )
        
        if "instrumental_mag_r1.5" in _matched_table.columns:
            _matched_table["calib_mag"] = (
                _matched_table["instrumental_mag_r1.5"] + zero_point_value + 0.09 * air
            )

        st.session_state["final_phot_table"] = _phot_table

        fig, (ax, ax_resid) = plt.subplots(1, 2, figsize=(14, 6), dpi=100)

        # Calculate residuals
        _matched_table["residual"] = (
            _matched_table[filter_band] - _matched_table["calib_mag"]
        )

        # Left plot: Zero point calibration
        # Create bins for magnitude ranges
        bin_width = 0.5  # 0.5 magnitude width bins
        min_mag = _matched_table[filter_band].min()
        max_mag = _matched_table[filter_band].max()
        bins = np.arange(np.floor(min_mag), np.ceil(max_mag) + bin_width, bin_width)

        # Group data by magnitude bins
        grouped = _matched_table.groupby(pd.cut(_matched_table[filter_band], bins))
        bin_centers = [(bin.left + bin.right) / 2 for bin in grouped.groups.keys()]
        bin_means = grouped["calib_mag"].mean().values
        bin_stds = grouped["calib_mag"].std().values

        # Plot individual points
        ax.scatter(
            _matched_table[filter_band],
            _matched_table["calib_mag"],
            alpha=0.5,
            label="Matched sources",
            color="blue",
        )

        # Plot binned means with error bars showing standard deviation
        valid_bins = ~np.isnan(bin_means) & ~np.isnan(bin_stds)
        ax.errorbar(
            np.array(bin_centers)[valid_bins],
            bin_means[valid_bins],
            yerr=bin_stds[valid_bins],
            fmt="ro-",
            label="Mean ± StdDev (binned)",
            capsize=5,
        )

        # Add a diagonal line for reference
        # Create ideal y=x reference line spanning the full range of magnitudes
        mag_range = [min(_matched_table[filter_band].min(), _matched_table["calib_mag"].min()),
                     max(_matched_table[filter_band].max(), _matched_table["calib_mag"].max())]
        ideal_mag = np.linspace(mag_range[0], mag_range[1], 100)
        ax.plot(ideal_mag, ideal_mag, "k--", alpha=0.7, label="y=x")

        ax.set_xlabel(f"Gaia {filter_band}")
        ax.set_ylabel("Calib mag")
        ax.set_title("Gaia magnitude vs Calibrated magnitude")
        ax.legend()
        ax.grid(True, alpha=0.5)

        # Right plot: Residuals
        mag_cat = _matched_table[filter_band]
        mag_inst = _matched_table["instrumental_mag"]
        zp_mean = zero_point_value
        residuals = mag_cat - (mag_inst + zp_mean)
        if "aperture_mag_err" in _matched_table.columns:
            aperture_mag_err = _matched_table["aperture_mag_err"].values
        else:
            aperture_mag_err = np.zeros_like(residuals)
        zp_err = zero_point_std if zero_point_std is not None else 0.0
        yerr = np.sqrt(aperture_mag_err**2 + zp_err**2)

        ax_resid.errorbar(mag_cat, residuals, yerr=yerr, fmt='o', markersize=5,
                          alpha=0.7, label='Residuals')
        ax_resid.axhline(0, color='gray', ls='--')
        ax_resid.set_xlabel('Calibrated magnitude')
        ax_resid.set_ylabel('Residual (catalog - calibrated)')
        ax_resid.set_title('Photometric Residuals')
        ax_resid.grid(True, alpha=0.5)
        ax_resid.legend()

        # Adjust layout and display
        fig.tight_layout()
        st.pyplot(fig)

        st.success(
            f"Calculated Zero Point: {zero_point_value:.2f} ± {zero_point_std:.2f}"
        )

        try:
            base_name = st.session_state.get("base_filename", "photometry")
            username = st.session_state.get("username", "anonymous")
            output_dir = ensure_output_directory(f"{username}_rpp_results")
            zero_point_plot_path = os.path.join(
                output_dir, f"{base_name}_zero_point_plot.png"
            )
            fig.savefig(zero_point_plot_path)
        except Exception as e:
            st.warning(f"Could not save plot to file: {e}")

        return round(zero_point_value, 2), round(zero_point_std, 2), fig
    except Exception as e:
        st.error(f"Error calculating zero point: {e}")
        return None, None, None


def enhance_catalog(
    api_key,
    final_table,
    matched_table,
    header,
    pixel_scale_arcsec,
    search_radius_arcsec=60,
):
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

    # Ensure we have valid RA/Dec coordinates in final_table
    if "ra" not in final_table.columns or "dec" not in final_table.columns:
        st.error("final_table must contain 'ra' and 'dec' columns for cross-matching")
        return final_table
    
    # Filter out sources with NaN coordinates at the beginning
    valid_coords_mask = (
        pd.notna(final_table["ra"]) & 
        pd.notna(final_table["dec"]) & 
        np.isfinite(final_table["ra"]) & 
        np.isfinite(final_table["dec"])
    )
    
    if not valid_coords_mask.any():
        st.error("No sources with valid RA/Dec coordinates found for cross-matching")
        return final_table
    
    num_invalid = len(final_table) - valid_coords_mask.sum()
    if num_invalid > 0:
        st.warning(f"Excluding {num_invalid} sources with invalid coordinates from cross-matching")

    # Compute field of view (arcmin) ONCE and use everywhere
    field_center_ra = None
    field_center_dec = None
    field_width_arcmin = 30.0  # fallback default
    if header is not None:
        if "CRVAL1" in header and "CRVAL2" in header:
            field_center_ra = float(header["CRVAL1"])
            field_center_dec = float(header["CRVAL2"])
        elif "RA" in header and "DEC" in header:
            field_center_ra = float(header["RA"])
            field_center_dec = float(header["DEC"])
        elif "OBJRA" in header and "OBJDEC" in header:
            field_center_ra = float(header["OBJRA"])
            field_center_dec = float(header["OBJDEC"])
        # Compute field width if possible
        if "NAXIS1" in header and "NAXIS2" in header and pixel_scale_arcsec:
            field_width_arcmin = (
                max(header.get("NAXIS1", 1000), header.get("NAXIS2", 1000))
                * pixel_scale_arcsec
                / 60.0
            )

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

    if field_center_ra is not None and field_center_dec is not None:
        if not (-360 <= field_center_ra <= 360) or not (-90 <= field_center_dec <= 90):
            st.warning(
                f"Invalid coordinates: RA={field_center_ra}, DEC={field_center_dec}"
            )
        else:
            pass
    else:
        st.warning("Could not extract field center coordinates from header")

    st.info("Querying Astro-Colibri API...")

    if api_key is None:
        api_key = os.environ.get("ASTROCOLIBRI_API")
        if api_key is None:
            st.warning("No API key for ASTRO-COLIBRI provided or found")
            pass

    try:
        try:
            # Base URL of the API
            url = URL + "cone_search"

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
                    base_date = datetime.fromisoformat(
                        observation_date.replace("T", " ").split(".")[0]
                    )
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
                    "position": {"ra": field_center_ra, "dec": field_center_dec},
                    "radius": field_width_arcmin / 2.0,
                },
            }

            # Perform the POST request
            response = requests.post(url, headers=headers, data=json.dumps(body))

            # Process the response
            try:
                if response.status_code == 200:
                    events = response.json()["voevents"]
                else:
                    st.warning(f"url: {url}")
                    st.warning(
                        f"Request failed with status code: {response.status_code}"
                    )

            except json.JSONDecodeError:
                st.error("Request did NOT succeed : ", response.status_code)
                st.error("Error message : ", response.content.decode("UTF-8"))

        except Exception as e:
            st.error(f"Error querying Astro-Colibri API: {str(e)}")
            # Continue with function instead of returning None

        if response is not None and response.status_code == 200:
            sources = {
                "ra": [],
                "dec": [],
                "discoverer_internal_name": [],
                "type": [],
                "classification": [],
            }

            # astrostars = pd.DataFrame(source)
            final_table["astrocolibri_name"] = None
            final_table["astrocolibri_type"] = None
            final_table["astrocolibri_classification"] = None
            for event in events:
                if "ra" in event and "dec" in event:
                    sources["ra"].append(event["ra"])
                    sources["dec"].append(event["dec"])
                    sources["discoverer_internal_name"].append(
                        event["discoverer_internal_name"]
                    )
                    sources["type"].append(event["type"])
                    sources["classification"].append(event["classification"])
            astrostars = pd.DataFrame(sources)
            st.success(f"Found {len(astrostars)} Astro-Colibri sources in field.")
            st.dataframe(astrostars)

            # Filter valid coordinates for astro-colibri matching
            valid_final_coords = final_table[valid_coords_mask]
            
            if len(valid_final_coords) > 0 and len(astrostars) > 0:
                source_coords = SkyCoord(
                    ra=valid_final_coords["ra"].values,
                    dec=valid_final_coords["dec"].values,
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
                matches = d2d < (15 * u.arcsec)

                # Map matches back to the original table indices
                valid_indices = valid_final_coords.index
                
                for i, (match, match_idx) in enumerate(zip(matches, idx)):
                    if match:
                        original_idx = valid_indices[i]
                        final_table.loc[original_idx, "astrocolibri_name"] = astrostars[
                            "discoverer_internal_name"
                        ][match_idx]
                        final_table.loc[original_idx, "astrocolibri_type"] = astrostars["type"][match_idx]
                        final_table.loc[original_idx, "astrocolibri_classification"] = astrostars[
                            "classification"][match_idx]
                
                st.success("Astro-Colibri matched objects in field.")
            else:
                st.info("No valid coordinates available for Astro-Colibri matching")
        else:
            st.write("No Astro-Colibri sources found in the field.")

    except Exception as e:
        st.error(f"Error querying Astro-Colibri: {str(e)}")
        st.write("No Astro-Colibri sources found.")

    status_text.write("Querying SIMBAD for object identifications...")

    custom_simbad = Simbad()
    custom_simbad.add_votable_fields("otype", "main_id", "ids", "B", "V")

    st.info("Querying SIMBAD")

    try:
        center_coord = SkyCoord(ra=field_center_ra, dec=field_center_dec, unit="deg")
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

                # Filter valid coordinates for SIMBAD matching
                valid_final_coords = final_table[valid_coords_mask]
                
                if len(valid_final_coords) > 0:
                    source_coords = SkyCoord(
                        ra=valid_final_coords["ra"].values,
                        dec=valid_final_coords["dec"].values,
                        unit="deg",
                    )

                    if all(col in simbad_result.colnames for col in ["ra", "dec"]):
                        try:
                            # Filter out NaN coordinates in SIMBAD result
                            simbad_valid_mask = (
                                pd.notna(simbad_result["ra"]) & 
                                pd.notna(simbad_result["dec"]) &
                                np.isfinite(simbad_result["ra"]) &
                                np.isfinite(simbad_result["dec"])
                            )
                            
                            if not simbad_valid_mask.any():
                                st.warning("No SIMBAD sources with valid coordinates found")
                            else:
                                simbad_filtered = simbad_result[simbad_valid_mask]
                                
                                simbad_coords = SkyCoord(
                                    ra=simbad_filtered["ra"],
                                    dec=simbad_filtered["dec"],
                                    unit=(u.hourangle, u.deg),
                                )

                                idx, d2d, _ = source_coords.match_to_catalog_sky(simbad_coords)
                                matches = d2d <= (10 * u.arcsec)

                                # Map matches back to the original table indices
                                valid_indices = valid_final_coords.index

                                for i, (match, match_idx) in enumerate(zip(matches, idx)):
                                    if match:
                                        original_idx = valid_indices[i]
                                        final_table.loc[original_idx, "simbad_main_id"] = simbad_filtered[
                                            "main_id"
                                        ][match_idx]
                                        final_table.loc[original_idx, "simbad_otype"] = simbad_filtered[
                                            "otype"
                                        ][match_idx]
                                        final_table.loc[original_idx, "simbad_B"] = simbad_filtered["B"][
                                            match_idx
                                        ]
                                        final_table.loc[original_idx, "simbad_V"] = simbad_filtered["V"][
                                            match_idx
                                        ]
                                        if "ids" in simbad_filtered.colnames:
                                            final_table.loc[original_idx, "simbad_ids"] = simbad_filtered[
                                                "ids"
                                            ][match_idx]

                                st.success(f"Found {sum(matches)} SIMBAD objects in field.")
                        except Exception as e:
                            st.error(
                                f"Error creating SkyCoord objects from SIMBAD data: {str(e)}"
                            )
                            st.write(f"Available SIMBAD columns: {simbad_result.colnames}")
                    else:
                        available_cols = ", ".join(simbad_result.colnames)
                        st.error(
                            f"SIMBAD result missing required columns. Available columns: {available_cols}"
                        )
                else:
                    st.info("No valid coordinates available for SIMBAD matching")
            else:
                st.write("No SIMBAD objects found in the field.")
    except Exception as e:
        st.error(f"SIMBAD query execution failed: {str(e)}")

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

                                # Filter valid coordinates for SkyBoT matching
                                valid_final_coords = final_table[valid_coords_mask]
                                
                                if len(valid_final_coords) > 0:
                                    source_coords = SkyCoord(
                                        ra=valid_final_coords["ra"].values,
                                        dec=valid_final_coords["dec"].values,
                                        unit=u.deg,
                                    )

                                    idx, d2d, _ = source_coords.match_to_catalog_sky(
                                        skybot_coords
                                    )
                                    matches = d2d <= (10 * u.arcsec)

                                    # Map matches back to the original table indices
                                    valid_indices = valid_final_coords.index

                                    for i, (match, match_idx) in enumerate(
                                        zip(matches, idx)
                                    ):
                                        if match:
                                            original_idx = valid_indices[i]
                                            obj = skybot_result["data"][match_idx]
                                            final_table.loc[original_idx, "skybot_NAME"] = obj["NAME"]
                                            final_table.loc[original_idx, "skybot_OBJECT_TYPE"] = obj[
                                                "OBJECT_TYPE"
                                            ]
                                            if "MAGV" in obj:
                                                final_table.loc[original_idx, "skybot_MAGV"] = obj[
                                                    "MAGV"
                                                ]

                                    st.success(
                                        f"Found {sum(matches)} solar system objects in field."
                                    )
                                else:
                                    st.info("No valid coordinates available for SkyBoT matching")
                            else:
                                st.warning("No solar system objects found in the field.")
                        except ValueError as e:
                            st.warning(
                                f"No solar system objects found (no valid JSON data returned). {str(e)}"
                            )
                    else:
                        st.warning("No solar system objects found in the field.")
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
                
                # Filter valid coordinates for AAVSO matching
                valid_final_coords = final_table[valid_coords_mask]
                
                if len(valid_final_coords) > 0:
                    source_coords = SkyCoord(
                        ra=valid_final_coords["ra"].values,
                        dec=valid_final_coords["dec"].values,
                        unit=u.deg,
                    )

                    idx, d2d, _ = source_coords.match_to_catalog_sky(vsx_coords)
                    matches = d2d <= (10 * u.arcsec)

                    final_table["aavso_Name"] = None
                    final_table["aavso_Type"] = None
                    final_table["aavso_Period"] = None

                    # Map matches back to the original table indices
                    valid_indices = valid_final_coords.index

                    for i, (match, match_idx) in enumerate(zip(matches, idx)):
                        if match:
                            original_idx = valid_indices[i]
                            final_table.loc[original_idx, "aavso_Name"] = vsx_table["Name"][match_idx]
                            final_table.loc[original_idx, "aavso_Type"] = vsx_table["Type"][match_idx]
                            if "Period" in vsx_table.colnames:
                                final_table.loc[original_idx, "aavso_Period"] = vsx_table["Period"][
                                    match_idx
                                ]

                    st.success(f"Found {sum(matches)} variable stars in field.")
                else:
                    st.info("No valid coordinates available for AAVSO matching")
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

                # Filter valid coordinates for QSO matching
                valid_final_coords = final_table[valid_coords_mask]
                
                if len(valid_final_coords) > 0:
                    # Create source coordinates for matching
                    source_coords = SkyCoord(
                        ra=valid_final_coords["ra"], dec=valid_final_coords["dec"], unit=u.deg
                    )

                    # Perform cross-matching
                    idx, d2d, _ = source_coords.match_to_catalog_3d(qso_coords)
                    matches = d2d.arcsec <= 10

                    # Add matched quasar information to the final table
                    final_table["qso_name"] = None
                    final_table["qso_redshift"] = None
                    final_table["qso_Rmag"] = None

                    # Initialize catalog_matches column if it doesn't exist
                    if "catalog_matches" not in final_table.columns:
                        final_table["catalog_matches"] = ""

                    # Map matches back to the original table indices
                    valid_indices = valid_final_coords.index
                    matched_sources = np.where(matches)[0]
                    matched_qsos = idx[matches]

                    for i, qso_idx in zip(matched_sources, matched_qsos):
                        original_idx = valid_indices[i]
                        final_table.loc[original_idx, "qso_name"] = qso_df.iloc[qso_idx][
                            "Name"
                        ]
                        final_table.loc[original_idx, "qso_redshift"] = qso_df.iloc[qso_idx][
                            "z"
                        ]
                        final_table.loc[original_idx, "qso_Rmag"] = qso_df.iloc[qso_idx][
                            "Rmag"
                        ]

                    # Update the catalog_matches column for matched quasars
                    has_qso = final_table["qso_name"].notna()
                    final_table.loc[has_qso, "catalog_matches"] += "QSO; "

                    st.success(
                        f"Found {sum(has_qso)} quasars in field from Milliquas catalog."
                    )
                    write_to_log(
                        st.session_state.get("log_buffer"),
                        f"Found {sum(has_qso)} quasar matches in Milliquas catalog",
                        "INFO",
                    )
                else:
                    st.info("No valid coordinates available for QSO matching")
            else:
                st.warning("No quasars found in field from Milliquas catalog.")
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
    final_table.loc[final_table["catalog_matches"] == "",
                    "catalog_matches"] = None

    matches_count = final_table["catalog_matches"].notna().sum()
    if matches_count > 0:
        st.subheader(f"Matched Objects Summary ({matches_count} sources)")
        matched_df = final_table[final_table["catalog_matches"].notna()].copy()

        display_cols = [
            "xcenter",
            "ycenter",
            "ra",
            "dec",
            "aperture_mag",
            "catalog_matches",
        ]
        display_cols = [col for col in display_cols 
                        if col in matched_df.columns]

        st.dataframe(matched_df[display_cols])

    if "match_id" in final_table.columns:
        final_table.drop("match_id", axis=1, inplace=True)

    return final_table


def validate_wcs_orientation(original_header, solved_header, test_pixel_coords):
    """
    Validate that WCS transformation preserves expected orientation
    """
    try:
        orig_wcs = WCS(original_header)
        solved_wcs = WCS(solved_header)
        
        # Test a few pixel positions
        orig_sky = orig_wcs.pixel_to_world_values(test_pixel_coords[:, 0], test_pixel_coords[:, 1])
        solved_sky = solved_wcs.pixel_to_world_values(test_pixel_coords[:, 0], test_pixel_coords[:, 1])
        
        # Check if coordinates are consistent (within reasonable tolerance)
        ra_diff = np.abs(orig_sky[0] - solved_sky[0])
        dec_diff = np.abs(orig_sky[1] - solved_sky[1])
        
        if np.any(ra_diff > 0.1) or np.any(dec_diff > 0.1):  # 0.1 degree tolerance
            st.warning("WCS orientation may have changed during plate solving")
            return False
            
        return True
    except Exception as e:
        st.warning(f"Could not validate WCS orientation: {e}")
        return True  # Assume OK if validation fails


def validate_cross_match_results(phot_table, matched_table, header):
    """
    Validate that cross-matching results make sense
    """
    if len(matched_table) == 0:
        return False
        
    # Check if matched sources are distributed across the field
    # (not clustered in one corner, which might indicate flipping)
    ra_range = matched_table["ra"].max() - matched_table["ra"].min()
    dec_range = matched_table["dec"].max() - matched_table["dec"].min()
    
    # Expect some reasonable spread for a real field
    if ra_range < 0.001 or dec_range < 0.001:  # Less than ~4 arcsec
        st.warning("Matched sources seem too clustered - possible coordinate issue")
        return False
        
    # Check separation distribution
    separations = matched_table.get("gaia_separation_arcsec", [])
    if len(separations) > 0:
        median_sep = np.median(separations)
        if median_sep > 10:  # More than 10 arcsec median separation
            st.warning(f"Large median separation ({median_sep:.1f}) suggests coordinate problems")
            return False

    return True


def get_field_center_coordinates(header):
    """
    Consistently extract field center coordinates with priority order
    """
    # Priority order for coordinate keywords
    coord_keywords = [
        ("CRVAL1", "CRVAL2"),  # Standard WCS
        ("RA", "DEC"),         # Common telescope keywords
        ("OBJRA", "OBJDEC"),   # Object coordinates
    ]
    
    for ra_key, dec_key in coord_keywords:
        if ra_key in header and dec_key in header:
            try:
                ra = float(header[ra_key])
                dec = float(header[dec_key])
                if 0 <= ra <= 360 and -90 <= dec <= 90:
                    return ra, dec
            except (ValueError, TypeError):
                continue
    
    return None, None
