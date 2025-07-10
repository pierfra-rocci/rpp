import os
import json

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
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
# from astropy.modeling import models, fitting
from photutils.background import LocalBackground, MMMBackground
from photutils.psf import fit_fwhm
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
from photutils.psf import EPSFBuilder, extract_stars, PSFPhotometry
from photutils.psf import SourceGrouper
from stdpipe import photometry, astrometry, catalogs

from src.tools import (FIGURE_SIZES, URL, safe_catalog_query,
                       safe_wcs_create, ensure_output_directory,
                       write_to_log)

from typing import Union, Any, Optional, Dict, Tuple


def solve_with_astrometrynet(file_path):
    """
    Solve astrometric plate using local Astrometry.Net installation via stdpipe.
    This function loads a FITS image, detects objects using photutils, and uses stdpipe's
    blind_match_objects wrapper around Astrometry.Net to determine accurate WCS information.

    Parameters
    ----------
    file_path : str
        Path to the FITS image file that needs astrometric solving

    Returns
    -------
    tuple
        (wcs_object, updated_header) where:
        - wcs_object: astropy.wcs.WCS object containing the WCS solution
        - updated_header: Original header updated with WCS keywords

    Notes
    -----
    This function requires:
    - Local Astrometry.Net installation with solve-field binary
    - Appropriate index files for the field scale
    - The image should contain enough stars for plate solving (typically >10)
    
    Uses stdpipe.astrometry.blind_match_objects for cleaner interface.
    """
    try:
        if not os.path.exists(file_path):
            st.error(f"File {file_path} does not exist.")
            return None, None
            
        # Load the FITS file
        st.write("Loading FITS file for local plate solving...")
        with fits.open(file_path) as hdul:
            image_data = hdul[0].data
            header = hdul[0].header.copy()
            
        if image_data is None:
            st.error("No image data found in FITS file")
            return None, None
            
        # Ensure data is float32 for better compatibility
        if image_data.dtype != np.float32:
            image_data = image_data.astype(np.float32)
            
        # Check if WCS already exists
        try:
            existing_wcs = WCS(header)
            if existing_wcs.is_celestial:
                st.info("Valid WCS already exists in header. Proceeding with blind solve anyway...")
        except Exception:
            st.info("No valid WCS found in header. Proceeding with blind solve...")
            
        # Estimate background
        st.write("Detecting objects for plate solving using photutils...")
        bkg, bkg_error = estimate_background(image_data, figure=False)
        if bkg is None:
            st.error(f"Failed to estimate background: {bkg_error}")
            return None, None
            
        image_sub = image_data - bkg.background
        
        # Helper function to try different detection parameters
        def _try_source_detection(image_sub, fwhm_estimates,
                                  threshold_multipliers, min_sources=10):
            """Helper function to try different detection parameters."""
            for fwhm_est in fwhm_estimates:
                for thresh_mult in threshold_multipliers:
                    threshold = thresh_mult * np.std(image_sub)
                    
                    try:
                        st.write(f"Trying FWHM={fwhm_est}, threshold={threshold:.1f}...")
                        daofind = DAOStarFinder(fwhm=fwhm_est, threshold=threshold)
                        temp_sources = daofind(image_sub)
                        
                        if temp_sources is not None and len(temp_sources) >= min_sources:
                            st.success(f"Photutils found {len(temp_sources)} sources with "
                                     f"FWHM={fwhm_est}, threshold={threshold:.1f}")
                            return temp_sources
                        elif temp_sources is not None:
                            st.write(f"Found {len(temp_sources)} sources (need at least {min_sources})")
                            
                    except Exception as e:
                        st.write(f"Detection failed with FWHM={fwhm_est}, threshold={threshold:.1f}: {e}")
                        continue
            return None

        # Try standard detection parameters first
        sources = _try_source_detection(
            image_sub,
            fwhm_estimates=[3.0, 4.0, 5.0],
            threshold_multipliers=[3.0, 4.0, 5.0],
            min_sources=10
        )
        
        # If that fails, try more aggressive parameters
        if sources is None:
            st.warning("Standard detection failed. Trying more aggressive parameters...")
            sources = _try_source_detection(
                image_sub,
                fwhm_estimates=[1.5, 2.0, 2.5],
                threshold_multipliers=[2.0, 2.5],
                min_sources=5
            )
        
        if sources is None or len(sources) < 5:
            st.error("Failed to detect sufficient sources for plate solving.")
            st.error("Possible solutions:")
            st.error("1. Check if the image contains stars (not empty field)")
            st.error("2. Verify image is not severely over/under-exposed")
            st.error("3. Try adjusting the detection threshold manually")
            st.error("4. Consider pre-processing the image (cosmic ray removal, etc.)")
            
            # Show a small region of the image for inspection
            try:
                center_y, center_x = image_data.shape[0] // 2, image_data.shape[1] // 2
                sample_region = image_data[center_y-50:center_y+50,
                                           center_x-50:center_x+50]
                
                fig_sample, ax_sample = plt.subplots(figsize=(6, 6))
                im = ax_sample.imshow(sample_region, origin='lower',
                                      cmap='gray')
                ax_sample.set_title('Central 100x100 pixel region')
                plt.colorbar(im, ax=ax_sample)
                st.pyplot(fig_sample)
                
            except Exception as plot_error:
                st.warning(f"Could not display sample region: {plot_error}")
            
            return None, None
        
        # Prepare sources for astrometry solving
        st.write(f"Successfully detected {len(sources)} sources")
        
        # Filter for best sources if too many
        if len(sources) > 500:
            sources.sort('flux')
            sources.reverse()
            sources = sources[:500]
            st.write(f"Using brightest {len(sources)} sources for plate solving")
        
        st.success(f"Ready for plate solving with {len(sources)} sources")
        
        # Convert sources to the format expected by stdpipe
        obj_table = Table()
        obj_table['x'] = sources['xcentroid']
        obj_table['y'] = sources['ycentroid'] 
        obj_table['flux'] = sources['flux']
        
        # Add signal-to-noise ratio if available
        if 'peak' in sources.colnames:
            # Estimate SNR from peak/background ratio
            background_std = np.std(image_sub)
            obj_table['sn'] = sources['peak'] / background_std
            obj_table['fluxerr'] = sources['flux'] / obj_table['sn']
        else:
            # Use flux as proxy for SNR
            obj_table['sn'] = sources['flux'] / np.median(sources['flux'])
            obj_table['fluxerr'] = np.sqrt(np.abs(sources['flux']))
        
        # Ensure fluxerr is positive and finite
        obj_table['fluxerr'] = np.where(
            (obj_table['fluxerr'] <= 0) | ~np.isfinite(obj_table['fluxerr']),
            sources['flux'] * 0.1,
            obj_table['fluxerr']
        )

        # Get pixel scale estimate
        pixel_scale_estimate = None
        if header:
            # Try to get pixel scale from various header keywords
            for key in ['PIXSCALE', 'PIXSIZE', 'SECPIX']:
                if key in header:
                    st.write(f"Found pixel scale in header: {key}")
                    pixel_scale_estimate = float(header[key])
                    break
            
            # If still not found, try FOCALLEN + pixel size calculation
            if pixel_scale_estimate is None:
                try:
                    focal_length = None
                    pixel_size = None
                    
                    # Check for focal length keywords
                    for focal_key in ['FOCALLEN', 'FOCAL', 'FOCLEN', 'FL']:
                        if focal_key in header:
                            focal_length = float(header[focal_key])
                            st.write(f"Found focal length: {focal_length} mm from {focal_key}")
                            break
                    
                    # Check for pixel size keywords (in microns)
                    for pixel_key in ['PIXELSIZE', 'PIXSIZE', 'XPIXSZ', 'YPIXSZ', 'PIXELMICRONS']:
                        if pixel_key in header:
                            pixel_size = float(header[pixel_key])
                            st.write(f"Found pixel size: {pixel_size} microns from {pixel_key}")
                            break
                    
                    # Calculate pixel scale using the formula from the guide scope reference
                    if focal_length is not None and pixel_size is not None:
                        pixel_scale_estimate = 206 * pixel_size / focal_length
                        
                        # Sanity check - typical astronomical pixel scales
                        if not (0.1 <= pixel_scale_estimate <= 30.0):
                            st.warning(f"Unusual pixel scale calculated: {pixel_scale_estimate:.2f} arcsec/pixel")
                            st.warning("Please verify focal length and pixel size values in header")
                        else:
                            st.success(f"Pixel scale calculated from focal length and pixel size: {pixel_scale_estimate:.2f} arcsec/pixel")
                    elif focal_length is not None:
                        st.warning(f"Found focal length ({focal_length} mm) but no pixel size in header")
                    elif pixel_size is not None:
                        st.warning(f"Found pixel size ({pixel_size} µm) but no focal length in header")
                        
                except Exception as e:
                    st.warning(f"Error calculating pixel scale from focal length/pixel size: {e}")
                    pass
            
            # If not found, try to calculate from CD matrix
            if pixel_scale_estimate is None:
                try:
                    cd_values = [header.get(f'CD{i}_{j}', 0)
                                 for i in [1, 2] for j in [1, 2]]
                    if any(x != 0 for x in cd_values):
                        cd11, cd12, cd21, cd22 = cd_values
                        det = abs(cd11 * cd22 - cd12 * cd21)
                        pixel_scale_estimate = 3600 * np.sqrt(det)
                        st.write("Calculated pixel scale from CD matrix")
                except Exception:
                    pass
 
        # Prepare parameters for stdpipe blind_match_objects
        kwargs = {
            'order': 2,  # SIP distortion order
            'update': False,  # Don't update object list in place
            'sn': 5,
            'get_header': False,  # Return WCS object, not header
            'width': image_data.shape[1],
            'height': image_data.shape[0],
            'verbose': True
        }
        
        # Add pixel scale constraints if available
        if pixel_scale_estimate and pixel_scale_estimate > 0:
            scale_low = pixel_scale_estimate * 0.8
            scale_high = pixel_scale_estimate * 1.2
            kwargs.update({
                'scale_lower': scale_low,
                'scale_upper': scale_high,
                'scale_units': 'arcsecperpix'
            })
            st.write(f"Using pixel scale estimate: {pixel_scale_estimate:.2f} arcsec/pixel")
        else:
            # Use broad scale range if no estimate available
            kwargs.update({
                'scale_lower': 0.1,
                'scale_upper': 10.0,
                'scale_units': 'arcsecperpix'
            })
            st.write("No pixel scale estimate available, using broad range")
        
        # Add RA/DEC hint if available
        if header and 'RA' in header and 'DEC' in header:
            try:
                ra_hint = float(header['RA'])
                dec_hint = float(header['DEC'])
                if 0 <= ra_hint <= 360 and -90 <= dec_hint <= 90:
                    kwargs.update({
                        'center_ra': ra_hint,
                        'center_dec': dec_hint,
                        'radius': 0.95  # 5 degree search radius
                    })
                    st.write(f"Using RA/DEC hint: {ra_hint:.3f}, {dec_hint:.3f}")
            except Exception:
                st.write("Could not parse RA/DEC from header")
        
        st.write("Running blind_match_objects...")
        st.write(f"Using {len(obj_table)} sources for plate solving")

        try:
            # Call stdpipe's blind_match_objects function
            solved_wcs = astrometry.blind_match_objects(obj_table, **kwargs)
            
            if solved_wcs is not None:
                st.success("Plate solving successful!")
                
                # Update original header with WCS solution
                updated_header = header.copy()
                
                # Clear any existing WCS keywords to avoid conflicts
                astrometry.clear_wcs(updated_header, remove_comments=True,
                                     remove_underscored=True,
                                     remove_history=True)
                
                # Add new WCS keywords from solution
                wcs_header = solved_wcs.to_header(relax=True)
                updated_header.update(wcs_header)
                
                # Add solution metadata
                updated_header['COMMENT'] = 'WCS solution from stdpipe/astrometry.net'
                updated_header['STDPIPE'] = (True, 'Solved with stdpipe.astrometry.blind_match_objects')
                
                # Display solution information
                try:
                    center_ra, center_dec = solved_wcs.pixel_to_world_values(
                        image_data.shape[1]/2, image_data.shape[0]/2
                    )
                    pixel_scale = astrometry.get_pixscale(wcs=solved_wcs) * 3600
                    
                    st.write(f"Solution center: RA={center_ra:.6f}°, DEC={center_dec:.6f}°")
                    st.write(f"Pixel scale: {pixel_scale:.3f} arcsec/pixel")
                    
                    # Validate solution makes sense
                    if 0 <= center_ra <= 360 and -90 <= center_dec <= 90:
                        st.success("Solution coordinates are valid")
                    else:
                        st.warning("Solution coordinates seem invalid!")
                        
                except Exception as coord_error:
                    st.warning(f"Could not extract solution coordinates: {coord_error}")
                
                return solved_wcs, updated_header
                
            else:
                st.error("Plate solving failed!")
                st.error("Possible issues:")
                st.error("- Not enough stars detected")
                st.error("- Incorrect pixel scale estimate")
                st.error("- Field not in astrometry.net index files")
                st.error("- solve-field not found in PATH")
                
                return None, None
                
        except Exception as solve_error:
            st.error(f"Error during stdpipe plate solving: {solve_error}")
            st.error("Requirements:")
            st.error("- solve-field binary must be in PATH")
            st.error("- Appropriate astrometry.net index files must be installed")
            st.error("- Sufficient sources with good S/N ratio")
            
            return None, None
        
    except Exception as e:
        st.error(f"Error in plate solving setup: {str(e)}")
        return None, None


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
            top = bottom = border[0]
            left = right = border[2]
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


def estimate_background(image_data, box_size=100, filter_size=5, figure=True):
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
    adjusted_box_size = max(box_size, min(height // 10, width // 10, 128))
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
        if figure:
            fig_bkg = None
            try:
                # Create a figure with two subplots side by side for background/RMS
                fig_bkg, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
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
                base_filename = st.session_state.get("base_filename",
                                                     "photometry")
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
            finally:
                # Clean up matplotlib figure to prevent memory leaks
                if fig_bkg is not None:
                    plt.close(fig_bkg)

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
    # def compute_fwhm_marginal_sums(image_data, center_row, center_col, box_size):
    #     half_box = box_size // 2

    #     if box_size < 5:
    #         return None, None

    #     row_start = center_row - half_box
    #     row_end = center_row + half_box + 1
    #     col_start = center_col - half_box
    #     col_end = center_col + half_box + 1

    #     if (
    #         row_start < 0
    #         or row_end > image_data.shape[0]
    #         or col_start < 0
    #         or col_end > image_data.shape[1]
    #     ):
    #         return None, None

    #     box_data = image_data[row_start:row_end, col_start:col_end]

    #     if box_data.shape[0] < 5 or box_data.shape[1] < 5:
    #         return None

    #     sum_rows = np.sum(box_data, axis=1)
    #     sum_cols = np.sum(box_data, axis=0)

    #     if np.max(sum_rows) < 5 * np.median(sum_rows) or np.max(
    #         sum_cols
    #     ) < 5 * np.median(sum_cols):
    #         return None, None

    #     row_indices = np.arange(box_data.shape[0])
    #     col_indices = np.arange(box_data.shape[1])

    #     fitter = fitting.LevMarLSQFitter()

    #     try:
    #         row_max_idx = np.argmax(sum_rows)
    #         row_max_val = sum_rows[row_max_idx]

    #         model_row = models.Gaussian1D(
    #             amplitude=row_max_val, mean=row_max_idx, stddev=box_size / 6
    #         )

    #         fitted_row = fitter(model_row, row_indices, sum_rows)
    #         center_row_fit = fitted_row.mean.value + row_start
    #         fwhm_row = 2 * np.sqrt(2 * np.log(2)) * fitted_row.stddev.value
    #     except Exception:
    #         return None, None

    #     try:
    #         col_max_idx = np.argmax(sum_cols)
    #         col_max_val = sum_cols[col_max_idx]

    #         model_col = models.Gaussian1D(
    #             amplitude=col_max_val, mean=col_max_idx, stddev=box_size / 6
    #         )

    #         fitted_col = fitter(model_col, col_indices, sum_cols)
    #         center_col_fit = fitted_col.mean.value + col_start
    #         fwhm_col = 2 * np.sqrt(2 * np.log(2)) * fitted_col.stddev.value
    #     except Exception:
    #         return None, None

    #     # Calculate relative flux in the box
    #     box_flux = np.sum(box_data)
    #     total_flux = np.sum(image_data)
    #     relative_flux = box_flux / total_flux if total_flux != 0 else np.nan

    #     return fwhm_row, fwhm_col, center_row_fit, center_col_fit, relative_flux

    try:
        peak = 0.95 * np.nanmax(_img)
        # Show what happens if we exclude bright pixels
        _, _, clipped_std = sigma_clipped_stats(_img, sigma=3.0)

        daofind = DAOStarFinder(
            fwhm=1.6 * fwhm, threshold=7 * clipped_std, peakmax=peak
        )
        sources = daofind(_img, mask=mask)
        if sources is None:
            st.warning("No sources found !")
            return None, None

        flux = sources["flux"]
        median_flux = np.mean(flux)
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
        
        xypos = list(zip(filtered_sources['x_center'],
                         filtered_sources['y_center']))
        fwhms = fit_fwhm(_img, xypos=xypos, fit_shape=box_size)

        # fwhm_values = []
        # relative_fluxes = []
        # skipped_sources = 0

        # for source in filtered_sources:
        #     try:
        #         x_cen = int(source["xcentroid"])
        #         y_cen = int(source["ycentroid"])

        #         fwhm_results = compute_fwhm_marginal_sums(_img, y_cen, x_cen, box_size)
        #         if fwhm_results is None:
        #             skipped_sources += 1
        #             continue

        #         fwhm_row, fwhm_col, _, _, relative_flux = fwhm_results

        #         fwhm_source = np.mean([fwhm_row, fwhm_col])
        #         fwhm_values.append(fwhm_source)
        #         relative_fluxes.append(relative_flux)

        #     except Exception:
        #         skipped_sources += 1
        #         continue

        # if skipped_sources > 0:
        #     st.write(
        #         f"FWHM failed for {skipped_sources} sources out of {len(filtered_sources)}."
        #     )

        # if len(fwhm_values) == 0:
        #     msg = "No valid sources for FWHM fitting after marginal sums adjustment."
        #     st.error(msg)
        #     raise ValueError(msg)

        # fwhm_values_arr = np.array(fwhm_values)
        # relative_fluxes_arr = np.array(relative_fluxes)
        # valid = ~np.isnan(fwhm_values_arr) & ~np.isinf(fwhm_values_arr) & \
        #         ~np.isnan(relative_fluxes_arr) & ~np.isinf(relative_fluxes_arr)
        # if not np.any(valid):
        #     msg = "All FWHM or relative flux values are NaN or infinite after marginal sums calculation."
        #     st.error(msg)
        #     raise ValueError(msg)

        # Plot histogram of FWHM
        fig_fwhm, ax_fwhm = plt.subplots(figsize=FIGURE_SIZES["medium"])
        ax_fwhm.hist(fwhms, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax_fwhm.set_xlabel('FWHM (pixels)')
        ax_fwhm.set_ylabel('Number of sources')
        ax_fwhm.set_title('FWHM Distribution')
        ax_fwhm.grid(True, alpha=0.3)

        # Add median FWHM line in red
        median_fwhm = np.median(fwhms)
        ax_fwhm.axvline(median_fwhm, color='red', linestyle='--',
                        linewidth=2,
                        label=f"Median FWHM = {median_fwhm:.2f}")
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

        mean_fwhm = np.median(fwhms)
        st.success(f"FWHM based on Gaussian model: {round(mean_fwhm, 2)} pixels")

        return round(mean_fwhm, 2), clipped_std
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
    error : numpy.ndarray, optional
        Error array for the image

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
    # Initial validation
    try:
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

        if not callable(daostarfind):
            raise ValueError(
                "The 'finder' parameter must be a callable star finder, such as DAOStarFinder."
            )

        nddata = NDData(data=img)
    except Exception as e:
        st.error(f"Error in initial validation: {e}")
        raise

    # Filter photo_table to select only the best stars for PSF model
    try:
        st.write("Filtering stars for PSF model construction...")
        
        # Ensure all arrays are numpy arrays and handle NaN values first
        flux = np.asarray(photo_table["flux"])
        roundness2 = np.asarray(photo_table["roundness2"])
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
        valid_roundness = np.isfinite(roundness2)
        valid_sharpness = np.isfinite(sharpness)
        valid_xcentroid = np.isfinite(xcentroid)
        valid_ycentroid = np.isfinite(ycentroid)
        
        # Combine validity checks
        valid_all = (valid_flux & valid_roundness & valid_sharpness &
                     valid_xcentroid & valid_ycentroid)
        
        if not np.any(valid_all):
            raise ValueError("No sources with all valid parameters found")

        flux_criteria = np.zeros_like(valid_flux, dtype=bool)
        flux_criteria[valid_flux] = (flux[valid_flux] >= flux_min) & (flux[valid_flux] <= flux_max)
        
        roundness_criteria = np.zeros_like(valid_roundness, dtype=bool)
        roundness_criteria[valid_roundness] = np.abs(roundness2[valid_roundness]) < 0.25
        
        sharpness_criteria = np.zeros_like(valid_sharpness, dtype=bool)
        sharpness_criteria[valid_sharpness] = np.abs(sharpness[valid_sharpness]) > 0.5
        
        # Edge criteria
        edge_criteria = np.zeros_like(valid_xcentroid, dtype=bool)
        valid_coords = valid_xcentroid & valid_ycentroid
        edge_criteria[valid_coords] = (
            (xcentroid[valid_coords] > 2 * fwhm) &
            (xcentroid[valid_coords] < img.shape[1] - 2 * fwhm) &
            (ycentroid[valid_coords] > 2 * fwhm) &
            (ycentroid[valid_coords] < img.shape[0] - 2 * fwhm)
        )
        
        # Combine all criteria with validity checks
        good_stars_mask = (
            valid_all &
            flux_criteria &
            roundness_criteria &
            sharpness_criteria &
            edge_criteria
        )
        
        # Apply filters
        filtered_photo_table = photo_table[good_stars_mask]
        
        st.write(f"Original sources: {len(photo_table)}")
        st.write(f"Filtered sources for PSF model: {len(filtered_photo_table)}")
        st.write(f"Flux range for PSF stars: {flux_min:.1f} - {flux_max:.1f}")
        
        # Check if we have enough stars for PSF construction
        if len(filtered_photo_table) < 10:
            st.warning(f"Only {len(filtered_photo_table)} stars available for PSF model. Relaxing criteria...")
            
            # FIXED: Relax criteria using the same approach
            roundness_criteria_relaxed = np.zeros_like(valid_roundness, dtype=bool)
            roundness_criteria_relaxed[valid_roundness] = np.abs(roundness1[valid_roundness]) < 0.5
            
            sharpness_criteria_relaxed = np.zeros_like(valid_sharpness, dtype=bool)
            sharpness_criteria_relaxed[valid_sharpness] = np.abs(sharpness[valid_sharpness]) < 1.0
            
            flux_criteria_relaxed = np.zeros_like(valid_flux, dtype=bool)
            flux_criteria_relaxed[valid_flux] = (
                (flux[valid_flux] >= flux_median - 2*flux_std) &
                (flux[valid_flux] <= flux_median + 2*flux_std)
            )
            
            good_stars_mask = (
                valid_all &
                flux_criteria_relaxed &
                roundness_criteria_relaxed &
                sharpness_criteria_relaxed &
                edge_criteria
            )
            
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

        # FIX: If extract_stars returns a list, convert to EPSFStars
        from photutils.psf import EPSFStars
        if isinstance(stars, list):
            if len(stars) == 0:
                raise ValueError("No stars extracted for PSF model. Check your selection criteria.")
            stars = EPSFStars(stars)
        n_stars = len(stars)
        st.write(f"{n_stars} stars extracted for PSF model.")

        if hasattr(stars, 'data') and stars.data is not None:

            # For EPSFStars, data is a list of individual star arrays
            if isinstance(stars.data, list) and len(stars.data) > 0:
                st.write(f"Number of star cutouts: {len(stars.data)}")
                # Check for NaN in all star data
                has_nan = any(np.isnan(star_data).any() for star_data in stars.data)
                st.write(f"NaN in star data: {has_nan}")
            else:
                st.write("Stars data is empty or not a list")
        else:
            st.write("Stars object has no data attribute")
            
        if n_stars == 0:
            raise ValueError("No stars extracted for PSF model. Check your selection criteria.")
    except Exception as e:
        st.error(f"Error extracting stars: {e}")
        raise

    try:
        # Remove stars with NaN or all-zero data
        mask_valid = []
        if hasattr(stars, 'data') and isinstance(stars.data, list):
            for star_data in stars.data:
                if np.isnan(star_data).any():
                    mask_valid.append(False)
                elif np.all(star_data == 0):
                    mask_valid.append(False)
                else:
                    mask_valid.append(True)
        else:
            try:
                for i in range(len(stars)):
                    star = stars[i]
                    if np.isnan(star.data).any():
                        mask_valid.append(False)
                    elif np.all(star.data == 0):
                        mask_valid.append(False)
                    else:
                        mask_valid.append(True)
            except Exception as iter_error:
                st.warning(f"Could not iterate through stars for validation: {iter_error}")
                # If we can't validate, assume all stars are valid
                mask_valid = [True] * len(stars)

        mask_valid = np.array(mask_valid)

        # Only filter if we have any invalid stars
        if not np.all(mask_valid):
            # Always wrap the result as EPSFStars
            from photutils.psf import EPSFStars
            filtered_stars = stars[mask_valid]
            # If filtered_stars is not EPSFStars, wrap it
            if not isinstance(filtered_stars, EPSFStars):
                filtered_stars = EPSFStars(list(filtered_stars))
            stars = filtered_stars
            n_stars = len(stars)
            st.write(f"{n_stars} valid stars remain for PSF model after filtering invalid data.")
        else:
            st.write(f"All {len(stars)} stars are valid for PSF model.")

        if len(stars) == 0:
            raise ValueError("No valid stars for PSF model after filtering.")
    except Exception as e:
        st.error(f"Error filtering stars for PSF model: {e}")
        raise

    try:
        epsf_builder = EPSFBuilder(oversampling=3, maxiters=3,
                                   progress_bar=False)
        # Try the EPSF building with error handling
        try:
            epsf, fitted_stars = epsf_builder(stars)
        except Exception as build_error:
            st.error(f"EPSFBuilder failed: {build_error}")
            # Try with even more conservative parameters
            st.write("Retrying with more conservative EPSFBuilder parameters...")
            try:
                epsf_builder_conservative = EPSFBuilder(
                    oversampling=2,  # Lower oversampling
                    maxiters=2,      # Fewer iterations
                    progress_bar=False,
                    smoothing_kernel='quartic',
                    recentering_maxiters=3,
                    recentering_boxsize=3
                )
                epsf, fitted_stars = epsf_builder_conservative(stars)
            except Exception as conservative_error:
                st.error(f"Conservative EPSFBuilder also failed: {conservative_error}")
                raise
        
        st.write("EPSF building completed successfully")

        if epsf is None:
            raise ValueError("EPSFBuilder returned None")
            
        if not hasattr(epsf, 'data'):
            raise ValueError("EPSF has no data attribute")
            
        if epsf.data is None:
            raise ValueError("EPSF data is None")
            
        if epsf.data.size == 0:
            raise ValueError("EPSF data is empty")
        
        try:
            # Ensure epsf.data is a numpy array before checking for NaN
            epsf_array = np.asarray(epsf.data)
            if epsf_array.size > 0 and np.isnan(epsf_array).any():
                st.warning("EPSF data contains NaN values")
        except Exception as nan_check_error:
            st.warning(f"Could not check for NaN values in EPSF data: {nan_check_error}")
            
        st.write(f"Shape of epsf.data: {epsf.data.shape}")
        
        # FIXED: More robust NaN reporting
        try:
            epsf_array = np.asarray(epsf.data)
            has_nan = np.isnan(epsf_array).any() if epsf_array.size > 0 else False
            st.write(f"NaN in epsf.data: {has_nan}")
        except Exception as nan_report_error:
            st.write(f"Could not check NaN status: {nan_report_error}")
        
        st.session_state["epsf_model"] = epsf
    except Exception as e:
        st.error(f"Error fitting PSF model: {e}")
        raise

    # Check for valid PSF
    if epsf is not None and hasattr(epsf, 'data') and epsf.data is not None and epsf.data.size > 0:
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
        raise ValueError("EPSF model creation failed - no valid PSF data")

    # Create PSF photometry object and perform photometry
    try:
        if error is not None and not isinstance(error, np.ndarray):
            st.warning("Invalid error array provided, proceeding without error estimation")
            error = None
        elif error is not None and error.shape != img.shape:
            st.warning("Error array shape mismatch, proceeding without error estimation")
            error = None

        # Create a SourceGrouper with a minimum separation distance
        # This groups sources that are closer than min_separation pixels
        min_separation = 1.9 * fwhm  # Adjust based on your FWHM
        grouper = SourceGrouper(min_separation=min_separation)
        bkgstat = MMMBackground()
        localbkg_estimator = LocalBackground(2.1*fwhm, 2.5*fwhm, bkgstat)

        psfphot = PSFPhotometry(
            psf_model=epsf,
            fit_shape=fit_shape,
            finder=daostarfind,
            aperture_radius=fit_shape / 2,
            grouper=grouper,
            localbkg_estimator=localbkg_estimator,
            progress_bar=False
        )

        # FIXED: Filter out sources that are in masked regions
        # Create initial guess table with correct parameter name
        initial_params = Table()
        initial_params['x_0'] = photo_table["xcentroid"]
        initial_params['y_0'] = photo_table["ycentroid"]
        
        # Filter out sources that fall in masked regions
        if mask is not None:
            # Convert coordinates to integers for mask indexing
            x_int = np.round(initial_params['x_0']).astype(int)
            y_int = np.round(initial_params['y_0']).astype(int)
            
            # Check bounds
            valid_bounds = (
                (x_int >= 0) & (x_int < img.shape[1]) &
                (y_int >= 0) & (y_int < img.shape[0])
            )
            
            # Check if sources are not in masked regions
            # mask=True means masked (bad) pixels, so we want mask=False
            valid_mask = np.ones(len(initial_params), dtype=bool)
            valid_mask[valid_bounds] = ~mask[y_int[valid_bounds], x_int[valid_bounds]]
            
            # Filter the initial parameters
            initial_params_filtered = initial_params[valid_mask]
            
            st.write(f"Filtered out {len(initial_params) - len(initial_params_filtered)} "
                    f"sources that fall in masked regions")
            st.write(f"Proceeding with {len(initial_params_filtered)} sources for PSF photometry")
            
            if len(initial_params_filtered) == 0:
                st.error("All sources fall in masked regions. Cannot perform PSF photometry.")
                return None, epsf
                
            initial_params = initial_params_filtered
        else:
            st.write("No mask provided, using all sources for PSF photometry")

        st.write("Performing PSF photometry on all sources...")
        phot_epsf_result = psfphot(img, init_params=initial_params, mask=mask, error=error)
        st.session_state["epsf_photometry_result"] = phot_epsf_result
        st.write("PSF photometry completed successfully.")
        
        return phot_epsf_result, epsf
        
    except Exception as e:
        st.error(f"Error executing PSF photometry: {e}")
        raise


def refine_astrometry_with_stdpipe(
    image_data: np.ndarray,
    science_header: dict,
    wcs: WCS,
    fwhm_estimate: float,
    pixel_scale: float,
    filter_band: str
) -> Optional[WCS]:
    """
    Perform astrometry refinement using stdpipe SCAMP and GAIA DR3 catalog.
    
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
        st.write("Doing astrometry refinement using SCAMP...")

        # Convert image data to float32 for stdpipe compatibility
        if image_data.dtype not in [np.float32, np.float64]:
            st.info(f"Converting image from {image_data.dtype} to float32 for stdpipe compatibility")
            image_data = image_data.astype(np.float32)

        # Clean and prepare header - remove problematic WCS distortion parameters
        clean_header = science_header.copy()
        
        # Remove problematic keywords that might interfere with stdpipe
        problematic_keys = ['HISTORY', 'COMMENT', 'CONTINUE']
        
        # Remove DSS distortion parameters that cause the "coefficient scale is zero" error
        dss_distortion_keys = [key for key in clean_header.keys() if any(pattern in str(key) for pattern in ['DSS', 'CNPIX', 'A_', 'B_', 'AP_', 'AMD',
                                                                                                             'BP_', 'B_', 'PV', 'SIP', 'DISTORT'])]
        
        if dss_distortion_keys:
            st.info(f"Removing {len(dss_distortion_keys)} problematic distortion parameters")
            for key in dss_distortion_keys:
                if key in clean_header:
                    del clean_header[key]
        
        # Remove other problematic keys
        for key in problematic_keys:
            if key in clean_header:
                del clean_header[key]
        
        # ADDED: Remove CDELTM1 and CDELTM2 keys that can cause issues with stdpipe
        cdeltm_keys = ['CDELTM1', 'CDELTM2']
        for key in cdeltm_keys:
            if key in clean_header:
                del clean_header[key]
                st.info(f"Removed problematic WCS key: {key}")

        # Validate and fix basic WCS parameters
        try:
            # Ensure CTYPE values are valid
            if 'CTYPE1' not in clean_header or not clean_header['CTYPE1']:
                clean_header['CTYPE1'] = 'RA---TAN'
            if 'CTYPE2' not in clean_header or not clean_header['CTYPE2']:
                clean_header['CTYPE2'] = 'DEC--TAN'
            
            # Ensure reference pixel is valid
            if 'CRPIX1' not in clean_header or not np.isfinite(clean_header['CRPIX1']):
                clean_header['CRPIX1'] = image_data.shape[1] / 2.0
            if 'CRPIX2' not in clean_header or not np.isfinite(clean_header['CRPIX2']):
                clean_header['CRPIX2'] = image_data.shape[0] / 2.0
            
            # Check CD matrix validity
            cd_keys = ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
            cd_values = [clean_header.get(key, 0) for key in cd_keys]
            
            if any(not np.isfinite(val) or val == 0 for val in cd_values):
                st.warning("Invalid CD matrix detected, attempting reconstruction")
                # Try to reconstruct from CDELT if available
                if 'CDELT1' in clean_header and 'CDELT2' in clean_header:
                    cdelt1 = clean_header['CDELT1']
                    cdelt2 = clean_header['CDELT2']
                    crota = clean_header.get('CROTA2', 0.0)
                    
                    cos_rot = np.cos(np.radians(crota))
                    sin_rot = np.sin(np.radians(crota))
                    
                    clean_header['CD1_1'] = cdelt1 * cos_rot
                    clean_header['CD1_2'] = -cdelt1 * sin_rot
                    clean_header['CD2_1'] = cdelt2 * sin_rot
                    clean_header['CD2_2'] = cdelt2 * cos_rot
                    
                    st.info("Reconstructed CD matrix from CDELT/CROTA")
                else:
                    # Use pixel scale estimate to create basic CD matrix
                    pixel_scale_deg = pixel_scale / 3600.0
                    clean_header['CD1_1'] = -pixel_scale_deg
                    clean_header['CD1_2'] = 0.0
                    clean_header['CD2_1'] = 0.0
                    clean_header['CD2_2'] = pixel_scale_deg
                    st.info("Created basic CD matrix from pixel scale estimate")
            
        except Exception as header_fix_error:
            st.warning(f"Error fixing header: {header_fix_error}")
        
        # Test the cleaned WCS before proceeding
        try:
            test_wcs = WCS(clean_header)
            # Try a test transformation
            test_x, test_y = image_data.shape[1] // 2, image_data.shape[0] // 2
            test_coords = test_wcs.pixel_to_world_values(test_x, test_y)
            st.info("Cleaned WCS passes basic validation")
        except Exception as wcs_test_error:
            st.error(f"Cleaned WCS still has issues: {wcs_test_error}")
            return None

        # Get objects using stdpipe with corrected parameters
        try:
            # Use correct parameter names based on stdpipe documentation
            obj = photometry.get_objects_sep(
                image_data,
                header=clean_header,
                thresh=3,
                minarea=7,
                aper=1.5 * fwhm_estimate,
                use_fwhm=True,
                use_mask_large=True,
                subtract_bg=False,
                verbose=False
            )
            
            if obj is None or len(obj) == 0:
                raise ValueError("No objects detected")
                
        except Exception as obj_error:
            # Try with more conservative parameters
            st.warning(f"Standard detection failed: {obj_error}. Trying with relaxed parameters...")
            try:
                obj = photometry.get_objects_sep(
                    image_data,
                    header=clean_header,
                    thresh=1.5,
                    minarea=7,
                    aper=1.5 * fwhm_estimate,
                    use_fwhm=True,
                    use_mask_large=True,
                    subtract_bg=False,
                    verbose=False
                )
                
                if obj is None or len(obj) == 0:
                    raise ValueError("No objects detected with relaxed parameters")
                    
            except Exception as final_error:
                st.error(f"Failed to extract objects: {final_error}")
                st.error("Try adjusting FWHM estimate or check image quality")
                return None

        st.info(f"Detected {len(obj)} objects for astrometry refinement")

        # Get frame center using the cleaned header and test WCS
        try:
            # Use the cleaned header instead of original
            center_ra, center_dec, radius = astrometry.get_frame_center(
                header=clean_header,
                wcs=test_wcs,  # Use the validated test WCS
                width=image_data.shape[1],
                height=image_data.shape[0]
            )
        except Exception as center_error:
            st.error(f"Failed to get frame center: {center_error}")
            # Try fallback method using header coordinates
            try:
                center_ra = clean_header.get('CRVAL1') or clean_header.get('RA') or clean_header.get('OBJRA')
                center_dec = clean_header.get('CRVAL2') or clean_header.get('DEC') or clean_header.get('OBJDEC')
                
                if center_ra is None or center_dec is None:
                    st.error("Could not determine field center coordinates")
                    return None
                
                # Calculate radius from image dimensions
                radius = max(image_data.shape) * pixel_scale / 3600.0 / 2.0
                st.info(f"Using fallback field center: RA={center_ra:.3f}, DEC={center_dec:.3f}, radius={radius:.3f}°")
                
            except Exception as fallback_error:
                st.error(f"Fallback coordinate extraction failed: {fallback_error}")
                return None
        
        # Map filter band to correct GAIA EDR3 column names
        gaia_band_mapping = {
            "phot_bp_mean_mag": "BPmag",
            "phot_rp_mean_mag": "RPmag",
            "phot_g_mean_mag": "Gmag"
        }
        
        gaia_band = gaia_band_mapping.get(filter_band, "Gmag")
        
        # Get GAIA catalog with correct parameters and error handling
        try:
            # Use correct catalog name and filters
            cat = catalogs.get_cat_vizier(
                center_ra,
                center_dec,
                radius,
                "I/350/gaiaedr3",  # Correct GAIA EDR3 catalog identifier
                filters={gaia_band: "<20.0"}
            )
            
            if cat is None or len(cat) == 0:
                st.warning("No GAIA catalog sources found in field")
                return None
                
        except Exception as cat_error:
            st.error(f"Failed to get GAIA catalog: {cat_error}")
            # Try with a smaller search radius
            try:
                smaller_radius = min(radius, 0.5)  # Limit to 0.5 degrees
                st.info(f"Retrying GAIA query with smaller radius: {smaller_radius:.3f}°")
                cat = catalogs.get_cat_vizier(
                    center_ra,
                    center_dec,
                    smaller_radius,
                    "I/350/gaiaedr3",
                    filters={gaia_band: "<19.0"}  # Also reduce magnitude limit
                )
                
                if cat is None or len(cat) == 0:
                    st.warning("No GAIA catalog sources found even with reduced search radius")
                    return None
                    
            except Exception as retry_error:
                st.error(f"GAIA catalog query retry failed: {retry_error}")
                return None
        
        st.info(f"Retrieved {len(cat)} GAIA catalog sources")
        
        # Apply additional filtering after retrieval if needed
        try:
            # Filter out sources with poor parallax measurements if the column exists
            if 'parallax' in cat.colnames:
                # Keep sources with reasonable parallax values (not extreme negative values)
                parallax_filter = cat['parallax'] > -100  # Basic parallax filter
                cat = cat[parallax_filter]
                st.info(f"After parallax filtering: {len(cat)} GAIA catalog sources")
        except Exception as filter_error:
            st.warning(f"Could not apply additional filtering: {filter_error}")
        
        # Ensure we still have enough sources for refinement
        if len(cat) < 10:
            st.warning(f"Too few GAIA sources ({len(cat)}) for reliable astrometry refinement")
            return None
        
        # Calculate matching radius in degrees
        try:
            match_radius_deg = 2.0 * fwhm_estimate * pixel_scale / 3600.0
            
            # Try SCAMP refinement with conservative parameters
            wcs_result = astrometry.refine_wcs_scamp(
                obj,
                cat,
                wcs=test_wcs,  # Use the validated test WCS
                sr=match_radius_deg,
                order=2,
                cat_col_ra='RA_ICRS',
                cat_col_dec='DE_ICRS',
                cat_col_mag=gaia_band,
                cat_mag_lim=19,  # Conservative magnitude limit
                verbose=True
            )
            
        except Exception as refine_error:
            st.warning(f"SCAMP astrometry refinement failed: {refine_error}")
            # Try with even more conservative parameters
            try:
                st.info("Retrying with ultra-conservative SCAMP parameters...")
                wcs_result = astrometry.refine_wcs_scamp(
                    obj,
                    cat,
                    wcs=test_wcs,
                    sr=match_radius_deg * 2.0,  # Double the search radius
                    order=1,
                    cat_col_ra='RA_ICRS',
                    cat_col_dec='DE_ICRS',
                    cat_col_mag=gaia_band,
                    cat_mag_lim=19.,
                    verbose=True
                )
            except Exception as final_refine_error:
                st.error(f"Final SCAMP attempt failed: {final_refine_error}")
                return None
        
        # Validate and return the refined WCS
        if wcs_result is not None:
            st.success("WCS refinement successful using SCAMP")

            # Test the refined WCS
            try:
                center_x, center_y = image_data.shape[1] // 2, image_data.shape[0] // 2
                test_coords = wcs_result.pixel_to_world_values(center_x, center_y)
                
                if isinstance(test_coords, (list, tuple)) and len(test_coords) >= 2:
                    test_ra, test_dec = test_coords[0], test_coords[1]
                else:
                    test_ra, test_dec = test_coords, 0  # Fallback
                
                # Validate coordinates
                if not (0 <= test_ra <= 360 and -90 <= test_dec <= 90):
                    st.warning(f"Refined WCS produces invalid coordinates: RA={test_ra}, DEC={test_dec}")
                    return None
                
                # Additional validation: check if the solution is reasonable
                original_ra = clean_header.get('CRVAL1') or clean_header.get('RA')
                original_dec = clean_header.get('CRVAL2') or clean_header.get('DEC')
                
                if original_ra is not None and original_dec is not None:
                    ra_diff = abs(test_ra - original_ra)
                    dec_diff = abs(test_dec - original_dec)
                    
                    # Allow for coordinate wrapping around 0/360
                    if ra_diff > 180:
                        ra_diff = 360 - ra_diff
                    
                    if ra_diff > 1.0 or dec_diff > 1.0:  # More than 1 degree difference
                        st.warning(f"Large coordinate shift detected: ΔRA={ra_diff:.3f}°, ΔDEC={dec_diff:.3f}°")
                        st.warning("This might indicate a plate solving error")
                        # Don't return None, but warn the user
                    
                st.success(f"Verified refined WCS: center at RA={test_ra:.6f}°, DEC={test_dec:.6f}°")
                
                # Update the science header with refined WCS if validation passes
                try:
                    # Clear old WCS and add new keywords
                    astrometry.clear_wcs(science_header)
                    wcs_header = wcs_result.to_header(relax=True)
                    science_header.update(wcs_header)
                    st.info("Updated header with refined WCS keywords")
                    
                except Exception as header_error:
                    st.warning(f"Could not update header: {header_error}")
                    # Return WCS anyway as it's still usable
                
                return wcs_result
                
            except Exception as test_error:
                st.warning(f"WCS verification failed: {test_error}")
                return None
        else:
            st.warning("SCAMP did not return a valid WCS solution")
            return None
            
    except ImportError as import_error:
        st.error(f"stdpipe import error: {import_error}")
        st.error("Make sure stdpipe is properly installed: pip install stdpipe")
        return None
    except Exception as e:
        st.warning(f"WCS refinement failed: {str(e)}")
        return None


def detection_and_photometry(
    image_data,
    science_header,
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
        Estimated FWHM in pixels, used for aperture and PSF sizing
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
        w, wcs_error = safe_wcs_create(science_header)
        if w is None:
            st.error(f"Error creating WCS: {wcs_error}")
            return None, None, daofind, None, None
    except Exception as e:
        st.error(f"Error creating WCS: {e}")
        return None, None, daofind, None, None

    pixel_scale = science_header.get(
        "PIXSCALE",
        science_header.get("PIXSIZE", science_header.get("PIXELSCAL", 1.0)),
    )

    bkg, bkg_error = estimate_background(
        image_data, box_size=64,
        filter_size=9
    )
    if bkg is None:
        st.error(f"Error estimating background: {bkg_error}")
        return None, None, daofind, None, None

    mask = make_border_mask(image_data, border=detection_mask)

    # Ensure image_sub is float64 to avoid casting errors
    image_sub = image_data.astype(np.float64) - bkg.background.astype(np.float64)

    show_subtracted_image(image_sub)

    # Ensure bkg_error is also float64
    bkg_error = np.full_like(image_sub, bkg.background_rms.astype(np.float64),
                             dtype=np.float64)

    exposure_time = 1.0
    if (np.max(image_data) - np.min(image_data)) > 1:
        exposure_time = science_header.get("EXPTIME",
                                           science_header.get("EXPOSURE",
                                                              science_header.get("EXP_TIME", 1.0)))

    # Ensure effective_gain is float64
    effective_gain = np.float64(2.5/np.std(image_data) * exposure_time)

    # Convert to float64 to ensure compatibility with calc_total_error
    total_error = calc_total_error(
        image_sub.astype(np.float64),
        bkg_error.astype(np.float64),
        effective_gain
    )

    st.write("Estimating FWHM...")
    fwhm_estimate, clipped_std = fwhm_fit(image_sub, mean_fwhm_pixel, mask)

    if fwhm_estimate is None:
        st.warning("Failed to estimate FWHM. Using the initial estimate.")
        fwhm_estimate = mean_fwhm_pixel

    # median_bkg_rms = np.median(bkg.background_rms)
    peak_max = 0.95 * np.max(image_sub)
    daofind = DAOStarFinder(
        fwhm=1.5 * fwhm_estimate,
        threshold=(threshold_sigma) * clipped_std,
        peakmax=peak_max)

    sources = daofind(image_sub,
                      mask=mask)

    if sources is None or len(sources) == 0:
        st.warning("No sources found!")
        return None, None, daofind, bkg, None

    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))

    # Check Astrometry option before refinement
    if hasattr(st, "session_state") and st.session_state.get("astrometry_check", False):
        refined_wcs = refine_astrometry_with_stdpipe(
            image_data=image_sub,
            science_header=science_header,
            wcs=w,
            fwhm_estimate=fwhm_estimate,
            pixel_scale=pixel_scale,
            filter_band=filter_band
        )

        # Validate WCS after refinement
        if refined_wcs:
            # Test a few source positions to ensure coordinates make sense
            test_coords = positions[:min(5, len(positions))]
            if not validate_wcs_orientation(science_header, science_header, test_coords):
                st.warning("WCS refinement may have introduced coordinate issues")
            w = refined_wcs
    else:
        st.info("Refine Astrometry is disabled. Skipping astrometry refinement.")

    # Create multiple circular apertures with different radii
    aperture_radii = [1.5, 2.0, 2.5]
    apertures = [CircularAperture(positions, r=radius * fwhm_estimate)
                 for radius in aperture_radii]

    # Create circular annulus apertures for background estimation
    from photutils.aperture import CircularAnnulus
    annulus_apertures = []
    for radius in aperture_radii:
        r_in = 1.5 * radius * fwhm_estimate
        r_out = 2.0 * radius * fwhm_estimate
        annulus_apertures.append(CircularAnnulus(positions,
                                                 r_in=r_in, r_out=r_out))

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
        elif "CTYPE1" in science_header:
            try:
                # Fallback to header WCS if no refined WCS available
                wcs_obj = WCS(science_header)
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
                    key in science_header for key in ["RA", "DEC", "NAXIS1", "NAXIS2"]
                ):
                    st.info("Using simple linear approximation for RA/DEC coordinates")
                    center_ra = science_header["RA"]
                    center_dec = science_header["DEC"]
                    width = science_header["NAXIS1"]
                    height = science_header["NAXIS2"]

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


# Visual check of the background-subtracted image
def show_subtracted_image(image_sub):
    fig, ax = plt.subplots(figsize=(7, 7))
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_sub)
    im = ax.imshow(image_sub, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set_title("Background-subtracted image")
    plt.colorbar(im, ax=ax, label='Flux')
    st.pyplot(fig)
    plt.close(fig)


def cross_match_with_gaia(
    _phot_table,
    science_header,
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

    if science_header is None:
        st.warning("No header information available. Cannot cross-match with Gaia.")
        return None

    try:
        # Use refined WCS if available, otherwise fallback to header WCS
        if refined_wcs is not None:
            w = refined_wcs
            st.info("Using refined WCS for Gaia cross-matching, if possible.")
        else:
            w = WCS(science_header)
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
        if "RA" not in science_header or "DEC" not in science_header:
            st.error("Missing RA/DEC coordinates in header")
            return None

        image_center_ra_dec = [science_header["RA"],
                               science_header["DEC"]]

        # Validate coordinate values
        if not (0 <= image_center_ra_dec[0] <= 360) or not (-90 <= image_center_ra_dec[1] <= 90):
            st.error(f"Invalid coordinates: RA={image_center_ra_dec[0]}, DEC={image_center_ra_dec[1]}")
            return None

        # Calculate search radius (divided by 1.5 to avoid field edge effects)
        gaia_search_radius_arcsec = (
            max(science_header["NAXIS1"], science_header["NAXIS2"])
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

        combined_filter = (mag_filter &
                           var_filter &
                           color_index_filter &
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
        Table of GAIA cross-matched sources (used for calibration)
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
        # Plot individual points
        ax.scatter(
            _matched_table[filter_band],
            _matched_table["calib_mag"],
            alpha=0.5,
            label="Matched sources",
            color="blue",
        )

        # Calculate and plot regression line with variance
        x_data = _matched_table[filter_band].values
        y_data = _matched_table["calib_mag"].values
        
        # Remove any NaN values for regression
        valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_clean = x_data[valid_mask]
        y_clean = y_data[valid_mask]
        
        if len(x_clean) > 1:
            # Calculate linear regression
            coeffs = np.polyfit(x_clean, y_clean, 1)
            slope, intercept = coeffs
            
            # Create regression line points
            x_reg = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_reg = slope * x_reg + intercept
            
            # Calculate residuals and standard deviation
            y_pred = slope * x_clean + intercept
            residuals = y_clean - y_pred
            std_residuals = np.std(residuals)
            
            # Plot regression line
            ax.plot(x_reg, y_reg, 'r-', linewidth=2,
                    label=f'Regression (slope={slope:.3f})')
            
            # Plot variance bands (±1σ and ±2σ)
            ax.fill_between(x_reg, y_reg - std_residuals, y_reg + std_residuals,
                            alpha=0.3, color='red', label=f'±1σ ({std_residuals:.3f} mag)')
            ax.fill_between(x_reg, y_reg - 2*std_residuals, y_reg + 2*std_residuals,
                            alpha=0.15, color='red', label=f'±2σ ({2*std_residuals:.3f} mag)')

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
    # Add input validation at the beginning
    if final_table is None:
        st.error("final_table is None - cannot enhance catalog")
        return None
    
    if len(final_table) == 0:
        st.warning("No sources to cross-match with catalogs.")
        return final_table

    # Ensure we have valid RA/Dec coordinates in final_table
    if "ra" not in final_table.columns or "dec" not in final_table.columns:
        st.error("final_table must contain 'ra' and 'dec' columns for cross-matching")
        return final_table
    
    # Make a copy to avoid modifying the original table
    enhanced_table = final_table.copy()

    # Filter out sources with NaN coordinates at the beginning
    valid_coords_mask = (
        pd.notna(enhanced_table["ra"]) & 
        pd.notna(enhanced_table["dec"]) & 
        np.isfinite(enhanced_table["ra"]) & 
        np.isfinite(enhanced_table["dec"])
    )
    
    if not valid_coords_mask.any():
        st.error("No sources with valid RA/Dec coordinates found for cross-matching")
        return enhanced_table
    
    num_invalid = len(enhanced_table) - valid_coords_mask.sum()
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

        if "xcenter" in enhanced_table.columns and "ycenter" in enhanced_table.columns:
            enhanced_table["match_id"] = (
                enhanced_table["xcenter"].round(2).astype(str)
                               + "_"
                + enhanced_table["ycenter"].round(2).astype(str)
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

            enhanced_table = pd.merge(enhanced_table, gaia_subset, on="match_id", how="left")

            enhanced_table["gaia_calib_star"] = enhanced_table["match_id"].isin(
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
            enhanced_table["astrocolibri_name"] = None
            enhanced_table["astrocolibri_type"] = None
            enhanced_table["astrocolibri_classification"] = None
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
            valid_final_coords = enhanced_table[valid_coords_mask]
            
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
                        enhanced_table.loc[original_idx, "astrocolibri_name"] = astrostars[
                            "discoverer_internal_name"
                        ][match_idx]
                        enhanced_table.loc[original_idx, "astrocolibri_type"] = astrostars["type"][match_idx]
                        enhanced_table.loc[original_idx, "astrocolibri_classification"] = astrostars[
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
                enhanced_table["simbad_main_id"] = None
                enhanced_table["simbad_otype"] = None
                enhanced_table["simbad_ids"] = None
                enhanced_table["simbad_B"] = None
                enhanced_table["simbad_V"] = None

                # Filter valid coordinates for SIMBAD matching
                valid_final_coords = enhanced_table[valid_coords_mask]
                
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
                                        enhanced_table.loc[original_idx, "simbad_main_id"] = simbad_filtered[
                                            "main_id"
                                        ][match_idx]
                                        enhanced_table.loc[original_idx, "simbad_otype"] = simbad_filtered[
                                            "otype"
                                        ][match_idx]
                                        enhanced_table.loc[original_idx, "simbad_B"] = simbad_filtered["B"][
                                            match_idx
                                        ]
                                        enhanced_table.loc[original_idx, "simbad_V"] = simbad_filtered["V"][
                                            match_idx
                                        ]
                                        if "ids" in simbad_filtered.colnames:
                                            enhanced_table.loc[original_idx, "simbad_ids"] = simbad_filtered[
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
                enhanced_table["skybot_NAME"] = None
                enhanced_table["skybot_OBJECT_TYPE"] = None
                enhanced_table["skybot_MAGV"] = None

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
                                valid_final_coords = enhanced_table[valid_coords_mask]
                                
                                if len(valid_final_coords) > 0:
                                    # Create source coordinates for matching
                                    source_coords = SkyCoord(
                                        ra=valid_final_coords["ra"], dec=valid_final_coords["dec"], unit=u.deg
                                    )

                                    # Perform cross-matching
                                    idx, d2d, _ = source_coords.match_to_catalog_3d(skybot_coords)
                                    matches = d2d.arcsec <= 10

                                    # Add matched solar system object information to the final table
                                    enhanced_table["skybot_NAME"] = None
                                    enhanced_table["skybot_OBJECT_TYPE"] = None
                                    enhanced_table["skybot_MAGV"] = None

                                    # Initialize catalog_matches column if it doesn't exist
                                    if "catalog_matches" not in enhanced_table.columns:
                                        enhanced_table["catalog_matches"] = ""

                                    # Map matches back to the original table indices
                                    valid_indices = valid_final_coords.index
                                    matched_sources = np.where(matches)[0]
                                    matched_skybots = idx[matches]

                                    for i, skybot_idx in zip(matched_sources, matched_skybots):
                                        original_idx = valid_indices[i]
                                        enhanced_table.loc[original_idx, "skybot_NAME"] = skybot_result["data"][skybot_idx]["NAME"]
                                        enhanced_table.loc[original_idx, "skybot_OBJECT_TYPE"] = skybot_result["data"][skybot_idx]["OBJECT_TYPE"]
                                        enhanced_table.loc[original_idx, "skybot_MAGV"] = skybot_result["data"][skybot_idx]["MAGV"]

                                    # Update the catalog_matches column for matched solar system objects
                                    has_skybot = enhanced_table["skybot_NAME"].notna()
                                    enhanced_table.loc[has_skybot, "catalog_matches"] += "SkyBoT; "

                                    st.success(
                                        f"Found {sum(has_skybot)} solar system objects in field."
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
                valid_final_coords = enhanced_table[valid_coords_mask]
                
                if len(valid_final_coords) > 0:
                    source_coords = SkyCoord(
                        ra=valid_final_coords["ra"].values,
                        dec=valid_final_coords["dec"].values,
                        unit="deg",
                    )

                    idx, d2d, _ = source_coords.match_to_catalog_sky(vsx_coords)
                    matches = d2d <= (10 * u.arcsec)

                    enhanced_table["aavso_Name"] = None
                    enhanced_table["aavso_Type"] = None
                    enhanced_table["aavso_Period"] = None

                    # Map matches back to the original table indices
                    valid_indices = valid_final_coords.index

                    for i, (match, match_idx) in enumerate(zip(matches, idx)):
                        if match:
                            original_idx = valid_indices[i]
                            enhanced_table.loc[original_idx, "aavso_Name"] = vsx_table["Name"][match_idx]
                            enhanced_table.loc[original_idx, "aavso_Type"] = vsx_table["Type"][match_idx]
                            if "Period" in vsx_table.colnames:
                                enhanced_table.loc[original_idx, "aavso_Period"] = vsx_table["Period"][
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
                valid_final_coords = enhanced_table[valid_coords_mask]
                
                if len(valid_final_coords) > 0:
                    # Create source coordinates for matching
                    source_coords = SkyCoord(
                        ra=valid_final_coords["ra"], dec=valid_final_coords["dec"], unit=u.deg
                    )

                    # Perform cross-matching
                    idx, d2d, _ = source_coords.match_to_catalog_3d(qso_coords)
                    matches = d2d.arcsec <= 10

                    # Add matched quasar information to the final table
                    enhanced_table["qso_name"] = None
                    enhanced_table["qso_redshift"] = None
                    enhanced_table["qso_Rmag"] = None

                    # Initialize catalog_matches column if it doesn't exist
                    if "catalog_matches" not in enhanced_table.columns:
                        enhanced_table["catalog_matches"] = ""

                    # Map matches back to the original table indices
                    valid_indices = valid_final_coords.index
                    matched_sources = np.where(matches)[0]
                    matched_qsos = idx[matches]

                    for i, qso_idx in zip(matched_sources, matched_qsos):
                        original_idx = valid_indices[i]
                        enhanced_table.loc[original_idx, "qso_name"] = qso_df.iloc[qso_idx][
                            "Name"
                        ]
                        enhanced_table.loc[original_idx, "qso_redshift"] = qso_df.iloc[qso_idx][
                            "z"
                        ]
                        enhanced_table.loc[original_idx, "qso_Rmag"] = qso_df.iloc[qso_idx][
                            "Rmag"
                        ]

                    # Update the catalog_matches column for matched quasars
                    has_qso = enhanced_table["qso_name"].notna()
                    enhanced_table.loc[has_qso, "catalog_matches"] += "QSO; "

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

    return enhanced_table


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
