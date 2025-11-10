import os
import streamlit as st

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astroscrappy
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
from photutils.psf import fit_fwhm
from astropy.visualization import ZScaleInterval

import astropy.units as u
from photutils.utils import calc_total_error
from photutils.detection import DAOStarFinder
from photutils.aperture import (CircularAperture, CircularAnnulus,
                                aperture_photometry)

from src.tools import safe_wcs_create, ensure_output_directory

from typing import Union, Optional, Dict, Tuple

from src.psf import perform_psf_photometry
from src.utils_common import (build_minimal_tan_wcs,
                              estimate_background, refine_wcs_in_memory)


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
    mask[top : height - bottom, left : width - right] = True

    return ~mask if invert else mask


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
        mat: {
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
    airmass_keywords = ["AIRMASS", "SECZ", "AIRMASS_START", "AIRMASS_END"]
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
                    st.warning(
                        f"Invalid airmass value in header ({existing_airmass}), calculating from coordinates"
                    )
                    break
            except (ValueError, TypeError):
                st.warning(
                    f"Could not parse airmass value from header keyword {keyword}"
                )
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
        date_keywords = ["DATE-OBS", "DATE", "DATE_OBS", "DATEOBS", "MJD-OBS"]
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
                    clean_date = obstime_str.replace("Z", "").replace("T", " ")
                    # Try ISO format
                    obstime = Time(clean_date, format="iso")
                else:
                    # If it's not a string, convert it
                    obstime = Time(str(obstime_str))
            except Exception:
                # Last resort: try as MJD if it's a number
                try:
                    mjd_value = float(obstime_str)
                    obstime = Time(mjd_value, format="mjd")
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Could not parse observation time '{obstime_str}' in any recognized format"
                    )

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

        st.write(f"Date and time of observation (UTC) : {obstime.iso}")
        st.write(
            f"Altitude : {details['altaz']['altitude']}° \n"
            f"Azimuth : {details['altaz']['azimuth']}°"
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
    try:
        peak = 0.95 * np.nanmax(_img)
        _, _, clipped_std = sigma_clipped_stats(_img, sigma=3.0)

        daofind = DAOStarFinder(
            fwhm=1.6 * fwhm, threshold=8 * clipped_std, peakmax=peak
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

        st.write(f"Sources after filtering : {len(filtered_sources)}")

        if len(filtered_sources) == 0:
            msg = "No valid sources for fitting found after filtering."
            st.error(msg)
            raise ValueError(msg)

        box_size = int(6 * round(fwhm))
        if box_size % 2 == 0:
            box_size += 1

        xypos = list(zip(filtered_sources["xcentroid"], filtered_sources["ycentroid"]))
        fwhms = fit_fwhm(_img, xypos=xypos, fit_shape=box_size)

        mean_fwhm = np.median(fwhms)
        st.success(f"FWHM using gaussian model : {round(mean_fwhm, 2)} pixels")

        return round(mean_fwhm, 2), clipped_std
    except ValueError as e:
        raise e
    except Exception as e:
        st.error(f"Unexpected error in fwhm_fit: {e}")
        raise ValueError(f"Unexpected error in fwhm_fit: {e}")


def detection_and_photometry(
    image_data,
    science_header,
    mean_fwhm_pixel,
    threshold_sigma,
    detection_mask,
    filter_band,
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

    bkg, bkg_error = estimate_background(image_data, box_size=64, filter_size=9)
    if bkg is None:
        st.error(f"Error estimating background: {bkg_error}")
        return None, None, daofind, None, None

    mask = make_border_mask(image_data, border=detection_mask)

    # Ensure image_sub is float64 to avoid casting errors
    image_sub = image_data.astype(np.float64) - bkg.background.astype(np.float64)

    show_subtracted_image(image_sub)

    # Ensure bkg_error is also float64
    bkg_error = np.full_like(
        image_sub, bkg.background_rms.astype(np.float64), dtype=np.float64
    )

    exposure_time = 1.0
    if (np.max(image_data) - np.min(image_data)) > 1:
        exposure_time = science_header.get(
            "EXPTIME",
            science_header.get("EXPOSURE", science_header.get("EXP_TIME", 1.0)),
        )

    # Ensure effective_gain is float64
    effective_gain = np.float64(2.5 / np.std(image_data) * exposure_time)

    # Convert to float64 to ensure compatibility with calc_total_error
    total_error = calc_total_error(
        image_sub.astype(np.float64), bkg_error.astype(np.float64), effective_gain
    )

    st.write("Estimating FWHM ...")
    fwhm_estimate, clipped_std = fwhm_fit(image_sub, mean_fwhm_pixel, mask)

    if fwhm_estimate is None:
        st.warning("Failed to estimate FWHM. Using the initial estimate.")
        fwhm_estimate = mean_fwhm_pixel

    peak_max = 0.95 * np.max(image_sub)
    daofind = DAOStarFinder(
        fwhm=1.5 * fwhm_estimate,
        threshold=(threshold_sigma + 0.5) * clipped_std,
        peakmax=peak_max,
    )

    sources = daofind(image_sub, mask=mask)

    if sources is None or len(sources) == 0:
        st.warning("No sources found!")
        return None, None, daofind, bkg, None

    positions = np.transpose((sources["xcentroid"], sources["ycentroid"]))

    # Check Astrometry option before refinement
    if hasattr(st, "session_state") and st.session_state.get("astrometry_check",
                                                             False):

        # _, refined_wcs = refine_astrometry_with_stdpipe(
        #     image_data=image_sub,
        #     science_header=science_header,
        #     wcs=w,
        #     fwhm_estimate=fwhm_estimate,
        #     pixel_scale=pixel_scale,
        #     filter_band=filter_band,
        # )
        

        # Step 1: Build minimal safe WCS
        wcs_safe, header_clean = build_minimal_tan_wcs(header, data, wcs=w,
                                                       pixel_scale_arcsec=pixel_scale)

        # Step 2: Pass wcs_safe to SCAMP for refinement
         _, refined_wcs = refine_wcs_in_memory(image_sub,
                                               wcs_safe,
                                               header_clean,
                                               pixel_scale=pixel_scale)


        w = refined_wcs
    else:
        st.info("Refine Astrometry is disabled. Skipping astrometry refinement")

    # Create multiple circular apertures with different radii
    aperture_radii = [1.5, 2.0, 2.5]
    apertures = [
        CircularAperture(positions, r=radius * fwhm_estimate)
        for radius in aperture_radii
    ]

    # Create circular annulus apertures for background estimation
    annulus_apertures = []
    for radius in aperture_radii:
        r_in = 1.5 * radius * fwhm_estimate
        r_out = 2.0 * radius * fwhm_estimate
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
                image_sub, aperture, error=total_error, wcs=wcs_obj
            )

            # Background estimation from annulus with robust outlier rejection
            bkg_result = aperture_photometry(
                image_sub, annulus, error=total_error, wcs=wcs_obj
            )

            # Process each source in this aperture size
            n_sources = len(bkg_result)

            for source_idx in range(n_sources):
                # Get individual apertures for this source
                if hasattr(annulus, "__getitem__"):  # Multiple sources
                    single_annulus = annulus[source_idx]
                    single_aperture = aperture[source_idx]
                else:  # Single source
                    single_annulus = annulus
                    single_aperture = aperture

                # Get the annulus mask for this specific source
                try:
                    annulus_mask = single_annulus.to_mask(method="center")
                    if annulus_mask is None:
                        continue

                    annulus_data = annulus_mask.multiply(image_sub)
                    if annulus_data is None or annulus_data.size == 0:
                        continue

                    # Extract only the non-zero pixels (pixels within the annulus)
                    annulus_pixels = annulus_data[annulus_data != 0]

                    if (
                        len(annulus_pixels) > 10
                    ):  # Need sufficient pixels for statistics
                        # Remove saturated pixels
                        saturation_level = science_header.get(
                            "SATURATE", np.max(image_data) * 0.95
                        )
                        valid_pixels = annulus_pixels[
                            annulus_pixels < saturation_level * 0.8
                        ]

                        if len(valid_pixels) == 0:
                            continue

                        # Iterative sigma clipping
                        clipped_pixels = sigma_clip(
                            valid_pixels, sigma=2.5, maxiters=3, masked=False
                        )

                        if len(clipped_pixels) > 5:
                            median_val = np.median(clipped_pixels)
                            mad_val = np.median(np.abs(clipped_pixels - median_val))
                            threshold = median_val + 3 * mad_val

                            final_pixels = clipped_pixels[clipped_pixels <= threshold]

                            if len(final_pixels) > 3:
                                robust_bkg_per_pixel = np.mean(final_pixels)
                                robust_bkg_std = np.std(final_pixels)

                                # Calculate total background for this aperture
                                aperture_area = single_aperture.area
                                robust_total_bkg = robust_bkg_per_pixel * aperture_area

                                # Update the background result for this source
                                bkg_result["aperture_sum"][source_idx] = (
                                    robust_total_bkg
                                )

                                if "aperture_sum_err" in bkg_result.colnames:
                                    bkg_uncertainty = robust_bkg_std * np.sqrt(
                                        aperture_area
                                    )
                                    bkg_result["aperture_sum_err"][source_idx] = (
                                        bkg_uncertainty
                                    )
                            elif len(clipped_pixels) > 0:
                                # Fallback to sigma-clipped mean
                                robust_bkg_per_pixel = np.mean(clipped_pixels)
                                robust_total_bkg = (
                                    robust_bkg_per_pixel * single_aperture.area
                                )
                                bkg_result["aperture_sum"][source_idx] = (
                                    robust_total_bkg
                                )

                except Exception as e:
                    print(
                        f"Warning: Background estimation failed for source {source_idx}: {e}"
                    )
                    continue

            # Add radius information and process results
            radius_suffix = f"_r{aperture_radii[i]:.1f}"

            # Rename aperture columns
            if "aperture_sum" in phot_result.colnames:
                phot_result.rename_column(
                    "aperture_sum", f"aperture_sum{radius_suffix}"
                )
            if "aperture_sum_err" in phot_result.colnames:
                phot_result.rename_column(
                    "aperture_sum_err", f"aperture_sum_err{radius_suffix}"
                )

            # Calculate background-corrected photometry
            if "aperture_sum" in bkg_result.colnames:
                # Ensure we handle both scalar and array cases
                annulus_area = getattr(annulus, "area", [a.area for a in annulus])
                aperture_area = getattr(aperture, "area", [a.area for a in aperture])

                # Avoid division by zero
                with np.errstate(divide="ignore", invalid="ignore"):
                    bkg_per_pixel = np.divide(
                        bkg_result["aperture_sum"],
                        annulus_area,
                        out=np.zeros_like(bkg_result["aperture_sum"]),
                        where=annulus_area != 0,
                    )

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

            if (
                aperture_sum_col in phot_table.colnames
                and aperture_err_col in phot_table.colnames
            ):
                # SNR for raw aperture sum
                phot_table[f"snr{radius_suffix}"] = np.round(
                    phot_table[aperture_sum_col] / phot_table[aperture_err_col]
                )
                m_err = 1.0857 / phot_table[f"snr{radius_suffix}"]
                phot_table[f"aperture_mag_err{radius_suffix}"] = m_err

                # Instrumental magnitude for raw aperture sum
                instrumental_mags = -2.5 * np.log10(
                    phot_table[aperture_sum_col] / exposure_time
                )
                phot_table[f"instrumental_mag{radius_suffix}"] = instrumental_mags

                # If background-corrected flux is available, calculate its magnitude too
                if bkg_corr_col in phot_table.colnames:
                    # Handle negative or zero background-corrected fluxes
                    valid_flux = phot_table[bkg_corr_col] > 0
                    phot_table[f"instrumental_mag_bkg_corr{radius_suffix}"] = np.nan
                    phot_table[f"instrumental_mag_bkg_corr{radius_suffix}"][
                        valid_flux
                    ] = -2.5 * np.log10(
                        phot_table[bkg_corr_col][valid_flux] / exposure_time
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

            epsf_table["snr"] = np.round(
                epsf_table["flux_fit"] / np.sqrt(epsf_table["flux_err"])
            )
            m_err = 1.0857 / epsf_table["snr"]
            epsf_table["psf_mag_err"] = m_err

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


def show_subtracted_image(image_sub):
    """
    Display the background-subtracted image using Streamlit.
    Parameters
    ----------
    image_sub : numpy.ndarray
        The background-subtracted image data.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    zscale = ZScaleInterval()
    vmin, vmax = zscale.get_limits(image_sub)
    im = ax.imshow(image_sub, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title("Background-subtracted image")
    plt.colorbar(im, ax=ax, label="Flux")
    st.pyplot(fig)
    plt.close(fig)


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
        valid = np.isfinite(_matched_table["instrumental_mag"]) & np.isfinite(
            _matched_table[filter_band]
        )

        zero_points = (
            _matched_table[filter_band][valid]
            - _matched_table["instrumental_mag"][valid]
        )
        _matched_table["zero_point"] = zero_points
        _matched_table["zero_point_error"] = np.std(zero_points)

        clipped_zero_points = sigma_clip(
            zero_points, sigma=3, cenfunc="mean", masked=False
        )

        zero_point_value = np.median(clipped_zero_points)
        zero_point_std = np.std(clipped_zero_points)

        if np.ma.is_masked(zero_point_value) or np.isnan(zero_point_value):
            zero_point_value = float("nan")
        if np.ma.is_masked(zero_point_std) or np.isnan(zero_point_std):
            zero_point_std = float("nan")

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
            ax.plot(
                x_reg, y_reg, "r-", linewidth=2, label=f"Regression (slope={slope:.3f})"
            )

            # Plot variance bands (±1σ and ±2σ)
            ax.fill_between(
                x_reg,
                y_reg - std_residuals,
                y_reg + std_residuals,
                alpha=0.3,
                color="red",
                label=f"±1σ ({std_residuals:.3f} mag)",
            )
            ax.fill_between(
                x_reg,
                y_reg - 2 * std_residuals,
                y_reg + 2 * std_residuals,
                alpha=0.15,
                color="red",
                label=f"±2σ ({2 * std_residuals:.3f} mag)",
            )

        # Add a diagonal line for reference
        # Create ideal y=x reference line spanning the full range of magnitudes
        mag_range = [
            min(_matched_table[filter_band].min(), _matched_table["calib_mag"].min()),
            max(_matched_table[filter_band].max(), _matched_table["calib_mag"].max()),
        ]
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

        ax_resid.errorbar(
            mag_cat,
            residuals,
            yerr=yerr,
            fmt="o",
            markersize=5,
            alpha=0.7,
            label="Residuals",
        )
        ax_resid.axhline(0, color="gray", ls="--")
        ax_resid.set_xlabel("Calibrated magnitude")
        ax_resid.set_ylabel("Residual (catalog - calibrated)")
        ax_resid.set_title("Photometric Residuals")
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
