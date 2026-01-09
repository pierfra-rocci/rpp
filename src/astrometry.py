import os
import streamlit as st
from typing import Optional

import numpy as np

from astropy.wcs import WCS
from astropy.io import fits

# stdpipe imports
from stdpipe import astrometry
from stdpipe import photometry
from stdpipe import catalogs
from stdpipe import pipeline as stdpipe_pipeline

from src.progress import ProgressReporter, get_default_reporter


def _try_source_detection(
    image,
    header,
    fwhm_estimates,
    threshold_multi,
    min_sources=10,
    progress: Optional[ProgressReporter] = None,
):
    """Helper function to try different detection parameters.
    Refactored to use stdpipe (SExtractor/SEP + photutils measurement)
    """
    progress = progress or get_default_reporter()
    
    # We iterate through parameters to find a good set
    for fwhm_est in fwhm_estimates:
        for thresh in threshold_multi:
            try:
                progress.info(f"Trying detection with thresh={thresh}, FWHM={fwhm_est}")

                # 1. Detect objects using SExtractor/SEP via stdpipe
                # get_objects_sextractor handles background estimation internally
                # but we pass the background-subtracted image 'image_sub'.
                # 'thresh' corresponds to the detection threshold (sigma).
                if header.get("GAIN"):
                    gain = header.get("GAIN")
                else:
                    gain = 65635 / np.max(image)

                sources = photometry.get_objects_sextractor(
                    image,
                    thresh=thresh,
                    aper=1.5 * fwhm_est,
                    gain=gain,
                    edge=10,
                    bg_size=64,
                )

                if sources is not None and len(sources) >= min_sources:
                    # 2. Measure objects using photutils via stdpipe
                    # This refines the photometry and measurements
                    sources = photometry.measure_objects(
                        sources,
                        image,
                        fwhm=fwhm_est,
                        aper=1.0,  # Aperture radius in FWHM units
                        bkgann=[2.0, 3.0],  # Background annulus in FWHM units
                        verbose=False,
                        sn=5.0,  # Minimum S/N
                    )

                    if sources is not None and len(sources) >= min_sources:
                        progress.success(
                            f"Found {len(sources)} sources with "
                            f"thresh={thresh}, FWHM={fwhm_est}"
                        )
                        return sources

                if sources is not None:
                    progress.write(
                        f"Found {len(sources)} sources (need at least {min_sources})"
                    )

            except Exception as e:
                progress.write(f"Detection failed with FWHM={fwhm_est} : {e}")
                continue
    return None


def solve_with_astrometrynet(file_path, progress: Optional[ProgressReporter] = None):
    """
    Solve astrometric plate using local Astrometry.Net installation via stdpipe.
    This function loads a FITS image, detects objects using photutils, and uses stdpipe's
    blind_match_objects wrapper around Astrometry.Net to determine accurate WCS information.

    Refactored to use stdpipe for detection, measurement, and refinement (Gaia DR2).

    Parameters
    ----------
    file_path : str
        Path to the FITS image file that needs astrometric solving

    Returns
    -------
    tuple
        (wcs_object, updated_header, log_messages, error) where:
        - wcs_object: astropy.wcs.WCS object containing the WCS solution
        - updated_header: Original header updated with WCS keywords
        - log_messages: List of log messages
        - error: Error message if solving fails

    Notes
    -----
    This function requires:
    - Local Astrometry.Net installation with solve-field binary
    - Appropriate index files for the field scale
    - The image should contain enough stars for plate solving (typically >10)

    Uses stdpipe.astrometry.blind_match_objects for cleaner interface.
    """
    progress = progress or get_default_reporter()
    log_messages = []
    try:
        if not os.path.exists(file_path):
            return None, None, log_messages, f"File {file_path} does not exist"

        # Load the FITS file
        log_messages.append("INFO: Loading FITS file for local plate solving")
        progress.info("Loading FITS file for plate solving...")
        with fits.open(file_path) as hdul:
            image_data = hdul[0].data
            header = hdul[0].header.copy()

        if image_data is None:
            return None, None, log_messages, "No image data found in FITS file"

        # Check if WCS already exists
        try:
            existing_wcs = WCS(header)
            if existing_wcs.is_celestial:
                log_messages.append("INFO: Proceeding with blind plate solve")
        except Exception:
            log_messages.append(
                "INFO: No valid WCS found in header. Proceeding with blind solve"
            )

        # Try standard detection parameters first
        sources = _try_source_detection(
            image_data,
            header,
            fwhm_estimates=[3.5, 4.0, 5.0],
            threshold_multi=[2.0, 1.5],
            min_sources=50,
            progress=progress,
        )

        # If that fails, try more aggressive parameters
        if sources is None:
            log_messages.append(
                "WARNING: Standard detection failed. Trying more aggressive parameters"
            )
            sources = _try_source_detection(
                image_data,
                header,
                fwhm_estimates=[2.0, 2.5, 3.0],
                threshold_multi=[1.0, 0.5],
                min_sources=25,
                progress=progress,
            )

        if sources is None or len(sources) < 5:
            error_message = "Failed to detect sufficient sources for plate solving."
            log_messages.append(f"ERROR: {error_message}")
            return None, None, log_messages, error_message

        # Filter for best sources if too many
        if len(sources) > 500:
            sources.sort("flux")
            sources.reverse()
            sources = sources[:500]
            log_messages.append(
                f"INFO: Using brightest {len(sources)} sources for plate solving"
            )

        log_messages.append(
            f"SUCCESS: Ready for plate solving with {len(sources)} sources"
        )

        # Get pixel scale estimate
        pixel_scale_estimate = None
        if header:
            # Try to get pixel scale from various header keywords
            for key in ["PIXSCALE", "PIXSIZE", "SECPIX"]:
                if key in header:
                    log_messages.append(f"INFO: Found pixel scale in header: {key}")
                    pixel_scale_estimate = float(header[key])
                    break

            # If still not found, try FOCALLEN + pixel size calculation
            if pixel_scale_estimate is None:
                try:
                    focal_length = None
                    pixel_size = None

                    # Check for focal length keywords
                    for focal_key in ["FOCALLEN", "FOCAL", "FOCLEN", "FL"]:
                        if focal_key in header:
                            focal_length = float(header[focal_key])
                            log_messages.append(
                                f"INFO: Found focal length: {focal_length} mm from {focal_key}"
                            )
                            break

                    # Check for pixel size keywords (in microns)
                    for pixel_key in [
                        "PIXELSIZE",
                        "PIXSIZE",
                        "XPIXSZ",
                        "YPIXSZ",
                        "PIXELMICRONS",
                    ]:
                        if pixel_key in header:
                            pixel_size = float(header[pixel_key])
                            log_messages.append(
                                f"INFO: Found pixel size: {pixel_size} microns from {pixel_key}"
                            )
                            break

                    # Calculate pixel scale using the formula from the guide scope reference
                    if focal_length is not None and pixel_size is not None:
                        pixel_scale_estimate = 206 * pixel_size / focal_length

                        # Sanity check - typical astronomical pixel scales
                        if not (0.1 <= pixel_scale_estimate <= 30.0):
                            log_messages.append(
                                f"WARNING: Unusual pixel scale calculated: {pixel_scale_estimate:.2f} arcsec/pixel"
                            )
                        else:
                            log_messages.append(
                                f"SUCCESS: Pixel scale calculated from focal length and pixel size: {pixel_scale_estimate:.2f} arcsec/pixel"
                            )
                    elif focal_length is not None:
                        log_messages.append(
                            f"WARNING: Found focal length ({focal_length} mm) but no pixel size in header"
                        )
                    elif pixel_size is not None:
                        log_messages.append(
                            f"WARNING: Found pixel size ({pixel_size} µm) but no focal length in header"
                        )

                except Exception as e:
                    log_messages.append(
                        f"WARNING: Error calculating pixel scale from focal length/pixel size: {e}"
                    )
                    pass

            # If not found, try to calculate from CD matrix
            if pixel_scale_estimate is None:
                try:
                    cd_values = [
                        header.get(f"CD{i}_{j}", 0) for i in [1, 2] for j in [1, 2]
                    ]
                    if any(x != 0 for x in cd_values):
                        cd11, cd12, cd21, cd22 = cd_values
                        det = abs(cd11 * cd22 - cd12 * cd21)
                        pixel_scale_estimate = 3600 * np.sqrt(det)
                        log_messages.append(
                            "INFO: Calculated pixel scale from CD matrix"
                        )
                except Exception:
                    pass

        # Prepare parameters for stdpipe blind_match_objects
        kwargs = {
            "order": 3,  # SIP distortion order
            "update": False,  # Don't update object list in place
            "sn": 5,
            "get_header": False,  # Return WCS object, not header
            "width": image_data.shape[1],
            "height": image_data.shape[0],
            "verbose": True,
        }

        # Add pixel scale constraints if available
        if pixel_scale_estimate and pixel_scale_estimate > 0:
            scale_low = pixel_scale_estimate * 0.8
            scale_high = pixel_scale_estimate * 1.2
            kwargs.update(
                {
                    "scale_lower": scale_low,
                    "scale_upper": scale_high,
                    "scale_units": "arcsecperpix",
                }
            )
            log_messages.append(
                f"INFO: Using pixel scale estimate: {pixel_scale_estimate:.2f} arcsec/pixel"
            )
        else:
            # Use broad scale range if no estimate available
            kwargs.update(
                {"scale_lower": 0.1, "scale_upper": 10.0, "scale_units": "arcsecperpix"}
            )
            log_messages.append(
                "INFO: No pixel scale estimate available, using broad range"
            )

        # Add RA/DEC hint if available
        if header and "RA" in header and "DEC" in header:
            try:
                ra_hint = float(header["RA"])
                dec_hint = float(header["DEC"])
                if 0 <= ra_hint <= 360 and -90 <= dec_hint <= 90:
                    kwargs.update(
                        {
                            "center_ra": ra_hint,
                            "center_dec": dec_hint,
                            "radius": 0.95,  # 5 degree search radius
                        }
                    )
                    log_messages.append(
                        f"INFO: Using RA/DEC hint: {ra_hint:.3f}, {dec_hint:.3f}"
                    )
            except Exception:
                log_messages.append("WARNING: Could not parse RA/DEC from header")

        try:
            # Call stdpipe's blind_match_objects function
            # We pass 'sources' directly as it is an Astropy Table compatible with stdpipe
            solved_wcs = astrometry.blind_match_objects(sources, **kwargs)

            if solved_wcs is not None:
                log_messages.append("SUCCESS: Plate solving successful")

                # --- Refinement Step (Gaia DR2) ---
                try:
                    log_messages.append(
                        "INFO: Attempting WCS refinement using Gaia DR2"
                    )

                    # 1. Get image parameters from the initial solution
                    height, width = image_data.shape

                    # Get frame center and radius
                    center_ra, center_dec, center_sr = astrometry.get_frame_center(
                        wcs=solved_wcs, width=width, height=height
                    )

                    # 2. Fetch Gaia DR2 catalog
                    log_messages.append(
                        f"INFO: Fetching Gaia DR2 catalog around RA={center_ra:.4f}, DEC={center_dec:.4f}, r={center_sr:.2f} deg"
                    )
                    cat = catalogs.get_cat_vizier(
                        center_ra, center_dec, center_sr, "ps1", filters={"rmag": "<19"}
                    )

                    if cat is not None and len(cat) > 10:
                        log_messages.append(f"INFO: Retrieved {len(cat)} catalog stars")

                        # 3. Refine Astrometry
                        # Get pixel scale from solved WCS for matching radius
                        pixscale_deg = astrometry.get_pixscale(wcs=solved_wcs)

                        refined_wcs = stdpipe_pipeline.refine_astrometry(
                            sources,
                            cat,
                            sr=7 * pixscale_deg,  # Matching radius ~ 5 pixels
                            wcs=solved_wcs,
                            method="scamp",  # Prefer SCAMP as per notebook, falls back if not avail?
                            cat_col_mag="rmag",
                            verbose=True,
                        )

                        if refined_wcs is not None:
                            solved_wcs = refined_wcs
                            log_messages.append(
                                "SUCCESS: Astrometry refined using Gaia DR2"
                            )
                        else:
                            log_messages.append(
                                "WARNING: Astrometric refinement failed, keeping initial solution"
                            )
                    else:
                        log_messages.append(
                            "WARNING: Could not retrieve sufficient catalog stars for refinement"
                        )

                except Exception as refine_error:
                    log_messages.append(
                        f"WARNING: Refinement process failed: {refine_error}"
                    )

                # Update original header with WCS solution (refined or initial)
                updated_header = header.copy()

                # Clear any existing WCS keywords to avoid conflicts
                astrometry.clear_wcs(
                    updated_header,
                    remove_comments=True,
                    remove_underscored=True,
                    remove_history=True,
                )

                # Add new WCS keywords from solution
                wcs_header = solved_wcs.to_header(relax=True)
                updated_header.update(wcs_header)

                # Add solution metadata
                updated_header["COMMENT"] = (
                    "WCS solution from astrometry.net (refined with Gaia DR2)"
                )
                updated_header["STDPIPE"] = (
                    True,
                    "Solved with astrometry.blind_match_objects + refinement",
                )

                # Display solution information
                try:
                    center_ra, center_dec = solved_wcs.pixel_to_world_values(
                        image_data.shape[1] / 2, image_data.shape[0] / 2
                    )
                    pixel_scale = astrometry.get_pixscale(wcs=solved_wcs) * 3600

                    log_messages.append(
                        f"INFO: Solution center: RA={center_ra:.6f}°, DEC={center_dec:.6f}°"
                    )
                    log_messages.append(
                        f"INFO: Pixel scale: {pixel_scale:.3f} arcsec/pixel"
                    )

                    # Validate solution makes sense
                    if 0 <= center_ra <= 360 and -90 <= center_dec <= 90:
                        log_messages.append("SUCCESS: Solution coordinates are valid")
                    else:
                        log_messages.append(
                            "WARNING: Solution coordinates seem invalid"
                        )

                except Exception as coord_error:
                    log_messages.append(
                        f"WARNING: Could not extract solution coordinates: {coord_error}"
                    )

                return solved_wcs, updated_header, log_messages, None

            else:
                error_message = "Plate solving failed"
                log_messages.append(f"ERROR: {error_message}")
                return None, None, log_messages, error_message

        except Exception as solve_error:
            error_message = f"Error during plate solving: {solve_error}"
            log_messages.append(f"ERROR: {error_message}")
            return None, None, log_messages, error_message

    except Exception as e:
        error_message = f"Error in plate solving setup: {str(e)}"
        log_messages.append(f"ERROR: {error_message}")
        return None, None, log_messages, error_message
