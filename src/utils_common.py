import os
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS
from astropy.stats import SigmaClip
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval

from stdpipe import photometry, astrometry, catalogs

from astropy.io import fits
from photutils.detection import DAOStarFinder
from photutils.background import Background2D, SExtractorBackground

from src.tools import ensure_output_directory, write_to_log

from typing import Optional


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
                im1 = ax1.imshow(
                    bkg.background, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
                )
                ax1.set_title("Estimated Background")
                fig_bkg.colorbar(im1, ax=ax1, label="Flux")

                # Plot the background RMS
                vmin_rms, vmax_rms = zscale.get_limits(bkg.background_rms)
                im2 = ax2.imshow(
                    bkg.background_rms,
                    origin="lower",
                    cmap="viridis",
                    vmin=vmin_rms,
                    vmax=vmax_rms,
                )
                ax2.set_title("Background RMS")
                fig_bkg.colorbar(im2, ax=ax2, label="Flux")

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
                hdu_bkg.header["COMMENT"] = (
                    "Background model created with photutils.Background2D"
                )
                hdu_bkg.header["BOXSIZE"] = (
                    adjusted_box_size,
                    "Box size for background estimation",
                )
                hdu_bkg.header["FILTSIZE"] = (
                    adjusted_filter_size,
                    "Filter size for background smoothing",
                )

                # Add RMS as extension
                hdu_rms = fits.ImageHDU(data=bkg.background_rms)
                hdu_rms.header["EXTNAME"] = "BACKGROUND_RMS"

                hdul = fits.HDUList([hdu_bkg, hdu_rms])
                hdul.writeto(bkg_filepath, overwrite=True)

                # Write to log if available
                log_buffer = st.session_state.get("log_buffer")
                if log_buffer is not None:
                    write_to_log(
                        log_buffer, f"Background model saved to {bkg_filename}"
                    )

            except Exception as e:
                st.warning(f"Error creating or saving background plot: {str(e)}")
            finally:
                # Clean up matplotlib figure to prevent memory leaks
                if fig_bkg is not None:
                    plt.close(fig_bkg)

        return bkg, None
    except Exception as e:
        return None, f"Background estimation error: {str(e)}"


def refine_astrometry_with_stdpipe(
    image_data: np.ndarray,
    science_header: dict,
    wcs: WCS,
    fwhm_estimate: float,
    pixel_scale: float,
    filter_band: str,
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
    log_buffer = st.session_state.get("log_buffer")
    write_to_log(log_buffer, "Starting astrometry refinement", "INFO")
    try:
        st.write("Doing astrometry refinement using SCAMP...")

        # Convert image data to float32 for stdpipe compatibility
        if image_data.dtype not in [np.float32, np.float64]:
            st.info(
                f"Converting image from {image_data.dtype} to float32 for stdpipe compatibility"
            )
            image_data = image_data.astype(np.float32)

        # Clean and prepare header for stdpipe
        clean_header = science_header.copy()

        # Remove known problematic keywords
        keys_to_remove = [
            "HISTORY",
            "COMMENT",
            "CONTINUE",  # General metadata
            "XPIXELSZ",
            "YPIXELSZ",
            "CDELTM1",
            "CDELTM2",
        ]

        # Add distortion-related keywords to the removal list
        for key in list(clean_header.keys()):
            if any(
                pattern in str(key).upper()
                for pattern in ["DSS", "SIP", "PV", "DISTORT", "A_", "B_", "AP_", "BP_"]
            ):
                keys_to_remove.append(key)

        removed_count = 0
        for key in set(keys_to_remove):  # Use set to avoid duplicates
            if key in clean_header:
                del clean_header[key]
                removed_count += 1

        if removed_count > 0:
            st.info(
                f"Removed {removed_count} problematic or distortion-related keywords from header."
            )

        # Validate and, if necessary, fix the core WCS parameters
        try:
            # Ensure CTYPEs are present and valid
            if not clean_header.get("CTYPE1"):
                clean_header["CTYPE1"] = "RA---TAN"
            if not clean_header.get("CTYPE2"):
                clean_header["CTYPE2"] = "DEC--TAN"

            # Ensure reference pixels are valid
            if not np.isfinite(clean_header.get("CRPIX1", np.nan)):
                clean_header["CRPIX1"] = image_data.shape[1] / 2.0
            if not np.isfinite(clean_header.get("CRPIX2", np.nan)):
                clean_header["CRPIX2"] = image_data.shape[0] / 2.0

            # Check for a valid CD matrix, if it exists
            cd_keys = ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]
            if all(key in clean_header for key in cd_keys):
                cd_values = [clean_header.get(key, 0) for key in cd_keys]
                # Check for non-finite or zero values in the matrix
                if any(not np.isfinite(v) or v == 0 for v in cd_values):
                    st.warning("Invalid CD matrix detected. It will be ignored.")
                    # Remove the invalid CD matrix so astropy can build a new one if possible
                    for key in cd_keys:
                        if key in clean_header:
                            del clean_header[key]

            # If no CD matrix, ensure CDELT is present for WCS creation
            if "CD1_1" not in clean_header:
                if "CDELT1" not in clean_header:
                    clean_header["CDELT1"] = -pixel_scale / 3600.0
                    st.info("CDELT1 not found. Setting from pixel scale.")
                if "CDELT2" not in clean_header:
                    clean_header["CDELT2"] = pixel_scale / 3600.0
                    st.info("CDELT2 not found. Setting from pixel scale.")

        except Exception as header_fix_error:
            st.warning(f"An error occurred while fixing the header: {header_fix_error}")

        # Test the cleaned WCS before proceeding
        try:
            test_wcs = WCS(clean_header)
            st.info("Cleaned WCS passes basic validation")
        except Exception as wcs_test_error:
            st.error(f"Cleaned WCS still has issues: {wcs_test_error}")
            return None

        def _ensure_native_byteorder(arr: np.ndarray) -> np.ndarray:
            # SEP requires native byteorder arrays (see SEP docs)
            if not arr.dtype.isnative:
                try:
                    return arr.byteswap().newbyteorder()
                except Exception:
                    return arr.astype(arr.dtype.newbyteorder("="))
            return arr

        image_for_det = image_data.copy()
        # make float32 for detection routines (many expect float32)
        if image_for_det.dtype != np.float32:
            image_for_det = image_for_det.astype(np.float32)
        image_for_det = _ensure_native_byteorder(image_for_det)

        # Use SExtractor-only detection strategy (remove SEP attempts)
        detection_candidates = None
        tried_methods = []

        st.info("Using SExtractor-only detection strategy")
        sex_param_grid = [
            {"thresh": 3.0, "minarea": 5, "r0": 0.0},
            {"thresh": 2.5, "minarea": 5, "r0": 0.8},
            {"thresh": 2.0, "minarea": 3, "r0": 1.0},
        ]

        for params in sex_param_grid:
            try:
                msg = (
                    f"Attempting SExtractor detection: DETECT_THRESH={params['thresh']}, "
                    f"DETECT_MINAREA={params['minarea']}, smoothing={params['r0']}"
                )
                st.info(msg)
                obj_try = photometry.get_objects_sextractor(
                    image_for_det,
                    header=clean_header,
                    thresh=params["thresh"],
                    r0=params["r0"],
                    minarea=params["minarea"],
                    aper=max(1.0, 1.5 * fwhm_estimate),
                    bg_size=64,
                    edge=10,
                    gain=1.0,
                    mask_to_nans=False,
                    verbose=False,
                )
                tried_methods.append(("sex", params))
                # get_objects_sextractor may return (table, checkimages...). Accept table
                if isinstance(obj_try, (list, tuple)):
                    obj_try = obj_try[0]
                if obj_try is not None and len(obj_try) > 0:
                    detection_candidates = obj_try
                    st.success(
                        f"Detected {len(detection_candidates)} objects with SExtractor"
                    )
                    break
            except Exception as e_sex:
                err_str = str(e_sex)
                st.write(f"SExtractor attempt failed: {err_str}")
                # record exception for diagnostics
                tried_methods.append(("sex-exc", params, err_str))
                continue

        obj = detection_candidates
        # If SExtractor found nothing, attempt a robust photutils DAOStarFinder fallback
        if obj is None or len(obj) == 0:
            st.warning(
                "SExtractor returned no detections; attempting DAOStarFinder fallback..."
            )
            # Log attempted methods and exceptions
            st.write(tried_methods)

            try:
                # Ensure image has finite values for stats
                finite_mask = np.isfinite(image_for_det)
                n_finite = int(np.sum(finite_mask))
                if n_finite < 10:
                    st.error(
                        "Image contains too few finite pixels for source detection."
                    )
                    return None

                # Compute robust background stats
                _, median_bkg, std_bkg = sigma_clipped_stats(
                    image_for_det[finite_mask], sigma=3.0
                )
                if not np.isfinite(std_bkg) or std_bkg <= 0:
                    st.error(
                        "Background noise estimate invalid; cannot run DAOStarFinder."
                    )
                    return None

                st.info(
                    f"DAO fallback: median_bkg={median_bkg:.3g}, std_bkg={std_bkg:.3g}, "
                    f"finite_pixels={n_finite}"
                )

                dao_thresholds = [5.0, 4.0, 3.0]
                dao_found = None
                for t in dao_thresholds:
                    try:
                        abs_thresh = t * std_bkg
                        daofind = DAOStarFinder(
                            threshold=abs_thresh,
                            fwhm=max(1.0, 1.5 * fwhm_estimate),
                        )
                        sources = daofind(image_for_det - median_bkg)
                        if sources is not None and len(sources) > 0:
                            dao_found = sources
                            st.success(
                                f"Detected {len(sources)} objects with DAOStarFinder "
                                f"(threshold={t}σ)"
                            )
                            break
                    except Exception as e_dao:
                        st.write(
                            f"DAOStarFinder attempt (threshold={t}) failed: {e_dao}"
                        )
                        continue

                if dao_found is None:
                    st.error("No objects detected by DAOStarFinder either.")
                    return None

                # Use DAOStarFinder results as obj for downstream processing
                obj = dao_found
            except Exception as fallback_exc:
                st.error(f"Fallback detection failed: {fallback_exc}")
                return None

        st.info(f"Detected {len(obj)} objects for astrometry refinement")

        # Get frame center using the cleaned header and test WCS
        try:
            # Use the cleaned header instead of original
            center_ra, center_dec, radius = astrometry.get_frame_center(
                header=clean_header,
                wcs=test_wcs,  # Use the validated test WCS
                width=image_data.shape[1],
                height=image_data.shape[0],
            )
        except Exception as center_error:
            st.error(f"Failed to get frame center: {center_error}")

            # Try fallback method using header coordinates
            try:
                center_ra = (
                    clean_header.get("CRVAL1")
                    or clean_header.get("RA")
                    or clean_header.get("OBJRA")
                )
                center_dec = (
                    clean_header.get("CRVAL2")
                    or clean_header.get("DEC")
                    or clean_header.get("OBJDEC")
                )

                if center_ra is None or center_dec is None:
                    st.error("Could not determine field center coordinates")
                    return None

                # Calculate radius from image dimensions
                radius = max(image_data.shape) * pixel_scale / 3600.0 / 2.0
                st.info(
                    f"Using fallback field center: RA={center_ra:.3f}, "
                    f"DEC={{center_dec:.3f}}, radius={{radius:.3f}}"
                    ""
                )

            except Exception as fallback_error:
                st.error(f"Fallback coordinate extraction failed: {fallback_error}")
                return None

        # Map filter band to correct GAIA EDR3 column names
        gaia_band_mapping = {
            "phot_bp_mean_mag": "BPmag",
            "phot_rp_mean_mag": "RPmag",
            "phot_g_mean_mag": "Gmag",
        }

        gaia_band = gaia_band_mapping.get(filter_band, "Gmag")

        # Get GAIA catalog with correct parameters and error handling
        try:
            cat = catalogs.get_cat_vizier(
                center_ra,
                center_dec,
                radius,
                "I/350/gaiaedr3",  # Correct GAIA EDR3 catalog identifier
                filters={gaia_band: "< 20.0"},
            )

            if cat is None or len(cat) == 0:
                st.warning("No GAIA catalog sources found in field")
                return None

        except Exception as cat_error:
            st.error(f"Failed to get GAIA catalog: {cat_error}")

            # Try with a smaller search radius
            try:
                smaller_radius = min(radius, 0.5)  # Limit to 0.5 degrees
                st.info(
                    f"Retrying GAIA query with smaller radius: {smaller_radius:.3f}°"
                )
                cat = catalogs.get_cat_vizier(
                    center_ra,
                    center_dec,
                    smaller_radius,
                    "I/350/gaiaedr3",
                    filters={gaia_band: "< 19.0"},
                )

                if cat is None or len(cat) == 0:
                    st.warning(
                        "No GAIA catalog sources found even with reduced search radius"
                    )
                    return None

            except Exception as retry_error:
                st.error(f"GAIA catalog query retry failed: {retry_error}")
                return None

        st.info(f"Retrieved {len(cat)} GAIA catalog sources")

        try:
            # Filter out sources with poor parallax measurements if the column exists
            if "parallax" in cat.colnames:
                parallax_filter = cat["parallax"] > -100  # Basic parallax filter
                cat = cat[parallax_filter]
                st.info(f"After parallax filtering: {len(cat)} GAIA catalog sources")
        except Exception as filter_error:
            st.warning(f"Could not apply additional filtering: {filter_error}")

        # Ensure we still have enough sources for refinement
        if len(cat) < 10:
            st.warning(
                f"Too few GAIA sources ({len(cat)}) for reliable astrometry refinement"
            )
            return None

        # Calculate matching radius in degrees
        try:
            match_radius_deg = 1.5 * fwhm_estimate * pixel_scale / 3600.0

            # Try SCAMP refinement with conservative parameters
            wcs_result = astrometry.refine_wcs_scamp(
                obj,
                cat,
                wcs=test_wcs,  # Use the validated test WCS
                sr=match_radius_deg,
                order=2,
                cat_col_ra="RA_ICRS",
                cat_col_dec="DE_ICRS",
                cat_col_mag=gaia_band,
                cat_mag_lim=19,  # Conservative magnitude limit
                verbose=True,
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
                    cat_col_ra="RA_ICRS",
                    cat_col_dec="DE_ICRS",
                    cat_col_mag=gaia_band,
                    cat_mag_lim=19.0,
                    verbose=True,
                )
            except Exception as final_refine_error:
                st.error(f"Final SCAMP attempt failed: {final_refine_error}")
                return None

        # Validate and return the refined WCS
        if wcs_result is not None:
            st.success("WCS refinement successful using SCAMP")

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
