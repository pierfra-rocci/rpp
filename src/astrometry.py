import os
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats

from astropy.io import fits
from photutils.detection import DAOStarFinder
from stdpipe import photometry, astrometry, catalogs

from src.tools import write_to_log
from src.pipeline import estimate_background

from typing import Optional


def _try_source_detection(
    image_sub, fwhm_estimates, threshold_multipliers, min_sources=10
):
    """Helper function to try different detection parameters."""
    for fwhm_est in fwhm_estimates:
        for thresh_mult in threshold_multipliers:
            threshold = thresh_mult * np.std(image_sub)

            try:
                st.write(
                    f"Trying FWHM={fwhm_est}, threshold={threshold:.1f}..."
                )
                daofind = DAOStarFinder(fwhm=fwhm_est, threshold=threshold)
                temp_sources = daofind(image_sub)

                if (
                    temp_sources is not None
                    and len(temp_sources) >= min_sources
                ):
                    st.success(
                        f"Photutils found {len(temp_sources)} sources with "
                        f"FWHM={fwhm_est}, threshold={threshold:.1f}"
                    )
                    return temp_sources
                elif temp_sources is not None:
                    st.write(
                        f"Found {len(temp_sources)} sources (need at least {min_sources})"
                    )

            except Exception as e:
                st.write(
                    f"Detection failed with FWHM={fwhm_est}, threshold={threshold:.1f}: {e}"
                )
                continue
    return None


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
                st.info(
                    "Valid WCS already exists in header. Proceeding with blind solve anyway..."
                )
        except Exception:
            st.info("No valid WCS found in header. Proceeding with blind solve...")

        # Estimate background
        st.write("Detecting objects for plate solving using photutils...")
        bkg, bkg_error = estimate_background(image_data, figure=False)
        if bkg is None:
            st.error(f"Failed to estimate background: {bkg_error}")
            return None, None

        image_sub = image_data - bkg.background
        
        # Try standard detection parameters first
        sources = _try_source_detection(
            image_sub,
            fwhm_estimates=[3.0, 4.0, 5.0],
            threshold_multipliers=[3.0, 4.0, 5.0],
            min_sources=10,
        )

        # If that fails, try more aggressive parameters
        if sources is None:
            st.warning(
                "Standard detection failed. Trying more aggressive parameters..."
            )
            sources = _try_source_detection(
                image_sub,
                fwhm_estimates=[1.5, 2.0, 2.5],
                threshold_multipliers=[2.0, 2.5],
                min_sources=5,
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
                sample_region = image_data[
                    center_y - 50: center_y + 50, center_x - 50: center_x + 50
                ]

                fig_sample, ax_sample = plt.subplots(figsize=(6, 6))
                im = ax_sample.imshow(sample_region, origin="lower", cmap="gray")
                ax_sample.set_title("Central 100x100 pixel region")
                plt.colorbar(im, ax=ax_sample)
                st.pyplot(fig_sample)

            except Exception as plot_error:
                st.warning(f"Could not display sample region: {plot_error}")

            return None, None

        # Prepare sources for astrometric solving
        st.write(f"Successfully detected {len(sources)} sources")

        # Filter for best sources if too many
        if len(sources) > 500:
            sources.sort("flux")
            sources.reverse()
            sources = sources[:500]
            st.write(f"Using brightest {len(sources)} sources for plate solving")

        st.success(f"Ready for plate solving with {len(sources)} sources")

        # Convert sources to the format expected by stdpipe
        obj_table = Table()
        obj_table["x"] = sources["xcentroid"]
        obj_table["y"] = sources["ycentroid"]
        obj_table["flux"] = sources["flux"]

        # Add signal-to-noise ratio if available
        if "peak" in sources.colnames:
            # Estimate SNR from peak/background ratio
            background_std = np.std(image_sub)
            obj_table["sn"] = sources["peak"] / background_std
            obj_table["fluxerr"] = sources["flux"] / obj_table["sn"]
        else:
            # Use flux as proxy for SNR
            obj_table["sn"] = sources["flux"] / np.median(sources["flux"])
            obj_table["fluxerr"] = np.sqrt(np.abs(sources["flux"]))

        # Ensure fluxerr is positive and finite
        obj_table["fluxerr"] = np.where(
            (obj_table["fluxerr"] <= 0) | ~np.isfinite(obj_table["fluxerr"]),
            sources["flux"] * 0.1,
            obj_table["fluxerr"],
        )

        # Get pixel scale estimate
        pixel_scale_estimate = None
        if header:
            # Try to get pixel scale from various header keywords
            for key in ["PIXSCALE", "PIXSIZE", "SECPIX"]:
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
                    for focal_key in ["FOCALLEN", "FOCAL", "FOCLEN", "FL"]:
                        if focal_key in header:
                            focal_length = float(header[focal_key])
                            st.write(
                                f"Found focal length: {focal_length} mm from {focal_key}"
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
                            st.write(
                                f"Found pixel size: {pixel_size} microns from {pixel_key}"
                            )
                            break

                    # Calculate pixel scale using the formula from the guide scope reference
                    if focal_length is not None and pixel_size is not None:
                        pixel_scale_estimate = 206 * pixel_size / focal_length

                        # Sanity check - typical astronomical pixel scales
                        if not (0.1 <= pixel_scale_estimate <= 30.0):
                            st.warning(
                                f"Unusual pixel scale calculated: {pixel_scale_estimate:.2f} arcsec/pixel"
                            )
                            st.warning(
                                "Please verify focal length and pixel size values in header"
                            )
                        else:
                            st.success(
                                f"Pixel scale calculated from focal length and pixel size: {pixel_scale_estimate:.2f} arcsec/pixel"
                            )
                    elif focal_length is not None:
                        st.warning(
                            f"Found focal length ({focal_length} mm) but no pixel size in header"
                        )
                    elif pixel_size is not None:
                        st.warning(
                            f"Found pixel size ({pixel_size} µm) but no focal length in header"
                        )

                except Exception as e:
                    st.warning(
                        f"Error calculating pixel scale from focal length/pixel size: {e}"
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
                        st.write("Calculated pixel scale from CD matrix")
                except Exception:
                    pass

        # Prepare parameters for stdpipe blind_match_objects
        kwargs = {
            "order": 2,  # SIP distortion order
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
            st.write(
                f"Using pixel scale estimate: {pixel_scale_estimate:.2f} arcsec/pixel"
            )
        else:
            # Use broad scale range if no estimate available
            kwargs.update(
                {"scale_lower": 0.1, "scale_upper": 10.0, "scale_units": "arcsecperpix"}
            )
            st.write("No pixel scale estimate available, using broad range")

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
                updated_header["COMMENT"] = "WCS solution from stdpipe/astrometry.net"
                updated_header["STDPIPE"] = (
                    True,
                    "Solved with stdpipe.astrometry.blind_match_objects",
                )

                # Display solution information
                try:
                    center_ra, center_dec = solved_wcs.pixel_to_world_values(
                        image_data.shape[1] / 2, image_data.shape[0] / 2
                    )
                    pixel_scale = astrometry.get_pixscale(wcs=solved_wcs) * 3600

                    st.write(
                        f"Solution center: RA={center_ra:.6f}°, DEC={center_dec:.6f}°"
                    )
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


def refine_astrometry_with_stdpipe(
    image_data: np.ndarray,
    science_header: dict,
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
            "HISTORY", "COMMENT", "CONTINUE",  # General metadata
            "XPIXELSZ", "YPIXELSZ", "CDELTM1", "CDELTM2",
        ]

        # Add distortion-related keywords to the removal list
        for key in list(clean_header.keys()):
            if any(pattern in str(key).upper() for pattern in ["DSS", "SIP", "PV", "DISTORT", "A_", "B_", "AP_", "BP_"]):
                keys_to_remove.append(key)

        removed_count = 0
        for key in set(keys_to_remove): # Use set to avoid duplicates
            if key in clean_header:
                del clean_header[key]
                removed_count += 1

        if removed_count > 0:
            st.info(f"Removed {removed_count} problematic or distortion-related keywords from header.")

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
