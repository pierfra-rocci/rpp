import os
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.wcs import WCS

from astropy.io import fits
from photutils.detection import DAOStarFinder
from stdpipe import astrometry

from src.utils_common import estimate_background


def _try_source_detection(
    image_sub, fwhm_estimates, threshold_multipliers, min_sources=10
):
    """Helper function to try different detection parameters."""
    for fwhm_est in fwhm_estimates:
        for thresh_mult in threshold_multipliers:
            threshold = thresh_mult * np.std(image_sub)

            try:
                st.write(f"Trying FWHM={fwhm_est}, threshold={threshold:.1f}...")
                daofind = DAOStarFinder(fwhm=fwhm_est, threshold=threshold)
                temp_sources = daofind(image_sub)

                if temp_sources is not None and len(temp_sources) >= min_sources:
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
                center_y, center_x = image_data.shape[0] // 2,
                image_data.shape[1] // 2
                sample_region = image_data[
                    center_y - 50: center_y + 50,
                    center_x - 50: center_x + 50
                ]

                fig_sample, ax_sample = plt.subplots(figsize=(6, 6))
                im = ax_sample.imshow(sample_region, origin="lower",
                                      cmap="gray")
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
