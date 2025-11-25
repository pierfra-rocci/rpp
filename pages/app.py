# Standard Library Imports
import sys
import os
import json
import tempfile
import warnings
from datetime import datetime
from types import SimpleNamespace

# Third-Party Imports
import streamlit as st
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval, ImageNormalize

from src.tools_app import (
    initialize_session_state,
    load_fits_data,
    plot_magnitude_distribution,
    update_observatory_from_fits_header,
    display_catalog_in_aladin,
    display_archived_files_browser,
    provide_download_buttons,
    clear_all_caches,
    handle_log_messages,
    cleanup_temp_files)

# Local Application Imports
from src.tools_pipeline import (
    GAIA_BANDS,
    extract_coordinates,
    extract_pixel_scale,
    safe_wcs_create,
    add_calibrated_magnitudes,
    merge_photometry_catalogs,
    clean_photometry_table,
)
from src.utils import (
    FIGURE_SIZES,
    get_base_filename,
    ensure_output_directory,
    initialize_log,
    write_to_log,
    zip_results_on_exit,
    save_header_to_txt,
    save_catalog_files,
)

from src.pipeline import (
    calculate_zero_point,
    detection_and_photometry,
    airmass,
)

from src.astrometry import solve_with_astrometrynet
from src.xmatch_catalogs import cross_match_with_gaia, enhance_catalog
from src.transient import find_candidates

from src.__version__ import version


# Conditional Import (already present, just noting its location)
if getattr(sys, "frozen", False):
    try:
        import importlib.metadata

        importlib.metadata.distributions = lambda **kwargs: []
    except ImportError:
        st.warning(
            "Could not modify importlib.metadata, potential issues in frozen mode."
        )

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="RAPAS Photometry Pipeline", page_icon=":star:", layout="centered"
)

initialize_session_state()

# Custom CSS to control plot display size
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Redirect to login if not authenticated
if not st.session_state.logged_in:
    st.warning("You must log in to access this page.")
    st.switch_page("pages/login.py")

# Add application version to the sidebar
st.title("**RAPAS Photometry Pipeline**")
st.markdown("[**RAPAS Project**](https://rapas.imcce.fr/) / [**Github**](https://github.com/pierfra-rocci/rpp)",
            unsafe_allow_html=True)
# Added: Quick Start Tutorial link (now displayed in an expander)
with st.expander("📘 Quick Start Tutorial"):
    try:
        with open(os.path.join("docs", "TUTORIAL.md"), "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("TUTORIAL.md not found. It should be in the `doc` folder.")

st.sidebar.markdown(f"**Version:** _{version}_")

with st.sidebar.expander("🔭 Observatory Data", expanded=False):
    st.session_state.observatory_name = st.text_input(
        "Observatory Name",
        value=st.session_state.observatory_name,
        help="Name of the observatory (e.g., TJMS).",
    )

    # Latitude input with locale handling
    latitude_input = st.text_input(
        "Latitude (degrees)",
        value=str(st.session_state.observatory_latitude),
        help="Observatory latitude in decimal degrees (North positive). Accepts both comma and dot as decimal separator.",
    )
    try:
        latitude = float(latitude_input.replace(",", "."))
        if -90 <= latitude <= 90:
            st.session_state.observatory_latitude = latitude
        else:
            st.error("Latitude must be between -90 and 90 degrees")
    except ValueError:
        if latitude_input.strip():  # Only show error if input is not empty
            st.error("Please enter a valid number for latitude.")

    # Longitude input with locale handling
    longitude_input = st.text_input(
        "Longitude (degrees)",
        value=str(st.session_state.observatory_longitude),
        help="Observatory longitude in decimal degrees (East positive). Accepts both comma and dot as decimal separator.",
    )
    try:
        longitude = float(longitude_input.replace(",", "."))
        if -180 <= longitude <= 180:
            st.session_state.observatory_longitude = longitude
        else:
            st.error("Longitude must be between -180 and 180 degrees")
    except ValueError:
        if longitude_input.strip():
            st.error("Please enter a valid number for longitude.")

    # Elevation input with locale handling
    elevation_input = st.text_input(
        "Elevation (meters)",
        value=str(st.session_state.observatory_elevation),
        help="Observatory elevation above sea level in meters. Accepts both comma and dot as decimal separator.",
    )
    try:
        elevation = float(elevation_input.replace(",", "."))
        st.session_state.observatory_elevation = elevation
    except ValueError:
        if elevation_input.strip():
            st.error("Please enter a valid number for elevation.")

with st.sidebar.expander("⚙️ Analysis Parameters", expanded=False):
    st.session_state.analysis_parameters["seeing"] = st.slider(
        "Estimated Seeing (FWHM, arcsec)",
        min_value=1.0,
        max_value=6.0,
        value=st.session_state.analysis_parameters["seeing"],
        step=0.5,
        help=(
            "Initial guess for the Full Width at Half Maximum of stars in "
            "arcseconds. Will be refined."
        ),
    )
    st.session_state.analysis_parameters["threshold_sigma"] = st.slider(
        "Detection Threshold (sigma)",
        min_value=1.0,
        max_value=6.0,
        value=st.session_state.analysis_parameters["threshold_sigma"],
        step=0.5,
        help=("Source detection threshold in units of background standard deviation."),
    )
    st.session_state.analysis_parameters["detection_mask"] = st.number_input(
        "Border Mask Size (pixels)",
        min_value=0,
        max_value=200,
        value=st.session_state.analysis_parameters["detection_mask"],
        step=5,
        help=(
            "Size of the border region (in pixels) to exclude from source detection."
        ),
    )
    # Use GAIA_BANDS (list of (label, field)) to present a friendly menu
    try:
        band_labels = [b[0] for b in GAIA_BANDS]
        band_fields = [b[1] for b in GAIA_BANDS]
    except Exception:
        band_labels = [str(b) for b in GAIA_BANDS]
        band_fields = band_labels

    # Determine default selection index based on stored filter field
    current_field = st.session_state.analysis_parameters.get(
        "filter_band", band_fields[0] if band_fields else "phot_g_mean_mag"
    )
    if current_field in band_fields:
        default_index = band_fields.index(current_field)
    else:
        default_index = 0

    selected_label = st.selectbox(
        "Calibration Filter Band",
        options=band_labels,
        index=default_index,
        help="Filter Magnitude band used for photometric calibration.",
    )

    # Store the actual catalog field name (second element) in session state
    try:
        selected_field = band_fields[band_labels.index(selected_label)]
    except Exception:
        selected_field = current_field

    st.session_state.analysis_parameters["filter_band"] = selected_field
    st.session_state.analysis_parameters["filter_max_mag"] = st.slider(
        "Max Calibration Mag",
        min_value=15.0,
        max_value=21.0,
        value=st.session_state.analysis_parameters["filter_max_mag"],
        step=0.5,
        help="Faintest magnitude to use for calibration stars.",
    )
    st.session_state.analysis_parameters["astrometry_check"] = st.toggle(
        "Astrometry check",
        value=st.session_state.analysis_parameters["astrometry_check"],
        help=(
            "Attempt to plate solve and refine WCS before photometry. "
        ),
    )

with st.sidebar.expander("🔑 API Keys", expanded=False):
    st.session_state.colibri_api_key = st.text_input(
        "Astro-Colibri UID Key (Optional)",
        value=st.session_state.get("colibri_api_key", ""),
        type="password",
        help="key for Astro-Colibri query",
    )
    st.markdown("[Get your key](https://www.astro-colibri.science)")

# Add expander for the Transient Finder
with st.sidebar.expander("Transient Candidates (_beta phase_)", expanded=False):

    # Add a checkbox to enable/disable the transient finder
    st.session_state.analysis_parameters['run_transient_finder'] = st.checkbox(
        "Enable Transient Finder",
        value=st.session_state.analysis_parameters.get('run_transient_finder', False)
    )

    # Add survey and filter selection
    survey_options = ["PanSTARRS"]
    if "DSS2" in st.session_state.analysis_parameters.get('transient_survey', 'PanSTARRS'):
        survey_index = 0
    else:
        survey_index = survey_options.index(st.session_state.analysis_parameters.get('transient_survey', 'PanSTARRS'))
    st.session_state.analysis_parameters['transient_survey'] = st.selectbox(
        "Reference Survey",
        options=survey_options,
        index=survey_index,
        help="Survey to use for the reference image (PanSTARRS has a smaller field of view limit).",
    )

    filter_options = ["g", "r", "i"]
    if "Red" or "Blue" in st.session_state.analysis_parameters.get('transient_filter', 'g'):
        filter_index = 0
    else:
        filter_index = filter_options.index(st.session_state.analysis_parameters.get('transient_filter', 'g'))
    st.session_state.analysis_parameters['transient_filter'] = st.selectbox(
        "Reference Filter",
        options=filter_options,
        index=filter_index,
        help="Filter/band for the reference image. Options depend on the selected survey.",
    )

if st.sidebar.button("💾 Save Configuration"):
    analysis_params = dict(st.session_state.get("analysis_parameters", {}))
    observatory_params = dict(st.session_state.get("observatory_data", {}))
    colibri_api_key = st.session_state.get("colibri_api_key")

    # Add pre-process options to analysis_parameters
    params = {
        "analysis_parameters": analysis_params,
        "observatory_data": observatory_params,
        "colibri_api_key": colibri_api_key,
    }
    name = st.session_state.get("username", "user")

    try:
        backend_url = "http://localhost:5000/save_config"
        resp = requests.post(
            backend_url,
            json={"username": name, "config_json": json.dumps(params)},
        )
        if resp.status_code != 200:
            st.sidebar.warning(f"Could not save config to DB: {resp.text}")
    except Exception as e:
        st.sidebar.warning(f"Could not connect to backend: {e}")

# Add archived files browser to sidebar
with st.sidebar.expander("📁 Archived Results", expanded=False):
    username = st.session_state.get("username", "anonymous")
    output_dir = ensure_output_directory(directory=f"{username}_results")
    display_archived_files_browser(output_dir)

with st.sidebar:
    if st.button("🧹 Clear Cache & Reset Upload"):
        clear_all_caches()

# Add logout button at the top right if user is logged in
if st.session_state.logged_in:
    st.sidebar.markdown("")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Logged in as:** _{st.session_state.username}_")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("Logged out successfully.")
        st.switch_page("pages/login.py")
    st.sidebar.markdown("---")
    st.sidebar.markdown("_Report feedback and bugs to_ : [rpp_support](mailto:rpp_support@saf-astronomie.fr)")

###########################################################################

# Persistent uploader: keep uploaded file bytes across reruns until cleared
if "uploaded_bytes" not in st.session_state:
    uploaded = st.file_uploader(
        "Choose a FITS file for analysis",
        type=["fits", "fit", "fts", "fits.gz"],
        key="science_uploader",
    )
    if uploaded:
        try:
            st.session_state["uploaded_bytes"] = uploaded.getvalue()
            st.session_state["uploaded_name"] = uploaded.name
            st.session_state["uploaded"] = uploaded
            science_file = uploaded
        except Exception:
            st.error("Failed to persist uploaded file. Please re-upload.")
            science_file = None
    else:
        science_file = None
else:
    tmp = SimpleNamespace()
    tmp._bytes = st.session_state.get("uploaded_bytes")

    def _read():
        return tmp._bytes

    def _getvalue():
        return tmp._bytes

    tmp.read = _read
    tmp.getvalue = _getvalue
    tmp.name = st.session_state.get("uploaded_name", "uploaded.fits")
    science_file = tmp

science_file_path = None

# Update observatory_data dictionary with current session state values
st.session_state.observatory_data = {
    "name": st.session_state.observatory_name,
    "latitude": st.session_state.observatory_latitude,
    "longitude": st.session_state.observatory_longitude,
    "elevation": st.session_state.observatory_elevation,
}

catalog_name = f"{st.session_state['base_filename']}_catalog.csv"
username = st.session_state.get("username", "anonymous")
output_dir = ensure_output_directory(directory=f"{username}_results")
st.session_state["output_dir"] = output_dir

if st.session_state.get("final_phot_table") is not None:
    provide_download_buttons(output_dir)
    zip_results_on_exit(science_file, output_dir)


if science_file is not None:
    suffix = os.path.splitext(science_file.name)[1]

    # Get the system temp directory
    system_tmp = tempfile.gettempdir()
    username = st.session_state.get("username", "anonymous")
    user_tmp_dir = os.path.join(system_tmp, username)

    # Create the user-specific temp directory if it doesn't exist
    os.makedirs(user_tmp_dir, exist_ok=True)

    # Now use this directory for the temp file
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=user_tmp_dir
    ) as tmp_file:
        tmp_file.write(science_file.getvalue())
        science_file_path = tmp_file.name

    st.session_state["science_file_path"] = science_file_path
    st.session_state.files_loaded["science_file"] = science_file

    base_filename = get_base_filename(science_file)
    st.session_state["base_filename"] = base_filename
    st.session_state["log_buffer"] = initialize_log(science_file.name)

if science_file is not None:
    with st.spinner("Loading FITS data..."):
        raw_data, science_header = load_fits_data(science_file)
        science_data = raw_data

    if science_data is not None and science_header is not None:
        st.success(f"Loaded '{science_file.name}' successfully.")
        write_to_log(
            st.session_state.log_buffer, f"Loaded FITS file: {science_file.name}"
        )

        # Update observatory data from FITS header if available
        result = update_observatory_from_fits_header(science_header)
        if result:
            st.info("Observatory information updated from FITS header")
            write_to_log(
                st.session_state.log_buffer,
                f"Observatory data updated from FITS header: {st.session_state.observatory_data}",
            )

    # Test WCS creation with better error handling
    wcs_obj, wcs_error, log_messages = safe_wcs_create(science_header)
    handle_log_messages(log_messages)
    # Initialize force_plate_solve as False by default
    force_plate_solve = st.session_state.get("astrometry_check", False)

    # If WCS creation fails due to singular matrix, try to proceed without WCS for detection
    proceed_without_wcs = False
    if wcs_obj is None and "singular" in str(wcs_error).lower():
        st.warning(f"WCS header has issues: {wcs_error}")
        st.info(
            "Will attempt source detection without WCS and then try plate solving..."
        )
        proceed_without_wcs = True
    elif wcs_obj is None:
        st.warning(f"No valid WCS found : {wcs_error}")
        st.write("Attempt plate solving...")
        use_astrometry = True
    else:
        st.success("Valid WCS found.")

        if force_plate_solve:
            st.info("Astrometry check enabled - will re-solve astrometry")
            use_astrometry = True
            wcs_obj = None  # Reset to trigger plate solving
        else:
            log_buffer = st.session_state["log_buffer"]
            write_to_log(log_buffer, "Valid WCS found in header")
            proceed_without_wcs = False
            use_astrometry = False

    # Handle plate solving
    if (wcs_obj is None and not proceed_without_wcs) or force_plate_solve:
        use_astrometry = True
        if use_astrometry:
            plate_solve_reason = (
                "forced by user" if force_plate_solve else "no valid WCS found"
            )
            with st.spinner(
                "Running astrometry check and plate solving - this may take a while..."
            ):
                wcs_obj, science_header, log_messages, error = solve_with_astrometrynet(science_file_path)
                handle_log_messages(log_messages)
                if error:
                    st.error(error)

                if wcs_obj is None:
                    if force_plate_solve:
                        st.error(
                            "Plate solving failed. Will use original WCS if available."
                        )
                        # Try to restore original WCS by reloading header
                        _, original_header = load_fits_data(science_file)
                        wcs_obj, wcs_error, _ = safe_wcs_create(original_header)
                        if wcs_obj is not None:
                            science_header = original_header
                            st.info("Restored original WCS solution")
                        else:
                            proceed_without_wcs = True
                    else:
                        st.error("Plate solving failed. No WCS solution was returned.")
                        proceed_without_wcs = True
                else:
                    st.session_state["calibrated_header"] = science_header
                    st.session_state["wcs_obj"] = wcs_obj

                log_buffer = st.session_state["log_buffer"]

                if wcs_obj is not None:
                    solve_type = (
                        "Forced plate-solve" if force_plate_solve else "Initial solve"
                    )
                    st.success(f"{solve_type} successful!")
                    write_to_log(
                        log_buffer, f"Astrometry check completed ({plate_solve_reason})"
                    )

                    wcs_header_filename = (
                        f"{st.session_state['base_filename']}_wcs_header"
                    )
                    wcs_header_file_path = save_header_to_txt(
                        science_header, wcs_header_filename, output_dir
                    )
                    if wcs_header_file_path:
                        st.info("Updated WCS header saved")

                    # Re-extract pixel scale and recalculate seeing with updated header
                    pixel_size_arcsec, pixel_scale_source = extract_pixel_scale(
                        science_header
                    )
                    seeing = st.session_state.analysis_parameters["seeing"]
                    mean_fwhm_pixel = seeing / pixel_size_arcsec

                    # Update session state with the new values
                    st.session_state["pixel_size_arcsec"] = pixel_size_arcsec
                    st.session_state["mean_fwhm_pixel"] = mean_fwhm_pixel

                    write_to_log(
                        log_buffer,
                        f"Updated pixel scale: {pixel_size_arcsec:.2f} arcsec/pixel ({pixel_scale_source})",
                    )
                    write_to_log(
                        log_buffer,
                        f"Updated seeing FWHM: {seeing:.2f} arcsec ({mean_fwhm_pixel:.2f} pixels)",
                    )
                else:
                    st.error("Plate solving failed")
                    write_to_log(
                        log_buffer,
                        "Failed",
                        level="ERROR",
                    )
                    proceed_without_wcs = True

    # Store whether we're proceeding without WCS
    st.session_state["proceed_without_wcs"] = proceed_without_wcs
    if proceed_without_wcs:
        st.warning(
            "Proceeding without valid WCS - photometry will be limited to instrumental magnitudes"
        )

    else:
        # Extract pixel scale from existing header when not re-solving
        pixel_size_arcsec, pixel_scale_source = extract_pixel_scale(science_header)
        seeing = st.session_state.analysis_parameters["seeing"]
        mean_fwhm_pixel = seeing / pixel_size_arcsec

        # Update session state with the calculated values
        st.session_state["pixel_size_arcsec"] = pixel_size_arcsec
        st.session_state["mean_fwhm_pixel"] = mean_fwhm_pixel

    if science_header is not None:
        log_buffer = st.session_state["log_buffer"]

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        header_filename = f"{st.session_state['base_filename']}_header"
        header_file_path = os.path.join(output_dir, f"{header_filename}.txt")

        header_file = save_header_to_txt(science_header, header_filename, output_dir)
        if header_file:
            write_to_log(log_buffer, f"Saved header to {header_file}")

    log_buffer = st.session_state["log_buffer"]
    write_to_log(log_buffer, f"Loaded Image: {science_file.name}")

    if science_data is not None:
        st.header("Image", anchor="science-image")
        try:
            # Create a side-by-side plot using matplotlib subplots
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(2 * FIGURE_SIZES["medium"][0], FIGURE_SIZES["medium"][1])
            )

            ax1.set_title("ZScale Visualization")
            try:
                norm = ImageNormalize(science_data, interval=ZScaleInterval())
                im1 = ax1.imshow(
                    science_data, norm=norm, origin="lower", cmap="viridis"
                )
            except Exception as norm_error:
                st.warning(
                    f"ZScale normalization failed: {norm_error}. Using simple normalization."
                )
                vmin, vmax = np.percentile(science_data, [1, 99])
                im1 = ax1.imshow(
                    science_data, vmin=vmin, vmax=vmax, origin="lower", cmap="viridis"
                )
            fig.colorbar(im1, ax=ax1, label="Pixel Value")
            ax1.axis("off")

            # Histogram Equalization
            ax2.set_title("Histogram Equalization")
            try:
                data_finite = science_data[np.isfinite(science_data)]
                if len(data_finite) > 0:
                    vmin, vmax = np.percentile(data_finite, [0.5, 99.5])
                    data_scaled = np.clip(science_data, vmin, vmax)
                    data_scaled = (data_scaled - vmin) / (vmax - vmin)
                    im2 = ax2.imshow(data_scaled, origin="lower", cmap="viridis")
                    fig.colorbar(im2, ax=ax2, label="Normalized Value")
                    ax2.axis("off")

                    ax_inset = fig.add_axes([0.65, 0.15, 0.15, 0.25])
                    ax_inset.hist(
                        data_scaled.flatten(), bins=50, color="skyblue", alpha=0.8
                    )
                    ax_inset.set_title("Pixel Distribution", fontsize=8)
                    ax_inset.tick_params(axis="both", which="major", labelsize=6)
                else:
                    st.warning(
                        "No finite values in image data for histogram equalization"
                    )
            except Exception as hist_error:
                st.warning(f"Histogram equalization failed: {hist_error}")

            fig.suptitle(f"{science_file.name}")
            st.pyplot(fig)

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"{st.session_state['base_filename']}_image.png"
            image_path = os.path.join(output_dir, image_filename)

            try:
                fig.savefig(image_path, dpi=120, bbox_inches="tight")
                write_to_log(log_buffer, "Saved image plot")
            except Exception as save_error:
                write_to_log(
                    log_buffer,
                    f"Failed to save image plot: {str(save_error)}",
                    level="ERROR",
                )
                st.error(f"Error saving image: {str(save_error)}")

            plt.close(fig)

        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
            st.exception(e)

        with st.expander("Image Header"):
            if science_header:
                st.text(repr(science_header))
            else:
                st.warning("No header information available for Image.")

    st.subheader("Statistics")
    if science_data is not None:
        st.write("Mean: ", f"{np.mean(science_data):.2f}", )
        st.write("Median: ", f"{np.median(science_data):.2f}")
        st.write("Rms: ", f"{np.std(science_data):.3f}")
        st.write("Min: ", f"{np.min(science_data):.2f}")
        st.write("Max: ", f"{np.max(science_data):.2f}")

        # Write statistics to log
        write_to_log(log_buffer, "Image Statistics", level="INFO")
        write_to_log(log_buffer, f"Mean: {np.mean(science_data):.2f}")
        write_to_log(log_buffer, f"Median: {np.median(science_data):.2f}")
        write_to_log(log_buffer, f"RMS: {np.std(science_data):.3f}")
        write_to_log(log_buffer, f"Min: {np.min(science_data):.2f}")
        write_to_log(log_buffer, f"Max: {np.max(science_data):.2f}")

        # Use updated header if available, otherwise use original
        header_for_stats = st.session_state.get("calibrated_header", science_header)

        if (
            "pixel_size_arcsec" in st.session_state
            and "mean_fwhm_pixel" in st.session_state
        ):
            pixel_size_arcsec = st.session_state["pixel_size_arcsec"]
            mean_fwhm_pixel = st.session_state["mean_fwhm_pixel"]
            pixel_scale_source = "session_state"
        else:
            pixel_size_arcsec, pixel_scale_source = extract_pixel_scale(
                header_for_stats
            )
            seeing = st.session_state.analysis_parameters["seeing"]
            mean_fwhm_pixel = seeing / pixel_size_arcsec

            st.session_state["pixel_size_arcsec"] = pixel_size_arcsec
            st.session_state["mean_fwhm_pixel"] = mean_fwhm_pixel

        st.write(
            "Mean Pixel Scale (arcsec): ",
            f"{pixel_size_arcsec:.2f}",
        )
        write_to_log(
            log_buffer,
            f"Final pixel scale: {pixel_size_arcsec:.2f} arcsec/pixel ({pixel_scale_source})",
        )
        seeing = st.session_state.analysis_parameters["seeing"]
        st.write("Mean FWHM (pixels): ", f"{mean_fwhm_pixel:.2f}")
        write_to_log(
            log_buffer,
            f"Final seeing FWHM: {seeing:.2f} arcsec ({mean_fwhm_pixel:.2f} pixels)",
        )

        ra_val, dec_val, coord_source = extract_coordinates(science_header)
        if ra_val is not None and dec_val is not None:
            st.write(f"RA={round(ra_val, 4)}°\n"
                     f"DEC={round(dec_val, 4)}°")
            write_to_log(
                log_buffer,
                f"Target coordinates: RA={ra_val}°, DEC={dec_val}° ({coord_source})",
            )
            ra_missing = dec_missing = False
            ra_val = dec_val = None  # Add this line
        else:
            st.warning(f"Coordinate issue: {coord_source}")
            write_to_log(
                log_buffer, f"Coordinate issue: {coord_source}", level="WARNING"
            )
            ra_missing = dec_missing = True

        if "manual_ra" not in st.session_state:
            st.session_state["manual_ra"] = ""
        if "manual_dec" not in st.session_state:
            st.session_state["manual_dec"] = ""

        if ra_missing or dec_missing:
            st.warning(
                "Target coordinates (RA/DEC) not found in FITS header. Please enter them manually:"
            )

            coord_col1, coord_col2 = st.columns(2)

            with coord_col1:
                default_ra = st.session_state["manual_ra"]
                if (
                    not default_ra
                    and science_header is not None
                    and "NAXIS1" in science_header
                    and "CRPIX1" in science_header
                    and "CD1_1" in science_header
                ):
                    default_ra = str(science_header.get("CRVAL1", ""))

                manual_ra = st.text_input(
                    "Right Ascension (degrees)",
                    value=default_ra,
                    help="Enter RA in decimal degrees (0-360)",
                    key="ra_input",
                )
                st.session_state["manual_ra"] = manual_ra

            with coord_col2:
                default_dec = st.session_state["manual_dec"]
                if (
                    not default_dec
                    and science_header is not None
                    and "NAXIS2" in science_header
                    and "CRPIX2" in science_header
                    and "CD2_2" in science_header
                ):
                    default_dec = str(science_header.get("CRVAL2", ""))

                manual_dec = st.text_input(
                    "Declination (degrees)",
                    value=default_dec,
                    help="Enter DEC in decimal degrees (-90 to +90)",
                    key="dec_input",
                )
                st.session_state["manual_dec"] = manual_dec

            if manual_ra and manual_dec:
                try:
                    ra_val = float(manual_ra)
                    dec_val = float(manual_dec)

                    if not (0 <= ra_val < 360):
                        st.error("RA must be between 0 and 360 degrees")
                    elif not (-90 <= dec_val <= 90):
                        st.error("DEC must be between -90 and +90 degrees")
                    else:
                        if science_header is not None:
                            science_header["RA"] = ra_val
                            science_header["DEC"] = dec_val

                        st.session_state["valid_ra"] = ra_val
                        st.session_state["valid_dec"] = dec_val

                        st.success(
                            f"Using manual coordinates: RA={ra_val}°, DEC={dec_val}°"
                        )
                except ValueError:
                    st.error("RA and DEC must be valid numbers")
            else:
                st.warning("Please enter both RA and DEC coordinates")
        else:
            if science_header is not None:
                for ra_key in ["RA", "OBJRA", "RA---", "CRVAL1"]:
                    if ra_key in science_header:
                        st.session_state["valid_ra"] = science_header[ra_key]
                        break

                for dec_key in ["DEC", "OBJDEC", "DEC---", "CRVAL2"]:
                    if dec_key in science_header:
                        st.session_state["valid_dec"] = science_header[dec_key]
                        break

        if (
            "valid_ra" in st.session_state
            and "valid_dec" in st.session_state
            and science_header is not None
        ):
            science_header["RA"] = st.session_state["valid_ra"]
            science_header["DEC"] = st.session_state["valid_dec"]

        try:
            # Create observatory dictionary using user inputs
            observatory_data = {
                "name": st.session_state.observatory_data["name"],
                "latitude": st.session_state.observatory_data["latitude"],
                "longitude": st.session_state.observatory_data["longitude"],
                "elevation": st.session_state.observatory_data["elevation"],
            }

            air = airmass(science_header, observatory=observatory_data)
            st.write(f"Airmass : {air:.2f}")
        except Exception as e:
            st.warning(f"Error calculating airmass: {e}")
            air = 0.0
            st.write(f"Using default airmass: {air:.2f}")

        zero_point_button_disabled = science_file is None

        if st.button(
                "Photometric Calibration",
                disabled=zero_point_button_disabled,
                key="run_zp"):

            image_to_process = science_data
            header_to_process = science_header.copy()

            # Validate that we have a header
            if header_to_process is None:
                st.error(
                    "No header available for processing. Cannot proceed with photometry."
                )
                st.stop()

            # Get the correct pixel scale and FWHM values
            if (
                "pixel_size_arcsec" in st.session_state
                and "mean_fwhm_pixel" in st.session_state
            ):
                pixel_size_arcsec = st.session_state["pixel_size_arcsec"]
                mean_fwhm_pixel = st.session_state["mean_fwhm_pixel"]
            else:
                pixel_size_arcsec, _ = extract_pixel_scale(header_to_process)
                seeing = st.session_state.analysis_parameters["seeing"]
                mean_fwhm_pixel = seeing / pixel_size_arcsec

            if image_to_process is not None:
                try:
                    with st.spinner(
                        "Background Extraction, FWHM Computation, Sources Detection and Photometry..."
                    ):
                        # Extract parameters from session state
                        threshold_sigma = st.session_state.analysis_parameters[
                            "threshold_sigma"
                        ]
                        detection_mask = st.session_state.analysis_parameters[
                            "detection_mask"
                        ]
                        filter_band = st.session_state.analysis_parameters[
                            "filter_band"
                        ]
                        filter_max_mag = st.session_state.analysis_parameters[
                            "filter_max_mag"
                        ]

                        result = detection_and_photometry(
                            image_to_process,
                            header_to_process,
                            mean_fwhm_pixel,
                            threshold_sigma,
                            detection_mask
                        )

                        if isinstance(result, tuple) and len(result) == 6:
                            phot_table_qtable, epsf_table, daofind, bkg, w, bkg_fig = result
                        else:
                            st.error(
                                f"detection_and_photometry returned unexpected result: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}"
                            )
                            phot_table_qtable = epsf_table = daofind = bkg = w = bkg_fig = None

                        if phot_table_qtable is not None:
                            phot_table_df = phot_table_qtable.to_pandas().copy(
                                deep=True
                            )
                        else:
                            st.error("No sources detected in the image.")
                            phot_table_df = None

                    if phot_table_df is not None:
                        with st.spinner("Cross-matching with Gaia..."):
                            matched_table, log_messages = cross_match_with_gaia(
                                phot_table_qtable,
                                header_to_process,
                                pixel_size_arcsec,
                                mean_fwhm_pixel,
                                filter_band,
                                filter_max_mag,
                                refined_wcs=w,
                            )
                            handle_log_messages(log_messages)

                        if matched_table is not None:
                            st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
                            st.dataframe(matched_table.head(10))

                            with st.spinner("Calculating zero point..."):
                                zero_point_value, zero_point_std, zp_plot = (
                                    calculate_zero_point(
                                        phot_table_qtable,
                                        matched_table,
                                        filter_band,
                                        air,
                                    )
                                )

                                if zero_point_value is not None:
                                    write_to_log(
                                        log_buffer,
                                        f"Zero point: {zero_point_value:.2f} ± {zero_point_std:.2f}",
                                    )
                                    write_to_log(log_buffer, f"Airmass: {air:.2f}")

                                    try:
                                        if "epsf_photometry_result" in st.session_state and epsf_table is not None:

                                            # Convert tables to pandas DataFrames
                                            epsf_df = epsf_table.to_pandas() if not isinstance(epsf_table, pd.DataFrame) else epsf_table
                                            final_table = st.session_state["final_phot_table"]

                                            # Merge keeping all sources
                                            final_table, log_messages = merge_photometry_catalogs(
                                                aperture_table=final_table,
                                                psf_table=epsf_df,
                                                tolerance_pixels=2.
                                            )
                                            handle_log_messages(log_messages)

                                            # Add calibrated magnitudes
                                            final_table = add_calibrated_magnitudes(
                                                final_table,
                                                zero_point=zero_point_value,
                                                airmass=air
                                            )

                                            # Add metadata
                                            final_table['zero_point'] = zero_point_value
                                            final_table['zero_point_std'] = zero_point_std
                                            final_table['airmass'] = air

                                            # Clean the table
                                            final_table, log_messages = clean_photometry_table(final_table, require_magnitude=True)
                                            handle_log_messages(log_messages)

                                            # Save to session state
                                            st.session_state["final_phot_table"] = final_table
                                            st.success(f"Catalog includes {len(final_table)} sources.")

                                    except Exception as e:
                                        st.error(f"Error merging photometry: {e}")
                                        import traceback
                                        st.code(traceback.format_exc())

                                    st.subheader("Final Photometry Catalog")
                                    st.dataframe(final_table.head(10))

                                    st.success(
                                        f"Catalog includes {len(final_table)} sources."
                                    )

                                    st.subheader(
                                        "Magnitude Distribution (Aperture & PSF)"
                                    )

                                    fig_mag = plot_magnitude_distribution(
                                        final_table, log_buffer
                                    )
                                    st.pyplot(fig_mag)

                                    try:
                                        base_filename = st.session_state.get(
                                            "base_filename", "photometry"
                                        )
                                        username = st.session_state.get(
                                            "username", "anonymous"
                                        )
                                        output_dir = ensure_output_directory(
                                            directory=f"{username}_results"
                                        )
                                        hist_filename = (
                                            f"{base_filename}_histogram_mag.png"
                                        )
                                        hist_filepath = os.path.join(
                                            output_dir, hist_filename
                                        )
                                        fig_mag.savefig(
                                            hist_filepath,
                                            dpi=120,
                                            bbox_inches="tight",
                                        )
                                        write_to_log(
                                            log_buffer,
                                            f"Saved magnitude histogram plot: {hist_filename}",
                                        )
                                    except Exception as e:
                                        st.warning(
                                            f"Could not save magnitude histogram plot: {e}"
                                        )

                                    # Clean up the figure
                                    plt.close(fig_mag)

                                    if (
                                        final_table is not None
                                        and "ra" in final_table.columns
                                        and "dec" in final_table.columns
                                    ):
                                        st.subheader(
                                            "Cross-matching with Astronomical Catalogs"
                                        )
                                        search_radius = (
                                            max(
                                                header_to_process["NAXIS1"],
                                                header_to_process["NAXIS2"],
                                            )
                                            * pixel_size_arcsec
                                            / 2.0
                                        )
                                        # Get colibri API key from session state
                                        colibri_api_key = st.session_state.get(
                                            "colibri_api_key", ""
                                        )
                                        # Ensure final_table exists
                                        if final_table is None:
                                            st.error(
                                                "Final photometry table is None - cannot perform catalog enhancement"
                                            )
                                            final_table = st.session_state.get(
                                                "final_phot_table"
                                            )

                                        if (
                                            final_table is not None
                                            and len(final_table) > 0
                                        ):
                                            final_table, log_messages = enhance_catalog(
                                                colibri_api_key,
                                                final_table,
                                                matched_table,
                                                header_to_process,
                                                pixel_size_arcsec,
                                                search_radius_arcsec=search_radius,
                                            )
                                            handle_log_messages(log_messages)
                                    elif final_table is not None:
                                        st.warning(
                                            "RA/DEC coordinates not available for catalog cross-matching"
                                        )
                                    else:
                                        st.error(
                                            "Final photometry table is None - cannot perform cross-matching"
                                        )

                                    # Call the new function here
                                    success_messages, error_messages = save_catalog_files(
                                        final_table, catalog_name, output_dir
                                    )
                                    for msg in success_messages:
                                        st.success(msg)
                                    for msg in error_messages:
                                        st.error(msg)

                except Exception as e:
                    st.error(f"Error during zero point calibration: {str(e)}")
                    st.exception(e)

                ra_center = None
                dec_center = None

                # Use the correct header - prioritize calibrated header if available
                header_for_coords = st.session_state.get(
                    "calibrated_header", science_header
                )
                if header_for_coords is None:
                    header_for_coords = science_header

                coord_keys = [("CRVAL1", "CRVAL2"), ("RA", "DEC"), ("OBJRA", "OBJDEC")]

                for ra_key, dec_key in coord_keys:
                    if (
                        header_for_coords is not None
                        and ra_key in header_for_coords
                        and dec_key in header_for_coords
                    ):
                        try:
                            ra_center = float(header_for_coords[ra_key])
                            dec_center = float(header_for_coords[dec_key])
                            if ra_center is not None and dec_center is not None:
                                break
                        except (ValueError, TypeError):
                            continue

                # Fallback to session state coordinates if header failed
                if ra_center is None or dec_center is None:
                    ra_center = st.session_state.get("valid_ra")
                    dec_center = st.session_state.get("valid_dec")

                if ra_center is not None and dec_center is not None:
                    # Check if we have a valid final photometry table
                    final_phot_table = st.session_state.get("final_phot_table")

                    # Run Transient Finder if enabled
                    if st.session_state.analysis_parameters.get("run_transient_finder"):
                        if (
                            "science_file_path" in st.session_state
                            and st.session_state["science_file_path"]
                        ):
                            with st.spinner(
                                "Running Image Subtraction... This may take a moment."
                            ):
                                try:
                                    candidates = find_candidates(
                                                science_data,
                                                header_for_coords,
                                                mean_fwhm_pixel,
                                                pixel_size_arcsec,
                                                ra_center,
                                                dec_center,
                                                search_radius/3600,
                                                mask=None,
                                                catalog=st.session_state.analysis_parameters.get("transient_survey", "PanSTARRS"),
                                                filter_name=st.session_state.analysis_parameters.get("transient_filter", "r"),
                                                mag_limit='<20',
                                            )
                                    if candidates:
                                        st.subheader("Transient Candidates Found")
                                        for idx, cand in enumerate(candidates,
                                                                start=1):
                                            st.markdown(f"**Candidate {idx}:** RA={cand['ra']:.6f}°, DEC={cand['dec']:.6f}°, Mag={cand.get('mag', 'N/A')}, ")
                                    else:
                                        st.warning("No transient candidates found.")
                                except Exception as e:
                                    st.error(f"Error during transient finding: {str(e)}")

                    if (
                        final_phot_table is not None
                        and not final_phot_table.empty
                        and len(final_phot_table) > 0
                    ):
                        st.subheader("Aladin Catalog Viewer")

                        try:
                            display_catalog_in_aladin(
                                final_table=final_phot_table,
                                ra_center=ra_center,
                                dec_center=dec_center,
                                id_cols=[
                                    "id",
                                    "simbad_main_id",
                                    "skybot_NAME",
                                    "aavso_Name",
                                    "gaia_source_id",
                                    "astrocolibri_name",
                                    "qso_name",
                                ],
                            )
                        except Exception as e:
                            st.error(f"Error displaying Aladin viewer: {str(e)}")
                            st.info(
                                "Catalog data is available but cannot be displayed in interactive viewer."
                            )

                    st.link_button(
                        "ESA Sky Viewer",
                        f"https://sky.esa.int/esasky/?target={ra_center}%20{dec_center}&hips=DSS2+color&fov=1.0&projection=SIN&cooframe=J2000&sci=true&lang=en",
                        help="Open ESA Sky with the same target coordinates",
                    )

                    st.link_button(
                        "SIMBAD",
                        f"https://simbad.u-strasbg.fr/simbad/sim-coo?Coord={ra_center}+{dec_center}&Radius=5&Radius.unit=arcmin",
                        help="Open SIMBAD at CDS Strasbourg for these coordinates",
                    )

                    st.link_button(
                        "XMatch",
                        f"https://cdsxmatch.u-strasbg.fr/xmatch?request=doQuery&RA={ra_center}&DEC={dec_center}&radius=5",
                        help="Open CDS XMatch service for these coordinates",
                    )

                    # Only provide download buttons if processing was completed
                    if final_phot_table is not None:
                        provide_download_buttons(output_dir)
                        cleanup_temp_files()
                        zip_results_on_exit(science_file, output_dir)
                else:
                    st.warning(
                        "Could not determine coordinates from image header. Cannot display ESASky or Aladin Viewer."
                    )
else:
    st.text(
        "👆 Upload an Image FITS file to Start",
    )

if "log_buffer" in st.session_state and st.session_state["log_buffer"] is not None:
    log_buffer = st.session_state["log_buffer"]
    log_filename = f"{st.session_state['base_filename']}.log"
    log_filepath = os.path.join(output_dir, log_filename)

    # Get parameters from session state
    seeing = st.session_state.analysis_parameters["seeing"]
    threshold_sigma = st.session_state.analysis_parameters["threshold_sigma"]
    detection_mask = st.session_state.analysis_parameters["detection_mask"]
    filter_band = st.session_state.analysis_parameters["filter_band"]
    filter_max_mag = st.session_state.analysis_parameters["filter_max_mag"]

    write_to_log(
        log_buffer, f"Elevation: {st.session_state.observatory_data['elevation']} m"
    )

    write_to_log(log_buffer, "Gaia Parameters", level="INFO")
    write_to_log(log_buffer, f"Gaia Band: {filter_band}")
    write_to_log(log_buffer, f"Gaia Max Magnitude: {filter_max_mag}")

    write_to_log(log_buffer, "Calibration Options", level="INFO")

    # Finalize and save the log
    write_to_log(log_buffer, "Processing completed", level="INFO")
    with open(log_filepath, "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())
    write_to_log(log_buffer, f"Log saved to {log_filepath}")

