import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Standard Library Imports
import zipfile
import base64
import json
import tempfile
import warnings
import atexit
import subprocess
from datetime import datetime, timedelta
from urllib.parse import quote
from functools import partial
from io import StringIO, BytesIO
from typing import Union, Any, Optional, Dict, Tuple, List

# Third-Party Imports
import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# astroscrappy import moved to main_functions
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.stats import sigma_clip, SigmaClip, sigma_clipped_stats
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun
from astropy.time import Time
from astropy.modeling import models, fitting
from astropy.nddata import NDData
from astropy.visualization import (ZScaleInterval, ImageNormalize,
                                   PercentileInterval, simple_norm)
import astropy.units as u
# Astroquery imports moved to main_functions
# Photutils imports moved to main_functions
# stdpipe import moved to main_functions

# Local Application Imports
from tools import (FIGURE_SIZES, URL, GAIA_BAND, extract_coordinates,
                   extract_pixel_scale, get_base_filename, safe_catalog_query,
                   safe_wcs_create, ensure_output_directory, cleanup_temp_files,
                   initialize_log, write_to_log, zip_rpp_results_on_exit)
from __version__ import version
# Import moved functions using absolute import from project root
from main_functions import (
    solve_with_siril,
    detect_remove_cosmic_rays,
    make_border_mask,
    estimate_background,
    airmass,
    load_fits_data,
    fwhm_fit,
    perform_psf_photometry,
    detection_and_photometry,
    cross_match_with_gaia,
    calculate_zero_point,
    run_zero_point_calibration,
    enhance_catalog
)


# Conditional Import (already present, just noting its location)
if getattr(sys, "frozen", False):
        # Ensure importlib.metadata is available or handle appropriately
        try:
            import importlib.metadata
            importlib.metadata.distributions = lambda **kwargs: []
        except ImportError:
            st.warning("Could not modify importlib.metadata, potential issues in frozen mode.")


warnings.filterwarnings("ignore")


st.set_page_config(page_title="RAPAS Photometry Pipeline", page_icon="🔭",
                   layout="wide")


def initialize_session_state():
    """
    Initialize all session state variables for the application.
    Ensures all required keys have default values.
    """
    # Login/User State
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    # Core Data/Results State
    if "calibrated_header" not in st.session_state:
        st.session_state.calibrated_header = None
    if "final_phot_table" not in st.session_state:
        st.session_state.final_phot_table = None
    if "epsf_model" not in st.session_state:
        st.session_state.epsf_model = None
    if "epsf_photometry_result" not in st.session_state:
        st.session_state.epsf_photometry_result = None

    # Logging and File Handling State
    if "log_buffer" not in st.session_state:
        st.session_state.log_buffer = None
    if "base_filename" not in st.session_state:
        st.session_state.base_filename = "photometry"
    if "science_file_path" not in st.session_state:
        st.session_state.science_file_path = None
    if "output_dir" not in st.session_state:
        # Ensure the default output directory exists when initialized
        st.session_state.output_dir = ensure_output_directory("rpp_results")

    # Analysis Parameters State (consolidated)
    default_analysis_params = {
        "seeing": 3.0,
        "threshold_sigma": 3.0,
        "detection_mask": 25,
        "filter_band": "phot_g_mean_mag",  # Default Gaia band
        "filter_max_mag": 20.0,           # Default Gaia mag limit
        "astrometry_check": False,        # Astrometry refinement toggle
        "calibrate_cosmic_rays": False,   # CRR toggle
        "cr_gain": 1.0,                   # CRR default gain
        "cr_readnoise": 6.5,              # CRR default readnoise
        "cr_sigclip": 6.0,                # CRR default sigclip
    }
    if "analysis_parameters" not in st.session_state:
        st.session_state.analysis_parameters = default_analysis_params.copy()
    else:
        # Ensure all keys exist, adding defaults if missing from loaded config
        for key, value in default_analysis_params.items():
            if key not in st.session_state.analysis_parameters:
                st.session_state.analysis_parameters[key] = value

    # Observatory Parameters State
    default_observatory_data = {
        "name": "",
        "latitude": 0.,
        "longitude": 0.,
        "elevation": 0.,
    }
    if "observatory_data" not in st.session_state:
        st.session_state.observatory_data = default_observatory_data.copy()
    else:
        # Ensure all keys exist, adding defaults if missing
        for key, value in default_observatory_data.items():
            if key not in st.session_state.observatory_data:
                st.session_state.observatory_data[key] = value

    # Individual observatory keys for direct widget binding (synced later)
    if "observatory_name" not in st.session_state:
        st.session_state.observatory_name = st.session_state.observatory_data["name"]
    if "observatory_latitude" not in st.session_state:
        st.session_state.observatory_latitude = st.session_state.observatory_data["latitude"]
    if "observatory_longitude" not in st.session_state:
        st.session_state.observatory_longitude = st.session_state.observatory_data["longitude"]
    if "observatory_elevation" not in st.session_state:
        st.session_state.observatory_elevation = st.session_state.observatory_data["elevation"]

    # API Keys State
    if "colibri_api_key" not in st.session_state:
        st.session_state.colibri_api_key = None
    # File Loading State (Track which calibration files are loaded)
    if "files_loaded" not in st.session_state:
        st.session_state.files_loaded = {
            "science_file": None,
        }


# --- Initialize Session State Early ---
initialize_session_state()

# Redirect to login if not authenticated
if not st.session_state.logged_in:
    st.warning("You must log in to access this page.")
    st.switch_page("pages/login.py")  # Corrected path if login.py is in pages

# --- Main Application Title ---
st.title("🔭 RAPAS Photometry Pipeline")

# Add application version to the sidebar
st.sidebar.markdown(f"**App Version:** _{version}_")

# --- Sidebar Configuration ---
with st.sidebar.expander("🔭 Observatory Data", expanded=False):
    st.session_state.observatory_name = st.text_input(
        "Observatory Name",
        value=st.session_state.observatory_name,
        help="Name of the observatory (e.g., TJMS).",
    )
    st.session_state.observatory_latitude = st.number_input(
        "Latitude (degrees)",
        value=st.session_state.observatory_latitude,
        format="%.6f",
        help="Observatory latitude in decimal degrees (North positive).",
    )
    st.session_state.observatory_longitude = st.number_input(
        "Longitude (degrees)",
        value=st.session_state.observatory_longitude,
        format="%.6f",
        help="Observatory longitude in decimal degrees (East positive).",
    )
    st.session_state.observatory_elevation = st.number_input(
        "Elevation (meters)",
        value=st.session_state.observatory_elevation,
        format="%.1f",
        help="Observatory elevation above sea level in meters.",
    )

with st.sidebar.expander("⚙️ Analysis Parameters", expanded=False):
    st.session_state.analysis_parameters["seeing"] = st.number_input(
        "Estimated Seeing (FWHM, arcsec)",
        min_value=0.1,
        max_value=20.0,
        value=st.session_state.analysis_parameters["seeing"],
        step=0.1,
        format="%.1f",
        help="Initial guess for the Full Width at Half Maximum of stars in arcseconds. Will be refined.",
    )
    st.session_state.analysis_parameters["threshold_sigma"] = st.number_input(
        "Detection Threshold (sigma)",
        min_value=1.0,
        max_value=20.0,
        value=st.session_state.analysis_parameters["threshold_sigma"],
        step=0.5,
        format="%.1f",
        help="Source detection threshold in units of background standard deviation.",
    )
    st.session_state.analysis_parameters["detection_mask"] = st.number_input(
        "Border Mask Size (pixels)",
        min_value=0,
        max_value=500,
        value=st.session_state.analysis_parameters["detection_mask"],
        step=5,
        help="Size of the border region (in pixels) to exclude from source detection.",
    )
    st.session_state.analysis_parameters["filter_band"] = st.selectbox(
        "Calibration Filter Band (Gaia)",
        options=[
            "phot_g_mean_mag",
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
        ],
        index=[
            "phot_g_mean_mag",
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
        ].index(st.session_state.analysis_parameters["filter_band"]),
        help="Gaia magnitude band used for photometric calibration.",
    )
    st.session_state.analysis_parameters["filter_max_mag"] = st.number_input(
        "Max Calibration Mag (Gaia)",
        min_value=10.0,
        max_value=25.0,
        value=st.session_state.analysis_parameters["filter_max_mag"],
        step=0.5,
        format="%.1f",
        help="Faintest Gaia magnitude to use for calibration stars.",
    )
    st.session_state.analysis_parameters["astrometry_check"] = st.toggle(
        "Refine Astrometry (Stdpipe)",
        value=st.session_state.analysis_parameters["astrometry_check"],
        help="Attempt to refine WCS using detected sources before photometry (requires external solver like Siril or astrometry.net).",
    )
    st.session_state.analysis_parameters["calibrate_cosmic_rays"] = st.toggle(
        "Remove Cosmic Rays (Astroscrappy)",
        value=st.session_state.analysis_parameters["calibrate_cosmic_rays"],
        help="Detect and remove cosmic rays using the L.A.Cosmic algorithm.",
    )
    if st.session_state.analysis_parameters["calibrate_cosmic_rays"]:
        st.session_state.analysis_parameters["cr_gain"] = st.number_input("CRR Gain (e-/ADU)", value=st.session_state.analysis_parameters["cr_gain"], min_value=0.1)
        st.session_state.analysis_parameters["cr_readnoise"] = st.number_input("CRR Read Noise (e-)", value=st.session_state.analysis_parameters["cr_readnoise"], min_value=0.1)
        st.session_state.analysis_parameters["cr_sigclip"] = st.number_input("CRR Sigma Clip", value=st.session_state.analysis_parameters["cr_sigclip"], min_value=1.0)


with st.sidebar.expander("🔑 API Keys", expanded=False):
     st.session_state.colibri_api_key = st.text_input(
         "Colibri API Key (Optional)",
         value=st.session_state.get("colibri_api_key", ""),
         type="password",
         help="API key for Colibri GRB catalog queries."
     )


# Save Configuration Button
if st.sidebar.button("💾 Save Configuration"):
    save_config()

# Add quick links (Moved here)
st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Links")
st.sidebar.link_button("SIMBAD Database", "http://simbad.u-strasbg.fr/simbad/")
st.sidebar.link_button("VizieR Catalogues", "http://vizier.u-strasbg.fr/viz-bin/VizieR")
st.sidebar.link_button("Gaia Archive", "https://gea.esac.esa.int/archive/")
st.sidebar.link_button("Astro-Colibri", "https://astro-colibri.science/")
st.sidebar.link_button("RAPAS Project", "https://rapas.imcce.fr/")
st.sidebar.markdown("---")


# --- File Upload Section ---
uploaded_science_file = st.file_uploader(
    "Choose a FITS file for analysis", type=["fits", "fit"]
)

# --- Main Processing Logic ---
if uploaded_science_file is not None:
    # Initialize log buffer for this run
    st.session_state.log_buffer = StringIO()
    initialize_log(st.session_state.log_buffer, st.session_state.username)

    # Load data using the function from main_functions
    with st.spinner("Loading FITS data..."):
        normalized_data, raw_data, science_header = load_fits_data(uploaded_science_file)

    if raw_data is not None and science_header is not None:
        st.success(f"Loaded '{uploaded_science_file.name}' successfully.")
        write_to_log(st.session_state.log_buffer, f"Loaded FITS file: {uploaded_science_file.name}")

        # Store essential info in session state
        st.session_state.files_loaded["science_file"] = uploaded_science_file.name
        st.session_state.base_filename = get_base_filename(uploaded_science_file.name)
        st.session_state.science_file_path = os.path.join(tempfile.gettempdir(), uploaded_science_file.name) # Store path for potential external tools

        # Save uploaded file temporarily if needed by external tools like Siril
        try:
            with open(st.session_state.science_file_path, "wb") as f:
                f.write(uploaded_science_file.getbuffer())
            write_to_log(st.session_state.log_buffer, f"Saved temporary file to {st.session_state.science_file_path}")
            # Register cleanup for the temporary file
            atexit.register(cleanup_temp_files, [st.session_state.science_file_path])
        except Exception as e:
            st.error(f"Error saving temporary file: {e}")
            st.session_state.science_file_path = None # Invalidate path


        # --- Display Basic Image Info & Visualization ---
        st.subheader("Image Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Object: {science_header.get('OBJECT', 'N/A')}")
            st.write(f"Dimensions: {raw_data.shape}")
            st.write(f"Exposure Time: {science_header.get('EXPTIME', 'N/A')} s")
            st.write(f"Filter: {science_header.get('FILTER', 'N/A')}")
            st.write(f"Date Obs: {science_header.get('DATE-OBS', 'N/A')}")

            # Extract Pixel Scale
            pixel_scale_arcsec = extract_pixel_scale(science_header)
            if pixel_scale_arcsec:
                st.write(f"Pixel Scale: {pixel_scale_arcsec:.3f} arcsec/pixel")
                mean_fwhm_pixel = st.session_state.analysis_parameters["seeing"] / pixel_scale_arcsec
                st.write(f"Initial FWHM Estimate: {mean_fwhm_pixel:.2f} pixels")
            else:
                st.warning("Could not determine pixel scale from header. Using default FWHM.")
                pixel_scale_arcsec = 1.0 # Default or prompt user?
                mean_fwhm_pixel = 5.0 # Default guess in pixels

            # Calculate Airmass
            airmass_value, airmass_details = airmass(
                science_header, st.session_state.observatory_data, return_details=True
            )
            if airmass_value > 0:
                st.write(f"Calculated Airmass: {airmass_value:.2f}")
                if airmass_details:
                     st.write(f"Observation Type: {airmass_details.get('observation_type', 'N/A')}")
                     st.write(f"Sun Altitude: {airmass_details.get('sun_altitude', 'N/A')}°")
            else:
                st.warning("Could not calculate airmass from header/observatory data.")
                airmass_value = 1.0 # Assume default if calculation fails


        with col2:
            st.write("Image Preview (Normalized):")
            fig_preview, ax_preview = plt.subplots(figsize=FIGURE_SIZES["medium"])
            norm = simple_norm(normalized_data, 'sqrt', percent=99)
            ax_preview.imshow(normalized_data, cmap="gray", origin="lower", norm=norm)
            ax_preview.set_title(f"{st.session_state.base_filename}")
            ax_preview.set_xticks([])
            ax_preview.set_yticks([])
            st.pyplot(fig_preview)
            plt.close(fig_preview) # Close plot to free memory

        # --- Processing Steps ---
        image_to_process = raw_data.copy() # Start with raw data

        # 1. Cosmic Ray Removal (Optional)
        if st.session_state.analysis_parameters["calibrate_cosmic_rays"]:
            with st.spinner("Removing cosmic rays..."):
                cr_params = {k.replace('cr_', ''): v for k, v in st.session_state.analysis_parameters.items() if k.startswith('cr_')}
                cleaned_image, cr_mask = detect_remove_cosmic_rays(
                    image_to_process, **cr_params, verbose=False
                )
                if cr_mask is not None:
                    num_cr = np.sum(cr_mask)
                    st.success(f"Cosmic ray removal complete. Detected {num_cr} pixels.")
                    write_to_log(st.session_state.log_buffer, f"Removed {num_cr} cosmic ray pixels.")
                    image_to_process = cleaned_image # Update image for next steps

                    # Display CR mask (optional)
                    # fig_cr, ax_cr = plt.subplots()
                    # ax_cr.imshow(cr_mask, origin='lower', cmap='binary')
                    # ax_cr.set_title("Cosmic Ray Mask")
                    # st.pyplot(fig_cr)
                    # plt.close(fig_cr)
                else:
                    st.warning("Cosmic ray removal failed or was skipped.")
                    write_to_log(st.session_state.log_buffer, "Cosmic ray removal failed or skipped.")


        # 2. Astrometry Refinement (Optional, using Siril via PowerShell)
        calibrated_header = science_header.copy() # Start with original header
        if st.session_state.analysis_parameters["astrometry_check"]:
             if st.session_state.science_file_path and os.path.exists(st.session_state.science_file_path):
                 with st.spinner("Refining astrometry using Siril..."):
                     try:
                         wcs_obj_solved, solved_header = solve_with_siril(st.session_state.science_file_path)
                         if wcs_obj_solved and solved_header:
                             st.success("Astrometry refinement successful.")
                             write_to_log(st.session_state.log_buffer, "Astrometry refined using Siril.")
                             calibrated_header = solved_header # Use the updated header
                             st.session_state.calibrated_header = calibrated_header # Store solved header

                             # Update pixel scale and FWHM based on refined WCS
                             new_pixel_scale = extract_pixel_scale(calibrated_header)
                             if new_pixel_scale:
                                 pixel_scale_arcsec = new_pixel_scale
                                 mean_fwhm_pixel = st.session_state.analysis_parameters["seeing"] / pixel_scale_arcsec
                                 st.info(f"Updated Pixel Scale: {pixel_scale_arcsec:.3f} arcsec/pixel")
                                 st.info(f"Updated FWHM Estimate: {mean_fwhm_pixel:.2f} pixels")
                             else:
                                 st.warning("Could not extract pixel scale from refined header.")

                         else:
                             st.warning("Astrometry refinement with Siril failed. Using original WCS.")
                             write_to_log(st.session_state.log_buffer, "Astrometry refinement with Siril failed.")
                             st.session_state.calibrated_header = science_header # Store original if failed
                     except Exception as e:
                         st.error(f"Error during Siril execution: {e}")
                         write_to_log(st.session_state.log_buffer, f"Error during Siril execution: {e}")
                         st.session_state.calibrated_header = science_header # Store original on error
             else:
                 st.warning("Cannot refine astrometry: Science file path not available.")
                 write_to_log(st.session_state.log_buffer, "Skipped astrometry refinement: file path missing.")
                 st.session_state.calibrated_header = science_header # Store original if skipped
        else:
             st.session_state.calibrated_header = science_header # Store original if not requested


        # 3. Run Photometry and Calibration Pipeline
        st.header("Photometric Calibration")
        zp_value, zp_std, final_phot_table = run_zero_point_calibration(
            image_to_process=image_to_process,
            header=st.session_state.calibrated_header, # Use potentially updated header
            pixel_size_arcsec=pixel_scale_arcsec,
            mean_fwhm_pixel=mean_fwhm_pixel, # Use potentially updated FWHM
            threshold_sigma=st.session_state.analysis_parameters["threshold_sigma"],
            detection_mask=st.session_state.analysis_parameters["detection_mask"],
            filter_band=st.session_state.analysis_parameters["filter_band"],
            filter_max_mag=st.session_state.analysis_parameters["filter_max_mag"],
            air=airmass_value,
        )

        # Store results in session state (already done within run_zero_point_calibration)
        # st.session_state.final_phot_table = final_phot_table

        # 4. Enhance Catalog (Optional)
        if final_phot_table is not None:
            with st.spinner("Enhancing catalog (placeholder)..."):
                 # Retrieve matched table if needed for enhancement
                 # Note: matched_table is currently local to run_zero_point_calibration
                 # Need to either return it or re-run crossmatch if enhancement needs it.
                 # For now, passing None as matched_table.
                 enhanced_table = enhance_catalog(
                     api_key=st.session_state.get("colibri_api_key"),
                     final_table=final_phot_table,
                     matched_table=None, # Placeholder
                     header=st.session_state.calibrated_header,
                     pixel_scale_arcsec=pixel_scale_arcsec
                 )
                 if enhanced_table is not None:
                     st.session_state.final_phot_table = enhanced_table # Update with enhanced version
                     st.success("Catalog enhancement complete (placeholder).")
                     write_to_log(st.session_state.log_buffer, "Catalog enhancement step completed (placeholder).")
                 else:
                     st.warning("Catalog enhancement skipped or failed.")
                     write_to_log(st.session_state.log_buffer, "Catalog enhancement skipped or failed.")


        # --- Display Results ---
        st.header("Results")
        if st.session_state.final_phot_table is not None:
            st.dataframe(st.session_state.final_phot_table)

            # Provide download button for the final catalog
            csv_buffer = StringIO()
            st.session_state.final_phot_table.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Catalog as CSV",
                data=csv_buffer.getvalue(),
                file_name=f"{st.session_state.base_filename}_catalog.csv",
                mime="text/csv",
            )
            write_to_log(st.session_state.log_buffer, f"Final catalog generated with {len(st.session_state.final_phot_table)} sources.")

            # Save catalog to output directory
            try:
                output_dir = ensure_output_directory("rpp_results")
                catalog_filename = os.path.join(output_dir, f"{st.session_state.base_filename}_catalog.csv")
                st.session_state.final_phot_table.to_csv(catalog_filename, index=False)
                write_to_log(st.session_state.log_buffer, f"Saved final catalog to {catalog_filename}")
            except Exception as e:
                st.error(f"Error saving catalog file: {e}")
                write_to_log(st.session_state.log_buffer, f"Error saving catalog file: {e}")

        else:
            st.warning("No final photometry table generated.")
            write_to_log(st.session_state.log_buffer, "Processing finished, but no final catalog was generated.")


        # --- Display Log ---
        st.subheader("Processing Log")
        log_content = st.session_state.log_buffer.getvalue()
        st.text_area("Log Output", log_content, height=200)

        # Save log to file
        try:
            output_dir = ensure_output_directory("rpp_results")
            log_filename = os.path.join(output_dir, f"{st.session_state.base_filename}.log")
            with open(log_filename, "w") as f:
                f.write(log_content)
            st.info(f"Log saved to {log_filename}")
        except Exception as e:
            st.error(f"Error saving log file: {e}")

        # --- Cleanup and Exit Hook ---
        # Register the zip function to run on exit (if not already registered)
        # Check if already registered might be complex, maybe register unconditionally?
        # Or use a flag in session state.
        if 'zip_registered' not in st.session_state:
             atexit.register(zip_rpp_results_on_exit, st.session_state.output_dir, st.session_state.base_filename)
             st.session_state.zip_registered = True


    else:
        st.error("Failed to load FITS file.")
        if st.session_state.get("log_buffer"):
             write_to_log(st.session_state.log_buffer, "Failed to load FITS file.")

# --- Footer or other static elements ---
st.markdown("---")
