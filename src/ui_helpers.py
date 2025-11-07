# src/ui_helpers.py
import base64
import json
import os
import zipfile
from io import BytesIO
from datetime import datetime
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from jinja2 import Template
import matplotlib.pyplot as plt
import numpy as np
from src.tools import write_to_log


def display_catalog_in_aladin(
    final_table: pd.DataFrame,
    ra_center: float,
    dec_center: float,
    fov: float = 1.5,
    ra_col: str = "ra",
    dec_col: str = "dec",
    mag_col: str = "psf_mag",
    alt_mag_col: str = "aperture_mag",
    catalog_col: str = "catalog_matches",
    id_cols: list[str] = ["simbad_main_id", "skybot_NAME", "aavso_Name"],
    fallback_id_prefix: str = "Source",
    survey: str = "CDS/P/DSS2/color",
) -> None:
    """
    Display a DataFrame catalog in an embedded Aladin Lite interactive sky viewer.

    This function creates an interactive astronomical image with catalog overlay
    that allows exploring detected sources and their cross-matches.

    Parameters
    ----------
    final_table : pandas.DataFrame
        DataFrame containing catalog data with coordinates and photometry
    ra_center : float
        Right Ascension center coordinate for the Aladin view (degrees)
    dec_center : float
        Declination center coordinate for the Aladin view (degrees)
    fov : float, optional
        Initial field of view in degrees, default=1.5
    ra_col : str, optional
        Name of the column containing Right Ascension values, default='ra'
    dec_col : str, optional
        Name of the column containing Declination values, default='dec'
    mag_col : str, optional
        Name of the primary magnitude column, default='psf_mag'
    alt_mag_col : str, optional
        Name of an alternative (preferred) magnitude column, default='aperture_mag'
    catalog_col : str, optional
        Name of the column containing catalog match information, default='catalog_matches'
    id_cols : list[str], optional
        List of column names (in order of preference) to use for source identifiers
    fallback_id_prefix : str, optional
        Prefix to use for source names if no ID is found, default="Source"
    survey : str, optional
        The initial sky survey to display in Aladin Lite, default="CDS/P/DSS2/color"

    Notes
    -----
    The function creates an interactive HTML component embedded in Streamlit that
    shows a DSS image of the field with catalog sources overlaid as interactive markers.
    Each marker shows a popup with detailed source information when clicked.

    Handles errors gracefully and provides feedback in the Streamlit interface
    if any issues occur.
    """
    if not (
        isinstance(ra_center, (int, float)) and isinstance(dec_center, (int, float))
    ):
        st.error("Missing or invalid center coordinates (RA/Dec) for Aladin display.")
        return

    if not isinstance(final_table, pd.DataFrame) or final_table.empty:
        st.warning("Input table is empty or not a DataFrame. Cannot display in Aladin.")
        return

    if ra_col not in final_table.columns or dec_col not in final_table.columns:
        st.error(
            f"Required columns '{ra_col}' or '{dec_col}' not found in the DataFrame."
        )
        return

    catalog_sources = []
    required_cols = {ra_col, dec_col}
    optional_cols = {mag_col, alt_mag_col, catalog_col}.union(set(id_cols))
    available_cols = set(final_table.columns)

    present_optional_cols = optional_cols.intersection(available_cols)
    cols_to_iterate = list(required_cols.union(present_optional_cols))

    for row in final_table[cols_to_iterate].itertuples():
        ra_val = getattr(row, ra_col)
        dec_val = getattr(row, dec_col)
        if pd.notna(ra_val) and pd.notna(dec_val):
            try:
                source = {"ra": float(ra_val), "dec": float(dec_val)}
            except (ValueError, TypeError):
                continue

            # Handle magnitude - collect both PSF and aperture magnitudes
            psf_mag = None
            aperture_mag = None

            # Get PSF magnitude
            if mag_col in present_optional_cols and pd.notna(getattr(row, mag_col, None)):
                try:
                    psf_mag = float(getattr(row, mag_col))
                    source["psf_mag"] = psf_mag
                except (ValueError, TypeError):
                    pass

            # Get aperture magnitude (try multiple aperture columns)
            aperture_mag_cols = ["aperture_mag_r1.5", "aperture_mag", "calib_mag"]
            for ap_col in aperture_mag_cols:
                if ap_col in present_optional_cols and pd.notna(getattr(row, ap_col, None)):
                    try:
                        aperture_mag = float(getattr(row, ap_col))
                        source["aperture_mag"] = aperture_mag
                        break
                    except (ValueError, TypeError):
                        continue

            # Handle catalog matches - collect individual catalog IDs
            catalog_matches = {}

            # SIMBAD matches
            if "simbad_main_id" in present_optional_cols and pd.notna(
                getattr(row, "simbad_main_id", None)
            ):
                simbad_id = str(getattr(row, "simbad_main_id")).strip()
                if simbad_id and simbad_id not in ["", "nan", "None"]:
                    catalog_matches["SIMBAD"] = simbad_id

            # SkyBoT matches (solar system objects)
            if "skybot_NAME" in present_optional_cols and pd.notna(getattr(row, "skybot_NAME", None)):
                skybot_name = str(getattr(row, "skybot_NAME")).strip()
                if skybot_name and skybot_name not in ["", "nan", "None"]:
                    catalog_matches["SkyBoT"] = skybot_name

            # AAVSO VSX matches (variable stars)
            if "aavso_Name" in present_optional_cols and pd.notna(getattr(row, "aavso_Name", None)):
                aavso_name = str(getattr(row, "aavso_Name")).strip()
                if aavso_name and aavso_name not in ["", "nan", "None"]:
                    catalog_matches["AAVSO"] = aavso_name

            # Astro-Colibri matches
            if "astrocolibri_name" in present_optional_cols and pd.notna(
                getattr(row, "astrocolibri_name", None)
            ):
                colibri_name = str(getattr(row, "astrocolibri_name")).strip()
                if colibri_name and colibri_name not in ["", "nan", "None"]:
                    catalog_matches["Astro-Colibri"] = colibri_name

            # Quasar matches
            if "qso_name" in present_optional_cols and pd.notna(getattr(row, "qso_name", None)):
                qso_name = str(getattr(row, "qso_name")).strip()
                if qso_name and qso_name not in ["", "nan", "None"]:
                    catalog_matches["QSO"] = qso_name

            # GAIA matches (if calibration star)
            if "gaia_calib_star" in present_optional_cols and getattr(row, "gaia_calib_star", False):
                catalog_matches["GAIA"] = "Calibration Star"

            source["catalog_matches"] = catalog_matches

            # FIXED: Handle source identification - prioritize the real "id" column first
            source_id = f"{fallback_id_prefix} {row.Index + 1}"  # Default fallback

            # First, check if there's an "id" column in the table (the real catalog ID)
            id_val = getattr(row, "id", None)
            if "id" in final_table.columns and pd.notna(id_val):
                id_value = str(id_val).strip()
                if id_value and id_value not in ["nan", "None", ""]:
                    source_id = f"ID: {id_value}"
            # Only if no "id" column exists, fall back to other catalog identifiers
            elif id_cols:
                for id_col in id_cols:
                    if id_col in present_optional_cols and pd.notna(getattr(row, id_col, None)):
                        id_value = str(getattr(row, id_col)).strip()
                        if id_value and id_value not in ["nan", "None", ""]:
                            source_id = id_value
                            break

            source["name"] = source_id
            source["source_number"] = row.Index + 1

            # Add the raw ID value separately for display in popup
            id_val = getattr(row, "id", None)
            if "id" in final_table.columns and pd.notna(id_val):
                source["catalog_id"] = str(id_val)

            # ... rest of existing code ...

            # Add additional useful information
            for info_col in ["snr", "flux_fit", "fwhm"]:
                if info_col in present_optional_cols and pd.notna(getattr(row, info_col, None)):
                    try:
                        source[info_col] = float(getattr(row, info_col))
                    except (ValueError, TypeError):
                        source[info_col] = str(getattr(row, info_col))

            catalog_sources.append(source)

    if not catalog_sources:
        st.warning("No valid sources with RA/Dec found in the table to display.")
        return

    with st.spinner("Loading Aladin Lite viewer..."):
        try:
            sources_json_b64 = base64.b64encode(
                json.dumps(catalog_sources).encode("utf-8")
            ).decode("utf-8")

            # Load the HTML template from file
            template_path = os.path.join(os.path.dirname(__file__), "..", "templates", "aladin_template.html")
            with open(template_path, "r") as f:
                template_str = f.read()

            # Create and render the Jinja2 template
            template = Template(template_str)
            html_content = template.render(
                ra_center=ra_center,
                dec_center=dec_center,
                fov=fov,
                survey=survey,
                sources_json_b64=sources_json_b64,
            )

            components.html(
                html_content,
                height=600,
                scrolling=True,
            )

        except Exception as e:
            st.error(f"Failed to render Aladin HTML component: {str(e)}")
            # Fallback: show a simple coordinate table
            st.subheader("Source Coordinates (Aladin viewer unavailable)")
            display_df = pd.DataFrame(catalog_sources)
            if not display_df.empty:
                st.dataframe(display_df, use_container_width=True)

def plot_magnitude_distribution(final_table, log_buffer=None):
    """
    Create magnitude distribution plots (histogram and error scatter plot).

    Parameters
    ----------
    final_table : pandas.DataFrame
        DataFrame containing photometry results with magnitude columns
    log_buffer : StringIO, optional
        Log buffer for writing status messages

    Returns
    -------
    matplotlib.figure.Figure
        Figure object containing the plots
    """
    fig_mag, (ax_mag, ax_err) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

    # Check if we have magnitude columns - use the new multi-aperture columns
    has_aperture = "aperture_mag_r1.5" in final_table.columns
    has_psf = "psf_mag" in final_table.columns

    if not has_aperture and not has_psf:
        # Create empty plots with message
        ax_mag.text(
            0.5,
            0.5,
            "No magnitude data available",
            ha="center",
            va="center",
            transform=ax_mag.transAxes,
        )
        ax_err.text(
            0.5,
            0.5,
            "No magnitude error data available",
            ha="center",
            va="center",
            transform=ax_err.transAxes,
        )
        return fig_mag

    # Calculate bins for magnitude distribution
    mag_values = []
    if has_aperture:
        mag_values.extend(final_table["aperture_mag_r1.5"].dropna().tolist())
    if has_psf:
        mag_values.extend(final_table["psf_mag"].dropna().tolist())

    if mag_values:
        bins = np.linspace(min(mag_values), max(mag_values), 40)
    else:
        bins = 40

    # Magnitude distribution histogram (left panel)
    if has_aperture:
        ax_mag.hist(
            final_table["aperture_mag_r1.5"].dropna(),
            bins=bins,
            alpha=0.6,
            label="Aperture Calib Mag (1.5×FWHM)",
            color="tab:blue",
        )
    if has_psf:
        ax_mag.hist(
            final_table["psf_mag"].dropna(),
            bins=bins,
            alpha=0.6,
            label="PSF Calib Mag",
            color="tab:orange",
        )

    ax_mag.set_xlabel("Calibrated Magnitude")
    ax_mag.set_ylabel("Number of Sources")
    ax_mag.set_title("Distribution of Calibrated Magnitudes")
    ax_mag.legend()
    ax_mag.grid(True, alpha=0.3)

    # Scatter plot of magnitude vs error (right panel)
    if has_aperture and "aperture_mag_err_r1.5" in final_table.columns:
        ax_err.scatter(
            final_table["aperture_mag_r1.5"],
            final_table["aperture_mag_err_r1.5"],
            alpha=0.7,
            label="Aperture (1.5×FWHM)",
            color="tab:blue",
            s=18,
        )
    if has_psf and "psf_mag_err" in final_table.columns:
        ax_err.scatter(
            final_table["psf_mag"],
            final_table["psf_mag_err"],
            alpha=0.7,
            label="PSF",
            color="tab:orange",
            s=18,
        )

    ax_err.set_xlabel("Calibrated Magnitude")
    ax_err.set_ylabel("Magnitude Error")
    ax_err.set_title("Magnitude Error vs Magnitude")
    ax_err.legend()
    ax_err.grid(True, alpha=0.3)

    fig_mag.tight_layout()

    # Log the plot creation if log_buffer is provided
    if log_buffer:
        write_to_log(log_buffer, "Created magnitude distribution plots")

    return fig_mag

def provide_download_buttons(folder_path):
    """
    Creates a single download button for a zip file containing all files in the specified folder.

    This function compresses all files in the given folder into a single zip archive
    and provides a download button for the archive.

    Args:
        folder_path (str): The path to the folder containing files to be zipped and made downloadable

    Returns:
        None: The function creates a Streamlit download button directly in the app interface
    """
    try:
        base_filename = st.session_state.get("base_filename", "")

        files = [
            f
            for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.startswith(base_filename)
            and not f.lower().endswith(".zip")
        ]
        if not files:
            st.write("No files found in output directory")
            return

        # Create a timestamp for the zip filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = st.session_state.get("base_filename", "rpp_results")
        zip_filename = f"{base_name}_{timestamp}.zip"

        # Create in-memory zip file
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file in files:
                file_path = os.path.join(folder_path, file)
                zip_file.write(file_path, arcname=file)

        # Reset buffer position to the beginning
        zip_buffer.seek(0)
        st.caption(f"Archive contains {len(files)} files")
        # Create download button for the zip file
        if st.download_button(
            label="📦 Download Results (ZIP)",
            data=zip_buffer,
            file_name=zip_filename,
            mime="application/zip",
            on_click="ignore",
        ):
            st.success("File downloaded !")

    except Exception as e:
        st.error(f"Error creating zip archive: {str(e)}")


def display_archived_files_browser(output_dir):
    """
    Display a file browser for archived ZIP files in the user's results directory.

    This function creates a secure file browser that only allows access to ZIP files
    within the specified output directory. Also automatically cleans up old files
    (except .json) that are older than 1 month.

    Parameters
    ----------
    output_dir : str
        Path to the user's results directory

    Returns
    -------
    None
        Creates Streamlit interface elements directly
    """
    if not os.path.exists(output_dir):
        st.warning("No results directory found.")
        return

    try:
        # Automatically clean old files (excluding .json files)
        cutoff_date = datetime.now() - pd.Timedelta(days=30)
        deleted_count = 0

        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path):
                # Skip .json files from cleanup
                if item.lower().endswith(".json"):
                    continue

                try:
                    mod_time = datetime.fromtimestamp(os.path.getmtime(item_path))
                    if mod_time < cutoff_date:
                        os.remove(item_path)
                        deleted_count += 1
                except Exception:
                    # Silently continue if file can't be deleted
                    pass

        # Collect only ZIP files
        zip_files = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path) and item.lower().endswith(".zip"):
                # Get file stats
                try:
                    stat = os.stat(item_path)
                    file_size = stat.st_size
                    mod_time = datetime.fromtimestamp(stat.st_mtime)

                    zip_files.append(
                        {
                            "name": item,
                            "size": file_size,
                            "modified": mod_time,
                            "path": item_path,
                        }
                    )
                except Exception:
                    # Skip files that can't be accessed
                    continue

        if not zip_files:
            st.info("No ZIP archives found.")
            if deleted_count > 0:
                st.caption(f"Auto-cleaned {deleted_count} old files.")
            return

        # Sort files by modification time (newest first)
        zip_files.sort(key=lambda x: x["modified"], reverse=True)

        st.write(f"**📦 {len(zip_files)} ZIP Archive(s)**")

        if deleted_count > 0:
            st.caption(f"Auto-cleaned {deleted_count} old files.")

        # Compact display with download buttons
        for file_info in zip_files:
            # Format file size
            size = file_info["size"]
            if size > 1024 * 1024:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size}B"

            # Format date
            date_str = file_info["modified"].strftime("%m/%d %H:%M")

            # Truncate filename if too long
            display_name = file_info["name"]
            if len(display_name) > 30:
                display_name = display_name[:27] + "..."

            # Create compact row with download button
            col1, col2 = st.columns([3, 1])

            with col1:
                st.text(f"{display_name}")
                st.caption(f"{size_str} • {date_str}")

            with col2:
                try:
                    with open(file_info["path"], "rb") as f:
                        file_data = f.read()

                    st.download_button(
                        label="📥",
                        data=file_data,
                        file_name=file_info["name"],
                        mime="application/zip",
                        key=f"download_zip_{file_info['name']}",
                        help=f"Download {file_info['name']}",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(f"Error: {str(e)[:15]}...")

    except Exception as e:
        st.error(f"Error accessing results directory: {str(e)}")

def clear_all_caches():
    """Clear all Streamlit caches and reset file upload state"""
    try:
        st.cache_data.clear()
        if hasattr(st.cache_resource, "clear"):
            st.cache_resource.clear()

        # Clear file upload related session state
        upload_keys = [
            key for key in st.session_state.keys() if "uploader" in key.lower()
        ]
        for key in upload_keys:
            del st.session_state[key]

        # Also clear persisted uploaded file (if present)
        if "uploaded" in st.session_state:
            try:
                del st.session_state["uploaded"]
            except Exception:
                pass
        # Also clear persisted raw upload bytes and name if present
        if "uploaded_bytes" in st.session_state:
            try:
                del st.session_state["uploaded_bytes"]
            except Exception:
                pass
        if "uploaded_name" in st.session_state:
            try:
                del st.session_state["uploaded_name"]
            except Exception:
                pass
            
        # Clear file-related session state
        if "science_file_path" in st.session_state:
            try:
                if os.path.exists(st.session_state["science_file_path"]):
                    os.unlink(st.session_state["science_file_path"])
            except Exception:
                pass
            del st.session_state["science_file_path"]

        if "files_loaded" in st.session_state:
            st.session_state["files_loaded"] = {"science_file": None}

        st.success("All caches cleared successfully!")
        st.rerun()
    except Exception as e:
        st.error(f"Error clearing caches: {e}")
