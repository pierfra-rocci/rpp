# Standard Library Imports
import sys
import os
import zipfile
import base64
import json
import tempfile
import warnings
from datetime import datetime
from io import StringIO, BytesIO

# Third-Party Imports
import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import (ZScaleInterval, ImageNormalize)

# Local Application Imports
from src.tools import (FIGURE_SIZES, GAIA_BANDS, extract_coordinates,
                       extract_pixel_scale, get_base_filename,
                       safe_wcs_create, ensure_output_directory,
                       cleanup_temp_files, initialize_log, write_to_log,
                       zip_rpp_results_on_exit, save_header_to_txt)

from src.pipeline import (solve_with_siril, cross_match_with_gaia,
                          calculate_zero_point, detection_and_photometry,
                          detect_remove_cosmic_rays, enhance_catalog,
                          airmass)

from src.__version__ import version

# Conditional Import (already present, just noting its location)
if getattr(sys, "frozen", False):
    try:
        import importlib.metadata
        importlib.metadata.distributions = lambda **kwargs: []
    except ImportError:
        st.warning(
            "Could not modify importlib.metadata, "
            "potential issues in frozen mode."
        )

warnings.filterwarnings("ignore")


@st.cache_data
def load_fits_data(file):
    """
    Load image data and header from a FITS file with robust error handling.

    Handles multiple FITS formats including multi-extension files, data cubes,
    and RGB images. For multi-dimensional data, extracts a 2D plane.

    Parameters
    ----------
    file : Streamlit UploadedFile
        FITS file object from Streamlit file uploader

    Returns
    -------
    tuple
        (image_data, header) where:
        - image_data: 2D numpy array of pixel values, or None if loading failed
        - header: FITS header dictionary, or None if loading failed

    Notes
    -----
    - For multi-extension FITS files, uses the primary HDU if it contains data,
      otherwise uses the first HDU with valid data.
    - For 3D data (RGB or data cube), extracts the first 2D plane with appropriate warnings.
    - For higher dimensional data, takes the first slice along all extra dimensions.
    """
    if file is not None:
        file_content = file.read()
        hdul = fits.open(BytesIO(file_content), mode="readonly")
        try:
            data = hdul[0].data
            hdul.verify("fix")
            header = hdul[0].header

            if data is None:
                for i, hdu in enumerate(hdul[1:], 1):
                    if hasattr(hdu, "data") and hdu.data is not None:
                        data = hdu.data
                        header = hdu.header
                        # Only log it, don't show in UI
                        if st.session_state.get("log_buffer"):
                            write_to_log(st.session_state["log_buffer"], 
                                       f"Primary HDU has no data. Using data from HDU #{i}.")
                        break

            if data is None:
                st.warning("No image data found in the FITS file.")
                return None, None

            if len(data.shape) == 3:
                if data.shape[0] == 3 or data.shape[0] == 4:
                    st.info(
                        f"Detected RGB data with shape {data.shape}. Using first color channel."
                    )
                    data = data[0]
                elif data.shape[2] == 3 or data.shape[2] == 4:
                    st.info(
                        f"Detected RGB data with shape {data.shape}. Using first color channel."
                    )
                    data = data[:, :, 0]
                else:
                    st.info(
                        f"Detected 3D data with shape {data.shape}. Using first plane."
                    )
                    data = data[0]
            elif len(data.shape) > 3:
                st.warning(
                    f"Data has {len(data.shape)} dimensions. Using first slice only."
                )
                sliced_data = data
                while len(sliced_data.shape) > 2:
                    sliced_data = sliced_data[0]
                data = sliced_data

            return data, header

        except Exception as e:
            st.error(f"Error loading FITS file: {str(e)}")
            return None, None
        finally:
            hdul.close()

    return None, None


def display_catalog_in_aladin(
    final_table: pd.DataFrame,  # Remove unused 'image' parameter
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

    for idx, row in final_table[cols_to_iterate].iterrows():
        ra_val = row[ra_col]
        dec_val = row[dec_col]
        if pd.notna(ra_val) and pd.notna(dec_val):
            try:
                source = {"ra": float(ra_val), "dec": float(dec_val)}
            except (ValueError, TypeError):
                continue

            # Handle magnitude - prefer alt_mag_col over mag_col
            mag_to_use = None
            mag_source = ""
            
            # Check for multi-aperture columns first
            for aperture_col in ["aperture_mag_r1.5", "aperture_mag_r2.0", "aperture_mag_r2.5"]:
                if aperture_col in present_optional_cols and pd.notna(row[aperture_col]):
                    try:
                        mag_to_use = float(row[aperture_col])
                        mag_source = aperture_col.replace("aperture_mag_r", "Ap(") + "×FWHM)"
                        break
                    except (ValueError, TypeError):
                        pass
            
            # Fall back to standard magnitude columns
            if mag_to_use is None:
                if alt_mag_col in present_optional_cols and pd.notna(row[alt_mag_col]):
                    try:
                        mag_to_use = float(row[alt_mag_col])
                        mag_source = "Aperture"
                    except (ValueError, TypeError):
                        pass
                elif mag_col in present_optional_cols and pd.notna(row[mag_col]):
                    try:
                        mag_to_use = float(row[mag_col])
                        mag_source = "PSF"
                    except (ValueError, TypeError):
                        pass

            if mag_to_use is not None:
                source["mag"] = mag_to_use
                source["mag_source"] = mag_source

            # Handle catalog matches
            catalog_info = ""
            if catalog_col in present_optional_cols and pd.notna(row[catalog_col]):
                catalog_info = str(row[catalog_col])
            
            # Check for specific catalog columns
            catalog_matches = []
            for cat_prefix in ["simbad_", "skybot_", "aavso_", "gaia_"]:
                cat_cols = [col for col in present_optional_cols if col.startswith(cat_prefix)]
                if cat_cols:
                    for cat_col in cat_cols:
                        if pd.notna(row[cat_col]) and str(row[cat_col]).strip() not in ["", "nan", "None"]:
                            catalog_matches.append(f"{cat_prefix.rstrip('_').upper()}: {row[cat_col]}")
            
            if catalog_matches:
                catalog_info = "; ".join(catalog_matches)
            
            if catalog_info:
                source["catalog"] = catalog_info

            # Handle source identification with priority
            source_id = f"{fallback_id_prefix} {idx + 1}"
            if id_cols:
                for id_col in id_cols:
                    if id_col in present_optional_cols and pd.notna(row[id_col]):
                        id_value = str(row[id_col]).strip()
                        if id_value and id_value not in ["nan", "None", ""]:
                            source_id = id_value
                            break
            
            source["name"] = source_id
            source["source_number"] = idx + 1

            # Add additional useful information
            for info_col in ["fwhm", "flux_fit", "sky_center"]:
                if info_col in present_optional_cols and pd.notna(row[info_col]):
                    try:
                        source[info_col] = float(row[info_col])
                    except (ValueError, TypeError):
                        source[info_col] = str(row[info_col])

            catalog_sources.append(source)

    if not catalog_sources:
        st.warning("No valid sources with RA/Dec found in the table to display.")
        return

    with st.spinner("Loading Aladin Lite viewer..."):
        try:
            try:
                sources_json_b64 = base64.b64encode(
                    json.dumps(catalog_sources).encode("utf-8")
                ).decode("utf-8")
            except Exception as e:
                st.error(f"Failed to encode catalog data: {str(e)}")
                return

            # Fix JavaScript error handling structure
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <title>Aladin Lite</title>
            </head>
            <body>
                <div id="aladin-lite-div" style="width:100%;height:550px;"></div>
                <script type="text/javascript" src="https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.js" charset="utf-8"></script>
                <script type="text/javascript">
                    document.addEventListener("DOMContentLoaded", function(event) {{
                        try {{
                            let aladin = A.aladin('#aladin-lite-div', {{
                                target: '{ra_center} {dec_center}',
                                fov: {fov},
                                survey: '{survey}',
                                cooFrame: 'J2000',
                                showReticle: false,
                                showZoomControl: true,
                                showFullscreenControl: true,
                                showLayersControl: true,
                                showGotoControl: true,
                                showSimbadPointerControl: true
                            }});

                            let cat = A.catalog({{
                                name: 'Photometry Results',
                                sourceSize: 12,
                                shape: 'circle',
                                color: '#00ff88',
                                onClick: 'showPopup'
                            }});
                            aladin.addCatalog(cat);
                            
                            let sourcesData = JSON.parse(atob("{sources_json_b64}"));
                            let aladinSources = [];

                            sourcesData.forEach(function(source) {{
                                // Add HTML escaping for source data
                                let escapedName = (source.name || '').replace(/[<>&"']/g, function(m) {{
                                    return {{'<':'&lt;', '>':'&gt;', '&':'&amp;', '"':'&quot;', "'":"&#39;"}}[m];
                                }});
                                
                                let popupContent = '<div style="padding:5px;">';
                                if(escapedName) {{
                                    popupContent += '<b>' + escapedName + '</b><br/>';
                                }}
                                popupContent += 'RA: ' + (typeof source.ra === 'number' ? source.ra.toFixed(6) : source.ra) + '<br/>';
                                popupContent += 'Dec: ' + (typeof source.dec === 'number' ? source.dec.toFixed(6) : source.dec) + '<br/>';

                                if(source.mag) {{
                                    popupContent += 'Mag: ' + (typeof source.mag === 'number' ? source.mag.toFixed(2) : source.mag) + '<br/>';
                                }}
                                if(source.catalog) {{
                                    let escapedCatalog = source.catalog.replace(/[<>&"']/g, function(m) {{
                                        return {{'<':'&lt;', '>':'&gt;', '&':'&amp;', '"':'&quot;', "'":"&#39;"}}[m];
                                    }});
                                    popupContent += 'Catalogs: ' + escapedCatalog + '<br/>';
                                }}
                                popupContent += '</div>';

                                let aladinSource = A.source(
                                    source.ra,
                                    source.dec,
                                    {{ description: popupContent }}
                                );
                                aladinSources.push(aladinSource);
                            }});

                            if (aladinSources.length > 0) {{
                                cat.addSources(aladinSources);
                            }}
                        }} catch (error) {{
                            console.error("Error initializing Aladin Lite or adding sources:", error);
                            document.getElementById('aladin-lite-div').innerHTML = '<p style="color:red;">Error loading Aladin viewer. Check console for details.</p>';
                        }}
                    }});
                </script>
            </body>
            </html>
            """

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

        # Filter files to only include those starting with the base filename prefix
        # and exclude files ending with '.zip'
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
            label="📦 Download All Results (ZIP)",
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
    Display a file browser for archived files in the user's results directory.
    
    This function creates a secure file browser that only allows access to files
    within the specified output directory. Users can view and download individual
    files or select multiple files for download.
    
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
        all_files = []
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if os.path.isfile(item_path):
                # Get file stats
                stat = os.stat(item_path)
                file_size = stat.st_size
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                
                all_files.append({
                    'name': item,
                    'size': file_size,
                    'modified': mod_time,
                    'path': item_path
                })
        
        if not all_files:
            st.info("No archived files found in results directory.")
            return
        
        # Sort files by modification time (newest first)
        all_files.sort(key=lambda x: x['modified'], reverse=True)
        
        st.subheader(f"📁 Archived Files ({len(all_files)} files)")
        
        # Create a DataFrame for better display
        display_data = []
        for file_info in all_files:
            size_str = f"{file_info['size']:,} bytes"
            if file_info['size'] > 1024:
                size_str = f"{file_info['size']/1024:.1f} KB"
            if file_info['size'] > 1024*1024:
                size_str = f"{file_info['size']/(1024*1024):.1f} MB"
            
            display_data.append({
                'File Name': file_info['name'],
                'Size': size_str,
                'Modified': file_info['modified'].strftime('%Y-%m-%d %H:%M:%S'),
                'Type': os.path.splitext(file_info['name'])[1] or 'No extension'
            })
        
        files_df = pd.DataFrame(display_data)
        st.dataframe(files_df, use_container_width=True)
        
        # File selection for individual downloads
        st.subheader("📥 Download Individual Files")
        
        # Group files by type for easier selection
        file_types = {}
        for file_info in all_files:
            ext = os.path.splitext(file_info['name'])[1].lower()
            if ext not in file_types:
                file_types[ext] = []
            file_types[ext].append(file_info)
        
        # Create tabs for different file types
        if len(file_types) > 1:
            type_tabs = st.tabs([f"{ext or 'No ext'} ({len(files)})" for ext, files in file_types.items()])
            
            for tab, (ext, files) in zip(type_tabs, file_types.items()):
                with tab:
                    for file_info in files:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.text(file_info['name'])
                        
                        with col2:
                            size_str = f"{file_info['size']:,} bytes"
                            if file_info['size'] > 1024:
                                size_str = f"{file_info['size']/1024:.1f} KB"
                            if file_info['size'] > 1024*1024:
                                size_str = f"{file_info['size']/(1024*1024):.1f} MB"
                            st.text(size_str)
                        
                        with col3:
                            # Create download button for individual file
                            try:
                                with open(file_info['path'], 'rb') as f:
                                    file_data = f.read()
                                
                                # Determine MIME type based on extension
                                mime_type = "application/octet-stream"
                                if ext == '.zip':
                                    mime_type = "application/zip"
                                
                                st.download_button(
                                    label="📥",
                                    data=file_data,
                                    file_name=file_info['name'],
                                    mime=mime_type,
                                    key=f"download_{file_info['name']}",
                                    help=f"Download {file_info['name']}",
                                    on_click="ignore"
                                )
                            except Exception as e:
                                st.error(f"Error reading file {file_info['name']}: {str(e)}")
        else:
            # Single file type, display directly
            for file_info in all_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.text(file_info['name'])
                
                with col2:
                    size_str = f"{file_info['size']:,} bytes"
                    if file_info['size'] > 1024:
                        size_str = f"{file_info['size']/1024:.1f} KB"
                    if file_info['size'] > 1024*1024:
                        size_str = f"{file_info['size']/(1024*1024):.1f} MB"
                    st.text(size_str)
                
                with col3:
                    try:
                        with open(file_info['path'], 'rb') as f:
                            file_data = f.read()
                        
                        ext = os.path.splitext(file_info['name'])[1].lower()
                        mime_type = "application/octet-stream"
                        if ext == '.zip':
                            mime_type = "application/zip"
                        
                        st.download_button(
                            label="📥",
                            data=file_data,
                            file_name=file_info['name'],
                            mime=mime_type,
                            key=f"download_{file_info['name']}",
                            help=f"Download {file_info['name']}",
                            on_click="ignore"
                        )
                    except Exception as e:
                        st.error(f"Error reading file {file_info['name']}: {str(e)}")
        
        # Directory cleanup option
        st.subheader("🗑️ File Management")
        
        if st.button("🧹 Clean Old Files",
                     help="Remove files older than 30 days"):
            try:
                cutoff_date = datetime.now() - pd.Timedelta(days=30)
                deleted_count = 0
                
                for file_info in all_files:
                    if file_info['modified'] < cutoff_date:
                        try:
                            os.remove(file_info['path'])
                            deleted_count += 1
                        except Exception as e:
                            st.warning(f"Could not delete {file_info['name']}: {str(e)}")
                
                if deleted_count > 0:
                    st.success(f"Deleted {deleted_count} old files.")
                    st.experimental_rerun()
                else:
                    st.info("No old files found to delete.")
                    
            except Exception as e:
                st.error(f"Error during cleanup: {str(e)}")
                
    except Exception as e:
        st.error(f"Error accessing results directory: {str(e)}")


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
        ax_mag.text(0.5, 0.5, "No magnitude data available",
                    ha='center', va='center', transform=ax_mag.transAxes)
        ax_err.text(0.5, 0.5, "No magnitude error data available",
                    ha='center', va='center', transform=ax_err.transAxes)
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
            s=18
        )
    if has_psf and "psf_mag_err" in final_table.columns:
        ax_err.scatter(
            final_table["psf_mag"],
            final_table["psf_mag_err"],
            alpha=0.7,
            label="PSF",
            color="tab:orange",
            s=18
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
    defaults = {
        "calibrated_header": None,
        "final_phot_table": None,
        "epsf_model": None,
        "epsf_photometry_result": None,
        "log_buffer": None,
        "base_filename": "photometry",
        "science_file_path": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Output directory
    if "output_dir" not in st.session_state:
        username = st.session_state.get("username", "anonymous")
        st.session_state.output_dir = ensure_output_directory(f"{username}_rpp_results")

    # Analysis Parameters (consolidated)
    default_analysis_params = {
        "seeing": 3.0,
        "threshold_sigma": 3.0,
        "detection_mask": 10,
        "filter_band": "phot_g_mean_mag",
        "filter_max_mag": 20.0,
        "astrometry_check": False,
        "calibrate_cosmic_rays": False,
        "cr_gain": 1.0,
        "cr_readnoise": 2.5,
        "cr_sigclip": 6.0
    }
    
    if "analysis_parameters" not in st.session_state:
        st.session_state.analysis_parameters = default_analysis_params.copy()
    else:
        # Ensure all keys exist
        for key, value in default_analysis_params.items():
            if key not in st.session_state.analysis_parameters:
                st.session_state.analysis_parameters[key] = value

    # Observatory Parameters (keep only dictionary, remove individual keys)
    default_observatory_data = {
        "name": "Obs",
        "latitude": 0.,
        "longitude": 0.,
        "elevation": 0.,
    }
    
    if "observatory_data" not in st.session_state:
        st.session_state.observatory_data = default_observatory_data.copy()
    else:
        for key, value in default_observatory_data.items():
            if key not in st.session_state.observatory_data:
                st.session_state.observatory_data[key] = value

    # API Keys and File Loading
    if "colibri_api_key" not in st.session_state:
        st.session_state.colibri_api_key = None
    if "files_loaded" not in st.session_state:
        st.session_state.files_loaded = {"science_file": None}

###################################################################
# Main Streamlit app
###################################################################


st.set_page_config(page_title="RAPAS Photometry Pipeline", page_icon="🔭",
                   layout="wide")

# --- Initialize Session State Early ---
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
st.title("🔭 RAPAS Photometry Pipeline")
st.sidebar.markdown(f"**App Version:** _{version}_")

# --- Sync session state with loaded config before creating widgets ---
if "observatory_data" in st.session_state:
    obs = st.session_state["observatory_data"]
    st.session_state["observatory_name"] = obs.get("name", "")
    st.session_state["observatory_latitude"] = obs.get("latitude", 0.0)
    st.session_state["observatory_longitude"] = obs.get("longitude", 0.0)
    st.session_state["observatory_elevation"] = obs.get("elevation", 0.0)

if "analysis_parameters" in st.session_state:
    ap = st.session_state["analysis_parameters"]
    for key in [
        "seeing", "threshold_sigma", "detection_mask",
        "astrometry_check", "calibrate_cosmic_rays",
        "cr_gain", "cr_readnoise", "cr_sigclip",
        "filter_band", "filter_max_mag"
    ]:
        if key in ap:
            st.session_state[key] = ap[key]

if "filter_parameters" in st.session_state:
    gaia = st.session_state["filter_parameters"]
    st.session_state["filter_band"] = gaia.get("filter_band", "phot_g_mean_mag")
    st.session_state["filter_max_mag"] = gaia.get("filter_max_mag", 20.0)

if "colibri_api_key" in st.session_state:
    st.session_state["colibri_api_key"] = st.session_state["colibri_api_key"]

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
        min_value=1.0,
        max_value=6.0,
        value=st.session_state.analysis_parameters["seeing"],
        step=0.1,
        format="%.1f",
        help=(
            "Initial guess for the Full Width at Half Maximum of stars in "
            "arcseconds. Will be refined."
        ),
    )
    st.session_state.analysis_parameters["threshold_sigma"] = st.number_input(
        "Detection Threshold (sigma)",
        min_value=0.5,
        max_value=4.5,
        value=st.session_state.analysis_parameters["threshold_sigma"],
        step=0.5,
        format="%.1f",
        help=(
            "Source detection threshold in units of background "
            "standard deviation."
        ),
    )
    st.session_state.analysis_parameters["detection_mask"] = st.number_input(
        "Border Mask Size (pixels)",
        min_value=0,
        max_value=200,
        value=st.session_state.analysis_parameters["detection_mask"],
        step=5,
        help=(
            "Size of the border region (in pixels) to exclude from "
            "source detection."
        ),
    )
    st.session_state.analysis_parameters["filter_band"] = st.selectbox(
        "Calibration Filter Band",
        options=GAIA_BANDS,
        index=GAIA_BANDS.index(st.session_state.analysis_parameters["filter_band"]),
        help="Filter Magnitude band used for photometric calibration.",
    )
    st.session_state.analysis_parameters["filter_max_mag"] = st.number_input(
        "Max Calibration Mag",
        min_value=15.0,
        max_value=21.0,
        value=st.session_state.analysis_parameters["filter_max_mag"],
        step=0.5,
        format="%.1f",
        help="Faintest magnitude to use for calibration stars.",
    )
    st.session_state.analysis_parameters["astrometry_check"] = st.toggle(
        "Refine Astrometry",
        value=st.session_state.analysis_parameters["astrometry_check"],
        help=(
            "Attempt to refine WCS using detected sources before photometry "
            "(requires external solver)."
        ),
    )
    st.session_state.analysis_parameters["calibrate_cosmic_rays"] = st.toggle(
        "Remove Cosmic Rays",
        value=st.session_state.analysis_parameters["calibrate_cosmic_rays"],
        help="Detect and remove cosmic rays using the L.A.Cosmic algorithm.",
    )
    if st.session_state.analysis_parameters["calibrate_cosmic_rays"]:
        st.session_state.analysis_parameters["cr_gain"] = st.number_input(
            "CRR Gain (e-/ADU)",
            value=st.session_state.analysis_parameters["cr_gain"],
            min_value=0.1
        )
        st.session_state.analysis_parameters["cr_readnoise"] = st.number_input(
            "CRR Read Noise (e-)",
            value=st.session_state.analysis_parameters["cr_readnoise"],
            min_value=1.0
        )
        st.session_state.analysis_parameters["cr_sigclip"] = st.number_input(
            "CRR Sigma Clip",
            value=st.session_state.analysis_parameters["cr_sigclip"],
            min_value=4.0
        )

with st.sidebar.expander("🔑 API Keys", expanded=False):
    st.session_state.colibri_api_key = st.text_input(
        "Astro-Colibri UID Key (Optional)",
        value=st.session_state.get("colibri_api_key", ""),
        type="password",
        help="key for Astro-Colibri query",
    )
    st.markdown("[Get your key](https://www.astro-colibri.science)")

if st.sidebar.button("💾 Save Configuration"):
    analysis_params = dict(st.session_state.get("analysis_parameters", {}))
    observatory_params = dict(st.session_state.get("observatory_data", {}))
    colibri_api_key = st.session_state.get("colibri_api_key")
    
    # Add pre-process options to analysis_parameters
    analysis_params["astrometry_check"] = st.session_state.get(
        "astrometry_check", False
    )
    analysis_params["calibrate_cosmic_rays"] = st.session_state.get(
        "calibrate_cosmic_rays", False
    )
    params = {"analysis_parameters": analysis_params,
              "observatory_data": observatory_params,
              "colibri_api_key": colibri_api_key,
              }
    name = st.session_state.get("username", "user")
    config_filename = f"{name}_config.json"
    config_path = os.path.join(
        st.session_state.get("output_dir", f"{name}_rpp_results"), config_filename
    )
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)
        st.sidebar.success("Parameters Saved")
    except Exception as e:
        st.sidebar.error(f"Failed to save config: {e}")

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

# Add logout button at the top right if user is logged in
if st.session_state.logged_in:
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("Logged out successfully.")
        st.switch_page("pages/login.py")

# Add archived files browser to sidebar
with st.expander("📁 Archived Images", expanded=False):
    username = st.session_state.get("username", "anonymous")
    output_dir = ensure_output_directory(f"{username}_rpp_results")
    
    if os.path.exists(output_dir):
        try:
            all_files = []
            for item in os.listdir(output_dir):
                item_path = os.path.join(output_dir, item)
                if os.path.isfile(item_path):
                    # Get file stats
                    stat = os.stat(item_path)
                    file_size = stat.st_size
                    mod_time = datetime.fromtimestamp(stat.st_mtime)
                    
                    all_files.append({
                        'name': item,
                        'size': file_size,
                        'modified': mod_time,
                        'path': item_path
                    })
            
            if all_files:
                # Sort files by modification time (newest first)
                all_files.sort(key=lambda x: x['modified'], reverse=True)
                
                st.write(f"**{len(all_files)} files found**")
                
                # Show only the 5 most recent files in sidebar
                recent_files = all_files[:5]
                
                for file_info in recent_files:
                    with st.container():
                        # File name (truncated if too long)
                        display_name = file_info['name']
                        if len(display_name) > 25:
                            display_name = display_name[:22] + "..."
                        
                        st.text(display_name)
                        
                        # Size and download button in columns
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            if file_info['size'] > 1024*1024:
                                size_str = f"{file_info['size']/(1024*1024):.1f}MB"
                            elif file_info['size'] > 1024:
                                size_str = f"{file_info['size']/1024:.1f}KB"
                            else:
                                size_str = f"{file_info['size']}B"
                            st.caption(size_str)
                        
                        with col2:
                            try:
                                with open(file_info['path'], 'rb') as f:
                                    file_data = f.read()
                                
                                ext = os.path.splitext(file_info['name'])[1].lower()
                                mime_type = "application/octet-stream"
                                if ext == '.zip':
                                    mime_type = "application/zip"
                                elif ext == '.csv':
                                    mime_type = "text/csv"
                                elif ext == '.txt':
                                    mime_type = "text/plain"
                                elif ext == '.png':
                                    mime_type = "image/png"
                                elif ext == '.fits':
                                    mime_type = "application/fits"
                                
                                st.download_button(
                                    label="📥",
                                    data=file_data,
                                    file_name=file_info['name'],
                                    mime=mime_type,
                                    key=f"sidebar_download_{file_info['name']}",
                                    help=f"Download {file_info['name']}",
                                    on_click="ignore"
                                )
                            except Exception as e:
                                st.error(f"Error: {str(e)[:20]}")
                        
                        st.divider()
                
                if len(all_files) > 5:
                    st.caption(f"... and {len(all_files) - 5} more files")
                
                # Bulk download option
                if st.button("📦 Download All as ZIP", use_container_width=True):
                    try:
                        # Create timestamp for zip filename
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        zip_filename = f"{username}_results_{timestamp}.zip"
                        
                        # Create in-memory zip
                        zip_buffer = BytesIO()
                        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                            for file_info in all_files:
                                zip_file.write(file_info['path'], arcname=file_info['name'])
                        
                        zip_buffer.seek(0);
                        
                        st.download_button(
                            label=f"📦 {zip_filename}",
                            data=zip_buffer,
                            file_name=zip_filename,
                            mime="application/zip",
                            key="sidebar_bulk_download",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error creating archive: {str(e)}")
                
                # Cleanup option
                if st.button("🧹 Clean Old Files (30+ days)", use_container_width=True):
                    try:
                        cutoff_date = datetime.now() - pd.Timedelta(days=30)
                        deleted_count = 0
                        
                        for file_info in all_files:
                            if file_info['modified'] < cutoff_date:
                                try:
                                    os.remove(file_info['path'])
                                    deleted_count += 1
                                except Exception:
                                    pass
                        
                        if deleted_count > 0:
                            st.success(f"Deleted {deleted_count} old files")
                            st.rerun()
                        else:
                            st.info("No old files to delete")
                    except Exception as e:
                        st.error(f"Cleanup error: {str(e)}")
            else:
                st.info("No archived files found")
        except Exception as e:
            st.error(f"Error accessing files: {str(e)}")
    else:
        st.info("No results directory yet")

science_file = st.file_uploader(
    "Choose a FITS file for analysis", type=["fits", "fit", "fts", "fits.gz"],
    key="science_uploader"
)
science_file_path = None

st.session_state["calibrate_cosmic_rays"] = st.session_state.analysis_parameters["calibrate_cosmic_rays"]

# Update observatory_data dictionary with current session state values
st.session_state.observatory_data = {
    "name": st.session_state.observatory_name,
    "latitude": st.session_state.observatory_latitude,
    "longitude": st.session_state.observatory_longitude,
    "elevation": st.session_state.observatory_elevation,
}

# These lines are duplicating values already in analysis_parameters
st.session_state["seeing"] = st.session_state.analysis_parameters["seeing"]
st.session_state["threshold_sigma"] = st.session_state.analysis_parameters["threshold_sigma"]
st.session_state["detection_mask"] = st.session_state.analysis_parameters["detection_mask"]

# # You can remove it completely or keep it if you want to ensure consistency:
# st.session_state["analysis_parameters"].update(
#     {
#         "seeing": st.session_state.analysis_parameters["seeing"],
#         "threshold_sigma": st.session_state.analysis_parameters["threshold_sigma"],
#         "detection_mask": st.session_state.analysis_parameters["detection_mask"],
#     }
# )

catalog_name = f"{st.session_state['base_filename']}_catalog.csv"
username = st.session_state.get("username", "anonymous")
output_dir = ensure_output_directory(f"{username}_rpp_results")
st.session_state["output_dir"] = output_dir


if science_file is not None:
    suffix = os.path.splitext(science_file.name)[1]

    # Get the system temp directory
    system_tmp = tempfile.gettempdir()
    username = st.session_state.get("username", "anonymous")
    user_tmp_dir = os.path.join(system_tmp, username)

    # Create the user-specific temp directory if it doesn't exist
    os.makedirs(user_tmp_dir, exist_ok=True)

    # Now use this directory for the temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=user_tmp_dir) as tmp_file:
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
            st.session_state.log_buffer,
            f"Loaded FITS file: {science_file.name}"
        )

    # Apply cosmic ray removal if enabled
    if st.session_state.get("calibrate_cosmic_rays", False):
        st.info("Applying cosmic ray removal...")
        try:
            cr_gain = st.session_state.get("cr_gain", 1.0)
            cr_readnoise = st.session_state.get("cr_readnoise", 2.5)
            cr_sigclip = st.session_state.get("cr_sigclip", 6.5)
            clean_data, _ = detect_remove_cosmic_rays(
                science_data,
                gain=cr_gain,
                readnoise=cr_readnoise,
                sigclip=cr_sigclip,
            )
            if clean_data is not None:
                science_data = clean_data
                st.success("Cosmic ray removal applied.")
            else:
                st.warning("Cosmic ray removal did not return valid data.")
        except Exception as e:
            st.error(f"Error during cosmic ray removal: {e}")

    wcs_obj, wcs_error = safe_wcs_create(science_header)
    if wcs_obj is None:
        st.warning(f"No valid WCS found in the FITS header: {wcs_error}")

        st.write("Attempt plate solving...")
        use_astrometry = True

        if use_astrometry:
            with st.spinner("Running plate solve (this may take a while)..."):
                result = solve_with_siril(science_file_path)
                if result is None:
                    st.error("Plate solving failed. No WCS solution was returned.")
                    wcs_obj, science_header = None, None
                else:
                    wcs_obj, science_header = result
                    st.session_state["calibrated_header"] = science_header
                    st.session_state["wcs_obj"] = wcs_obj

                log_buffer = st.session_state["log_buffer"]

                if wcs_obj is not None:
                    st.success("plate solving successful!")
                    write_to_log(
                        log_buffer,
                        "Solved",
                    )

                    wcs_header_filename = (
                        f"{st.session_state['base_filename']}_wcs_header"
                    )
                    wcs_header_file_path = save_header_to_txt(
                        science_header, wcs_header_filename
                    )
                    if wcs_header_file_path:
                        st.info("Updated WCS header saved")
                else:
                    st.error("Plate solving failed")
                    write_to_log(
                        log_buffer,
                        "Failed to solve plat",
                        level="ERROR",
                    )
    else:
        st.success("Valid WCS found in the FITS header.")
        log_buffer = st.session_state["log_buffer"]
        write_to_log(log_buffer, "Valid WCS found in header")

        # Allow user to attempt plate solving even if WCS is valid
        replate_solve = st.checkbox(
            "Re-run plate solving (overwrite existing WCS)?",
            value=False,
            help="Force a new plate solve, replacing the current WCS solution."
        )

        if replate_solve:
            with st.spinner("Running plate solve (this may take a while)..."):
                result = solve_with_siril(science_file_path)
                if result is None:
                    st.error("Plate solving failed. No WCS solution was returned.")
                    wcs_obj, science_header = None, None
                else:
                    wcs_obj, science_header = result
                    st.session_state["calibrated_header"] = science_header
                    st.session_state["wcs_obj"] = wcs_obj

                log_buffer = st.session_state["log_buffer"]

                if wcs_obj is not None:
                    st.success("Plate solving successful!")
                    write_to_log(
                        log_buffer,
                        "Solved plate (forced re-solve)",
                    )

                    wcs_header_filename = (
                        f"{st.session_state['base_filename']}_wcs_header"
                    )
                    wcs_header_file_path = save_header_to_txt(
                        science_header, wcs_header_filename
                    )
                    if wcs_header_file_path:
                        st.info("Updated WCS header saved")
                else:
                    st.error("Plate solving failed")
                    write_to_log(
                        log_buffer,
                        "Failed to solve plate (forced re-solve)",
                        level="ERROR",
                    )

    if science_header is not None:
        log_buffer = st.session_state["log_buffer"]

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        header_filename = f"{st.session_state['base_filename']}_header"
        header_file_path = os.path.join(output_dir, f"{header_filename}.txt")

        header_file = save_header_to_txt(science_header, header_filename)
        if header_file:
            write_to_log(log_buffer, f"Saved header to {header_file}")

    log_buffer = st.session_state["log_buffer"]
    write_to_log(log_buffer, f"Loaded Image: {science_file.name}")

    if science_header is None:
        science_header = {}

    st.header("Image", anchor="science-image")

    if science_data is not None:
        try:
            # Create a side-by-side plot using matplotlib subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * FIGURE_SIZES["medium"][0], FIGURE_SIZES["medium"][1]))
            # ZScale Visualization
            ax1.set_title("ZScale Visualization")
            try:
                norm = ImageNormalize(science_data, interval=ZScaleInterval())
                im1 = ax1.imshow(science_data, norm=norm, origin="lower", cmap="viridis")
            except Exception as norm_error:
                st.warning(f"ZScale normalization failed: {norm_error}. Using simple normalization.")
                vmin, vmax = np.percentile(science_data, [1, 99])
                im1 = ax1.imshow(science_data, vmin=vmin, vmax=vmax, origin="lower", cmap="viridis")
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
                    # Add small histogram inset
                    ax_inset = fig.add_axes([0.65, 0.15, 0.15, 0.25])
                    ax_inset.hist(data_scaled.flatten(), bins=50, color='skyblue', alpha=0.8)
                    ax_inset.set_title('Pixel Distribution', fontsize=8)
                    ax_inset.tick_params(axis='both', which='major', labelsize=6)
                else:
                    st.warning("No finite values in image data for histogram equalization")
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

    st.subheader("Image Statistics")
    if science_data is not None:
        stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
        stats_col1.metric("Mean", f"{np.mean(science_data):.3f}")
        stats_col2.metric("Median", f"{np.median(science_data):.3f}")
        stats_col3.metric("Rms", f"{np.std(science_data):.3f}")
        stats_col4.metric("Min", f"{np.min(science_data):.3f}")
        stats_col5.metric("Max", f"{np.max(science_data):.3f}")

        pixel_size_arcsec, pixel_scale_source = extract_pixel_scale(science_header)
        st.metric(
            "Mean Pixel Scale (arcsec/pixel)",
            f"{pixel_size_arcsec:.2f}",
        )
        write_to_log(
            log_buffer,
            f"Pixel scale: {pixel_size_arcsec:.2f} arcsec/pixel ({pixel_scale_source})",
        )
        seeing = st.session_state.analysis_parameters["seeing"]
        mean_fwhm_pixel = seeing / pixel_size_arcsec
        st.metric("Mean FWHM from seeing (pixels)", f"{mean_fwhm_pixel:.2f}")
        write_to_log(
            log_buffer,
            f"Seeing FWHM: {seeing:.2f} arcsec ({mean_fwhm_pixel:.2f} pixels)",
        )

        ra_val, dec_val, coord_source = extract_coordinates(science_header)
        if ra_val is not None and dec_val is not None:
            st.write(f"Target: RA={round(ra_val, 4)}°, DEC={round(dec_val, 4)}°")
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
            for ra_key in ["RA", "OBJRA", "RA---", "CRVAL1"]:
                if ra_key in science_header:
                    st.session_state["valid_ra"] = science_header[ra_key]
                    break

            for dec_key in ["DEC", "OBJDEC", "DEC---", "CRVAL2"]:
                if dec_key in science_header:
                    st.session_state["valid_dec"] = science_header[dec_key]
                    break

        if "valid_ra" in st.session_state and "valid_dec" in st.session_state:
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
            st.write(f"Airmass: {air:.2f}")
        except Exception as e:
            st.warning(f"Error calculating airmass: {e}")
            air = 0.0
            st.write(f"Using default airmass: {air:.2f}")

        exposure_time_science = science_header.get(
            "EXPOSURE", science_header.get("EXPTIME", 1.0)
        )

        zero_point_button_disabled = science_file is None
        if st.button(
            "Photometric Calibration",
            disabled=zero_point_button_disabled,
            key="run_zp",
        ):
            image_to_process = science_data
            header_to_process = science_header

            if image_to_process is not None:
                try:
                    with st.spinner(
                        "Background Extraction, FWHM Computation, Sources Detection and Photometry..."
                    ):
                        # Extract parameters from session state
                        threshold_sigma = st.session_state.analysis_parameters["threshold_sigma"]
                        detection_mask = st.session_state.analysis_parameters["detection_mask"]
                        filter_band = st.session_state.analysis_parameters["filter_band"]
                        filter_max_mag = st.session_state.analysis_parameters["filter_max_mag"]

                        result = detection_and_photometry(
                            image_to_process,
                            header_to_process,
                            mean_fwhm_pixel,
                            threshold_sigma,
                            detection_mask,
                            filter_band,
                        )
                        
                        # Handle the return values - should be a tuple of 4 elements
                        if isinstance(result, tuple) and len(result) == 4:
                            phot_table_qtable, epsf_table, daofind, bkg = result
                        else:
                            st.error(f"detection_and_photometry returned unexpected result: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
                            phot_table_qtable = epsf_table = daofind = bkg = None

                        if phot_table_qtable is not None:
                            phot_table_df = phot_table_qtable.to_pandas().copy(
                                deep=True
                            )
                        else:
                            st.error("No sources detected in the image.")
                            phot_table_df = None

                    if phot_table_df is not None:
                        with st.spinner("Cross-matching with Gaia..."):
                            matched_table = cross_match_with_gaia(
                                phot_table_qtable,
                                header_to_process,
                                pixel_size_arcsec,
                                mean_fwhm_pixel,
                                filter_band,
                                filter_max_mag,
                            )

                        if matched_table is not None:
                            st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
                            st.dataframe(matched_table.head(10))

                            with st.spinner("Calculating zero point..."):
                                zero_point_value, zero_point_std, zp_plot = (
                                    calculate_zero_point(
                                        phot_table_qtable, matched_table, filter_band, air
                                    )
                                )

                                if zero_point_value is not None:
                                    write_to_log(
                                        log_buffer,
                                        f"Zero point: {zero_point_value:.3f} ± {zero_point_std:.3f}",
                                    )
                                    write_to_log(log_buffer, f"Airmass: {air:.2f}")

                                    if "final_phot_table" in st.session_state:
                                        final_table = st.session_state[
                                            "final_phot_table"
                                        ]

                                        try:
                                            if (
                                                "epsf_photometry_result"
                                                in st.session_state
                                                and epsf_table is not None
                                            ):
                                                epsf_df = (
                                                    epsf_table.to_pandas()
                                                    if not isinstance(
                                                        epsf_table, pd.DataFrame
                                                    )
                                                    else epsf_table
                                                )

                                                epsf_x_col = (
                                                    "x_init"
                                                    if "x_init" in epsf_df.columns
                                                    else "xcenter"
                                                )
                                                epsf_y_col = (
                                                    "y_init"
                                                    if "y_init" in epsf_df.columns
                                                    else "ycenter"
                                                )

                                                final_x_col = (
                                                    "xcenter"
                                                    if "xcenter" in final_table.columns
                                                    else "x_0"
                                                )
                                                final_y_col = (
                                                    "ycenter"
                                                    if "ycenter" in final_table.columns
                                                    else "y_0"
                                                )

                                                if (
                                                    epsf_x_col in epsf_df.columns
                                                    and epsf_y_col in epsf_df.columns
                                                ):
                                                    epsf_df["match_id"] = (
                                                        epsf_df[epsf_x_col]
                                                        .round(2)
                                                        .astype(str)
                                                        + "_"
                                                        + epsf_df[epsf_y_col]
                                                        .round(2)
                                                        .astype(str)
                                                    )

                                                if (
                                                    final_x_col in final_table.columns
                                                    and final_y_col
                                                    in final_table.columns
                                                ):
                                                    final_table["match_id"] = (
                                                        final_table[final_x_col]
                                                        .round(2)
                                                        .astype(str)
                                                        + "_"
                                                        + final_table[final_y_col]
                                                        .round(2)
                                                        .astype(str)
                                                    )
                                                else:
                                                    st.warning(
                                                        f"Coordinate columns not found in final table. Available columns: {final_table.columns.tolist()}"
                                                    )

                                                epsf_cols = {}
                                                epsf_cols["match_id"] = "match_id"
                                                epsf_cols["flux_fit"] = "psf_flux_fit"
                                                epsf_cols["flux_err"] = "psf_flux_err"
                                                epsf_cols["instrumental_mag"] = (
                                                    "psf_instrumental_mag"
                                                )

                                                if (
                                                    len(epsf_cols) > 1
                                                    and "match_id" in epsf_df.columns
                                                    and "match_id" in final_table.columns
                                                ):
                                                    epsf_subset = epsf_df[
                                                        [col for col in epsf_cols.keys() if col in epsf_df.columns]
                                                    ].rename(columns=epsf_cols)

                                                final_table = pd.merge(
                                                    final_table, epsf_df, on="match_id", how="outer", indicator=True
                                                )

                                                if (
                                                    "instrumental_mag_y"
                                                    in final_table.columns
                                                ):
                                                    final_table["psf_mag"] = (
                                                        final_table[
                                                            "instrumental_mag_y"
                                                        ]
                                                        + zero_point_value
                                                        - 0.1 * air
                                                    )
                                                    st.success(
                                                        "Added PSF photometry"
                                                    )

                                                if (
                                                    "instrumental_mag_x"
                                                    in final_table.columns
                                                ):
                                                    if (
                                                        "calib_mag"
                                                        not in final_table.columns
                                                    ):
                                                        final_table[
                                                            "aperture_instrumental_mag"
                                                        ] = final_table[
                                                            "instrumental_mag_x"
                                                        ]
                                                        final_table[
                                                            "aperture_mag"
                                                        ] = (
                                                            final_table[
                                                                "instrumental_mag_x"
                                                            ]
                                                            + zero_point_value
                                                            - 0.1 * air
                                                        )
                                                    else:
                                                        final_table = final_table.rename(
                                                            columns={
                                                                "instrumental_mag_x": "aperture_instrumental_mag",
                                                                "calib_mag": "aperture_mag",
                                                                "instrumental_mag_y": "psf_instrumental_mag",
                                                                "ra_y": "ra",
                                                                "dec_y": "dec",
                                                                "id_x": "id",
                                                            }
                                                        )
                                                    st.success(
                                                        "Added Aperture photometry"
                                                    )

                                                final_table.drop(
                                                    "match_id", axis=1, inplace=True
                                                )

                                            csv_buffer = StringIO()

                                            cols_to_drop = []
                                            for col_name in [
                                                "ra_x",
                                                "dec_x",
                                                "id_y",
                                                "group_id",
                                                "group_size",
                                                "x_init",
                                                "y_init",
                                                "flux_init",
                                                "sky_center.ra",
                                                "sky_center.dec",
                                                "inter_detected",
                                                "local_bkg",
                                                "npixfit",
                                                "x_err",
                                                "y_err"
                                            ]:
                                                if col_name in final_table.columns:
                                                    cols_to_drop.append(col_name)

                                            if cols_to_drop:
                                                final_table = final_table.drop(
                                                    columns=cols_to_drop
                                                )

                                            if "match_id" in final_table.columns:
                                                final_table.drop(
                                                    "match_id", axis=1, inplace=True
                                                )

                                            final_table["zero_point"] = zero_point_value
                                            final_table["zero_point_std"] = (
                                                zero_point_std
                                            )
                                            final_table["airmass"] = air

                                            st.session_state["final_phot_table"] = final_table

                                            st.subheader("Final Photometry Catalog")
                                            st.dataframe(final_table.head(10))

                                            st.success(
                                                f"Catalog includes {len(final_table)} sources."
                                            )

                                            # Plot histogram of aperture_mag and psf_mag before catalog enhancement
                                            st.subheader("Magnitude Distribution (Aperture vs PSF)")
                                            
                                            # Create and display the magnitude distribution plots
                                            fig_mag = plot_magnitude_distribution(final_table, log_buffer)
                                            st.pyplot(fig_mag)

                                            # Save the histogram as an image file
                                            try:
                                                base_filename = st.session_state.get("base_filename", "photometry")
                                                username = st.session_state.get("username", "anonymous")
                                                output_dir = ensure_output_directory(f"{username}_rpp_results")
                                                hist_filename = f"{base_filename}_histogram_mag.png"
                                                hist_filepath = os.path.join(output_dir, hist_filename)
                                                fig_mag.savefig(hist_filepath, dpi=120, bbox_inches="tight")
                                                write_to_log(log_buffer, f"Saved magnitude histogram plot: {hist_filename}")
                                            except Exception as e:
                                                st.warning(f"Could not save magnitude histogram plot: {e}")
                                            
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
                                                colibri_api_key = st.session_state.get("colibri_api_key", "")
                                                final_table = enhance_catalog(
                                                    colibri_api_key,
                                                    final_table,
                                                    matched_table,
                                                    header_to_process,
                                                    pixel_size_arcsec,
                                                    search_radius_arcsec=search_radius,
                                                )
                                            elif final_table is not None:
                                                st.warning(
                                                    "RA/DEC coordinates not available for catalog cross-matching"
                                                )
                                            else:
                                                st.error(
                                                    "Final photometry table is None - cannot perform cross-matching"
                                                )

                                            final_table.to_csv(csv_buffer, index=False)
                                            csv_data = csv_buffer.getvalue()

                                            timestamp_str = datetime.now().strftime(
                                                "%Y%m%d_%H%M%S"
                                            )
                                            base_catalog_name = catalog_name
                                            if base_catalog_name.endswith(".csv"):
                                                base_catalog_name = base_catalog_name[
                                                    :-4
                                                ]
                                            filename = f"{base_catalog_name}.csv"

                                            catalog_path = os.path.join(
                                                output_dir, filename
                                            )

                                            with open(catalog_path, "w") as f:
                                                f.write(csv_data)

                                        except Exception as e:
                                            st.error(f"Error preparing download: {e}")
                        else:
                            st.error(
                                "Failed to cross-match with Gaia catalog. Check WCS information in image header."
                            )
                except Exception as e:
                    st.error(f"Error during zero point calibration: {str(e)}")
                    st.exception(e)

                ra_center = None
                dec_center = None

                coord_keys = [("CRVAL1", "CRVAL2"),
                              ("RA", "DEC"),
                              ("OBJRA", "OBJDEC")]

                for ra_key, dec_key in coord_keys:
                    if ra_key in header_to_process and dec_key in header_to_process:
                        ra_center = header_to_process[ra_key]
                        dec_center = header_to_process[dec_key]
                        break

                if ra_center is not None and dec_center is not None:
                    # Check if we have a valid final photometry table
                    final_phot_table = st.session_state.get("final_phot_table")
                    
                    if (final_phot_table is not None and 
                        not final_phot_table.empty and 
                        len(final_phot_table) > 0):
                        
                        st.subheader("Aladin Catalog Viewer")
                        
                        try:
                            display_catalog_in_aladin(
                                final_table=final_phot_table,
                                ra_center=ra_center,
                                dec_center=dec_center,
                            )
                        except Exception as e:
                            st.error(f"Error displaying Aladin viewer: {str(e)}")
                            st.info("Catalog data is available but cannot be displayed in interactive viewer.")

                    st.link_button(
                        "ESA Sky Viewer",
                        f"https://sky.esa.int/esasky/?target={ra_center}%20{dec_center}&hips=DSS2+color&fov=1.0&projection=SIN&cooframe=J2000&sci=true&lang=en",
                        help="Open ESA Sky with the same target coordinates",
                    )

                    # Only provide download buttons if processing was completed
                    if final_phot_table is not None:
                        provide_download_buttons(output_dir)
                        cleanup_temp_files()
                        zip_rpp_results_on_exit(science_file, output_dir)
                else:
                    st.warning(
                        "Could not determine coordinates from image header. Cannot display ESASky or interactive viewer."
                    )
else:
    st.text(
        "👆 Please upload an image FITS file to start.",
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
    with open(log_filepath, "w", encoding='utf-8') as f:
        f.write(log_buffer.getvalue())
    write_to_log(log_buffer, f"Log saved to {log_filepath}")

st.markdown("---")
