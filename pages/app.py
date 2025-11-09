# Standard Library Imports
import sys
import os
import zipfile
import base64
import json
import tempfile
import warnings
from datetime import datetime
from io import BytesIO
from types import SimpleNamespace

# Third-Party Imports
import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize

# Local Application Imports
from src.tools import (
    FIGURE_SIZES,
    GAIA_BANDS,
    extract_coordinates,
    extract_pixel_scale,
    get_base_filename,
    safe_wcs_create,
    ensure_output_directory,
    cleanup_temp_files,
    initialize_log,
    write_to_log,
    zip_rpp_results_on_exit,
    save_header_to_txt,
    fix_header,
    save_catalog_files,
)

from src.pipeline import (
    calculate_zero_point,
    detection_and_photometry,
    detect_remove_cosmic_rays,
    airmass,
)

from src.astrometry import solve_with_astrometrynet

from src.xmatch_catalogs import cross_match_with_gaia, enhance_catalog

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


def load_fits_data(_file):
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
    if _file is not None:
        # Read bytes and guard against empty uploads
        try:
            file_content = _file.read()
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            return None, None

        if not file_content:
            st.error("Uploaded FITS file is empty.")
            return None, None

        # Try opening the FITS file and handle corrupt/invalid FITS gracefully
        try:
            hdul = fits.open(BytesIO(file_content), mode="readonly")
        except OSError as e:
            st.error(f"Invalid or corrupt FITS file: {e}")
            return None, None
        except Exception as e:
            st.error(f"Error opening FITS file: {e}")
            return None, None

        try:
            data = hdul[0].data
            hdul.verify("fix")
            header = hdul[0].header

            if data is None:
                for i, hdu in enumerate(hdul[1:], 1):
                    if hasattr(hdu, "data") and hdu.data is not None:
                        data = hdu.data
                        header = hdu.header
                        if st.session_state.get("log_buffer"):
                            write_to_log(
                                st.session_state["log_buffer"],
                                f"Primary HDU has no data. Using data from HDU #{i}.",
                            )
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

            # Apply header fixes before returning
            try:
                original_header = header.copy()
                fixed_header = fix_header(header)

                # Check if any fixes were applied
                fixes_applied = []

                # Check for specific fixes
                if "CTYPE1" in fixed_header and "CTYPE1" in original_header:
                    if str(fixed_header["CTYPE1"]) != str(original_header["CTYPE1"]):
                        fixes_applied.append(
                            f"CTYPE1: {original_header['CTYPE1']} → {fixed_header['CTYPE1']}"
                        )

                if "CTYPE2" in fixed_header and "CTYPE2" in original_header:
                    if str(fixed_header["CTYPE2"]) != str(original_header["CTYPE2"]):
                        fixes_applied.append(
                            f"CTYPE2: {original_header['CTYPE2']} → {fixed_header['CTYPE2']}"
                        )

                # Check for added keywords
                for key in ["CRPIX1", "CRPIX2", "EQUINOX", "RADESYS"]:
                    if key in fixed_header and key not in original_header:
                        fixes_applied.append(f"Added {key}: {fixed_header[key]}")

                # Check for CD matrix fixes
                for key in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]:
                    if key in fixed_header and key in original_header:
                        if abs(fixed_header[key] - original_header.get(key, 0)) > 1e-10:
                            fixes_applied.append(f"Fixed {key}")

                if fixes_applied:
                    st.info("Applied header fixes")
                    if st.session_state.get("log_buffer"):
                        write_to_log(
                            st.session_state["log_buffer"],
                            f"Header fixes applied: {'; '.join(fixes_applied)}",
                        )

                header = fixed_header

            except Exception as fix_error:
                st.warning(f"Header fixing encountered an issue: {str(fix_error)}")
                if st.session_state.get("log_buffer"):
                    write_to_log(
                        st.session_state["log_buffer"],
                        f"Header fixing failed: {str(fix_error)}",
                        level="WARNING",
                    )

            return data, header

        except Exception as e:
            st.error(f"Error loading FITS file: {str(e)}")
            return None, None
        finally:
            try:
                hdul.close()
            except Exception:
                pass

    return None, None


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

    for idx, row in final_table[cols_to_iterate].iterrows():
        ra_val = row[ra_col]
        dec_val = row[dec_col]
        if pd.notna(ra_val) and pd.notna(dec_val):
            try:
                source = {"ra": float(ra_val), "dec": float(dec_val)}
            except (ValueError, TypeError):
                continue

            psf_mag = None
            aperture_mag = None

            # Get PSF magnitude
            if mag_col in present_optional_cols and pd.notna(row[mag_col]):
                try:
                    psf_mag = float(row[mag_col])
                    source["psf_mag"] = psf_mag
                except (ValueError, TypeError):
                    pass

            # Get aperture magnitude (try multiple aperture columns)
            aperture_mag_cols = ["aperture_mag_r1.5", "aperture_mag", "calib_mag"]
            for ap_col in aperture_mag_cols:
                if ap_col in present_optional_cols and pd.notna(row[ap_col]):
                    try:
                        aperture_mag = float(row[ap_col])
                        source["aperture_mag"] = aperture_mag
                        break
                    except (ValueError, TypeError):
                        continue

            # Handle catalog matches - collect individual catalog IDs
            catalog_matches = {}

            # SIMBAD matches
            if "simbad_main_id" in present_optional_cols and pd.notna(
                row["simbad_main_id"]
            ):
                simbad_id = str(row["simbad_main_id"]).strip()
                if simbad_id and simbad_id not in ["", "nan", "None"]:
                    catalog_matches["SIMBAD"] = simbad_id

            # SkyBoT matches (solar system objects)
            if "skybot_NAME" in present_optional_cols and pd.notna(row["skybot_NAME"]):
                skybot_name = str(row["skybot_NAME"]).strip()
                if skybot_name and skybot_name not in ["", "nan", "None"]:
                    catalog_matches["SkyBoT"] = skybot_name

            # AAVSO VSX matches (variable stars)
            if "aavso_Name" in present_optional_cols and pd.notna(row["aavso_Name"]):
                aavso_name = str(row["aavso_Name"]).strip()
                if aavso_name and aavso_name not in ["", "nan", "None"]:
                    catalog_matches["AAVSO"] = aavso_name

            # Astro-Colibri matches
            if "astrocolibri_name" in present_optional_cols and pd.notna(
                row["astrocolibri_name"]
            ):
                colibri_name = str(row["astrocolibri_name"]).strip()
                if colibri_name and colibri_name not in ["", "nan", "None"]:
                    catalog_matches["Astro-Colibri"] = colibri_name

            # Quasar matches
            if "qso_name" in present_optional_cols and pd.notna(row["qso_name"]):
                qso_name = str(row["qso_name"]).strip()
                if qso_name and qso_name not in ["", "nan", "None"]:
                    catalog_matches["QSO"] = qso_name

            # GAIA matches (if calibration star)
            if "gaia_calib_star" in present_optional_cols and row.get(
                "gaia_calib_star", False
            ):
                catalog_matches["GAIA"] = "Calibration Star"

            source["catalog_matches"] = catalog_matches
            source_id = f"{fallback_id_prefix} {idx + 1}"

            # First, check if there's an "id" column in the table (the real catalog ID)
            if "id" in final_table.columns and pd.notna(row.get("id")):
                id_value = str(row["id"]).strip()
                if id_value and id_value not in ["nan", "None", ""]:
                    source_id = f"ID: {id_value}"

            elif id_cols:
                for id_col in id_cols:
                    if id_col in present_optional_cols and pd.notna(row[id_col]):
                        id_value = str(row[id_col]).strip()
                        if id_value and id_value not in ["nan", "None", ""]:
                            source_id = id_value
                            break

            source["name"] = source_id
            source["source_number"] = idx + 1

            # Add the raw ID value separately for display in popup
            if "id" in final_table.columns and pd.notna(row.get("id")):
                source["catalog_id"] = str(row["id"])

            # Add additional useful information
            for info_col in ["snr", "flux_fit", "fwhm"]:
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

            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Aladin Lite v3</title>
    <link rel="stylesheet" href="https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.min.css" />
</head>
<body>
    <div id="aladin-lite-div" style="width:100%;height:550px;"></div>
    <script type="text/javascript" src="https://code.jquery.com/jquery-3.6.0.min.js" charset="utf-8"></script>
    <script type="text/javascript" src="https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.js" charset="utf-8"></script>
    <script type="text/javascript">
        document.addEventListener("DOMContentLoaded", function(event) {{
            try {{
                // Initialize Aladin v3
                let aladin = A.aladin('#aladin-lite-div', {{
                    target: '{ra_center} {dec_center}',
                    fov: {fov},
                    survey: '{survey}',
                    reticleSize: 0,
                    showZoomControl: true,
                    showFullscreenControl: true,
                    showLayersControl: true,
                    showGotoControl: true,
                    showSimbadPointerControl: true
                }});

                // Create catalog overlay
                let cat = A.catalog({{
                    name: 'Photometry Results',
                    sourceSize: 12,
                    shape: 'circle',
                    color: '#00ff88',
                    onClick: 'showPopup'
                }});
                aladin.addCatalog(cat);
                
                // Decode and parse the base64 encoded JSON data
                let sourcesData = JSON.parse(atob("{sources_json_b64}"));
                let aladinSources = [];

                sourcesData.forEach(function(source) {{
                    // Build comprehensive popup content
                    let popupContent = '<div style="padding:8px; max-width:300px; font-family:Arial,sans-serif;">';
                    
                    // Header with catalog ID if available, otherwise source number
                    if(source.catalog_id) {{
                        popupContent += '<h4 style="margin:0 0 8px 0; color:#2c5282;">Source ID: ' + source.catalog_id + '</h4>';
                    }} else if(source.source_number) {{
                        popupContent += '<h4 style="margin:0 0 8px 0; color:#2c5282;">Source #' + source.source_number + '</h4>';
                    }}
                    
                    // Coordinates section
                    popupContent += '<div style="background:#f7fafc; padding:6px; margin:4px 0; border-radius:4px;">';
                    popupContent += '<strong>Coordinates:</strong><br/>';
                    popupContent += 'RA: ' + (typeof source.ra === 'number' ? source.ra.toFixed(6) : source.ra) + '°<br/>';
                    popupContent += 'Dec: ' + (typeof source.dec === 'number' ? source.dec.toFixed(6) : source.dec) + '°';
                    popupContent += '</div>';

                    // Photometry section - handle both PSF and aperture magnitudes
                    if(source.psf_mag || source.aperture_mag) {{
                        popupContent += '<div style="background:#edf2f7; padding:6px; margin:4px 0; border-radius:4px;">';
                        popupContent += '<strong>Photometry:</strong><br/>';
                        
                        // Show PSF magnitude if available
                        if(source.psf_mag && source.psf_mag !== null && source.psf_mag !== undefined) {{
                            popupContent += 'PSF Mag: ' + (typeof source.psf_mag === 'number' ? source.psf_mag.toFixed(2) : source.psf_mag) + '<br/>';
                        }}
                        
                        // Show aperture magnitude if available
                        if(source.aperture_mag && source.aperture_mag !== null && source.aperture_mag !== undefined) {{
                            popupContent += 'Aperture Mag: ' + (typeof source.aperture_mag === 'number' ? source.aperture_mag.toFixed(2) : source.aperture_mag) + '<br/>';
                        }}
                        
                        // If only aperture_mag is available and psf_mag is not, show it prominently
                        if(!source.psf_mag && source.aperture_mag) {{
                            popupContent += '<em style="color:#2d3748;">Primary magnitude: Aperture</em><br/>';
                        }}
                        
                        // Signal-to-noise ratio if available
                        if(source.snr) {{
                            popupContent += 'S/N: ' + (typeof source.snr === 'number' ? source.snr.toFixed(1) : source.snr);
                        }}
                        popupContent += '</div>';
                    }}
                    
                    // Catalog matches section
                    if(source.catalog_matches && 
                       Object.keys(source.catalog_matches).length > 0) {{
                        popupContent += '<div style="background:#e6fffa; ' +
                                       'padding:6px; margin:4px 0; border-radius:4px;">';
                        popupContent += '<strong>Catalog Matches:</strong><br/>';
                        for(let catalog in source.catalog_matches) {{
                            let catalogValue = source.catalog_matches[catalog];
                            if(catalogValue && catalogValue !== '' && 
                               catalogValue !== 'nan') {{
                                let escapedCatalogValue = catalogValue.toString()
                                    .replace(/[<>&"']/g, function(m) {{
                                    return {{'<':'&lt;', '>':'&gt;', '&':'&amp;', 
                                            '"':'&quot;', "'":"&#39;"}}[m];
                                }});
                                popupContent += '<span style="display:inline-block; ' +
                                               'background:#81e6d9; color:#234e52; ' +
                                               'padding:2px 6px; margin:1px; ' +
                                               'border-radius:3px; font-size:11px;">';
                                popupContent += catalog + ': ' + escapedCatalogValue + 
                                               '</span><br/>';
                            }}
                        }}
                        popupContent += '</div>';
                    }}

                    // Create source with v3 properties
                    let aladinSource = A.source(
                        source.ra,
                        source.dec,
                        {{ 
                            data: popupContent,
                        }}
                    );
                    aladinSources.push(aladinSource);
                }});

                // Add sources to catalog
                if (aladinSources.length > 0) {{
                    cat.addSources(aladinSources);
                    cat.updateShape();
                }}
            }} catch (error) {{
                console.error("Error initializing Aladin Lite v3 or adding sources:", error);
                document.getElementById('aladin-lite-div').innerHTML = '<p style="color:red;">Error loading Aladin v3 viewer. Try refreshing the page or use a modern browser.</p>';
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

    if log_buffer:
        write_to_log(log_buffer, "Created magnitude distribution plots")

    return fig_mag


def initialize_session_state():
    """
    Initialize all session state variables for the application.
    Ensures all required keys have default values.
    """
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
        "science_file_path": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Output directory
    if "output_dir" not in st.session_state:
        username = st.session_state.get("username", "anonymous")
        st.session_state.output_dir = ensure_output_directory(f"{username}_rpp_results")

    # Analysis Parameters
    default_analysis_params = {
        "seeing": 3.0,
        "threshold_sigma": 3.0,
        "detection_mask": 10,
        "filter_band": "phot_g_mean_mag",
        "filter_max_mag": 20.0,
        "astrometry_check": False,
        "force_plate_solve": False,
        "calibrate_cosmic_rays": False,
        "cr_gain": 1.0,
        "cr_readnoise": 2.5,
        "cr_sigclip": 6.0,
        "run_transient_finder": False,
        "transient_survey": "DSS2",
        "transient_filter": "red",
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
        "latitude": 0.0,
        "longitude": 0.0,
        "elevation": 0.0,
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

    # --- Sync session state with loaded config ---
    if "observatory_data" in st.session_state:
        obs = st.session_state["observatory_data"]
        st.session_state["observatory_name"] = obs.get("name", "")
        st.session_state["observatory_latitude"] = obs.get("latitude", 0.0)
        st.session_state["observatory_longitude"] = obs.get("longitude", 0.0)
        st.session_state["observatory_elevation"] = obs.get("elevation", 0.0)

    if "analysis_parameters" in st.session_state:
        ap = st.session_state["analysis_parameters"]
        for key in [
            "seeing",
            "threshold_sigma",
            "detection_mask",
            "astrometry_check",
            "calibrate_cosmic_rays",
            "cr_gain",
            "cr_readnoise",
            "cr_sigclip",
            "filter_band",
            "filter_max_mag",
            "run_transient_finder",
            "transient_survey",
            "transient_filter",
        ]:
            if key in ap:
                st.session_state[key] = ap[key]

    if "colibri_api_key" in st.session_state:
        st.session_state["colibri_api_key"] = st.session_state["colibri_api_key"]


def update_observatory_from_fits_header(header):
    """
    Update observatory data in session state from FITS header information.

    Parameters
    ----------
    header : astropy.io.fits.Header
        FITS header object to extract observatory information from

    Returns
    -------
    bool
        True if any observatory data was updated, False otherwise
    """
    updated = False

    # Check for observatory name
    for key in ["TELESCOP", "OBSERVER"]:
        if key in header and header[key]:
            observatory_name = str(header[key]).strip()
            if observatory_name and observatory_name not in ["", "UNKNOWN", "None"]:
                st.session_state.observatory_data["name"] = observatory_name
                st.session_state["observatory_name"] = observatory_name
                updated = True
                break

    # Check for site latitude - Fixed key name from SITELAN to SITELAT
    if "SITELAT" in header and header["SITELAT"] is not None:
        try:
            latitude = float(header["SITELAT"])
            if -90 <= latitude <= 90:  # Valid latitude range
                st.session_state.observatory_data["latitude"] = latitude
                st.session_state["observatory_latitude"] = latitude
                updated = True
        except (ValueError, TypeError):
            pass

    # Check for site longitude
    if "SITELONG" in header and header["SITELONG"] is not None:
        try:
            longitude = float(header["SITELONG"])
            if -180 <= longitude <= 180:  # Valid longitude range
                st.session_state.observatory_data["longitude"] = longitude
                st.session_state["observatory_longitude"] = longitude
                updated = True
        except (ValueError, TypeError):
            pass

    # Check for site elevation
    if "SITEELEV" in header and header["SITEELEV"] is not None:
        try:
            elevation = float(header["SITEELEV"])
            if elevation >= -400:  # Reasonable minimum (below sea level)
                st.session_state.observatory_data["elevation"] = elevation
                st.session_state["observatory_elevation"] = elevation
                updated = True
        except (ValueError, TypeError):
            pass

    return updated


######################
# Main Streamlit App #
######################

st.set_page_config(
    page_title="RAPAS Photometry Pipeline", page_icon="🔭")

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
st.markdown("[**RAPAS Project**](https://rapas.imcce.fr/) / [**Github**](https://github.com/pierfra-rocci/rpp)",
            unsafe_allow_html=True)

# Added: Quick Start Tutorial link (now displayed in an expander)
with st.expander("📘 Quick Start Tutorial"):
    try:
        with open(os.path.join("docs", "TUTORIAL.md"), "r", encoding="utf-8") as f:
            st.markdown(f.read(), unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("TUTORIAL.md not found. It should be in the `doc` folder.")

st.sidebar.markdown(f"**App Version:** _{version}_")

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
        max_value=5.0,
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
        st.session_state.analysis_parameters["cr_gain"] = st.slider(
            "CRR Gain (e-/ADU)",
            min_value=0.1,
            max_value=10.0,
            value=st.session_state.analysis_parameters["cr_gain"],
            step=0.5,
            help="Camera gain in electrons per ADU.",
        )
        st.session_state.analysis_parameters["cr_readnoise"] = st.slider(
            "CRR Read Noise (e-)",
            min_value=1.0,
            max_value=20.0,
            value=st.session_state.analysis_parameters["cr_readnoise"],
            step=0.5,
            help="Camera read noise in electrons.",
        )
        st.session_state.analysis_parameters["cr_sigclip"] = st.slider(
            "CRR Sigma Clip",
            min_value=4.0,
            max_value=10.0,
            value=st.session_state.analysis_parameters["cr_sigclip"],
            step=0.5,
            help="Sigma clipping threshold for cosmic ray detection.",
        )

with st.sidebar.expander("🔑 API Keys", expanded=False):
    st.session_state.colibri_api_key = st.text_input(
        "Astro-Colibri UID Key (Optional)",
        value=st.session_state.get("colibri_api_key", ""),
        type="password",
        help="key for Astro-Colibri query",
    )
    st.markdown("[Get your key](https://www.astro-colibri.science)")

# Add an expander for the Transient Finder
# with st.sidebar.expander("Transient Finder", expanded=False):

#     # Add a checkbox to enable/disable the transient finder
#     st.session_state.analysis_parameters['run_transient_finder'] = st.checkbox(
#         "Enable Transient Finder",
#         value=st.session_state.analysis_parameters.get('run_transient_finder', False)
#     )

#     # Add survey and filter selection
#     survey_options = ["PanSTARRS", "DSS2"]
#     survey_index = survey_options.index(st.session_state.analysis_parameters.get('transient_survey', 'DSS2'))
#     st.session_state.analysis_parameters['transient_survey'] = st.selectbox(
#         "Reference Survey",
#         options=survey_options,
#         index=survey_index,
#         help="Survey to use for the reference image (PanSTARRS has a smaller field of view limit).",
#     )

#     filter_options = ["g", "r", "i", "blue", "red"]
#     filter_index = filter_options.index(st.session_state.analysis_parameters.get('transient_filter', 'red'))
#     st.session_state.analysis_parameters['transient_filter'] = st.selectbox(
#         "Reference Filter",
#         options=filter_options,
#         index=filter_index,
#         help="Filter/band for the reference image. Options depend on the selected survey.",
#     )

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

# Add archived files browser to sidebar
with st.sidebar.expander("📁 Archived Results", expanded=False):
    username = st.session_state.get("username", "anonymous")
    output_dir = ensure_output_directory(f"{username}_rpp_results")
    display_archived_files_browser(output_dir)

with st.sidebar:
    if st.button("🧹 Clear Cache & Reset Upload"):
        clear_all_caches()

# Add logout button at the top right if user is logged in
if st.session_state.logged_in:
    st.sidebar.markdown("")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Logged in as:** {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.success("Logged out successfully.")
        st.switch_page("pages/login.py")

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

# Runtime logic to update the sessions state parameters
st.session_state["calibrate_cosmic_rays"] = st.session_state.analysis_parameters[
    "calibrate_cosmic_rays"
]

# Update observatory_data dictionary with current session state values
st.session_state.observatory_data = {
    "name": st.session_state.observatory_name,
    "latitude": st.session_state.observatory_latitude,
    "longitude": st.session_state.observatory_longitude,
    "elevation": st.session_state.observatory_elevation,
}

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

    # Apply cosmic ray removal if enabled
    if st.session_state.get("calibrate_cosmic_rays", False):
        st.info("Applying cosmic ray removal...")
        try:
            cr_gain = st.session_state.analysis_parameters.get("cr_gain", 1.0)
            cr_readnoise = st.session_state.analysis_parameters.get("cr_readnoise", 2.5)
            cr_sigclip = st.session_state.analysis_parameters.get("cr_sigclip", 6.0)
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

    # Test WCS creation with better error handling
    wcs_obj, wcs_error = safe_wcs_create(science_header)

    # Initialize force_plate_solve as False by default
    force_plate_solve = False

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

        # Show Force Plate Solving checkbox only when valid WCS exists
        force_plate_solve = st.checkbox(
            "🔄 Force Plate-Solve",
            value=st.session_state.analysis_parameters.get("force_plate_solve", False),
            help=(
                "Force plate solving even though a valid WCS is present. "
                "This will replace the existing WCS solution with a new one."
            ),
            key="force_plate_solve_main",
        )

        if force_plate_solve:
            st.info("🔄 Force plate solve enabled - will re-solve astrometry")
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
                f"Running plate solve ({plate_solve_reason}) - this may take a while..."
            ):
                result = solve_with_astrometrynet(science_file_path)
                if result is None:
                    if force_plate_solve:
                        st.error(
                            "Forced plate solving failed. Will use original WCS if available."
                        )
                        # Try to restore original WCS by reloading header
                        _, original_header = load_fits_data(science_file)
                        wcs_obj, wcs_error = safe_wcs_create(original_header)
                        if wcs_obj is not None:
                            science_header = original_header
                            st.info("Restored original WCS solution")
                        else:
                            proceed_without_wcs = True
                    else:
                        st.error("Plate solving failed. No WCS solution was returned.")
                        proceed_without_wcs = True
                else:
                    wcs_obj, science_header = result
                    st.session_state["calibrated_header"] = science_header
                    st.session_state["wcs_obj"] = wcs_obj

                log_buffer = st.session_state["log_buffer"]

                if wcs_obj is not None:
                    solve_type = (
                        "Forced plate-solve" if force_plate_solve else "Initial solve"
                    )
                    st.success(f"✅ {solve_type} successful!")
                    write_to_log(
                        log_buffer, f"Plate solving completed ({plate_solve_reason})"
                    )

                    wcs_header_filename = (
                        f"{st.session_state['base_filename']}_wcs_header"
                    )
                    wcs_header_file_path = save_header_to_txt(
                        science_header, wcs_header_filename
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
            "⚠️ Proceeding without valid WCS - photometry will be limited to instrumental magnitudes"
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

        header_file = save_header_to_txt(science_header, header_filename)
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

    st.subheader("Image Statistics")
    if science_data is not None:
        stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
        stats_col1.metric("Mean", f"{np.mean(science_data):.3f}")
        stats_col2.metric("Median", f"{np.median(science_data):.3f}")
        stats_col3.metric("Rms", f"{np.std(science_data):.3f}")
        stats_col4.metric("Min", f"{np.min(science_data):.3f}")
        stats_col5.metric("Max", f"{np.max(science_data):.3f}")

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

        st.metric(
            "Mean Pixel Scale (arcsec/pixel)",
            f"{pixel_size_arcsec:.2f}",
        )
        write_to_log(
            log_buffer,
            f"Final pixel scale: {pixel_size_arcsec:.2f} arcsec/pixel ({pixel_scale_source})",
        )
        seeing = st.session_state.analysis_parameters["seeing"]
        st.metric("Mean FWHM from seeing (pixels)", f"{mean_fwhm_pixel:.2f}")
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
            key="run_zp",
        ):
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
                            detection_mask,
                            filter_band,
                        )

                        if isinstance(result, tuple) and len(result) == 5:
                            phot_table_qtable, epsf_table, daofind, bkg, w = result
                        else:
                            st.error(
                                f"detection_and_photometry returned unexpected result: {type(result)}, length: {len(result) if hasattr(result, '__len__') else 'N/A'}"
                            )
                            phot_table_qtable = epsf_table = daofind = bkg = w = None

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
                                refined_wcs=w,
                            )

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
                                                    and "match_id"
                                                    in final_table.columns
                                                ):
                                                    epsf_subset = epsf_df[
                                                        [
                                                            col
                                                            for col in epsf_cols.keys()
                                                            if col in epsf_df.columns
                                                        ]
                                                    ].rename(columns=epsf_cols)

                                                final_table = pd.merge(
                                                    final_table,
                                                    epsf_df,
                                                    on="match_id",
                                                    how="outer",
                                                    indicator=True,
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
                                                    st.success("Added PSF photometry")

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
                                                        final_table["aperture_mag"] = (
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
                                                "y_err",
                                                "calib_mag",
                                                "instrumental_mag",
                                                "snr",
                                                "_merge",
                                                "flags",
                                                "qfit",
                                                "cfit",
                                                "match_id",
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

                                            st.session_state["final_phot_table"] = (
                                                final_table
                                            )

                                            st.subheader("Final Photometry Catalog")
                                            st.dataframe(final_table.head(10))

                                            st.success(
                                                f"Catalog includes {len(final_table)} sources."
                                            )

                                            # Plot histogram of aperture_mag and psf_mag before catalog enhancement
                                            st.subheader(
                                                "Magnitude Distribution (Aperture & PSF)"
                                            )

                                            # Create and display the magnitude distribution plots
                                            fig_mag = plot_magnitude_distribution(
                                                final_table, log_buffer
                                            )
                                            st.pyplot(fig_mag)

                                            # Save the histogram as an image file
                                            try:
                                                base_filename = st.session_state.get(
                                                    "base_filename", "photometry"
                                                )
                                                username = st.session_state.get(
                                                    "username", "anonymous"
                                                )
                                                output_dir = ensure_output_directory(
                                                    f"{username}_rpp_results"
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
                                                # Ensure final_table exists before enhancement
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

                                            # Call the new function here
                                            save_catalog_files(
                                                final_table, catalog_name, output_dir
                                            )

                                        except Exception as e:
                                            st.error(f"{e}")
                                else:
                                    st.error(
                                        "Failed to cross-match with Gaia catalog. Check WCS information in image header."
                                    )
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

                # Fallback to session state coordinates if header extraction failed
                if ra_center is None or dec_center is None:
                    ra_center = st.session_state.get("valid_ra")
                    dec_center = st.session_state.get("valid_dec")

                if ra_center is not None and dec_center is not None:
                    # Check if we have a valid final photometry table
                    final_phot_table = st.session_state.get("final_phot_table")

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
                                    # Initialize the TransientFinder
                                    finder = TransientFinder(
                                        science_fits_path=st.session_state[
                                            "science_file_path"
                                        ],
                                        output_dir=st.session_state["output_dir"],
                                    )

                                    # 1. Get reference image
                                    st.write(
                                        f"Retrieving reference image from {st.session_state.transient_survey} ({st.session_state.transient_filter} band)..."
                                    )
                                    if not finder.get_reference_image(
                                        survey=st.session_state.transient_survey,
                                        filter_band=st.session_state.transient_filter,
                                    ):
                                        st.error(
                                            "❌ Failed to retrieve the reference image. Please try another survey or filter."
                                        )
                                        st.stop()
                                    st.write("✅ Reference image retrieved.")

                                    # 2. Perform subtraction
                                    st.write("Performing image subtraction...")
                                    if not finder.perform_subtraction(method="proper"):
                                        st.error("❌ Image subtraction failed.")
                                        st.stop()
                                    st.write("✅ Image subtraction complete.")

                                    # 3. Detect transients
                                    st.write("Detecting transient sources...")
                                    transients = finder.detect_transients(threshold=5.0)

                                    if transients is not None and len(transients) > 0:
                                        st.success(
                                            f"🎉 Found {len(transients)} transient candidate(s)!"
                                        )
                                        st.dataframe(transients.to_pandas())

                                        # 4. Plot results
                                        st.write("Generating result plots...")
                                        plot_path = finder.plot_results(show=False)
                                        if plot_path and os.path.exists(plot_path):
                                            st.image(
                                                plot_path,
                                                caption="Transient Detection: Science, Reference, and Difference Images",
                                            )

                                        # 5. Plot cutouts
                                        cutout_paths = finder.plot_transient_cutouts(
                                            show=False
                                        )
                                        if cutout_paths:
                                            st.write("Cutouts for each transient:")
                                            for path in cutout_paths:
                                                if os.path.exists(path):
                                                    st.image(
                                                        path,
                                                        caption=os.path.basename(path),
                                                    )
                                    else:
                                        st.info(
                                            "✅ No significant transient sources were detected."
                                        )

                                    # 6. Cleanup temporary files
                                    if not st.session_state.get(
                                        "keep_temp_files", False
                                    ):
                                        finder.cleanup_temp_files()
                                        st.write("Temporary files cleaned up.")

                                except Exception as e:
                                    st.error(
                                        f"An error occurred during the transient finding process: {e}"
                                    )
                                    st.exception(e)
                        else:
                            st.warning(
                                "Please upload a science FITS file first before running the transient finder."
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
    with open(log_filepath, "w", encoding="utf-8") as f:
        f.write(log_buffer.getvalue())
    write_to_log(log_buffer, f"Log saved to {log_filepath}")

st.markdown("---")
