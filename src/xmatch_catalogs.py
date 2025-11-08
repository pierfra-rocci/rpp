import os
import json

import streamlit as st

from datetime import datetime, timedelta
from urllib.parse import quote
import requests

import numpy as np
import pandas as pd
from astropy.table import join

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.time import Time


import astropy.units as u
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from src.tools import (
    URL,
    safe_catalog_query,
    write_to_log,
)


def cross_match_with_gaia(
    _phot_table,
    science_header,
    pixel_size_arcsec,
    mean_fwhm_pixel,
    filter_band,
    filter_max_mag,
    refined_wcs=None,
):
    """
    Cross-match detected sources with the GAIA DR3 star catalog.
    This function queries the GAIA catalog for a region matching the image field of view,
    applies filtering based on magnitude range, and matches GAIA sources to the detected sources.
    It also applies quality filters including variability, color index, and astrometric quality
    to ensure reliable photometric calibration stars.

    Parameters
    ----------
    _phot_table : astropy.table.Table

        Table containing detected source positions (underscore prevents caching issues)
    _science_header : dict or astropy.io.fits.Header
        FITS header with WCS information (underscore prevents caching issues)
    pixel_size_arcsec : float

        Pixel scale in arcseconds per pixel
    mean_fwhm_pixel : float
        FWHM in pixels, used to determine matching radius
    filter_band : str
        GAIA magnitude band to use for filtering (e.g., 'phot_g_mean_mag', 'phot_bp_mean_mag',
        'phot_rp_mean_mag' or other synthetic photometry bands)
    filter_max_mag : float
        Maximum magnitude for GAIA source filtering

    Returns
    -------
    pandas.DataFrame or None
        DataFrame containing matched sources with both measured and GAIA catalog data,
        or None if the cross-match failed or found no matches

    Notes
    -----
    - The maximum separation for matching is set to twice the FWHM in arcseconds
    - Applies a simple atmospheric extinction correction of 0.1*airmass
    - Stores the calibrated photometry table in session state as 'final_phot_table'
    - Creates and saves a plot showing the relation between GAIA and calibrated magnitudes
    """
    st.write("Cross-matching with Gaia DR3...")

    if science_header is None:
        st.warning("No header information available. Cannot cross-match with Gaia.")
        return None

    try:
        if refined_wcs is not None:
            w = refined_wcs
            st.info("Using refined WCS for Gaia cross-matching, if possible.")
        else:
            w = WCS(science_header)
            st.info("Using header WCS for Gaia cross-matching")
    except Exception as e:
        st.error(f"Error creating WCS: {e}")
        return None

    try:
        source_positions_pixel = np.transpose(
            (_phot_table["xcenter"], _phot_table["ycenter"])
        )
        source_positions_sky = w.pixel_to_world(
            source_positions_pixel[:, 0], source_positions_pixel[:, 1]
        )
    except Exception as e:
        st.error(f"Error converting pixel positions to sky coordinates: {e}")
        return None

    try:
        # Validate RA/DEC coordinates before using them
        if "RA" not in science_header or "DEC" not in science_header:
            st.error("Missing RA/DEC coordinates in header")
            return None

        image_center_ra_dec = [science_header["RA"], science_header["DEC"]]

        # Validate coordinate values
        if not (0 <= image_center_ra_dec[0] <= 360) or not (
            -90 <= image_center_ra_dec[1] <= 90
        ):
            st.error(
                f"Invalid coordinates: RA={image_center_ra_dec[0]}, DEC={image_center_ra_dec[1]}"
            )
            return None

        # Calculate search radius (divided by 1.5 to avoid field edge effects)
        gaia_search_radius_arcsec = (
            max(science_header["NAXIS1"], science_header["NAXIS2"])
            * pixel_size_arcsec
            / 1.5
        )
        radius_query = gaia_search_radius_arcsec * u.arcsec

        st.write(
            f"Querying Gaia in a radius of {round(radius_query.value / 60.0, 2)} arcmin."
        )

        # Set Gaia data release
        Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

        # Create a SkyCoord object for more reliable coordinate handling
        center_coord = SkyCoord(
            ra=image_center_ra_dec[0], dec=image_center_ra_dec[1], unit="deg"
        )

        try:
            job = Gaia.cone_search(center_coord, radius=radius_query)
            gaia_table = job.get_results()

            st.info(
                f"Retrieved {len(gaia_table) if gaia_table is not None else 0} sources from Gaia"
            )

            # Different query strategies based on filter band
            if (
                filter_band
                not in ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
                and gaia_table is not None
                and len(gaia_table) > 0
            ):
                # Create a comma-separated list of source_ids (limit to 1000)
                max_sources = min(len(gaia_table), 1000)
                source_ids = list(gaia_table["source_id"][:max_sources])
                source_ids_str = ",".join(str(id) for id in source_ids)
                # Query synthetic photometry just for these specific sources
                synth_query = f"""
                SELECT source_id, c_star, u_jkc_mag, v_jkc_mag, b_jkc_mag,
                r_jkc_mag, i_jkc_mag, u_sdss_mag, g_sdss_mag, r_sdss_mag,
                i_sdss_mag, z_sdss_mag
                FROM gaiadr3.synthetic_photometry_gspc
                WHERE source_id IN ({source_ids_str})
                """
                synth_job = Gaia.launch_job(query=synth_query)
                synth_table = synth_job.get_results()

                # Join the two tables
                if synth_table is not None and len(synth_table) > 0:
                    st.info(
                        f"Retrieved {len(synth_table)} synthetic photometry entries"
                    )
                    gaia_table = join(
                        gaia_table, synth_table, keys="source_id", join_type="right"
                    )

        except Exception as cone_error:
            st.warning(f"Gaia query failed: {cone_error}")
            return None
    except KeyError as ke:
        st.error(f"Missing header keyword: {ke}")
        return None
    except Exception as e:
        st.error(f"Error querying Gaia: {e}")
        return None

    st.write(gaia_table)

    if gaia_table is None or len(gaia_table) == 0:
        st.warning("No Gaia sources found within search radius.")
        return None

    try:
        mag_filter = gaia_table[filter_band] < filter_max_mag
        var_filter = gaia_table["phot_variable_flag"] != "VARIABLE"
        color_index_filter = (gaia_table["bp_rp"] > -1) & (gaia_table["bp_rp"] < 2)
        astrometric_filter = gaia_table["ruwe"] < 1.6

        combined_filter = (
            mag_filter & var_filter & color_index_filter & astrometric_filter
        )

        gaia_table_filtered = gaia_table[combined_filter]

        if len(gaia_table_filtered) == 0:
            st.warning(
                f"No Gaia sources found within magnitude range {filter_band} < {filter_max_mag}."
            )
            return None

        st.write(f"Filtered Gaia catalog to {len(gaia_table_filtered)} sources.")
    except Exception as e:
        st.error(f"Error filtering Gaia catalog: {e}")
        return None

    try:
        gaia_skycoords = SkyCoord(
            ra=gaia_table_filtered["ra"], dec=gaia_table_filtered["dec"], unit="deg"
        )
        idx, d2d, _ = source_positions_sky.match_to_catalog_sky(gaia_skycoords)

        max_sep_constraint = 2 * mean_fwhm_pixel * pixel_size_arcsec * u.arcsec
        gaia_matches = d2d < max_sep_constraint

        matched_indices_gaia = idx[gaia_matches]
        matched_indices_phot = np.where(gaia_matches)[0]

        if len(matched_indices_gaia) == 0:
            st.warning("No Gaia matches found within the separation constraint.")
            return None

        matched_table_qtable = _phot_table[matched_indices_phot]

        matched_table = matched_table_qtable.to_pandas()
        matched_table["gaia_index"] = matched_indices_gaia
        matched_table["gaia_separation_arcsec"] = d2d[gaia_matches].arcsec

        # Add the filter_band column from the filtered Gaia table
        matched_table[filter_band] = gaia_table_filtered[filter_band][
            matched_indices_gaia
        ]

        valid_gaia_mags = np.isfinite(matched_table[filter_band])
        matched_table = matched_table[valid_gaia_mags]

        # Remove sources with SNR < 1 before zero point calculation
        if "snr" in matched_table.columns:
            matched_table = matched_table[matched_table["snr"] >= 1]

        st.success(f"Found {len(matched_table)} Gaia matches after filtering.")
        return matched_table

    except Exception as e:
        st.error(f"Error during cross-matching: {e}")
        return None


def enhance_catalog(
    api_key,
    final_table,
    matched_table,
    header,
    pixel_scale_arcsec,
    search_radius_arcsec=60,
):
    """
    Enhance a photometric catalog with cross-matches from multiple
    astronomical databases.

    This function queries several catalogs and databases including:
    - GAIA DR3 (using previously matched sources)
    - Astro-Colibri (for known astronomical sources)
    - SIMBAD (for object identifications and classifications)
    - SkyBoT (for solar system objects)
    - AAVSO VSX (for variable stars)
    - VizieR VII/294 (for quasars)

    Parameters
    ----------
    final_table : pandas.DataFrame
        Final photometry catalog with RA/Dec coordinates
    matched_table : pandas.DataFrame
        Table of already matched GAIA sources (used for calibration)
    header : dict or astropy.io.fits.Header
        FITS header with observation information
    pixel_scale_arcsec : float
        Pixel scale in arcseconds per pixel
    search_radius_arcsec : float, optional
        Search radius for cross-matching in arcseconds, default=60

    Returns
    -------
    pandas.DataFrame
        Input dataframe with added catalog information including:
        - GAIA data for calibration stars
        - Astro-Colibri source identifications
        - SIMBAD identifications and object types
        - Solar system object identifications from SkyBoT
        - Variable star information from AAVSO VSX
        - Quasar information from VizieR VII/294
        - A summary column 'catalog_matches' listing all catalog matches

    Notes
    -----
       The function shows progress updates in the Streamlit interface and creates
    a summary display of matched objects. Queries are made with appropriate
    error handling to prevent failures if any catalog service is unavailable.
    API requests are processed in batches to avoid overwhelming servers.
    """
    # Add input validation at the beginning
    if final_table is None:
        st.error("final_table is None - cannot enhance catalog")
        return None

    if len(final_table) == 0:
        st.warning("No sources to cross-match with catalogs.")
        return final_table

    # Ensure we have valid RA/Dec coordinates in final_table
    if "ra" not in final_table.columns or "dec" not in final_table.columns:
        st.error("final_table must contain 'ra' and 'dec' columns for cross-matching")
        return final_table

    # Make a copy to avoid modifying the original table
    enhanced_table = final_table.copy()

    # Filter out sources with NaN coordinates at the beginning
    valid_coords_mask = (
        pd.notna(enhanced_table["ra"])
        & pd.notna(enhanced_table["dec"])
        & np.isfinite(enhanced_table["ra"])
        & np.isfinite(enhanced_table["dec"])
    )

    if not valid_coords_mask.any():
        st.error("No sources with valid RA/Dec coordinates found for cross-matching")
        return enhanced_table

    num_invalid = len(enhanced_table) - valid_coords_mask.sum()
    if num_invalid > 0:
        st.warning(
            f"Excluding {num_invalid} sources with invalid coordinates from cross-matching"
        )

    # Compute field of view (arcmin) ONCE and use everywhere
    field_center_ra = None
    field_center_dec = None
    field_width_arcmin = 35.  # fallback default
    if header is not None:
        if "CRVAL1" in header and "CRVAL2" in header:
            field_center_ra = float(header["CRVAL1"])
            field_center_dec = float(header["CRVAL2"])
        elif "RA" in header and "DEC" in header:
            field_center_ra = float(header["RA"])
            field_center_dec = float(header["DEC"])
        elif "OBJRA" in header and "OBJDEC" in header:
            field_center_ra = float(header["OBJRA"])
            field_center_dec = float(header["OBJDEC"])
        # Compute field width if possible
        if "NAXIS1" in header and "NAXIS2" in header and pixel_scale_arcsec:
            field_width_arcmin = (
                max(header.get("NAXIS1", 1000), header.get("NAXIS2", 1000))
                * pixel_scale_arcsec
                / 60.0
            )

    status_text = st.empty()
    status_text.write("Starting cross-match process...")

    if matched_table is not None and len(matched_table) > 0:
        status_text.write("Adding Gaia calibration matches...")

        if "xcenter" in enhanced_table.columns and "ycenter" in enhanced_table.columns:
            enhanced_table["match_id"] = (
                enhanced_table["xcenter"].round(2).astype(str)
                + "_"
                + enhanced_table["ycenter"].round(2).astype(str)
            )

        if "xcenter" in matched_table.columns and "ycenter" in matched_table.columns:
            matched_table["match_id"] = (
                matched_table["xcenter"].round(2).astype(str)
                + "_"
                + matched_table["ycenter"].round(2).astype(str)
            )

            gaia_cols = [
                col
                for col in matched_table.columns
                if any(x in col for x in ["gaia", "phot_"])
            ]
            gaia_cols.append("match_id")

            gaia_subset = matched_table[gaia_cols].copy()

            rename_dict = {}
            for col in gaia_subset.columns:
                if col != "match_id" and not col.startswith("gaia_"):
                    rename_dict[col] = f"gaia_{col}"

            if rename_dict:
                gaia_subset = gaia_subset.rename(columns=rename_dict)

            enhanced_table = pd.merge(
                enhanced_table, gaia_subset, on="match_id", how="left"
            )

            enhanced_table["gaia_calib_star"] = enhanced_table["match_id"].isin(
                matched_table["match_id"]
            )

            st.success(f"Added {len(matched_table)} Gaia calibration stars to catalog")

    if field_center_ra is not None and field_center_dec is not None:
        if not (-360 <= field_center_ra <= 360) or not (-90 <= field_center_dec <= 90):
            st.warning(
                f"Invalid coordinates: RA={field_center_ra}, DEC={field_center_dec}"
            )
        else:
            pass
    else:
        st.warning("Could not extract field center coordinates from header")

    st.info("Querying Astro-Colibri API...")

    if api_key is None:
        api_key = os.environ.get("ASTROCOLIBRI_API")
        if api_key is None:
            st.warning("No API key for ASTRO-COLIBRI provided or found")
            pass

    try:
        try:
            # Base URL of the API
            url = URL + "cone_search"

            # Request parameters (headers, body)
            headers = {"Content-Type": "application/json"}

            # Define date range for the query
            observation_date = None
            if header is not None:
                if "DATE-OBS" in header:
                    observation_date = header["DATE-OBS"]
                elif "DATE" in header:
                    observation_date = header["DATE"]

            # Set time range to ±7 days from observation date or current date
            if observation_date:
                try:
                    base_date = datetime.fromisoformat(
                        observation_date.replace("T", " ").split(".")[0]
                    )
                except (ValueError, TypeError):
                    base_date = datetime.now()
            else:
                base_date = datetime.now()

            date_min = (base_date - timedelta(days=28)).isoformat()
            date_max = (base_date + timedelta(days=28)).isoformat()

            body = {
                "uid": api_key,
                "filter": None,
                "time_range": {
                    "max": date_max,
                    "min": date_min,
                },
                "properties": {
                    "type": "cone",
                    "position": {"ra": field_center_ra, "dec": field_center_dec},
                    "radius": field_width_arcmin / 2.0,
                },
            }

            # Perform the POST request
            response = requests.post(url, headers=headers, data=json.dumps(body))

            # Process the response
            try:
                if response.status_code == 200:
                    events = response.json()["voevents"]
                else:
                    st.warning(f"url: {url}")
                    st.warning(
                        f"Request failed with status code: {response.status_code}"
                    )
            except json.JSONDecodeError:
                st.error("Request did NOT succeed : ", response.status_code)
                st.error("Error message : ", response.content.decode("UTF-8"))

        except Exception as e:
            st.error(f"Error querying Astro-Colibri API: {str(e)}")
            # Continue with function instead of returning None

        if response is not None and response.status_code == 200:
            sources = {
                "ra": [],
                "dec": [],
                "discoverer_internal_name": [],
                "type": [],
                "classification": [],
            }

            # astrostars = pd.DataFrame(source)
            enhanced_table["astrocolibri_name"] = None
            enhanced_table["astrocolibri_type"] = None
            enhanced_table["astrocolibri_classification"] = None
            for event in events:
                if "ra" in event and "dec" in event:
                    sources["ra"].append(event["ra"])
                    sources["dec"].append(event["dec"])
                    sources["discoverer_internal_name"].append(
                        event["discoverer_internal_name"]
                    )
                    sources["type"].append(event["type"])
                    sources["classification"].append(event["classification"])
            astrostars = pd.DataFrame(sources)
            st.success(f"Found {len(astrostars)} Astro-Colibri sources in field.")
            st.dataframe(astrostars)

            # Filter valid coordinates for astro-colibri matching
            valid_final_coords = enhanced_table[valid_coords_mask]

            if len(valid_final_coords) > 0 and len(astrostars) > 0:
                source_coords = SkyCoord(
                    ra=valid_final_coords["ra"].values,
                    dec=valid_final_coords["dec"].values,
                    unit="deg",
                )

                astro_colibri_coords = SkyCoord(
                    ra=astrostars["ra"],
                    dec=astrostars["dec"],
                    unit=(u.deg, u.deg),
                )

                if not isinstance(search_radius_arcsec, (int, float)):
                    raise ValueError("Search radius must be a number")

                idx, d2d, _ = source_coords.match_to_catalog_sky(astro_colibri_coords)
                matches = d2d < (10 * u.arcsec)

                # Map matches back to the original table indices
                valid_indices = list(valid_final_coords.index)

                # Use enumerate to iterate indices cleanly and avoid unpacking errors
                for i, match_idx in enumerate(idx):
                    try:
                        if not bool(matches[i]):
                            continue
                    except Exception:
                        # fallback if matches is a masked/quantity array
                        if getattr(matches[i], 'value', False) is False:
                            continue

                    original_idx = valid_indices[i]
                    # Use iloc for safe pandas indexing
                    try:
                        enhanced_table.loc[original_idx, "astrocolibri_name"] = (
                            astrostars.iloc[int(match_idx)]["discoverer_internal_name"]
                        )
                    except Exception:
                        enhanced_table.loc[original_idx, "astrocolibri_name"] = (
                            astrostars["discoverer_internal_name"].iloc[int(match_idx)]
                        )

                    try:
                        enhanced_table.loc[original_idx, "astrocolibri_type"] = (
                            astrostars.iloc[int(match_idx)]["type"]
                        )
                    except Exception:
                        enhanced_table.loc[original_idx, "astrocolibri_type"] = (
                            astrostars["type"].iloc[int(match_idx)]
                        )

                    try:
                        enhanced_table.loc[original_idx, "astrocolibri_classification"] = (
                            astrostars.iloc[int(match_idx)]["classification"]
                        )
                    except Exception:
                        enhanced_table.loc[original_idx, "astrocolibri_classification"] = (
                            astrostars["classification"].iloc[int(match_idx)]
                        )

                st.success("Astro-Colibri matched objects in field.")
            else:
                st.info("No valid coordinates available for Astro-Colibri matching")
        else:
            st.write("No Astro-Colibri sources found in the field.")
    except Exception as e:
        st.error(f"Error querying Astro-Colibri: {str(e)}")
        st.write("No Astro-Colibri sources found.")

    status_text.write("Querying SIMBAD for object identifications...")

    custom_simbad = Simbad()
    custom_simbad.add_votable_fields("otype", "main_id", "ids", "B", "V")

    st.info("Querying SIMBAD")

    try:
        center_coord = SkyCoord(ra=field_center_ra, dec=field_center_dec, unit="deg")
        simbad_result, error = safe_catalog_query(
            custom_simbad.query_region,
            "SIMBAD query failed",
            center_coord,
            radius=field_width_arcmin * u.arcmin,
        )
        if error:
            st.warning(error)
        else:
            if simbad_result is not None and len(simbad_result) > 0:
                enhanced_table["simbad_main_id"] = None
                enhanced_table["simbad_otype"] = None
                enhanced_table["simbad_ids"] = None
                enhanced_table["simbad_B"] = None
                enhanced_table["simbad_V"] = None

                # Filter valid coordinates for SIMBAD matching
                valid_final_coords = enhanced_table[valid_coords_mask]

                if len(valid_final_coords) > 0:
                    source_coords = SkyCoord(
                        ra=valid_final_coords["ra"].values,
                        dec=valid_final_coords["dec"].values,
                        unit="deg",
                    )

                    if all(col in simbad_result.colnames for col in ["ra", "dec"]):
                        try:
                            # Filter out NaN coordinates in SIMBAD result
                            simbad_valid_mask = (
                                pd.notna(simbad_result["ra"])
                                & pd.notna(simbad_result["dec"])
                                & np.isfinite(simbad_result["ra"])
                                & np.isfinite(simbad_result["dec"])
                            )

                            if not simbad_valid_mask.any():
                                st.warning(
                                    "No SIMBAD sources with valid coordinates found"
                                )
                            else:
                                simbad_filtered = simbad_result[simbad_valid_mask]

                                simbad_coords = SkyCoord(
                                    ra=simbad_filtered["ra"],
                                    dec=simbad_filtered["dec"],
                                    unit=(u.hourangle, u.deg),
                                )

                                idx, d2d, _ = source_coords.match_to_catalog_sky(
                                    simbad_coords
                                )
                                matches = d2d <= (10 * u.arcsec)

                                # Map matches back to the original table indices
                                valid_indices = valid_final_coords.index

                                for i, (match, match_idx) in enumerate(
                                    zip(matches, idx)
                                ):
                                    if match:
                                        original_idx = valid_indices[i]
                                        enhanced_table.loc[
                                            original_idx, "simbad_main_id"
                                        ] = simbad_filtered["main_id"][match_idx]
                                        enhanced_table.loc[
                                            original_idx, "simbad_otype"
                                        ] = simbad_filtered["otype"][match_idx]
                                        enhanced_table.loc[original_idx, "simbad_B"] = (
                                            simbad_filtered["B"][match_idx]
                                        )
                                        enhanced_table.loc[original_idx, "simbad_V"] = (
                                            simbad_filtered["V"][match_idx]
                                        )
                                        if "ids" in simbad_filtered.colnames:
                                            enhanced_table.loc[
                                                original_idx, "simbad_ids"
                                            ] = simbad_filtered["ids"][match_idx]

                                st.success(
                                    f"Found {sum(matches)} SIMBAD objects in field."
                                )
                        except Exception as e:
                            st.error(
                                f"Error creating SkyCoord objects from SIMBAD data: {str(e)}"
                            )
                            st.write(
                                f"Available SIMBAD columns: {simbad_result.colnames}"
                            )
                    else:
                        available_cols = ", ".join(simbad_result.colnames)
                        st.error(
                            f"SIMBAD result missing required columns. Available columns: {available_cols}"
                        )
                else:
                    st.info("No valid coordinates available for SIMBAD matching")
            else:
                st.write("No SIMBAD objects found in the field.")
    except Exception as e:
        st.error(f"SIMBAD query execution failed: {str(e)}")

    try:
        if field_center_ra is not None and field_center_dec is not None:
            if "DATE-OBS" in header:
                obs_date = header["DATE-OBS"]
            elif "DATE" in header:
                obs_date = header["DATE"]
            else:
                obs_date = Time.now().isot

            obs_time = Time(obs_date).isot

            sr_value = min(field_width_arcmin / 60.0, 1.0)  # degrees (cap at 1°)
            # Keep endpoint you were using, but the documented API is at ssp.imcce.fr
            skybot_url = (
                f"http://vo.imcce.fr/webservices/skybot/skybotconesearch_query.php?"
                f"RA={field_center_ra}&DEC={field_center_dec}&SR={sr_value}&"
                f"EPOCH={quote(str(obs_time))}&mime=json"
            )

            st.info("Querying SkyBoT for solar system objects...")

            try:
                # Prepare output columns
                enhanced_table["skybot_NAME"] = None
                enhanced_table["skybot_OBJECT_TYPE"] = None
                enhanced_table["skybot_MAGV"] = None

                response = requests.get(skybot_url, timeout=15)

                if response.status_code != 200:
                    st.warning(
                        f"SkyBoT query failed with status code {response.status_code}"
                    )
                else:
                    response_text = response.text.strip()
                    if not (response_text.startswith("{") or response_text.startswith("[")):
                        st.info("SkyBoT returned non-JSON or empty response.")
                    else:
                        try:
                            skybot_result = response.json()
                        except Exception as e_json:
                            st.warning(f"SkyBoT JSON parse failed: {e_json}")
                            skybot_result = {}

                        data = skybot_result.get("data") or skybot_result.get("result") or []
                        if not data:
                            st.info("No solar system objects found in the field.")
                        else:
                            # Build SkyCoord from returned objects, skip invalid entries
                            ra_list = []
                            dec_list = []
                            good_indices = []
                            for i_obj, obj in enumerate(data):
                                try:
                                    # handle different possible key names and types
                                    ra_val = obj.get("RA") or obj.get("ra") or obj.get("raj2000")
                                    dec_val = obj.get("DEC") or obj.get("dec") or obj.get("decj2000")
                                    if ra_val is None or dec_val is None:
                                        continue
                                    ra_f = float(ra_val)
                                    dec_f = float(dec_val)
                                    ra_list.append(ra_f)
                                    dec_list.append(dec_f)
                                    good_indices.append(i_obj)
                                except Exception:
                                    continue

                            if len(ra_list) == 0:
                                st.info("SkyBoT returned entries but no usable coordinates.")
                            else:
                                skybot_coords = SkyCoord(ra=ra_list, dec=dec_list, unit=u.deg)

                                # Filter valid coordinates for SkyBoT matching
                                valid_final_coords = enhanced_table[valid_coords_mask]
                                if len(valid_final_coords) == 0:
                                    st.info("No valid coordinates available for SkyBoT matching")
                                else:
                                    source_coords = SkyCoord(
                                        ra=valid_final_coords["ra"].values,
                                        dec=valid_final_coords["dec"].values,
                                        unit=u.deg,
                                    )

                                    # Use 2D sky matching
                                    idx, d2d, _ = source_coords.match_to_catalog_sky(skybot_coords)
                                    max_sep = 10.0 * u.arcsec
                                    matches = d2d <= max_sep

                                    # Map matches back to the original table indices
                                    valid_indices = valid_final_coords.index
                                    matched_sources = np.where(matches)[0]

                                    # Initialize catalog_matches column if missing
                                    if "catalog_matches" not in enhanced_table.columns:
                                        enhanced_table["catalog_matches"] = ""

                                    for src_i in matched_sources:
                                        original_idx = valid_indices[src_i]
                                        skybot_idx = good_indices[int(idx[src_i])]
                                        rec = data[skybot_idx]
                                        # case-insensitive safe extraction
                                        name = rec.get("NAME") or rec.get("Name") or rec.get("name")
                                        objtype = rec.get("OBJECT_TYPE") or rec.get("OBJECT") or rec.get("object_type")
                                        magv = rec.get("MAGV") or rec.get("magv") or rec.get("VMAG")
                                        enhanced_table.loc[original_idx, "skybot_NAME"] = name
                                        enhanced_table.loc[original_idx, "skybot_OBJECT_TYPE"] = objtype
                                        enhanced_table.loc[original_idx, "skybot_MAGV"] = magv

                                    has_skybot = enhanced_table["skybot_NAME"].notna()
                                    enhanced_table.loc[has_skybot, "catalog_matches"] += "SkyBoT; "
                                    st.success(f"Found {int(has_skybot.sum())} solar system objects in field.")
            except requests.exceptions.RequestException as req_err:
                st.warning(f"Request to SkyBoT failed: {req_err}")
            except Exception as e:
                st.error(f"Unexpected error during SkyBoT processing: {e}")
        else:
            st.warning("Could not determine field center for SkyBoT query")
    except Exception as e:
        st.error(f"Error in SkyBoT processing: {str(e)}")

    st.info("Querying AAVSO VSX for variable stars...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            Vizier.ROW_LIMIT = -1
            vizier_result = Vizier.query_region(
                SkyCoord(ra=field_center_ra, dec=field_center_dec, unit=u.deg),
                radius=field_width_arcmin * u.arcmin,
                catalog=["B/vsx/vsx"],
            )

            if (
                vizier_result
                and "B/vsx/vsx" in vizier_result.keys()
                and len(vizier_result["B/vsx/vsx"]) > 0
            ):
                vsx_table = vizier_result["B/vsx/vsx"]

                vsx_coords = SkyCoord(
                    ra=vsx_table["RAJ2000"], dec=vsx_table["DEJ2000"], unit=u.deg
                )

                # Filter valid coordinates for AAVSO matching
                valid_final_coords = enhanced_table[valid_coords_mask]

                if len(valid_final_coords) > 0:
                    source_coords = SkyCoord(
                        ra=valid_final_coords["ra"].values,
                        dec=valid_final_coords["dec"].values,
                        unit="deg",
                    )

                    idx, d2d, _ = source_coords.match_to_catalog_sky(vsx_coords)
                    matches = d2d <= (10 * u.arcsec)

                    enhanced_table["aavso_Name"] = None
                    enhanced_table["aavso_Type"] = None
                    enhanced_table["aavso_Period"] = None

                    # Map matches back to the original table indices
                    valid_indices = valid_final_coords.index

                    for i, (match, match_idx) in enumerate(zip(matches, idx)):
                        if match:
                            original_idx = valid_indices[i]
                            enhanced_table.loc[original_idx, "aavso_Name"] = vsx_table[
                                "Name"
                            ][match_idx]
                            enhanced_table.loc[original_idx, "aavso_Type"] = vsx_table[
                                "Type"
                            ][match_idx]
                            enhanced_table.loc[original_idx, "aavso_Period"] = (
                                vsx_table["Period"][match_idx]
                            )

                    st.success(f"Found {sum(matches)} variable stars in field.")
                else:
                    st.info("No valid coordinates available for AAVSO matching")
            else:
                st.write("No variable stars found in the field.")
    except Exception as e:
        st.error(f"Error querying AAVSO VSX: {e}")

    st.info("Querying Milliquas Catalog for quasars...")
    try:
        if field_center_ra is not None and field_center_dec is not None:
            # Set columns to retrieve from the quasar catalog
            v = Vizier(columns=["RAJ2000", "DEJ2000", "Name", "z", "Rmag"])
            v.ROW_LIMIT = -1  # No row limit

            # Query the VII/294 catalog around the field center
            result = v.query_region(
                SkyCoord(ra=field_center_ra, dec=field_center_dec, unit=(u.deg, u.deg)),
                width=field_width_arcmin * u.arcmin,
                catalog="VII/294",
            )

            if result and len(result) > 0:
                qso_table = result[0]

                # Convert to pandas DataFrame for easier matching
                qso_df = qso_table.to_pandas()
                qso_coords = SkyCoord(
                    ra=qso_df["RAJ2000"], dec=qso_df["DEJ2000"], unit=u.deg
                )

                # Filter valid coordinates for QSO matching
                valid_final_coords = enhanced_table[valid_coords_mask]

                if len(valid_final_coords) > 0:
                    # Create source coordinates for matching
                    source_coords = SkyCoord(
                        ra=valid_final_coords["ra"],
                        dec=valid_final_coords["dec"],
                        unit=u.deg,
                    )

                    # Perform cross-matching
                    idx, d2d, _ = source_coords.match_to_catalog_3d(qso_coords)
                    matches = d2d.arcsec <= 10

                    # Add matched quasar information to the final table
                    enhanced_table["qso_name"] = None
                    enhanced_table["qso_redshift"] = None
                    enhanced_table["qso_Rmag"] = None

                    # Initialize catalog_matches column if it doesn't exist
                    if "catalog_matches" not in enhanced_table.columns:
                        enhanced_table["catalog_matches"] = ""

                    # Map matches back to the original table indices
                    valid_indices = valid_final_coords.index
                    matched_sources = np.where(matches)[0]
                    matched_qsos = idx[matches]

                    for i, qso_idx in zip(matched_sources, matched_qsos):
                        original_idx = valid_indices[i]
                        enhanced_table.loc[original_idx, "qso_name"] = qso_df.iloc[
                            qso_idx
                        ]["Name"]
                        enhanced_table.loc[original_idx, "qso_redshift"] = qso_df.iloc[
                            qso_idx
                        ]["z"]
                        enhanced_table.loc[original_idx, "qso_Rmag"] = qso_df.iloc[
                            qso_idx
                        ]["Rmag"]

                    # Update the catalog_matches column for matched quasars
                    has_qso = enhanced_table["qso_name"].notna()
                    enhanced_table.loc[has_qso, "catalog_matches"] += "QSO; "

                    st.success(
                        f"Found {sum(has_qso)} quasars in field from Milliquas catalog."
                    )
                    write_to_log(
                        st.session_state.get("log_buffer"),
                        f"Found {sum(has_qso)} quasar matches in Milliquas catalog",
                        "INFO",
                    )
                else:
                    st.info("No valid coordinates available for QSO matching")
            else:
                st.warning("No quasars found in field from Milliquas catalog.")
                write_to_log(
                    st.session_state.get("log_buffer"),
                    "No quasars found in field from Milliquas catalog",
                    "INFO",
                )
    except Exception as e:
        st.error(f"Error querying VizieR Milliquas: {str(e)}")
        write_to_log(
            st.session_state.get("log_buffer"),
            f"Error in Milliquas catalog processing: {str(e)}",
            "ERROR",
        )

    return enhanced_table