import os
import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.stats import SigmaClip
from photutils.background import LocalBackground, MMMBackground
from astropy.nddata import NDData
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.modeling.models import Gaussian2D

from photutils.psf import EPSFBuilder, extract_stars, PSFPhotometry
from photutils.psf import SourceGrouper, EPSFStars

from src.tools_pipeline import FIGURE_SIZES
from src.utils import ensure_output_directory

from typing import Any, Optional, Tuple


def create_gaussian_psf_from_stars(stars, fwhm):
    """
    Create a simple Gaussian PSF model from median-combined star cutouts.

    Parameters
    ----------
    stars : EPSFStars
        Extracted star cutouts
    fwhm : float
        Full Width at Half Maximum in pixels

    Returns
    -------
    numpy.ndarray
        2D Gaussian PSF model array
    """
    try:
        # Get all valid star data arrays
        star_arrays = []
        for star in stars:
            if hasattr(star, "data") and star.data is not None:
                arr = np.asarray(star.data)
                if arr.size > 0 and not np.isnan(arr).any() and not np.all(arr == 0):
                    star_arrays.append(arr)

        if len(star_arrays) == 0:
            raise ValueError("No valid star arrays for Gaussian PSF creation")

        # Get the shape from the first star (they should all be the same size)
        shape = star_arrays[0].shape

        # Median combine the stars (normalizing each first)
        normalized_stars = []
        for arr in star_arrays:
            peak = np.max(arr)
            if peak > 0:
                normalized_stars.append(arr / peak)

        median_psf = np.median(normalized_stars, axis=0)

        # Fit a 2D Gaussian to the median combined PSF
        y, x = np.mgrid[: shape[0], : shape[1]]

        # Initial guess for Gaussian parameters
        amplitude = np.max(median_psf)
        x_mean = shape[1] / 2.0
        y_mean = shape[0] / 2.0
        sigma = fwhm / 2.355  # Convert FWHM to sigma

        # Create the Gaussian model
        gaussian = Gaussian2D(
            amplitude=amplitude,
            x_mean=x_mean,
            y_mean=y_mean,
            x_stddev=sigma,
            y_stddev=sigma,
        )

        # Evaluate the Gaussian on the grid
        gaussian_psf = gaussian(x, y)

        # Normalize to have same peak as median PSF
        gaussian_psf = gaussian_psf / np.max(gaussian_psf) * np.max(median_psf)

        st.write(
            f"Created Gaussian PSF fallback with FWHM={fwhm:.2f} pixels (sigma={sigma:.2f})"
        )

        return gaussian_psf

    except Exception as e:
        st.error(f"Error creating Gaussian PSF fallback: {e}")
        raise


def perform_psf_photometry(
    img: np.ndarray,
    photo_table: Table,
    fwhm: float,
    daostarfind: Any,
    mask: Optional[np.ndarray] = None,
    error=None,
    max_sources_for_psf: int = 500,
    max_stars_for_epsf: int = 200,
) -> Tuple[Table, Any]:
    """
    Perform PSF (Point Spread Function) photometry using an empirically-constructed PSF model.

    This function builds an empirical PSF (EPSF) from bright stars in the image
    and then uses this model to perform PSF photometry on all detected sources.
    If EPSF building fails, falls back to a simple Gaussian PSF model.

    Parameters
    ----------
    img : numpy.ndarray
        Image with sky background subtracted
    photo_table : astropy.table.Table
        Table containing source positions
    fwhm : float
        Full Width at Half Maximum in pixels, used to define extraction and fitting sizes
    daostarfind : callable
        Star detection function used as "finder" in PSF photometry
    mask : numpy.ndarray, optional
        Mask to exclude certain image areas from analysis
    error : numpy.ndarray, optional
        Error array for the image
    max_sources_for_psf : int, optional
        Maximum number of sources to extract for PSF model construction (default: 500)
        This limits the initial extraction before quality filtering
    max_stars_for_epsf : int, optional
        Maximum number of stars to use for final EPSF building (default: 200)
        Only the brightest stars are selected if more are available

    Returns
    -------
    Tuple[astropy.table.Table, photutils.psf.EPSFModel or numpy.ndarray]
        - phot_epsf_result : Table containing PSF photometry results, including fitted fluxes and positions
        - epsf : The fitted EPSF model (or Gaussian fallback) used for photometry
    """
    # Initial validation
    try:
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError(
                "Invalid image data provided. Ensure the image is a non-empty numpy array."
            )

        if mask is not None and (
            not isinstance(mask, np.ndarray) or mask.shape != img.shape
        ):
            raise ValueError(
                "Invalid mask provided. Ensure the mask is a numpy array with the same shape as the image."
            )

        if not callable(daostarfind):
            raise ValueError(
                "The 'finder' parameter must be a callable star finder, such as DAOStarFinder."
            )

        # build NDData including mask/uncertainty when available (improves extract_stars)
        try:
            from astropy.nddata import StdDevUncertainty

            nd_unc = StdDevUncertainty(error) if (error is not None) else None
        except Exception:
            nd_unc = None

        nddata = NDData(
            data=img, mask=mask if mask is not None else None, uncertainty=nd_unc
        )
    except Exception as e:
        st.error(f"Error in initial validation: {e}")
        raise

    # Filter photo_table to select only the best stars for PSF model
    # IMPORTANT: Keep original photo_table for final photometry on ALL sources
    photo_table_all_sources = photo_table  # Store original table

    try:
        st.write("Filtering stars for PSF model construction...")
        st.write(f"Starting with {len(photo_table)} sources")

        # Required columns (raise helpful error if missing)
        if "flux" not in photo_table.colnames:
            raise ValueError("photo_table missing required column 'flux'")
        if (
            "xcentroid" not in photo_table.colnames
            or "ycentroid" not in photo_table.colnames
        ):
            raise ValueError(
                "photo_table must contain 'xcentroid' and 'ycentroid' columns"
            )

        # ========== EARLY FILTERING STEP ==========
        # Pre-select top brightest sources to avoid extracting thousands of stars
        # This is ONLY for PSF model construction, not for final photometry

        # Get initial arrays from full table
        flux_initial = np.asarray(photo_table["flux"])
        xcentroid_initial = np.asarray(photo_table["xcentroid"])
        ycentroid_initial = np.asarray(photo_table["ycentroid"])

        # If we have too many sources, keep only the brightest ones before detailed filtering
        photo_table_for_psf = photo_table  # Start with full table
        if len(photo_table) > max_sources_for_psf:
            st.write(
                f"Too many sources ({len(photo_table)}). Pre-selecting top {max_sources_for_psf} brightest stars for PSF model construction..."
            )

            # Create a simple score based on flux (higher is better)
            # Only consider sources with finite flux and valid positions
            valid_for_preselect = (
                np.isfinite(flux_initial)
                & np.isfinite(xcentroid_initial)
                & np.isfinite(ycentroid_initial)
            )

            if np.sum(valid_for_preselect) == 0:
                raise ValueError("No sources with valid flux and positions found")

            # Rank by flux (brightest first)
            flux_ranks = np.argsort(flux_initial[valid_for_preselect])[::-1]

            # Create index mapping back to original table
            valid_indices = np.where(valid_for_preselect)[0]
            top_indices = valid_indices[flux_ranks[:max_sources_for_psf]]

            # Pre-filter ONLY for PSF construction (not for final photometry)
            photo_table_for_psf = photo_table[top_indices]

            st.write(
                f"Pre-selected {len(photo_table_for_psf)} brightest sources for PSF model construction"
            )
            st.write(
                f"   (Final photometry will be performed on all {len(photo_table_all_sources)} original sources)"
            )

        # ---------- START REPLACEMENT BLOCK ----------
        # NOW work with photo_table_for_psf for all subsequent operations
        # Ensure we always have an 'orig_index' mapping so we can map back to the original full table.

        if len(photo_table) > max_sources_for_psf:
            # top_indices was computed above during pre-selection
            # make a copy so adding columns does not mutate the original user's table
            photo_table_for_psf = photo_table_for_psf.copy()
            if "orig_index" not in photo_table_for_psf.colnames:
                photo_table_for_psf["orig_index"] = top_indices
        else:
            # no preselection: create an explicit mapping 0..N-1
            photo_table_for_psf = photo_table_for_psf.copy()
            if "orig_index" not in photo_table_for_psf.colnames:
                photo_table_for_psf["orig_index"] = np.arange(len(photo_table_for_psf))

        n_sources = len(photo_table_for_psf)

        def _col_arr(tbl, name, default=None):
            if name in tbl.colnames:
                return np.asarray(tbl[name])
            if default is not None:
                return (
                    np.asarray(default)
                    if np.shape(default) == (len(tbl),)
                    else np.full(len(tbl), default)
                )
            return np.full(len(tbl), np.nan)

        # Get arrays safely from photo_table_for_psf
        flux = _col_arr(photo_table_for_psf, "flux")
        roundness1 = _col_arr(photo_table_for_psf, "roundness1", np.nan)
        sharpness = _col_arr(photo_table_for_psf, "sharpness", np.nan)
        xcentroid = _col_arr(photo_table_for_psf, "xcentroid")
        ycentroid = _col_arr(photo_table_for_psf, "ycentroid")
        
        # Additional attributes for improved filtering
        # Try to get source properties that indicate size and shape
        fwhm_sources = _col_arr(photo_table_for_psf, "fwhm", np.nan)  # if available
        # Ellipticity can be inferred from roundness or derived from semi-major/minor axes if available
        semimajor = _col_arr(photo_table_for_psf, "a", np.nan)  # semi-major axis if available
        semiminor = _col_arr(photo_table_for_psf, "b", np.nan)  # semi-minor axis if available
        # Peak value for saturation detection
        peak = _col_arr(photo_table_for_psf, "peak", np.nan)  # peak flux if available

        # Get flux statistics with NaN handling (on potentially pre-filtered data)
        flux_finite = flux[np.isfinite(flux)]
        if len(flux_finite) == 0:
            raise ValueError("No sources with finite flux values found")

        flux_median = np.median(flux_finite)
        flux_std = np.std(flux_finite)

        # Define flux filtering criteria: keep bright, unsaturated stars
        # Avoid very faint sources (low S/N) and saturated sources
        flux_min = flux_median - 2 * flux_std  # Don't go too faint
        flux_max = flux_median + 2 * flux_std  # Avoid the very brightest (likely saturated)
        
        st.write(
            f"Target flux range: {flux_min:.1f} "
            f"→ {flux_max:.1f} (median ± 2σ)"
        )

        # Create individual boolean masks with explicit NaN handling
        valid_flux = np.isfinite(flux)
        valid_roundness = np.isfinite(roundness1)
        valid_sharpness = np.isfinite(sharpness)
        valid_xcentroid = np.isfinite(xcentroid)
        valid_ycentroid = np.isfinite(ycentroid)

        # Combine validity checks
        valid_all = (
            valid_flux
            & valid_roundness
            & valid_sharpness
            & valid_xcentroid
            & valid_ycentroid
        )

        if not np.any(valid_all):
            # relax: accept stars where position and flux are valid at least
            valid_all = valid_flux & valid_xcentroid & valid_ycentroid
            if not np.any(valid_all):
                raise ValueError("No sources with required valid parameters found")

        # ========== FLUX FILTERING ==========
        flux_criteria = np.zeros(n_sources, dtype=bool)
        flux_criteria[valid_flux] = (flux[valid_flux] >= flux_min) & (
            flux[valid_flux] <= flux_max
        )
        n_flux_pass = np.sum(flux_criteria)
        st.write(f"  ✓ Flux range: {n_flux_pass} sources")

        # ========== SATURATION FILTERING ==========
        # Reject sources with peak values very close to image maximum (likely saturated)
        # Estimate saturation level: use 95th percentile of peak values or provide explicit limit
        saturation_criteria = np.ones(n_sources, dtype=bool)
        valid_peak = np.isfinite(peak)
        if np.any(valid_peak):
            peak_values = peak[valid_peak]
            # Use 95th percentile as a soft saturation limit
            peak_95 = np.percentile(peak_values, 95)
            saturation_threshold = peak_95 * 0.98  # 2% below 95th percentile
            saturation_criteria[valid_peak] = (
                peak[valid_peak] < saturation_threshold
            )
            n_sat_pass = np.sum(saturation_criteria)
            st.write(
                f"  ✓ Saturation check: {n_sat_pass} "
                f"unsaturated sources (limit: {saturation_threshold:.0f})"
            )
        else:
            # No peak data available; check image max value directly
            # (will be done after star extraction if possible)
            st.write(
                "  ⓘ No peak flux available in catalog; "
                "saturation check deferred to star extraction"
            )

        # ========== SIZE/FWHM FILTERING ==========
        # Select stars with FWHM close to the expected FWHM (point-like, not extended)
        # Allow a reasonable range around the measured FWHM
        size_criteria = np.ones(n_sources, dtype=bool)
        valid_fwhm = np.isfinite(fwhm_sources)
        if np.any(valid_fwhm):
            fwhm_median = np.median(fwhm_sources[valid_fwhm])
            # Accept sources with FWHM within 0.7–1.3× median
            # (rejects extended objects and under-sampled stars)
            size_min = fwhm_median * 0.7
            size_max = fwhm_median * 1.3
            size_criteria[valid_fwhm] = (
                (fwhm_sources[valid_fwhm] >= size_min)
                & (fwhm_sources[valid_fwhm] <= size_max)
            )
            n_size_pass = np.sum(size_criteria)
            st.write(
                f"  ✓ Size/FWHM filter: {n_size_pass} "
                f"point-like sources ({size_min:.2f}–{size_max:.2f} px)"
            )
        else:
            st.write(
                "  ⓘ No FWHM data available; size filtering skipped"
            )

        # ========== SHAPE/ELLIPTICITY FILTERING ==========
        # Reject blends and optical defects using roundness and axis ratio
        roundness_criteria = np.zeros(n_sources, dtype=bool)
        roundness_criteria[valid_roundness] = (
            np.abs(roundness1[valid_roundness]) < 0.2
        )  # Stricter than before
        n_roundness_pass = np.sum(roundness_criteria)
        st.write(
            f"  ✓ Roundness (|r₁| < 0.2): {n_roundness_pass} sources"
        )

        # Add axis-ratio check if semi-major/minor axes available
        ellipticity_criteria = np.ones(n_sources, dtype=bool)
        valid_axes = np.isfinite(semimajor) & np.isfinite(semiminor)
        if np.any(valid_axes):
            # Axis ratio b/a; keep near-circular (ratio ≈ 1)
            axis_ratio = semiminor[valid_axes] / semimajor[valid_axes]
            # Ellipticity: 0 = circular, ~1 = very elongated
            ellipticity = 1.0 - axis_ratio
            ellipticity_criteria[valid_axes] = (
                ellipticity < 0.15
            )  # Reject highly elongated objects
            n_ellipticity_pass = np.sum(ellipticity_criteria)
            st.write(
                f"  ✓ Ellipticity (ε < 0.15): "
                f"{n_ellipticity_pass} sources"
            )
        else:
            st.write(
                "  ⓘ No axis data available; "
                "ellipticity filter skipped"
            )

        sharpness_criteria = np.zeros(n_sources, dtype=bool)
        sharpness_criteria[valid_sharpness] = sharpness[valid_sharpness] > 0.5
        n_sharpness_pass = np.sum(sharpness_criteria)
        st.write(f"  ✓ Sharpness (sharp > 0.5): {n_sharpness_pass} sources")

        # ========== CROWDING/ISOLATION FILTERING ==========
        # Reject stars with bright neighbors within ~40 pixels
        crowding_criteria = np.ones(n_sources, dtype=bool)
        neighbor_radius = 40.0  # pixels; typical for imaging
        # mag; reject if bright neighbor within this limit
        magnitude_threshold = 4.0
        
        # Convert flux to magnitude for neighbor comparison
        # (simple: -2.5 log10(flux))
        # Avoid log of non-positive values
        flux_safe = np.where(flux > 0, flux, 1.0)
        magnitude = -2.5 * np.log10(flux_safe)
        
        for i in range(n_sources):
            if not (
                valid_xcentroid[i]
                and valid_ycentroid[i]
                and valid_flux[i]
            ):
                crowding_criteria[i] = False
                continue
            
            # Find all neighbors within the search radius
            dx = xcentroid - xcentroid[i]
            dy = ycentroid - ycentroid[i]
            dist = np.sqrt(dx**2 + dy**2)
            
            # Exclude self (dist=0) and find neighbors
            neighbor_mask = (dist > 0.5) & (dist < neighbor_radius)
            
            if np.any(neighbor_mask):
                neighbor_mags = magnitude[neighbor_mask]
                
                # Count bright neighbors (within magnitude_threshold)
                bright_neighbors = np.sum(
                    neighbor_mags < magnitude[i] + magnitude_threshold
                )
                
                if bright_neighbors > 0:
                    crowding_criteria[i] = False
                    # Optionally log severely crowded sources
                    if bright_neighbors > 2:
                        pass  # Too crowded, reject
            
        n_crowding_pass = np.sum(crowding_criteria)
        st.write(
            f"  ✓ Isolation (no bright neighbors within "
            f"{neighbor_radius}px, <{magnitude_threshold:.1f}Δmag): "
            f"{n_crowding_pass} sources"
        )

        # ========== EDGE CRITERIA ==========
        # Keep away from edges where PSF is affected
        edge_criteria = np.zeros(n_sources, dtype=bool)
        valid_coords = valid_xcentroid & valid_ycentroid
        edge_buffer = max(3 * fwhm, 20)  # At least 3×FWHM or 20 px
        edge_criteria[valid_coords] = (
            (xcentroid[valid_coords] > edge_buffer)
            & (xcentroid[valid_coords] < img.shape[1] - edge_buffer)
            & (ycentroid[valid_coords] > edge_buffer)
            & (ycentroid[valid_coords] < img.shape[0] - edge_buffer)
        )
        n_edge_pass = np.sum(edge_criteria)
        st.write(
            f"  ✓ Not near edges (>{edge_buffer:.0f}px): "
            f"{n_edge_pass} sources"
        )

        # Combine all criteria with validity checks
        good_stars_mask = (
            valid_all
            & flux_criteria
            & saturation_criteria
            & size_criteria
            & roundness_criteria
            & ellipticity_criteria
            & sharpness_criteria
            & crowding_criteria
            & edge_criteria
        )

        # ===== APPLY THE MASK =====
        filtered_photo_table = photo_table_for_psf[good_stars_mask]
        st.write(f"Flux range: {flux_min:.1f} -> {flux_max:.1f}")
        st.write(f"Stars after quality filtering: {len(filtered_photo_table)}")

        # Save indices mapping to original full table
        orig_indices_preselection = np.asarray(photo_table_for_psf["orig_index"])
        orig_indices_filtered = orig_indices_preselection[good_stars_mask]

        st.session_state["psf_preselected_indices"] = orig_indices_preselection
        st.session_state["psf_filtered_indices"] = orig_indices_filtered

        # If too few, relax criteria (apply relaxed_mask to same preselected table)
        if len(filtered_photo_table) < 10:
            st.warning(
                f"Only {len(filtered_photo_table)} stars available for PSF model. Relaxing quality criteria..."
            )

            # Relax thresholds but maintain saturation and isolation checks (critical for PSF quality)
            
            # Relax roundness: allow more elliptical shapes
            roundness_criteria_relaxed = np.zeros(n_sources, dtype=bool)
            roundness_criteria_relaxed[valid_roundness] = (
                np.abs(roundness1[valid_roundness]) < 0.35
            )

            # Relax ellipticity: allow slightly elongated objects
            ellipticity_criteria_relaxed = np.ones(n_sources, dtype=bool)
            if np.any(valid_axes):
                axis_ratio = semiminor[valid_axes] / semimajor[valid_axes]
                ellipticity = 1.0 - axis_ratio
                ellipticity_criteria_relaxed[valid_axes] = ellipticity < 0.25  # More permissive
            
            # Relax sharpness: accept slightly broader profiles
            sharpness_criteria_relaxed = np.zeros(n_sources, dtype=bool)
            sharpness_criteria_relaxed[valid_sharpness] = (
                sharpness[valid_sharpness] > 0.3
            )

            # Relax size criterion: accept wider range of FWHMs
            size_criteria_relaxed = np.ones(n_sources, dtype=bool)
            if np.any(valid_fwhm):
                fwhm_median = np.median(fwhm_sources[valid_fwhm])
                size_min_relax = fwhm_median * 0.5
                size_max_relax = fwhm_median * 1.5
                size_criteria_relaxed[valid_fwhm] = (fwhm_sources[valid_fwhm] >= size_min_relax) & (
                    fwhm_sources[valid_fwhm] <= size_max_relax
                )
            
            # Relax flux criterion: broader range but still avoid saturation
            flux_criteria_relaxed = np.zeros(n_sources, dtype=bool)
            flux_criteria_relaxed[valid_flux] = (
                flux[valid_flux] >= flux_median - 2.5 * flux_std
            ) & (flux[valid_flux] <= flux_median + 2.5 * flux_std)
            
            # Relax isolation: allow one bright neighbor
            crowding_criteria_relaxed = np.ones(n_sources, dtype=bool)
            neighbor_radius_relaxed = 50.0  # Wider search radius
            magnitude_threshold_relaxed = 3.0  # Tighter magnitude threshold
            for i in range(n_sources):
                if not (valid_xcentroid[i] and valid_ycentroid[i] and valid_flux[i]):
                    crowding_criteria_relaxed[i] = False
                    continue
                dx = xcentroid - xcentroid[i]
                dy = ycentroid - ycentroid[i]
                dist = np.sqrt(dx**2 + dy**2)
                neighbor_mask = (dist > 0.5) & (dist < neighbor_radius_relaxed)
                if np.any(neighbor_mask):
                    bright_neighbors = np.sum(magnitude[neighbor_mask] < magnitude[i] + magnitude_threshold_relaxed)
                    if bright_neighbors > 1:  # Allow one neighbor (relaxed)
                        crowding_criteria_relaxed[i] = False

            # Keep saturation check (critical) and edge buffer
            edge_criteria_relaxed = np.zeros(n_sources, dtype=bool)
            edge_buffer_relaxed = max(2 * fwhm, 15)  # Slightly closer to edge allowed when relaxing
            edge_criteria_relaxed[valid_coords] = (
                (xcentroid[valid_coords] > edge_buffer_relaxed)
                & (xcentroid[valid_coords] < img.shape[1] - edge_buffer_relaxed)
                & (ycentroid[valid_coords] > edge_buffer_relaxed)
                & (ycentroid[valid_coords] < img.shape[0] - edge_buffer_relaxed)
            )

            relaxed_mask = (
                (valid_flux & valid_xcentroid & valid_ycentroid)
                & flux_criteria_relaxed
                & saturation_criteria  # Keep saturation check!
                & size_criteria_relaxed
                & roundness_criteria_relaxed
                & ellipticity_criteria_relaxed
                & sharpness_criteria_relaxed
                & crowding_criteria_relaxed
                & edge_criteria_relaxed
            )

            # Apply relaxed mask to the same table (photo_table_for_psf)
            filtered_photo_table = photo_table_for_psf[relaxed_mask]

            # Update original indices for filtered stars
            orig_indices_filtered = orig_indices_preselection[relaxed_mask]
            st.session_state["psf_filtered_indices"] = orig_indices_filtered

            st.write(f"After relaxing criteria: {len(filtered_photo_table)} stars")

        if len(filtered_photo_table) < 5:
            raise ValueError(
                "Too few good stars for PSF model construction. Need at least 5 stars."
            )
        # ---------- END REPLACEMENT BLOCK ----------

    except Exception as e:
        st.error(f"Error filtering stars for PSF model: {e}")
        raise

    try:
        stars_table = Table()
        # extract_stars expects 'x' and 'y' column names
        stars_table["x"] = filtered_photo_table["xcentroid"]
        stars_table["y"] = filtered_photo_table["ycentroid"]
    except Exception as e:
        st.error(f"Error preparing star positions table: {e}")
        raise

    try:
        # Ensure fit_shape accommodates 1.5*FWHM aperture radius
        # aperture_radius = fit_shape / 2.0, so:
        # fit_shape >= 2 * 1.5 * FWHM = 3 * FWHM
        fit_shape = max(11, int(3 * fwhm + 1))
        if fit_shape % 2 == 0:
            fit_shape += 1
        aperture_radius = fit_shape / 2.0
        st.write(
            f"Fitting shape: {fit_shape} pixels "
            f"(aperture radius: {aperture_radius:.2f} px = "
            f"{aperture_radius/fwhm:.2f} × FWHM)."
        )
    except Exception as e:
        st.error(f"Error calculating fitting shape: {e}")
        raise

    try:
        stars = extract_stars(nddata, stars_table, size=fit_shape)

        if isinstance(stars, list):
            if len(stars) == 0:
                raise ValueError(
                    "No stars extracted for PSF model. Check your selection criteria."
                )

        n_stars = len(stars)
        st.write(f"{n_stars} stars extracted for PSF model.")

        # basic inspection of cutouts for NaN/all-zero
        if hasattr(stars, "data") and stars.data is not None:
            if isinstance(stars.data, list) and len(stars.data) > 0:
                has_nan = any(np.isnan(star_data).any() for star_data in stars.data)
                st.write(f"NaN : {has_nan}")
            else:
                st.write("Stars data is empty or not a list")
        else:
            st.write("Stars object has no data attribute")

        if n_stars == 0:
            raise ValueError(
                "No stars extracted for PSF model. Check your selection criteria."
            )
    except Exception as e:
        st.error(f"Error extracting stars: {e}")
        raise

    try:
        # Remove stars with NaN or all-zero data
        mask_valid = []
        if hasattr(stars, "data") and isinstance(stars.data, list):
            for star_data in stars.data:
                if np.isnan(star_data).any():
                    mask_valid.append(False)
                elif np.all(star_data == 0):
                    mask_valid.append(False)
                else:
                    mask_valid.append(True)
        else:
            try:
                for i in range(len(stars)):
                    star = stars[i]
                    if np.isnan(star.data).any():
                        mask_valid.append(False)
                    elif np.all(star.data == 0):
                        mask_valid.append(False)
                    else:
                        mask_valid.append(True)
            except Exception as iter_error:
                st.warning(
                    f"Could not iterate through stars for validation: {iter_error}"
                )
                mask_valid = [True] * len(stars)

        mask_valid = np.array(mask_valid)

        # Only filter if we have any invalid stars
        if not np.all(mask_valid):
            filtered_stars = stars[mask_valid]
            if not isinstance(filtered_stars, EPSFStars):
                filtered_stars = EPSFStars(list(filtered_stars))
            stars = filtered_stars
            n_stars = len(stars)
            st.write(
                f"{n_stars} valid stars remain for PSF model after filtering invalid data."
            )
        else:
            st.write(f"All {len(stars)} stars are valid for PSF model.")

        if len(stars) == 0:
            raise ValueError("No valid stars for PSF model after filtering.")
    except Exception as e:
        st.error(f"Error filtering stars for PSF model: {e}")
        raise

    # Build EPSF model with Gaussian fallback
    try:
        # Filter out invalid Star objects and select brightest for EPSFBuilder
        valid_stars = []
        try:
            for star in stars:  # stars is already an EPSFStars object
                if hasattr(star, "data") and star.data is not None:
                    arr = np.asarray(star.data)
                    if arr.size == 0 or np.isnan(arr).any() or np.all(arr == 0):
                        continue
                    valid_stars.append(star)
        except Exception as e:
            st.warning(f"Error during initial filtering of stars for ePSF: {e}")
            valid_stars = []

        n_valid = len(valid_stars)
        st.write(f"Using {n_valid} valid Star objects for ePSF construction")

        if n_valid < 5:
            raise ValueError("Too few valid Star objects for ePSF building")

        # If there are a lot of stars, pick the brightest (by peak)
        if n_valid > max_stars_for_epsf:
            st.write(
                f"⚡ Selecting top {max_stars_for_epsf} brightest stars from {n_valid} valid stars..."
            )
            peaks = np.array([s.data.max() for s in valid_stars])
            top_idx = np.argsort(peaks)[-max_stars_for_epsf:][::-1]
            stars_for_builder = EPSFStars([valid_stars[i] for i in top_idx])
            st.write(
                f"✓ Selected {len(stars_for_builder)} brightest stars for ePSF construction."
            )
        else:
            stars_for_builder = EPSFStars(valid_stars)

        # Try EPSFBuilder with retries and progressively simpler parameters
        epsf = None
        epsf_data = None
        use_gaussian_fallback = False

        builder_attempts = [
            dict(oversampling=2, maxiters=10),
        ]

        for params in builder_attempts:
            try:
                epsf_builder = EPSFBuilder(progress_bar=False, **params)
                # Pass the wrapped object (not a raw Python list) to EPSFBuilder
                epsf, _ = epsf_builder(stars_for_builder)
                if epsf is not None:
                    st.write(
                        f"EPSFBuilder succeeded with oversampling={params.get('oversampling')}"
                    )
                    epsf_data = np.asarray(epsf.data)
                    break
            except Exception as build_error:
                st.warning(
                    f"EPSFBuilder attempt oversampling={params.get('oversampling')} failed: {build_error}"
                )
                epsf = None

        # If EPSFBuilder failed completely, use Gaussian fallback
        if epsf is None or (
            hasattr(epsf, "data")
            and (epsf.data is None or np.asarray(epsf.data).size == 0)
        ):
            st.warning(
                "⚠️ EPSFBuilder failed. Creating Gaussian PSF fallback from median-combined stars..."
            )
            use_gaussian_fallback = True

            try:
                epsf_data = create_gaussian_psf_from_stars(stars_for_builder, fwhm)

                # Create a simple object to hold the Gaussian PSF data
                class GaussianPSF:
                    def __init__(self, data, fwhm):
                        self.data = data
                        self.fwhm = fwhm
                        self.oversampling = 1

                epsf = GaussianPSF(epsf_data, fwhm)
                st.success("✓ Gaussian PSF fallback created successfully")

            except Exception as fallback_error:
                st.error(f"Gaussian PSF fallback also failed: {fallback_error}")
                raise ValueError("Both ePSF building and Gaussian fallback failed")
        else:
            epsf_data = np.asarray(epsf.data)

        # Validate epsf_data
        if epsf_data is None or epsf_data.size == 0:
            raise ValueError(
                "ePSF data is invalid or empty after attempts and fallback"
            )

        if np.isnan(epsf_data).any():
            st.warning("ePSF data contains NaN values")

        st.write(f"Shape of PSF data: {epsf_data.shape}")
        st.session_state["epsf_model"] = epsf
        st.session_state["used_gaussian_fallback"] = use_gaussian_fallback

    except Exception as e:
        st.error(f"Error fitting PSF model: {e}")
        raise

    # Ensure we have a usable PSF model for PSFPhotometry
    try:
        try:
            from photutils.psf import ImagePSF

            psf_for_phot = ImagePSF(epsf_data)
        except Exception:
            # fallback to epsf object (works on newer photutils versions)
            psf_for_phot = epsf

        # Save / display epsf as FITS and figure (best-effort)
        try:
            hdu = fits.PrimaryHDU(data=epsf_data)
            model_type = "Gaussian PSF (fallback)" if use_gaussian_fallback else "EPSF"
            hdu.header["COMMENT"] = f"PSF model: {model_type}"
            hdu.header["PSFTYPE"] = (
                "GAUSSIAN" if use_gaussian_fallback else "EPSF",
                "Type of PSF model used",
            )

            # Ensure FWHM written as a scalar
            try:
                if np is not None and (
                    hasattr(fwhm, "__len__") and not np.isscalar(fwhm)
                ):
                    fwhm_val = float(np.asarray(fwhm).mean())
                else:
                    fwhm_val = float(fwhm)
            except Exception:
                fwhm_val = float(getattr(epsf, "fwhm", 0.0) or 0.0)
            hdu.header["FWHMPIX"] = (fwhm_val, "FWHM in pixels used for extraction")

            # OVERSAMP may be tuple/list/array; convert to safe FITS-friendly value
            oversamp = getattr(epsf, "oversampling", 1 if use_gaussian_fallback else 3)
            try:
                if isinstance(oversamp, (list, tuple, np.ndarray)):
                    # store as comma-separated string to avoid illegal array header values
                    oversamp_val = ",".join(
                        str(int(x)) for x in np.asarray(oversamp).ravel()
                    )
                else:
                    oversamp_val = int(oversamp)
            except Exception:
                oversamp_val = 1 if use_gaussian_fallback else 3
            hdu.header["OVERSAMP"] = (str(oversamp_val), "Oversampling factor(s)")

            # Number of stars used - ensure scalar int
            try:
                nstars_val = int(len(filtered_photo_table))
            except Exception:
                nstars_val = 0
            hdu.header["NSTARS"] = (nstars_val, "Number of stars used for PSF model")

            psf_filename = (
                f"{st.session_state.get('base_filename', 'psf_model')}_psf.fits"
            )
            username = st.session_state.get("username", "anonymous")
            psf_filepath = os.path.join(
                ensure_output_directory(directory=f"{username}_results"), psf_filename
            )
            hdu.writeto(psf_filepath, overwrite=True)
            st.write(f"PSF model saved as FITS file ({model_type})")

            norm_epsf = simple_norm(epsf_data, "log", percent=99.0)
            fig_epsf_model, ax_epsf_model = plt.subplots(
                figsize=FIGURE_SIZES["medium"], dpi=120
            )
            ax_epsf_model.imshow(
                epsf_data,
                norm=norm_epsf,
                origin="lower",
                cmap="viridis",
                interpolation="nearest",
            )
            title = f"Fitted PSF Model ({nstars_val} stars)"
            if use_gaussian_fallback:
                title += " - Gaussian Fallback"
            ax_epsf_model.set_title(title)
            st.pyplot(fig_epsf_model)
        except Exception as e:
            st.warning(f"Error working with PSF model display/save: {e}")
    except Exception as e:
        st.error(f"Error preparing PSF model for photometry: {e}")
        raise

    # Create PSF photometry object and perform photometry
    try:
        if error is not None and not isinstance(error, np.ndarray):
            st.warning(
                "Invalid error array provided, proceeding without error estimation"
            )
            error = None
        elif error is not None and error.shape != img.shape:
            st.warning(
                "Error array shape mismatch, proceeding without error estimation"
            )
            error = None

        # Create a SourceGrouper
        min_separation = 2.0 * fwhm
        grouper = SourceGrouper(min_separation=min_separation)
        sigma_clip = SigmaClip(sigma=3.0)
        bkgstat = MMMBackground(sigma_clip=sigma_clip)
        localbkg_estimator = LocalBackground(3.0 * fwhm, 5.0 * fwhm, bkgstat)

        psfphot = PSFPhotometry(
            psf_model=psf_for_phot,
            fit_shape=fit_shape,
            finder=daostarfind,
            aperture_radius=aperture_radius,
            grouper=grouper,
            localbkg_estimator=localbkg_estimator,
        )

        initial_params = Table()
        initial_params["x"] = photo_table["xcentroid"]
        initial_params["y"] = photo_table["ycentroid"]
        if "flux" in photo_table.colnames:
            initial_params["flux"] = photo_table["flux"]
        initial_params["x_0"] = initial_params["x"]
        initial_params["y_0"] = initial_params["y"]

        # Filter out sources that fall in masked regions (robust)
        mask_bool = None
        if mask is not None:
            mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool is not None:
            # compute rounded indices with clipping
            x_int = np.clip(
                np.round(initial_params["x"]).astype(int), 0, img.shape[1] - 1
            )
            y_int = np.clip(
                np.round(initial_params["y"]).astype(int), 0, img.shape[0] - 1
            )

            valid_bounds = (
                (initial_params["x"] >= 0)
                & (initial_params["x"] < img.shape[1])
                & (initial_params["y"] >= 0)
                & (initial_params["y"] < img.shape[0])
            )

            valid_mask = np.ones(len(initial_params), dtype=bool)
            valid_mask[valid_bounds] = ~mask_bool[
                y_int[valid_bounds], x_int[valid_bounds]
            ]

            initial_params_filtered = initial_params[valid_mask]

            st.write(
                f"Filtered out {len(initial_params) - len(initial_params_filtered)} sources that fall in masked regions"
            )
            st.write(
                f"Proceeding with {len(initial_params_filtered)} sources for PSF photometry"
            )

            if len(initial_params_filtered) == 0:
                st.error(
                    "All sources fall in masked regions. Cannot perform PSF photometry."
                )
                return None, epsf

            initial_params = initial_params_filtered
        else:
            st.write("No mask provided, using all sources for PSF photometry")

        st.write("Performing PSF photometry on sources...")
        # call PSFPhotometry: data first
        phot_epsf_result = psfphot(
            img, init_params=initial_params, mask=mask_bool, error=error
        )
        st.session_state["epsf_photometry_result"] = phot_epsf_result
        st.write("PSF photometry completed successfully.")
        return phot_epsf_result, epsf

    except Exception as e:
        st.error(f"Error executing PSF photometry: {e}")
        raise
