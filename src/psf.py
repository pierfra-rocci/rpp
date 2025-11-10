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

from photutils.psf import EPSFBuilder, extract_stars, PSFPhotometry
from photutils.psf import SourceGrouper, EPSFStars

from src.tools import FIGURE_SIZES, ensure_output_directory

from typing import Any, Optional, Tuple


def perform_psf_photometry(
    img: np.ndarray,
    photo_table: Table,
    fwhm: float,
    daostarfind: Any,
    mask: Optional[np.ndarray] = None,
    error=None,
) -> Tuple[Table, Any]:
    """
    Perform PSF (Point Spread Function) photometry using an empirically-constructed PSF model.

    This function builds an empirical PSF (EPSF) from bright stars in the image
    and then uses this model to perform PSF photometry on all detected sources.

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

    Returns
    -------
    Tuple[astropy.table.Table, photutils.psf.EPSFModel]
        - phot_epsf_result : Table containing PSF photometry results, including fitted fluxes and positions
        - epsf : The fitted EPSF model used for photometry
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
    try:
        st.write("Filtering stars for PSF model construction...")

        # Ensure table-like behavior for astropy Table / QTable (no .get)
        n_sources = len(photo_table)
        if n_sources == 0:
            raise ValueError("photo_table is empty")

        def _col_arr(tbl, name, default=None):
            if name in tbl.colnames:
                return np.asarray(tbl[name])
            if default is not None:
                # return default repeated to match table length
                return (
                    np.asarray(default)
                    if np.shape(default) == (n_sources,)
                    else np.full(n_sources, default)
                )
            return np.full(n_sources, np.nan)

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

        # Get arrays safely
        flux = _col_arr(photo_table, "flux")
        roundness1 = _col_arr(photo_table, "roundness1", np.nan)
        sharpness = _col_arr(photo_table, "sharpness", np.nan)
        xcentroid = _col_arr(photo_table, "xcentroid")
        ycentroid = _col_arr(photo_table, "ycentroid")

        # Get flux statistics with NaN handling
        flux_finite = flux[np.isfinite(flux)]
        if len(flux_finite) == 0:
            raise ValueError("No sources with finite flux values found")

        flux_median = np.median(flux_finite)
        flux_std = np.std(flux_finite)

        # Define flux filtering criteria
        flux_min = flux_median - 3 * flux_std
        flux_max = flux_median + 3 * flux_std

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

        flux_criteria = np.zeros_like(valid_flux, dtype=bool)
        flux_criteria[valid_flux] = (flux[valid_flux] >= flux_min) & (
            flux[valid_flux] <= flux_max
        )

        roundness_criteria = np.zeros_like(valid_roundness, dtype=bool)
        roundness_criteria[valid_roundness] = np.abs(roundness1[valid_roundness]) < 0.25

        sharpness_criteria = np.zeros_like(valid_sharpness, dtype=bool)
        sharpness_criteria[valid_sharpness] = sharpness[valid_sharpness] > 0.5

        # Edge criteria
        edge_criteria = np.zeros_like(valid_xcentroid, dtype=bool)
        valid_coords = valid_xcentroid & valid_ycentroid
        edge_criteria[valid_coords] = (
            (xcentroid[valid_coords] > 2 * fwhm)
            & (xcentroid[valid_coords] < img.shape[1] - 2 * fwhm)
            & (ycentroid[valid_coords] > 2 * fwhm)
            & (ycentroid[valid_coords] < img.shape[0] - 2 * fwhm)
        )

        # Combine all criteria with validity checks
        good_stars_mask = (
            valid_all
            & flux_criteria
            & roundness_criteria
            & sharpness_criteria
            & edge_criteria
        )

        # Apply filters
        filtered_photo_table = photo_table[good_stars_mask]
        st.write(f"Flux range for PSF stars : {flux_min:.1f} -> {flux_max:.1f}")

        # Check if we have enough stars for PSF construction
        if len(filtered_photo_table) < 10:
            st.warning(
                f"Only {len(filtered_photo_table)} stars available for PSF model. Relaxing criteria..."
            )

            # Relax criteria
            roundness_criteria_relaxed = np.zeros_like(valid_roundness, dtype=bool)
            roundness_criteria_relaxed[valid_roundness] = (
                np.abs(roundness1[valid_roundness]) < 0.5
            )

            sharpness_criteria_relaxed = np.zeros_like(valid_sharpness, dtype=bool)
            sharpness_criteria_relaxed[valid_sharpness] = (
                np.abs(sharpness[valid_sharpness]) < 1.0
            )

            flux_criteria_relaxed = np.zeros_like(valid_flux, dtype=bool)
            flux_criteria_relaxed[valid_flux] = (
                flux[valid_flux] >= flux_median - 2 * flux_std
            ) & (flux[valid_flux] <= flux_median + 2 * flux_std)

            good_stars_mask = (
                (valid_flux & valid_xcentroid & valid_ycentroid)
                & flux_criteria_relaxed
                & roundness_criteria_relaxed
                & sharpness_criteria_relaxed
                & edge_criteria
            )

            filtered_photo_table = photo_table[good_stars_mask]

            st.write(f"After relaxing criteria: {len(filtered_photo_table)} stars")

        if len(filtered_photo_table) < 5:
            raise ValueError(
                "Too few good stars for PSF model construction. Need at least 5 stars."
            )

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
        # ensure fit_shape is odd and reasonably large (small FWHM can produce too small boxes)
        fit_shape = max(9, 2 * int(round(fwhm)) + 1)
        if fit_shape % 2 == 0:
            fit_shape += 1
        st.write(f"Fitting shape: {fit_shape} pixels.")
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

    try:
        # Filter out invalid Star objects and select brightest for EPSFBuilder
        valid_stars = []
        try:
            for star in stars: # stars is already an EPSFStars object
                if hasattr(star, "data") and star.data is not None:
                    arr = np.asarray(star.data)
                    if arr.size == 0 or np.isnan(arr).any() or np.all(arr == 0):
                        continue
                    valid_stars.append(star)
        except Exception as e:
            st.warning(f"Error during initial filtering of stars for EPSF: {e}")
            valid_stars = []

        n_valid = len(valid_stars)
        st.write(f"Using {n_valid} valid Star objects for EPSF construction")

        if n_valid < 5:
            raise ValueError("Too few valid Star objects for EPSF building")

        # If there are a lot of stars, pick the brightest (by peak)
        max_use = 200
        if n_valid > max_use:
            peaks = np.array([s.data.max() for s in valid_stars])
            top_idx = np.argsort(peaks)[-max_use:][::-1]
            stars_for_builder = EPSFStars([valid_stars[i] for i in top_idx])
            st.write(f"Selected {len(stars_for_builder)} brightest stars for EPSF construction.")
        else:
            stars_for_builder = EPSFStars(valid_stars)

        # Try EPSFBuilder with retries and progressively simpler parameters
        epsf = None
        builder_attempts = [
            dict(oversampling=2, maxiters=3),
        ]
        from types import SimpleNamespace # Keep this import as it's used later for fallback

        for params in builder_attempts:
            try:
                epsf_builder = EPSFBuilder(progress_bar=False, **params)
                # Pass the wrapped object (not a raw Python list) to EPSFBuilder
                epsf, _ = epsf_builder(stars_for_builder)
                if epsf is not None:
                    st.write(
                        f"EPSFBuilder succeeded with oversampling={params.get('oversampling')}"
                    )
                    break
            except Exception as build_error:
                st.warning(
                    f"EPSFBuilder attempt oversampling={params.get('oversampling')} failed: {build_error}"
                )
                epsf = None

        # If EPSFBuilder failed entirely, fall back to a median-stacked empirical PSF
        if epsf is None:
            st.warning(
                "EPSFBuilder failed on all attempts — constructing median empirical PSF as fallback"
            )
            # normalize each cutout by its sum (avoid divide-by-zero)
            norm_cutouts = []
            for c in selected:
                ssum = np.nansum(c)
                if ssum <= 0 or not np.isfinite(ssum):
                    continue
                norm_cutouts.append(c.astype(float) / ssum)

            if len(norm_cutouts) == 0:
                raise ValueError(
                    "No valid normalized cutouts available for median PSF fallback"
                )

            # ensure consistent shapes by cropping/padding to median shape
            shapes = np.array([c.shape for c in norm_cutouts])
            # pick the most common shape
            uniq_shapes, counts = np.unique(
                shapes.reshape(len(shapes), -1), axis=0, return_counts=True
            )
            # fallback to using the first shape if uniqueness fails
            try:
                chosen_shape = tuple(uniq_shapes[np.argmax(counts)])
            except Exception:
                chosen_shape = norm_cutouts[0].shape

            # center-crop or pad to chosen_shape
            def fit_to_shape(arr, shp):
                out = np.zeros(shp, dtype=float)
                y0 = (arr.shape[0] - shp[0]) // 2
                x0 = (arr.shape[1] - shp[1]) // 2
                if y0 >= 0 and x0 >= 0:
                    # crop
                    out = arr[y0: y0 + shp[0],
                              x0: x0 + shp[1]]
                else:
                    # pad
                    y_off = max(0, -y0)
                    x_off = max(0, -x0)
                    yy = min(arr.shape[0], shp[0] - y_off)
                    xx = min(arr.shape[1], shp[1] - x_off)
                    out[y_off : y_off + yy, x_off : x_off + xx] = arr[0:yy, 0:xx]
                return out

            aligned = [fit_to_shape(c, chosen_shape) for c in norm_cutouts]
            epsf_array = np.median(np.stack(aligned, axis=0), axis=0)
            # renormalize
            s = np.nansum(epsf_array)
            if s > 0 and np.isfinite(s):
                epsf_array = epsf_array / s

            # Build a minimal EPSF-like object with attributes expected by downstream code
            epsf = SimpleNamespace()
            epsf.data = epsf_array
            epsf.oversampling = 1
            epsf.fwhm = getattr(epsf, "fwhm", float(fwhm))

        # Validate epsf.data
        if (
            not hasattr(epsf, "data")
            or epsf.data is None
            or np.asarray(epsf.data).size == 0
        ):
            raise ValueError(
                "EPSF data is invalid or empty after attempts and fallback"
            )

        epsf_array = np.asarray(epsf.data)
        if epsf_array.size > 0 and np.isnan(epsf_array).any():
            st.warning("EPSF data contains NaN values")

        st.write(f"Shape of epsf.data: {epsf_array.shape}")
        st.session_state["epsf_model"] = epsf
    except Exception as e:
        st.error(f"Error fitting PSF model: {e}")
        raise

    # Ensure we have a usable PSF model for PSFPhotometry
    try:
        # Try to wrap the EPSF into an ImagePSF (preferred for PSFPhotometry compatibility)
        try:
            from photutils.psf import ImagePSF

            psf_for_phot = ImagePSF(np.asarray(epsf.data))
        except Exception:
            # fallback to epsf object (works on newer photutils versions)
            psf_for_phot = epsf

        # Save / display epsf as FITS and figure (best-effort)
        try:
            hdu = fits.PrimaryHDU(data=np.asarray(epsf.data))
            hdu.header["COMMENT"] = "PSF model created with photutils.EPSFBuilder"
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
            oversamp = getattr(epsf, "oversampling", 3)
            try:
                if isinstance(oversamp, (list, tuple, np.ndarray)):
                    # store as comma-separated string to avoid illegal array header values
                    oversamp_val = ",".join(
                        str(int(x)) for x in np.asarray(oversamp).ravel()
                    )
                else:
                    oversamp_val = int(oversamp)
            except Exception:
                oversamp_val = 3
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
                ensure_output_directory(f"../rpp_results/{username}_results"), psf_filename
            )
            hdu.writeto(psf_filepath, overwrite=True)
            st.write("PSF model saved as FITS file")

            norm_epsf = simple_norm(epsf.data, "log", percent=99.0)
            fig_epsf_model, ax_epsf_model = plt.subplots(
                figsize=FIGURE_SIZES["medium"], dpi=120
            )
            ax_epsf_model.imshow(
                epsf.data,
                norm=norm_epsf,
                origin="lower",
                cmap="viridis",
                interpolation="nearest",
            )
            ax_epsf_model.set_title(f"Fitted PSF Model ({nstars_val} stars)")
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
        localbkg_estimator = LocalBackground(2.0 * fwhm, 2.5 * fwhm, bkgstat)

        psfphot = PSFPhotometry(
            psf_model=psf_for_phot,
            fit_shape=fit_shape,
            finder=daostarfind,
            aperture_radius=float(fit_shape) / 2.0,
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
