import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits

from photutils.background import LocalBackground, MMMBackground
from astropy.nddata import NDData
from astropy.visualization import simple_norm
from photutils.psf import EPSFBuilder, extract_stars, IterativePSFPhotometry
from photutils.psf import SourceGrouper

from src.tools import FIGURE_SIZES, ensure_output_directory
from typing import Any, Optional, Tuple


def _filter_psf_stars(photo_table: Table, fwhm: float) -> Table:
    """
    Filter a table of sources to select the best candidates for PSF modeling.

    This function filters stars based on their proximity to the image edge,
    photometric properties, and quality metrics like sharpness and roundness.

    Parameters
    ----------
    photo_table : astropy.table.Table
        Table of detected sources, expected to have columns like 'xcentroid',
        'ycentroid', 'flux', 'sharpness', and 'roundness1'/'roundness2'.
    fwhm : float
        Full Width at Half Maximum in pixels, used to define the border margin.

    Returns
    -------
    astropy.table.Table
        A new table containing the filtered star candidates, sorted by flux.
    """
    # Get image shape from session state
    try:
        img_shape = st.session_state.get("image_shape")
        if img_shape is None:
            st.warning("Image shape not found in session state. Skipping edge filtering.")
            use_border_filter = False
        else:
            use_border_filter = True
    except Exception:
        st.warning("Could not retrieve image shape. Skipping edge filtering.")
        use_border_filter = False

    # Start with a mask that includes all stars
    initial_mask = np.ones(len(photo_table), dtype=bool)

    # 1. Filter out stars near the image border
    if use_border_filter:
        border_margin = 2 * fwhm
        x = photo_table["xcentroid"]
        y = photo_table["ycentroid"]
        border_mask = (
            (x > border_margin) &
            (x < img_shape[1] - border_margin) &
            (y > border_margin) &
            (y < img_shape[0] - border_margin)
        )
        st.write(f"Kept {np.sum(border_mask)} of {len(photo_table)} stars after edge filtering.")
    else:
        border_mask = initial_mask

    # 2. Filter based on sharpness (if available)
    if "sharpness" in photo_table.colnames:
        sharp_mask = (photo_table["sharpness"] > 0.2) & (photo_table["sharpness"] < 1.0)
        st.write(f"Kept {np.sum(sharp_mask)} of {len(photo_table)} stars after sharpness filtering.")
    else:
        sharp_mask = initial_mask

    # 3. Filter based on roundness (if available)
    if "roundness1" in photo_table.colnames and "roundness2" in photo_table.colnames:
        round_mask = (np.abs(photo_table["roundness1"]) < 0.5) & (np.abs(photo_table["roundness2"]) < 0.5)
        st.write(f"Kept {np.sum(round_mask)} of {len(photo_table)} stars after roundness filtering.")
    else:
        round_mask = initial_mask

    # Combine all masks
    final_mask = border_mask & sharp_mask & round_mask
    filtered_table = photo_table[final_mask]
    st.write(f"Total {len(filtered_table)} stars selected after all quality filters.")

    if len(filtered_table) == 0:
        raise ValueError("No stars remained after filtering. Cannot build PSF model.")

    # 4. Sort by flux and select the brightest stars
    if "flux" in filtered_table.colnames:
        filtered_table.sort("flux", reverse=True)
    elif "peak" in filtered_table.colnames:
        filtered_table.sort("peak", reverse=True)
    else:
        st.warning("No 'flux' or 'peak' column for sorting PSF stars. Using table as is.")

    # Limit the number of stars for building the PSF model
    max_stars_for_psf = 200
    if len(filtered_table) > max_stars_for_psf:
        filtered_table = filtered_table[:max_stars_for_psf]
        st.write(f"Selected the brightest {len(filtered_table)} stars for PSF model.")

    return filtered_table


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
        st.session_state["image_shape"] = img.shape

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
        filtered_photo_table = _filter_psf_stars(photo_table, fwhm)
    except Exception as e:
        st.error(f"Error filtering stars for PSF model: {e}")
        raise

    try:
        stars_table = Table()
        # extract_stars expects 'x' and 'y' column names
        stars_table["x"] = filtered_photo_table["xcentroid"]
        stars_table["y"] = filtered_photo_table["ycentroid"]
        st.write("Star positions table prepared from filtered sources.")
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

        # If extract_stars returns list or EPSFStars; ensure a consistent EPSFStars object
        from photutils.psf import EPSFStars

        if isinstance(stars, list):
            if len(stars) == 0:
                raise ValueError(
                    "No stars extracted for PSF model. Check your selection criteria."
                )
            stars = EPSFStars(stars)
        n_stars = len(stars)
        st.write(f"{n_stars} stars extracted for PSF model.")

        # basic inspection of cutouts for NaN/all-zero
        if hasattr(stars, "data") and stars.data is not None:
            if isinstance(stars.data, list) and len(stars.data) > 0:
                has_nan = any(np.isnan(star_data).any() for star_data in stars.data)
                st.write(f"NaN in star data: {has_nan}")
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
            from photutils.psf import EPSFStars

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
        # Prepare a cleaned list of star cutouts suitable for EPSFBuilder
        clean_cutouts = []
        try:
            # Iterate robustly through stars and extract numeric arrays
            for i in range(len(stars)):
                try:
                    star = stars[i]
                    # star may be a simple array or an object with .data
                    if hasattr(star, "data"):
                        arr = np.asarray(star.data)
                    else:
                        arr = np.asarray(star)

                    # skip empty/NaN/all-zero cutouts
                    if arr.size == 0:
                        continue
                    if np.isnan(arr).all() or np.all(arr == 0):
                        continue

                    clean_cutouts.append(arr)
                except Exception:
                    # skip problematic entries rather than aborting
                    continue
        except Exception:
            clean_cutouts = []

        n_clean = len(clean_cutouts)
        st.write(f"Using {n_clean} valid cutouts for EPSF construction")

        if n_clean < 5:
            raise ValueError("Too few valid star cutouts for EPSF building")

        max_use = 200
        if n_clean > max_use:
            peaks = np.array([c.max() for c in clean_cutouts])
            top_idx = np.argsort(peaks)[-max_use:][::-1]
            selected = [clean_cutouts[i] for i in top_idx]
        else:
            selected = clean_cutouts

        # Try EPSFBuilder with retries and progressively simpler parameters
        epsf = None
        builder_attempts = [
            dict(oversampling=3, maxiters=3),
            dict(oversampling=2, maxiters=2),
            dict(oversampling=1, maxiters=1),
        ]
        from types import SimpleNamespace

        # DEBUG: report selected summary before wrapping
        try:
            st.write(f"DEBUG: selected type: {type(selected)},n_selected={len(selected)}")
        except Exception:
            st.write("DEBUG: selected summary unavailable")

        try:
            from photutils.psf import EPSFStars
            try:
                stars_for_builder = EPSFStars(selected)
                st.write("DEBUG: wrapped selected into EPSFStars")
            except Exception as epsfstars_err:
                stars_for_builder = [SimpleNamespace(data=np.asarray(c))
                                     for c in selected]
                st.write(f"DEBUG: using SimpleNamespace wrappers for selected (EPSFStars failed: {epsfstars_err})")
        except Exception as import_err:
            stars_for_builder = [SimpleNamespace(data=np.asarray(c))
                                 for c in selected]
            st.write(f"DEBUG: EPSFStars import failed ({import_err}); using SimpleNamespace wrappers")

        for params in builder_attempts:
            try:
                epsf_builder = EPSFBuilder(progress_bar=False, **params)
                epsf, _ = epsf_builder(stars_for_builder)
                if epsf is not None:
                    st.write(f"EPSFBuilder succeeded with oversampling={params.get('oversampling')}")
                    break
            except Exception as build_error:
                st.warning(f"EPSFBuilder attempt oversampling={params.get('oversampling')} failed: {build_error}")
                epsf = None

        # If EPSFBuilder failed entirely, fall back to a median-stacked empirical PSF
        if epsf is None:
            st.warning("EPSFBuilder failed on all attempts — constructing median empirical PSF as fallback")
            # normalize each cutout by its sum (avoid divide-by-zero)
            norm_cutouts = []
            for c in selected:
                ssum = np.nansum(c)
                if ssum <= 0 or not np.isfinite(ssum):
                    continue
                norm_cutouts.append(c.astype(float) / ssum)

            if len(norm_cutouts) == 0:
                raise ValueError("No valid normalized cutouts available for median PSF fallback")

            # ensure consistent shapes by cropping/padding to median shape
            shapes = np.array([c.shape for c in norm_cutouts])
            uniq_shapes, counts = np.unique(shapes.reshape(len(shapes), -1),
                                            axis=0, return_counts=True)
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
                    out = arr[y0: y0 + shp[0], x0: x0 + shp[1]]
                else:
                    # pad
                    y_off = max(0, -y0)
                    x_off = max(0, -x0)
                    yy = min(arr.shape[0], shp[0] - y_off)
                    xx = min(arr.shape[1], shp[1] - x_off)
                    out[y_off: y_off + yy, x_off: x_off + xx] = arr[0:yy, 0:xx]
                return out

            aligned = [fit_to_shape(c, chosen_shape) for c in norm_cutouts]
            epsf_array = np.median(np.stack(aligned, axis=0), axis=0)
            # renormalize
            s = np.nansum(epsf_array)
            if s > 0 and np.isfinite(s):
                epsf_array = epsf_array / s

            epsf = SimpleNamespace()
            epsf.data = epsf_array
            epsf.oversampling = 1
            epsf.fwhm = getattr(epsf, 'fwhm', float(fwhm))

        # Validate epsf.data
        if not hasattr(epsf, "data") or epsf.data is None or np.asarray(epsf.data).size == 0:
            raise ValueError("EPSF data is invalid or empty after attempts and fallback")

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
        try:
            from photutils.psf import ImagePSF

            psf_for_phot = ImagePSF(np.asarray(epsf.data))
        except Exception:
            psf_for_phot = epsf

        # Save / display epsf as FITS and figure (best-effort)
        try:
            hdu = fits.PrimaryHDU(data=np.asarray(epsf.data))
            hdu.header["COMMENT"] = "PSF model created with EPSFBuilder"
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

            oversamp = getattr(epsf, "oversampling", 3)
            try:
                if isinstance(oversamp, (list, tuple, np.ndarray)):
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
                ensure_output_directory(f"{username}_rpp_results"),
                psf_filename
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
        min_separation = 2. * fwhm
        grouper = SourceGrouper(min_separation=min_separation)
        bkgstat = MMMBackground()
        localbkg_estimator = LocalBackground(2. * fwhm, 2.5 * fwhm, bkgstat)
        
        psfphot = IterativePSFPhotometry(
            psf_model=psf_for_phot,
            fit_shape=fit_shape,
            finder=daostarfind,
            maxiters=2,
            aperture_radius=float(fit_shape) / 2.0,
            grouper=grouper,
            localbkg_estimator=localbkg_estimator,
        )

        # Prepare initial parameters (use 'x' and 'y' to be compatible with PSFPhotometry)
        initial_params = Table()
        initial_params["x"] = photo_table["xcentroid"]
        initial_params["y"] = photo_table["ycentroid"]
        if "flux" in photo_table.colnames:
            initial_params["flux"] = photo_table["flux"]
        # also provide legacy names for compatibility
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