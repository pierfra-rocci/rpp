import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import SigmaClip
from astropy.visualization import ZScaleInterval

from photutils.background import Background2D, SExtractorBackground


def estimate_background(image_data, box_size=100, filter_size=5, figure=True):
    """
    Estimate the background and background RMS of an astronomical image.

    Uses photutils.Background2D to create a 2D background model with sigma-clipping
    and the SExtractor background estimation algorithm. Includes error handling
    and automatic adjustment for small images.

    Parameters
    ----------
    image_data : numpy.ndarray
        The 2D image array
    box_size : int, optional
        The box size in pixels for the local background estimation.
        Will be automatically adjusted if the image is small.
    filter_size : int, optional
        Size of the filter for smoothing the background.

    Returns
    -------
    tuple
        (background_2d_object, error_message) where:
        - background_2d_object: photutils.Background2D object if successful, None if failed
        - error_message: None if successful, string describing the error if failed

    Notes
    -----
    The function automatically adjusts the box_size and filter_size parameters
    if the image is too small, and handles various edge cases to ensure robust
    background estimation.
    """
    if image_data is None:
        return None, "No image data provided"

    if not isinstance(image_data, np.ndarray):
        return None, f"Image data must be a numpy array, got {type(image_data)}"

    if len(image_data.shape) != 2:
        return None, f"Image must be 2D, got shape {image_data.shape}"

    height, width = image_data.shape
    adjusted_box_size = max(box_size, min(height // 10, width // 10, 128))
    adjusted_filter_size = min(filter_size, adjusted_box_size // 2)

    if adjusted_box_size < 10:
        return None, f"Image too small ({height}x{width}) for background estimation"

    try:
        sigma_clip = SigmaClip(sigma=3)
        bkg_estimator = SExtractorBackground()

        bkg = Background2D(
            data=image_data,
            box_size=adjusted_box_size,
            filter_size=adjusted_filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
        )

        # Plot the background model with ZScale and save as FITS
        if figure:
            fig_bkg = None
            try:
                # Create a figure with two subplots side by side for background/RMS
                fig_bkg, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                # Use ZScaleInterval for better visualization
                zscale = ZScaleInterval()
                vmin, vmax = zscale.get_limits(bkg.background)

                # Plot the background model
                im1 = ax1.imshow(
                    bkg.background, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax
                )
                ax1.set_title("Estimated Background")
                fig_bkg.colorbar(im1, ax=ax1, label="Flux")

                # Plot the background RMS
                vmin_rms, vmax_rms = zscale.get_limits(bkg.background_rms)
                im2 = ax2.imshow(
                    bkg.background_rms,
                    origin="lower",
                    cmap="viridis",
                    vmin=vmin_rms,
                    vmax=vmax_rms,
                )
                ax2.set_title("Background RMS")
                fig_bkg.colorbar(im2, ax=ax2, label="Flux")

                fig_bkg.tight_layout()
                st.pyplot(fig_bkg)

            except Exception as e:
                st.warning(f"Error creating plot: {str(e)}")
            finally:
                # Clean up matplotlib figure to prevent memory leaks
                if fig_bkg is not None:
                    plt.close(fig_bkg)

        return bkg, None
    except Exception as e:
        return None, f"Background estimation error: {str(e)}"
