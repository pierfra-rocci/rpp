from stdpipe import (pipeline, cutouts,
                     templates, plots, catalogs)
import sep
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
import numpy as np
import streamlit as st
from astropy.table import Table

from src.tools_pipeline import fix_header


def find_candidates(
    image,
    header,
    fwhm,
    pixel_scale,
    ra_center,
    dec_center,
    sr,
    mask=None,
    filter_name=None,
    mag_limit="<20",
    detect_thresh=1.1
):
    """Find transient candidates in the given image around the specified object.
    Parameters
    ----------
    obj : dict
        Dictionary with object information (e.g., coordinates).
    image : 2D array
        The image data where to search for transients.
    mask : 2D array
        The mask data corresponding to the image."""
    st.warning("⚠️ Transient detection is currently in Beta phase.")
    gain = header.get("GAIN", 1.0)

    header, _ = fix_header(header)
    wcs = WCS(header)

    # Check hemisphere using astropy
    _ = SkyCoord(ra=ra_center*u.degree, dec=dec_center*u.degree, frame='icrs')
    is_southern = dec_center < 0

    # Auto-select catalog based on hemisphere if not specified or if PanSTARRS
    if is_southern:
        catalog = "skymapper"
        cat_cutout = "SkyMapper/"
        st.info(f"🌏 Southern hemisphere detected (Dec={dec_center:.2f}°). Using SkyMapper catalog.")
    else:
        catalog = "ps1"
        cat_cutout = "PanSTARRS/DR2/"
        st.info(f"🌍 Northern hemisphere detected (Dec={dec_center:.2f}°). Using PanSTARRS catalog.")

    st.info("Extracting source objects from image using SEP...")
    image = image.astype(image.dtype.newbyteorder("="))
    bkg = sep.Background(image, mask=mask)
    obj = sep.extract(image-bkg.back(), detect_thresh,
                      gain=gain, mask=mask, err=bkg.globalrms)

    if obj is None:
        st.warning("No objects found in the image.")
        return []

    st.info(f"Found {len(obj)} objects in the image.")

    # Add RA, Dec coordinates to the obj table using WCS
    ra_obj, dec_obj = wcs.all_pix2world(obj['x'], obj['y'], 0)
    obj = np.lib.recfunctions.append_fields(
        obj, ['ra', 'dec'], [ra_obj, dec_obj], usemask=False
    )

    # Query the appropriate catalog
    st.info(
        f"Querying {catalog} catalog for reference stars (Filter: {filter_name}, Limit: {mag_limit})..."
    )
    try:
        cat = catalogs.get_cat_vizier(
            ra_center, dec_center, sr, catalog, filters={filter_name + "mag": mag_limit}
        )
    except Exception as e:
        st.error(f"Failed to query {catalog} catalog: {e}")
        return []

    st.info(
        "Filtering candidates against catalog and known databases (VSX, APASS, ATLAS)..."
    )

    obj = Table(obj)
    obj['flag'].name = 'flags'  # Rename to 'flags' for compatibility

    candidates = pipeline.filter_transient_candidates(
        obj,
        cat=cat,
        fwhm=fwhm,
        sr=1.5,
        skybot=True,
        vizier=["gaiaedr3", "ps1", "sdss", "skymapper",
                "apass", "atlas", "vsx", "usnob1"],
        ned=False,
        verbose=False,
        flagged=True
    )

    st.success(
        f"Candidate filtering complete. Found {len(candidates)} potential transients."
    )

    for _, cand in enumerate(candidates):
        # Create the cutout from image based on the candidate
        cutout = cutouts.get_cutout(
            image,
            cand,
            25,
            header=header
        )
        try:
            cutout['template'] = templates.get_hips_image(
                cat_cutout + filter_name,
                header=cutout['header'],
                get_header=False
            )
        except Exception as e:
            st.warning(f"Failed to retrieve HiPS template image for candidate: {e}")
            cutout['template'] = None

        # Now we have three image planes in the cutout - let's display them
        st.pyplot(plots.plot_cutout(
            cutout,
            # Image planes to display
            planes=['image', 'template'],
            # Percentile-based scaling and linear stretching
            qq=[0.5, 99.5],
            stretch='linear'))

    return candidates


def create_template_mask(image, wcs, band="r", survey="ps1"):
    """
    Create a mask for the template image based on survey data.
    Parameters
    ----------
    image : 2D array
        The image data for which the template mask is to be created.
    wcs : WCS object
        The WCS information corresponding to the image.
    band : str, optional
        Filter band (default: 'r').
    survey : str, optional
        Survey name, 'ps1' or 'ls' (default: 'ps1').
    Returns
    -------
    tmask : 2D array
        The mask for the template image, where True indicates masked pixels.
    """
    st.warning("⚠️ Template mask creation is in Beta phase.")

    # Get r band image from Pan-STARRS with the same resolution and orientation
    st.info(f"Retrieving survey template image (Band: {band}, Survey: {survey})...")
    try:
        tmpl = templates.get_survey_image(
            band=band,
            # One of 'image' or 'mask'
            ext="image",
            # Either 'ps1' for Pan-STARRS or 'ls' for Legacy Survey
            survey=survey,
            # pixel grid defined by WCS and image size
            wcs=wcs,
            width=image.shape[1],
            height=image.shape[0],
            verbose=True,
        )
    except Exception as e:
        st.error(f"Failed to retrieve survey template image for band '{band}', survey '{survey}': {e}")
        return np.zeros(image.shape, dtype=bool)  # Return an empty mask on failure

    # Also get proper mask
    st.info(f"Retrieving survey mask (Band: {band})...")
    try:
        tmask = templates.get_survey_image(
                            band=band,
                            # One of 'image' or 'mask'
                            ext="mask",
                            # Either 'ps1' for Pan-STARRS or 'ls' for Legacy Survey
                            survey=survey,
                            # pixel grid defined by WCS and image size
                            wcs=wcs,
                            width=image.shape[1],
                            height=image.shape[0],
                            verbose=True,
                        )
    except Exception as e:
        st.error(f"Failed to retrieve survey mask for band '{band}', survey '{survey}': {e}")
        return np.zeros(image.shape, dtype=bool)  # Return an empty mask on failure
    
    st.info("Processing mask logic...")
    # We will exclude pixels with any non-zero mask value
    tmask = tmask > 0
    tmask |= np.isnan(tmpl)

    # plots.imshow(tmask, show_colorbar=False)

    st.success("✅ Template mask created successfully.")

    return tmask
