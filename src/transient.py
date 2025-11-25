from stdpipe import (pipeline, cutouts, photometry,
                     templates, plots, catalogs)

from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.tools_pipeline import fix_header


def find_candidates(image, header, fwhm, pixel_scale, ra_center, dec_center, sr,
                    mask=None, catalog=None,
                    filter_name=None, mag_limit='<19'):
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
    gain = header.get('GAIN', 1.0)

    header, _ = fix_header(header)

    st.info("Extracting source objects from image using SExtractor...")
    image = image.astype(image.dtype.newbyteorder('='))
    obj = photometry.get_objects_sep(
                        image,
                        mask=mask,
                        aper=1.5*fwhm,
                        thresh=2.0,
                        sn=5,
                        gain=gain,
                        edge=10,
                        bg_size=64,
                        use_fwhm=True,
                        wcs=WCS(header)
                        )

    if obj is None:
        st.warning("No objects found in the image.")
        return []

    st.info(f"Found {len(obj)} objects in the image.")

    # Let's get PanSTARRS objects brighter than r=18 mag
    if catalog == 'PanSTARRS':
        catalog = 'ps1'

    st.info(f"Querying {catalog} catalog for reference stars (Filter: {filter_name}, Limit: {mag_limit})...")
    cat = catalogs.get_cat_vizier(
        ra_center,
        dec_center,
        sr,
        catalog,
        filters={filter_name+'mag': mag_limit}
        )

    st.info("Filtering candidates against catalog and known databases (VSX, APASS, ATLAS)...")
    candidates = pipeline.filter_transient_candidates(
        obj,
        cat=cat,
        sr=2*fwhm*pixel_scale,
        vizier=['vsx', 'apass', 'atlas'],
        skybot=True,
        ned=True,
        verbose=False
    )

    st.success(f"✅ Candidate filtering complete. Found {len(candidates)} potential transients.")

    # for _, cand in enumerate(candidates):
    #     # Create the cutout from image based on the candidate
    #     cutout = cutouts.get_cutout(
    #         image,
    #         # Candidate
    #         cand,
    #         # Cutout half-size in pixels
    #         25,
    #         header=header
    #     )

    #     # We did not do image subtraction yet, but we may already
    #     # directly download the "template" image for this cutout
    #     # from HiPS server
    #     cutout['template'] = templates.get_hips_image(
    #         'PanSTARRS/DR2/'+filter_name,
    #         header=cutout['header'],
    #         get_header=False
    #     )

    #     # Now we have three image planes in the cutout - let's display them
    #     plots.plot_cutout(
    #         cutout,
    #         # Image planes to display
    #         planes=['image', 'template'],
    #         # Percentile-based scaling and linear stretching
    #         qq=[0.5, 99.5],
    #         stretch='linear')
    #     plt.show()

    return candidates


def create_template_mask(image, wcs):
    """
    Create a mask for the template image based on Pan-STARRS data.
    Parameters
    ----------
    image : 2D array
        The image data for which the template mask is to be created.
    wcs : WCS object
        The WCS information corresponding to the image.
    Returns
    -------
    tmask : 2D array
        The mask for the template image, where True indicates masked pixels.
    """
    st.warning("⚠️ Template mask creation is in Beta phase.")
    
    # Get r band image from Pan-STARRS with the same resolution and orientation
    st.info("Retrieving survey template image (Band: r)...")
    tmpl = templates.get_survey_image(
        band='r',
        # One of 'image' or 'mask'
        ext='image',
        # Either 'ps1' for Pan-STARRS or 'ls' for Legacy Survey
        survey='ps1',
        # pixel grid defined by WCS and image size
        wcs=wcs,
        width=image.shape[1],
        height=image.shape[0],
        verbose=True
    )

    # Also get proper mask
    st.info("Retrieving survey mask (Band: r)...")
    tmask = templates.get_survey_image(
        band='r',
        # One of 'image' or 'mask'
        ext='mask',
        # Either 'ps1' for Pan-STARRS or 'ls' for Legacy Survey
        survey='ps1',
        # pixel grid defined by WCS and image size
        wcs=wcs,
        width=image.shape[1],
        height=image.shape[0],
        verbose=True
    )

    st.info("Processing mask logic...")
    # We will exclude pixels with any non-zero mask value
    tmask = tmask > 0
    # We will also mask the regions of the template filled with NaNs
    tmask |= np.isnan(tmpl)

    plt.subplot(122)
    plots.imshow(tmask, show_colorbar=False)
    plt.title('Template mask')
    
    st.success("✅ Template mask created successfully.")

    return tmask