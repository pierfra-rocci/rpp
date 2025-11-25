from stdpipe import (pipeline, cutouts,
                     templates, plots)

import matplotlib.pyplot as plt
import numpy as np


def find_candidates(obj, image, mask, header, cat):
    """Find transient candidates in the given image around the specified object.
    Parameters
    ----------
    obj : dict
        Dictionary with object information (e.g., coordinates).
    image : 2D array
        The image data where to search for transients.
    mask : 2D array
        The mask data corresponding to the image."""
    candidates = pipeline.filter_transient_candidates(
        obj,
        cat=cat,
        sr=2/3600,
        # We will check against AAVSO VSX
        vizier=['vsx'],
        verbose=True
    )

    for _, cand in enumerate(candidates):
        # Create the cutout from image based on the candidate
        cutout = cutouts.get_cutout(
            image,
            # Candidate
            cand,
            # Cutout half-size in pixels
            20,
            # Additional planes for the cutout
            mask=mask,
            header=header
        )

        # We did not do image subtraction yet, but we may already
        # directly download the "template" image for this cutout
        # from HiPS server
        cutout['template'] = templates.get_hips_image(
            'PanSTARRS/DR2/r',
            # FITS header with proper WCS and size
            header=cutout['header'],
            # Return only image
            get_header=False
        )

        # Now we have three image planes in the cutout - let's display them
        plots.plot_cutout(
            cutout,
            # Image planes to display - optional
            planes=['image', 'template', 'mask'],
            # Percentile-based scaling and linear stretching
            qq=[0.5, 99.5],
            stretch='linear')
        plt.show()

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
    # Get r band image from Pan-STARRS with the same resolution and orientation
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

    # We will exclude pixels with any non-zero mask value
    tmask = tmask > 0
    # We will also mask the regions of the template filled with NaNs
    tmask |= np.isnan(tmpl)

    plt.subplot(122)
    plots.imshow(tmask, show_colorbar=False)
    plt.title('Template mask')

    return tmask

