from stdpipe import (pipeline, cutouts,
                     templates, plots,
                     subtraction, photometry, psf)

import matplotlib.pyplot as plt
import numpy as np
from photutils import Background2D


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
# Filtering of transient candidates
    candidates = pipeline.filter_transient_candidates(
        obj,
        cat=cat,
        sr=2/3600,
        # We will check only against AAVSO VSX
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
        # from HiPS server - same scale and orientation, but different PSF shape!..
        cutout['template'] = templates.get_hips_image(
            'PanSTARRS/DR1/r',
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

    # plt.subplot(121)
    # plots.imshow(tmpl, show_colorbar=False)
    # plt.title('Template')

    plt.subplot(122)
    plots.imshow(tmask, show_colorbar=False)
    plt.title('Template mask')

    return tmask


def run_image_subtraction(image, mask, tmpl, tmask, fwhm,
                          gain):
    """
    Run image subtraction using HOTPANTS algorithm.
    Parameters
    ----------
    image : 2D array
        The science image data.
    mask : 2D array
        The mask data for the science image.
    tmpl : 2D array
        The template image data.
    tmask : 2D array
        The mask data for the template image.
    fwhm : float
        The FWHM of the science image PSF in pixels.
    gain : float
        The gain of the science image in e-/ADU.
    Returns
    -------
    diff : 2D array
        The difference image.
    conv : 2D array
        The convolved template image.
    sdiff : 2D array
        The scaled difference image.
    ediff : 2D array
        The noise model of the difference image.
    """
    # Prepare the subtraction by estimating image and template backgrounds
    bg = Background2D(
        image,
        128,  # Grid size in pixels
        mask=mask,
        exclude_percentile=30
    ).background

    template_bg = Background2D(
        tmpl,
        128,  # Grid size in pixels
        mask=tmask,
        exclude_percentile=30
    ).background

    # Run the subtraction and get all possible result planes
    diff, conv, sdiff, ediff = subtraction.run_hotpants(
        # Background-subtracted image
        image-bg,
        # Background-subtracted template
        tmpl-template_bg,
        # Masks
        mask=mask,
        template_mask=tmask,
        # FWHMs for the convolution kernel size
        image_fwhm=fwhm,
        template_fwhm=1.5,
        # Parameters for the noise model
        image_gain=gain,
        template_gain=1e6,
        # Estimate noise model for the image automatically
        err=True,
        template_err=True,
        # Output parameters
        get_convolved=True,
        get_scaled=True,
        get_noise=True,
        verbose=True
    )

    # Bad pixels as marked by HOTPANTS
    dmask = diff == 1e-30
    # Set to zero all masked pixels
    sdiff[mask | tmask | dmask] = 0

    plots.imshow(sdiff, vmin=-3, vmax=10)
    plt.title('Noise-scaled difference image')

    # Get PSF model and store it to temporary file
    psf_model = psf.run_psfex(
        image,
        mask=mask,
        # No spatial variance for PSF model
        order=0,
        gain=gain,
        # Store PSF model to this file
        psffile='/tmp/psf.psf',
        verbose=True
    )

    # Run SExtractor on difference image with custom noise model,
    # returning object footprints and some additional fields
    sobj, segm = photometry.get_objects_sextractor(
        # Operate on difference image
        diff,
        # Combined mask
        mask=mask | tmask | dmask,
        # Error model
        err=ediff,
        # Exclude everything closer than 20 pixels to the edge
        edge=20,
        wcs=wcs,
        aper=5.0,
        # Get extra object parameters
        extra_params=['CLASS_STAR', 'NUMBER'],
        # Pass extra configuration parameters to SExtractor
        extra={
            'SEEING_FWHM': fwhm,
            'STARNNW_NAME': '/Users/karpov/opt/miniconda3/envs/stdpipe/share/sextractor/default.nnw'
        },
        # Also return the segmentation map
        checkimages=['SEGMENTATION'],
        # Use PSF model we just built
        psf='/tmp/psf.psf',
        verbose=True)

    # Perform forced aperture photometry, again with custom noise model and forced zero background level
    sobj = photometry.measure_objects(
        sobj,
        # Operate on difference image
        diff,
        # Combined mask
        mask=mask | tmask | dmask,
        # Known noise model for difference image
        err=ediff,
        # Here we use the same parameters as for forced photometry on original image
        fwhm=fwhm,
        aper=1.0, 
        bkgann=[5, 7],
        # Filter out everything with S/N < 3
        sn=3,
        verbose=True
    )

    # The difference is in original image normalization, so we know photometric zero point
    sobj['mag_calib'] = sobj['mag'] + m['zero_fn'](
        sobj['x'],
        sobj['y']
    )
    sobj['mag_calib_err'] = np.hypot(
        sobj['magerr'], m['zero_fn'](
            sobj['x'],
            sobj['y'],
            get_err=True
        )
    )

    # We may immediately reject flagged objects as they correspond to imaging artefacts (masked regions)
    sobj = sobj[sobj['flags'] == 0]

    print(len(sobj), 'transient candidates found in difference image')
