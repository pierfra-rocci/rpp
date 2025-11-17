from stdpipe import pipeline, cutouts, templates, plots
import matplotlib.pyplot as plt


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

    for i, cand in enumerate(candidates):
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
            # Image planes to display - optional, by default displays everything
            planes=['image', 'template', 'mask'],
            # Percentile-based scaling and linear stretching
            qq=[0.5, 99.5],
            stretch='linear')
        plt.show()

    return candidates
