from stdpipe import (pipeline, cutouts,
                     templates, catalogs, photometry)
import sep
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.patches import Circle
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
    mag_limit="<19",
    detect_thresh=1.5
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
    # st.warning("⚠️ Transient detection is currently in Beta phase.")
    if header.get('CVF'):
        gain = 1/header.get('CVF')
    else:
        gain = 65635/np.max(image)

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
        "Filtering candidates against catalog and known databases"
    )

    obj = Table(obj)
    obj['flag'].name = 'flags'  # Rename to 'flags' for compatibility

    candidates = pipeline.filter_transient_candidates(
        obj,
        cat=cat,
        fwhm=0.5*fwhm*pixel_scale,
        time=header.get('DATE-OBS', None),
        skybot=True,
        vizier=["gaiaedr3", "ps1", "skymapper",
                "sdss", "vsx", "apass"],
        ned=False,
        verbose=True,
        flagged=True,
    )

    st.success(
        f"Candidate filtering complete. Found {len(candidates)} potential transients."
    )

    for _, cand in enumerate(candidates)[:10]:
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
        fig = plot_cutout(
            cutout,
            # Image planes to display
            planes=['image', 'template'],
            # Percentile-based scaling and linear stretching
            qq=[0.5, 99.5],
            stretch='linear')
        st.pyplot(fig)

    return candidates


def plot_cutout(
    cutout,
    planes=['image', 'template', 'diff', 'mask'],
    fig=None,
    axs=None,
    mark_x=None,
    mark_y=None,
    mark_r=5.0,
    mark_r2=None,
    mark_r3=None,
    mark_color='red',
    mark_lw=2,
    mark_ra=None,
    mark_dec=None,
    r0=None,
    show_title=True,
    title=None,
    additional_title=None,
    **kwargs,
):
    """Routine for displaying various image planes from the cutout structure returned by :func:`stdpipe.cutouts.get_cutout`.

    The cutout planes are displayed in a single row, in the order defined by `planes` paremeters. Optionally, circular mark may be overlayed over the planes at the specified pixel position inside the cutout.

    :param cutout: Cutout structure as returned by :func:`stdpipe.cutouts.get_cutout`
    :param planes: List of names of cutout planes to show
    :param fig: Matplotlib figure where to plot, optional
    :param axs: Matplotlib axes same length as planes, optional
    :param mark_x: `x` coordinate of the overlay mark in cutout coordinates, optional
    :param mark_y: `y` coordinate of the overlay mark in cutout coordinates, optional
    :param mark_r: Radius of the overlay mark in cutout coordinates in pixels, optional
    :param mark_color: Color of the overlay mark, optional
    :param mark_lw: Line width of the overlay mark, optional
    :param mark_ra: Sky coordinate of the overlay mark, overrides `mark_x` and `mark_y`, optional
    :param mark_dec: Sky coordinate of the overlay mark, overrides `mark_x` and `mark_y`, optional
    :param r0: Smoothing kernel size (sigma) to be applied to the image and template planes, optional
    :param show_title: Show title over cutout. Defaults to True.
    :param title: The title to show above the cutouts, optional. If not provided, the title will be constructed from various pieces of cutout metadata, plus the contents of `additoonal_title` field, if provided
    :param additional_title: Additional text to append to automatically generated title of the cutout figure.
    :param \**kwargs: All additional parameters will be directly passed to :func:`stdpipe.plots.imshow` calls on individual images

    """
    nplots = len([_ for _ in planes if _ in cutout])

    # Always create a new figure for Streamlit
    fig, axs = plt.subplots(1, nplots, figsize=(nplots * 4, 4 + 1.0), dpi=75,
                            tight_layout=True)
    if nplots == 1:
        axs = [axs]

    for ii, name in enumerate(planes):
        if name in cutout and cutout[name] is not None:
            ax = axs[ii]
            # Only pass supported arguments to matplotlib's imshow
            imshow_kwargs = {'cmap': 'Blues_r'}
            # Allow user to override cmap via kwargs
            if 'cmap' in kwargs:
                imshow_kwargs['cmap'] = kwargs['cmap']
            ax.imshow(cutout[name], **imshow_kwargs)
            ax.set_title(name.upper())

            # Mark overlays
            if mark_ra is not None and mark_dec is not None and cutout.get('wcs'):
                mark_x, mark_y = cutout['wcs'].all_world2pix(mark_ra, mark_dec, 0)

            if mark_x is not None and mark_y is not None:
                ax.add_artist(
                    Circle(
                        (mark_x, mark_y),
                        mark_r,
                        edgecolor=mark_color,
                        facecolor='none',
                        ls='-',
                        lw=mark_lw,
                    )
                )
                for _ in [mark_r2, mark_r3]:
                    if _ is not None:
                        ax.add_artist(
                            Circle(
                                (mark_x, mark_y),
                                _,
                                edgecolor=mark_color,
                                facecolor='none',
                                ls='--',
                                lw=mark_lw/2,
                            )
                        )

    if show_title:
        if title is None:
            title = cutout['meta'].get('name', 'unnamed')
            if 'time' in cutout['meta']:
                title += ' at %s' % cutout['meta']['time'].to_value('iso')
            if 'mag_filter_name' in cutout['meta']:
                title += ' : ' + cutout['meta']['mag_filter_name']
                if (
                    'mag_color_name' in cutout['meta']
                    and 'mag_color_term' in cutout['meta']
                    and cutout['meta']['mag_color_term'] is not None
                ):
                    title += ' ' + photometry.format_color_term(
                        cutout['meta']['mag_color_term'],
                        color_name=cutout['meta']['mag_color_name'],
                    )
            if 'mag_limit' in cutout['meta']:
                title += ' : limit %.2f' % cutout['meta']['mag_limit']
            if 'mag_calib' in cutout['meta']:
                title += ' : mag = %.2f $\pm$ %.2f' % (
                    cutout['meta'].get('mag_calib', np.nan),
                    cutout['meta'].get(
                        'mag_calib_err', cutout['meta'].get('magerr', np.nan)
                    ),
                )
            if additional_title:
                title += ' : ' + additional_title
        fig.suptitle(title)

    return fig


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
