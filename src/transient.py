from stdpipe import (pipeline, cutouts,
                     templates, catalogs, photometry)
import sep
from astropy.coordinates import SkyCoord
from astropy.visualization import (PercentileInterval, LinearStretch,
                                   AsinhStretch)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
from astropy.wcs import WCS
import numpy as np
import streamlit as st
from astropy.table import Table

from src.tools_pipeline import fix_header


# Conversion to AB mags, from https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
filter_ab_offset = {
    'U': 0.79,
    'B': -0.09,
    'V': 0.02,
    'R': 0.21,
    'I': 0.45,
    'u': 0,
    'g': 0,
    'r': 0,
    'i': 0,
    'z': 0,
    'y': 0,
    'G': 0,
    'BP': 0,
    'RP': 0,
}


def find_candidates(
    zero_point_value,
    image,
    header,
    fwhm,
    pixel_scale,
    ra_center,
    dec_center,
    sr,
    mask=None,
    filter_cat=None,
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
    st.warning("⚠️ Transient detection is working but there are too many candidates. Further development is ongoing.")
    if header.get('GAIN'):
        gain = header.get('GAIN')
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

    mag = np.round(-2.5*np.log(obj['flux']) + zero_point_value, 2)
    if 'err' in obj.dtype.names:
        mag_err = np.round(2.5 / np.log(10) * obj['err'] / obj['flux'], 3)
    else:
        # Fallback: use a default fractional error estimate (5%)
        mag_err = np.round(2.5 / np.log(10) * 0.05 * np.ones_like(obj['flux']), 3)
    obj = np.lib.recfunctions.append_fields(
        obj, ['mag_calib', 'mag_calib_err'], [mag, mag_err], usemask=False)

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
        fwhm=1.*fwhm*pixel_scale,
        time=header.get('DATE-OBS', None),
        skybot=True,
        vizier=['gaiaedr3', 'ps1', 'skymapper', 'sdss',
                'vsx', 'apass', 'atlas', 'gsc'],
        vizier_checker_fn=lambda xobj, xcat, catname: checker_fn(xobj, xcat, catname, filter_mag=filter_cat),
        ned=False,
        verbose=True,
        flagged=True,
    )

    st.success(
        f"Candidate filtering complete. Found {len(candidates)} potential transients."
    )

    if len(candidates) > 100:
        st.warning(
            "More than 100 candidates found. Displaying only the first 10 candidates."
        )
        candidates = candidates[:100]

    for _, cand in list(enumerate(candidates))[:10]:
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

        # Now we have two image planes in the cutout - let's display them
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
    planes=['image', 'template'],
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
    show_title=True,
    title=None,
    additional_title=None,
    qq=None,
    stretch='linear',
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

    fig, axs = plt.subplots(1, nplots, figsize=(nplots * 4, 4 + 1.0), dpi=75,
                            tight_layout=True)
    if nplots == 1:
        axs = [axs]

    for ii, name in enumerate(planes):
        if name in cutout and cutout[name] is not None:
            ax = axs[ii]
            img = cutout[name].copy()
            
            # Replace NaN values with min value for proper display
            if np.isnan(img).any():
                valid_mask = ~np.isnan(img)
                if valid_mask.any():
                    img[~valid_mask] = np.nanmin(img)
                else:
                    img = np.ones_like(img) * 0.5  # Fallback for all-NaN
            
            # Apply percentile-based scaling and stretch only to science image
            if name == 'image' and qq is not None:
                interval = PercentileInterval(int(qq[0]), int(qq[1]))
                img = interval(img)
                
                if stretch == 'linear':
                    img = LinearStretch()(img)
                elif stretch == 'asinh':
                    img = AsinhStretch()(img)
            else:
                # For template: apply simple min-max normalization
                valid = img[~np.isnan(img)]
                if len(valid) > 0:
                    vmin, vmax = np.min(valid), np.max(valid)
                    if vmax > vmin:
                        img = (img - vmin) / (vmax - vmin)
            
            imshow_kwargs = {'cmap': 'Blues_r', 'vmin': 0, 'vmax': 1}
            if 'cmap' in kwargs:
                imshow_kwargs['cmap'] = kwargs['cmap']
            ax.imshow(img, **imshow_kwargs)
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


def checker_fn(xobj, xcat, catname, filter_mag='r'):
    """Example of custom checker function for filtering candidates.
    This function filters out objects that have a magnitude difference
    """
    xidx = np.ones_like(xobj, dtype=bool)

    fname = filter_mag
    if fname.endswith('mag'):
        fname = fname[:-3]

    cat_col_mag, _ = guess_catalogue_mag_columns(fname, xcat)

    if cat_col_mag is not None:
        mag = xobj['mag_calib']
        if fname in ['U', 'B', 'V', 'R', 'I'] and cat_col_mag not in ['Umag', 'Bmag', 'Vmag', 'Rmag', 'Imag']:
            # Convert to AB mags if using AB reference catalogue
            mag += filter_ab_offset.get(fname, 0)

        diff = mag - xcat[cat_col_mag]

        if len(diff[np.isfinite(diff)]) > 10:
            # Adjust zeropoint
            diff -= np.nanmedian(diff)

        xidx = diff > 1.0

    return xidx


def guess_catalogue_mag_columns(fname, cat, augmented_only=False):
    cat_col_mag = None
    cat_col_mag_err = None

    # Most of augmented catalogues
    if f"{fname}mag" in cat.colnames:
        cat_col_mag = f"{fname}mag"

        if f"e_{fname}mag" in cat.colnames:
            cat_col_mag_err = f"e_{fname}mag"

    elif augmented_only:
        raise RuntimeError(f"Unsupported filter {fname} for this catalogue")

    # Non-augmented PS1 etc
    elif "gmag" in cat.colnames and "rmag" in cat.colnames:
        if fname in ['U', 'B', 'V', 'BP']:
            cat_col_mag = "gmag"
        if fname in ['R', 'G']:
            cat_col_mag = "rmag"
        if fname in ['I', 'RP']:
            cat_col_mag = "imag"

        if f"e_{cat_col_mag}" in cat.colnames:
            cat_col_mag_err = f"e_{cat_col_mag}"

    # SkyMapper
    elif f"{fname}PSF" in cat.colnames:
        cat_col_mag = f"{fname}PSF"

        if f"e_{fname}PSF" in cat.colnames:
            cat_col_mag_err = f"e_{fname}PSF"

    # Gaia DR2/eDR3/DR3 from Vizier
    elif "BPmag" in cat.colnames and "RPmag" in cat.colnames and "Gmag" in cat.colnames:
        if fname in ['U', 'B', 'V', 'R', 'u', 'g', 'r', 'BP']:
            cat_col_mag = "BPmag"
        elif fname in ['I', 'i', 'z', 'RP']:
            cat_col_mag = "RPmag"
        else:
            cat_col_mag = "Gmag"

        if f"e_{cat_col_mag}" in cat.colnames:
            cat_col_mag_err = f"e_{cat_col_mag}"

    # Gaia DR2/eDR3/DR3 from XMatch
    elif "phot_bp_mean_mag" in cat.colnames and "phot_rp_mean_mag" in cat.colnames and "phot_g_mean_mag" in cat.colnames:
        if fname in ['U', 'B', 'V', 'R', 'u', 'g', 'r', 'BP']:
            cat_col_mag = "phot_bp_mean_mag"
        elif fname in ['I', 'i', 'z', 'RP']:
            cat_col_mag = "phot_rp_mean_mag"
        else:
            cat_col_mag = "phot_g_mean_mag"

        if f"{cat_col_mag}_error" in cat.colnames:
            cat_col_mag_err = f"{cat_col_mag}_error"

    # else:
    #     raise RuntimeError(f"Unsupported filter {fname} and/or catalogue")

    return cat_col_mag, cat_col_mag_err
