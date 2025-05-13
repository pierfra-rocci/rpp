import numpy as np
import streamlit as st
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astroquery.vizier import Vizier
from astroquery.hips2fits import hips2fits
from photutils import CircularAperture
from photutils.psf import extract_stars, EPSFBuilder
from photutils.psf.matching import create_matching_kernel
from astropy.convolution import convolve_fft
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval


def get_template_and_masks(
    science_image: np.ndarray,
    wcs: WCS,
    shape: tuple,
    hips_survey: str,
    star_mag_limit: float = 14.0,
    catalog_radius: float = None,
    psf_radius: float = 3.0
) -> tuple:
    """
    Download a HiPS template via astroquery.hips2fits, ensure linear scale,
    create masks for bright stars and invalid science pixels, and verify WCS/shape.

    Returns
    -------
    hips_data : np.ndarray
        HiPS FITS cutout data.
    template_mask : np.ndarray (bool)
        Mask of bright stars in the template.
    sci_mask : np.ndarray (bool)
        Mask of invalid pixels in the science image.
    """
    # Validate inputs
    try:
        st.info("Validating inputs...")
        if not isinstance(science_image, np.ndarray):
            raise TypeError("science_image must be numpy array")
        if not hasattr(wcs, 'pixel_to_world'):
            raise TypeError("wcs must be astropy.wcs.WCS")
        if len(shape) != 2:
            raise ValueError("shape= (ny,nx)")
        ny, nx = shape
        if science_image.shape != (ny, nx):
            st.warning("science_image shape != shape parameter; proceeding.")
        st.success("Inputs validated.")
    except Exception as e:
        st.error(f"Validation error: {e}")
        raise

    # Determine center and radius
    try:
        center = wcs.pixel_to_world(nx/2, ny/2)
        if catalog_radius is None:
            corners = np.array([[0,0],[0,ny],[nx,0],[nx,ny]])
            sky = wcs.pixel_to_world(corners[:,0], corners[:,1])
            catalog_radius = center.separation(sky).max().deg
        st.success(f"Center and radius: {center}, {catalog_radius:.3f}°")
    except Exception as e:
        st.error(f"Center/radius error: {e}")
        raise

    # Fetch HiPS FITS cutout
    try:
        st.info(f"Querying hips2fits for {hips_survey}...")
        hdu = hips2fits.query_with_wcs(
            hips=hips_survey,
            wcs=wcs,
            format='fits'
        )
        hips_data = hdu[0].data.astype(float)
        hips_header = hdu[0].header
        st.success("HiPS FITS cutout retrieved.")
    except Exception as e:
        st.error(f"hips2fits query error: {e}")
        raise

    # Crop or pad to match science image
    hips_data = hips_data[:ny, :nx]

    # WCS and shape consistency
    try:
        st.info("Checking WCS consistency...")
        hips_wcs = WCS(hips_header)
        if hips_data.shape != (ny, nx):
            st.warning(f"Shape mismatch: {hips_data.shape} vs {(ny,nx)}")
        else:
            st.success("Shape match OK.")
        sci_cd = np.abs(wcs.wcs.cdelt)
        hips_cd = np.abs(hips_wcs.wcs.cdelt)
        if not np.allclose(sci_cd, hips_cd, rtol=1e-3):
            st.warning(f"Pixel scale mismatch: {sci_cd} vs {hips_cd}")
        else:
            st.success("Pixel scale OK.")
        if not np.allclose(wcs.wcs.crval, hips_wcs.wcs.crval, atol=1e-3):
            st.warning(f"CRVAL mismatch: {wcs.wcs.crval} vs {hips_wcs.wcs.crval}")
        else:
            st.success("CRVAL OK.")
    except Exception as e:
        st.error(f"WCS check error: {e}")

    # Mask bright stars
    template_mask = np.zeros_like(hips_data, bool)
    try:
        st.info(f"Querying stars < {star_mag_limit} mag...")
        v = Vizier(columns=['RAJ2000','DEJ2000'], column_filters={f'gmag': f'<{star_mag_limit}'})
        res = v.query_region(center, radius=catalog_radius*u.deg, catalog='V/PS1')
        if res and len(res[0]):
            sc = SkyCoord(res[0]['RAJ2000'], res[0]['DEJ2000'], unit='deg')
            x,y = wcs.world_to_pixel(sc)
            apertures = CircularAperture(np.vstack((x,y)).T, r=psf_radius)
            st.success(f"Masking {len(x)} stars.")
            for ap in apertures.to_mask(method='center'):
                mask_img = ap.to_image(shape)
                template_mask |= (mask_img > 0)
        else:
            st.info("No bright stars found.")
    except Exception as e:
        st.warning(f"Star mask error: {e}")

    # Science image invalid mask
    try:
        sci_mask = ~np.isfinite(science_image)
        st.success("Science mask done.")
    except Exception as e:
        st.error(f"Science mask error: {e}")
        raise

    st.success("Template & masks ready.")
    return hips_data, template_mask, sci_mask


def visualize_results(
    hips_data: np.ndarray,
    science_image: np.ndarray,
    template_mask: np.ndarray,
    sci_mask: np.ndarray
) -> None:
    z = ZScaleInterval()
    vmin_s, vmax_s = z.get_limits(science_image)
    vmin_h, vmax_h = z.get_limits(hips_data)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0,0].imshow(science_image, origin='lower', vmin=vmin_s, vmax=vmax_s)
    axs[0,0].set_title('Science (ZScale)')
    axs[0, 0].axis('off')
    axs[0,1].imshow(hips_data, origin='lower', vmin=vmin_h, vmax=vmax_h)
    axs[0,1].set_title('HiPS (ZScale)')
    axs[0, 1].axis('off')
    axs[1,0].imshow(template_mask, origin='lower', cmap='gray')
    axs[1,0].set_title('Template Mask')
    axs[1, 0].axis('off')
    axs[1,1].imshow(sci_mask, origin='lower', cmap='gray')
    axs[1,1].set_title('Science Mask')
    axs[1, 1].axis('off')

    plt.tight_layout()
    st.pyplot(fig)


def match_psf_and_apply(
    hips_data: np.ndarray,
    science_image: np.ndarray,
    star_positions: np.ndarray,
    cutout_size: int = 25
) -> tuple:
    """
    Build PSF models for both images using isolated stars, create a matching kernel,
    and apply it to the HiPS template and science image.

    Parameters
    ----------
    hips_data : 2D array
        HiPS template image.
    science_image : 2D array
        Science image.
    template_mask, sci_mask : 2D bool arrays
        Masks to exclude stars and bad pixels when estimating PSF.
    star_positions : N x 2 array
        Pixel positions (x, y) of stars to use for PSF extraction.
    cutout_size : int
        Size of square cutouts around each star for PSF building.

    Returns
    -------
    matched_template : 2D array
        Template convolved to match science PSF.
    matched_science : 2D array
        Science image convolved to match template PSF.
    kernel : 2D array
        Matching convolution kernel.
    """
    st.info("Extracting star cutouts for PSF modeling...")
    try:
        # create tiny table for extract_stars
        from astropy.table import Table
        tbl = Table(rows=star_positions, names=('x_mean', 'y_mean'))
        # extract
        hips_stars = extract_stars(hips_data, tbl, size=cutout_size)
        sci_stars = extract_stars(science_image, tbl, size=cutout_size)
    except Exception as e:
        st.error(f"Star extraction failed: {e}")
        raise

    st.success("Stars extracted. Building EPSF models...")
    try:
        builder = EPSFBuilder(oversampling=2, maxiters=3)
        epsf_hips = builder(hips_stars)
        epsf_sci = builder(sci_stars)
    except Exception as e:
        st.error(f"EPSF building failed: {e}")
        raise

    st.success("EPSF models built. Creating matching kernel...")
    try:
        kernel = create_matching_kernel(epsf_hips.data, epsf_sci.data)
    except Exception as e:
        st.error(f"Kernel creation failed: {e}")
        raise

    st.info("Convolving images with matching kernel...")
    try:
        matched_template = convolve_fft(hips_data, kernel, normalize_kernel=True)
        matched_science = convolve_fft(science_image, kernel[::-1, ::-1], normalize_kernel=True)
    except Exception as e:
        st.error(f"Convolution failed: {e}")
        raise

    st.success("PSF matching applied.")
    return matched_template, matched_science, kernel