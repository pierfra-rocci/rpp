from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import get_sun
from typing import Union, Any, Optional, Dict, Tuple

from astropy.modeling import models, fitting
import streamlit as st
from streamlit.components.v1 import html  # Add this import for Aladin Lite widget
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, SigmaClip

import astropy.units as u
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, SExtractorBackground
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table
from astropy.visualization import ZScaleInterval, ImageNormalize, simple_norm
from io import StringIO, BytesIO
from astropy.wcs import WCS

from photutils.psf import EPSFBuilder, extract_stars, IterativePSFPhotometry
from astropy.nddata import NDData

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="RAPAS Photometric Calibration",
    page_icon="🔭",
    layout="wide"
)

# Custom CSS to control plot display size
st.markdown("""
<style>
    .stPlot > div {
        display: flex;
        justify-content: center;
        min-height: 400px;
    }
    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .element-container img {
        max-width: 100% !important;
        height: auto !important;
    }
</style>
""", unsafe_allow_html=True)

# Define standard figure sizes to use throughout the app
FIGURE_SIZES = {
    'small': (6, 5),     # For small plots
    'medium': (8, 6),    # For medium plots
    'large': (10, 8),    # For large plots
    'wide': (12, 6),     # For wide plots
    'stars_grid': (10, 8)  # For grid of stars
}

# Function to create standardized matplotlib figures
def create_figure(size='medium', dpi=100):
    """Create a matplotlib figure with standardized size"""
    if size in FIGURE_SIZES:
        figsize = FIGURE_SIZES[size]
    else:
        figsize = FIGURE_SIZES['medium']
    return plt.figure(figsize=figsize, dpi=dpi)

@st.cache_data
def airmass(
    _header: Dict,
    observatory: Optional[Dict] = None,
    return_details: bool = False
) -> Union[float, Tuple[float, Dict]]:
    """
    Calcule la masse d'air pour un objet céleste à partir des données d'en-tête
    et des informations de l'observatoire.

    Cette fonction gère plusieurs formats de coordonnées et effectue des
    vérifications de validité physique des résultats.

    Paramètres
    ----------
    header : Dict
        En-tête FITS ou dictionnaire contenant les informations de l'observation.
        Doit contenir:
        - Coordonnées (RA/DEC ou OBJRA/OBJDEC)
        - Date d'observation (DATE-OBS)
    observatory : Dict, optionnel
        Informations sur l'observatoire. Si non fourni, utilise l'OHP par défaut.
        Format:
        {
            'name': str,           # Nom de l'observatoire
            'latitude': float,     # Latitude en degrés
            'longitude': float,    # Longitude en degrés
            'elevation': float     # Élévation en mètres
        }
    return_details : bool, optionnel
        Si True, retourne des informations supplémentaires sur l'observation.

    Retourne
    -------
    Union[float, Tuple[float, Dict]]
        - Si return_details=False: masse d'air (float)
        - Si return_details=True: (masse d'air, détails_observation)

    Notes
    -----
    La masse d'air est une mesure de la quantité d'atmosphère traversée
    par la lumière d'un objet céleste. Une masse d'air de 1 correspond
    à une observation au zénith.
    """
    # Observatoire par défaut (OHP)
    DEFAULT_OBSERVATORY = {
        'name': 'TJMS',
        'latitude': 48.29166,
        'longitude': 2.43805,
        'elevation': 94.0
    }

    # Utilisation de l'observatoire spécifié ou par défaut
    obs_data = observatory or DEFAULT_OBSERVATORY

    try:
        # Extraction des coordonnées avec gestion de différents formats
        ra = _header.get("RA", _header.get("OBJRA", _header.get("RA---")))
        dec = _header.get("DEC", _header.get("OBJDEC", _header.get("DEC---")))
        obstime_str = _header.get("DATE-OBS", _header.get("DATE"))

        if any(v is None for v in [ra, dec, obstime_str]):
            missing = []
            if ra is None: missing.append("RA")
            if dec is None: missing.append("DEC")
            if obstime_str is None: missing.append("DATE-OBS")
            raise KeyError(f"Missing required header keywords: {', '.join(missing)}")

        # Conversion des coordonnées et création de l'objet SkyCoord
        coord = SkyCoord(ra=ra, dec=dec, unit=u.deg, frame='icrs')

        # Création de l'objet Time
        obstime = Time(obstime_str)

        # Création de l'objet EarthLocation pour l'observatoire
        location = EarthLocation(
            lat=obs_data['latitude']*u.deg,
            lon=obs_data['longitude']*u.deg,
            height=obs_data['elevation']*u.m
        )

        # Calcul de l'altitude et de la masse d'air
        altaz_frame = AltAz(obstime=obstime, location=location)
        altaz = coord.transform_to(altaz_frame)
        airmass_value = float(altaz.secz)

        # Vérifications physiques
        if airmass_value < 1.0:
            st.warning("Calculated airmass is less than 1 (physically impossible)")
            airmass_value = 1.0
        elif airmass_value > 40.0:
            st.warning("Extremely high airmass (>40), object near horizon")

        # Calcul de la position du Soleil pour vérifier les conditions d'observation
        sun_altaz = get_sun(obstime).transform_to(altaz_frame)
        sun_alt = float(sun_altaz.alt.deg)

        # Création du dictionnaire de détails
        details = {
            'observatory': obs_data['name'],
            'datetime': obstime.iso,
            'target_coords': {
                'ra': coord.ra.to_string(unit=u.hour),
                'dec': coord.dec.to_string(unit=u.deg)
            },
            'altaz': {
                'altitude': round(float(altaz.alt.deg), 2),
                'azimuth': round(float(altaz.az.deg), 2)
            },
            'sun_altitude': round(sun_alt, 2),
            'observation_type': 'night' if sun_alt < -18 else 
                              'twilight' if sun_alt < 0 else 
                              'day'
        }

        # Affichage des informations
        st.write("Observation details:")
        st.write(f"Date & Local Time: {obstime.iso}")
        st.write(f"Target position: RA={details['target_coords']['ra']}, "
              f"DEC={details['target_coords']['dec']}")
        st.write(f"Altitude: {details['altaz']['altitude']}°, "
              f"Azimuth: {details['altaz']['azimuth']}°")

        if return_details:
            return round(airmass_value, 2), details
        return round(airmass_value, 2)

    except Exception as e:
        st.warning(f"Error calculating airmass: {str(e)}")
        if return_details:
            return 0.0, {}
        return 0.0


@st.cache_data
def load_fits_data(file):
    """Load FITS data from an uploaded file."""
    if file is not None:
        file_content = file.read()
        # Create HDUList explicitly to avoid typing issues
        hdul = fits.open(BytesIO(file_content), mode='readonly')
        try:
            data = hdul[0].data
            header = hdul[0].header
            return data, header
        finally:
            hdul.close()
    return None, None


def calibrate_image_streamlit(science_data, science_header, bias_data, dark_data, flat_data,
                           exposure_time_science, exposure_time_dark,
                           apply_bias, apply_dark, apply_flat):
    """Calibrates a science image using bias, dark, and flat frames according to user selections."""
    if not apply_bias and not apply_dark and not apply_flat:
        st.write("Calibration steps are disabled. Returning raw science data.")
        return science_data, science_header

    calibrated_science = science_data.copy()
    steps_applied = []
    
    # Initialize corrected data variables
    dark_data_corrected = dark_data
    flat_data_corrected = flat_data

    if apply_bias and bias_data is not None:
        st.write("Application de la soustraction du bias...")
        calibrated_science -= bias_data
        steps_applied.append("Soustraction du Bias")
        if dark_data is not None:
            dark_data_corrected = dark_data - bias_data
        if flat_data is not None:
            flat_data_corrected = flat_data - bias_data

    if apply_dark and dark_data_corrected is not None:
        st.write("Application de la soustraction du dark...")
        # Scale dark frame if exposure times are different
        if exposure_time_science != exposure_time_dark:
            dark_scale_factor = exposure_time_science / exposure_time_dark
            scaled_dark = dark_data_corrected * dark_scale_factor
        else:
            scaled_dark = dark_data_corrected
        calibrated_science -= scaled_dark
        steps_applied.append("Soustraction du Dark")

    if apply_flat and flat_data_corrected is not None:
        st.write("Application de la correction du flat field...")
        # Normalize the flat field
        normalized_flat = flat_data_corrected / np.median(flat_data_corrected)
        calibrated_science /= normalized_flat
        steps_applied.append("Correction du Flat Field")

    if not steps_applied:
        st.write("No calibration steps were applied because files are missing or options are disabled.")
        return science_data, science_header

    st.success(f"Calibration steps applied: {', '.join(steps_applied)}")
    return calibrated_science, science_header


@st.cache_data
def make_border_mask(
    image: np.ndarray,
    border: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 50,
    invert: bool = True,
    dtype: np.dtype = bool
) -> np.ndarray:
    """Creates a binary mask for an image excluding one or more borders."""
    # Validation du type d'entrée
    if not isinstance(image, np.ndarray):
        raise TypeError("L'image doit être un numpy.ndarray")

    if image.size == 0:
        raise ValueError("L'image ne peut pas être vide")

    height, width = image.shape[:2]

    # Conversion et validation des bordures
    if isinstance(border, (int, float)):
        border = int(border)
        top = bottom = left = right = border
    elif isinstance(border, tuple):
        if len(border) == 2:
            vert, horiz = border
            top = bottom = vert
            left = right = horiz
        elif len(border) == 4:
            top, bottom, left, right = border
        else:
            raise ValueError("border doit être un int ou un tuple de 2 ou 4 éléments")
    else:
        raise TypeError("border doit être un int ou un tuple")

    # Validation des dimensions
    if any(b < 0 for b in (top, bottom, left, right)):
        raise ValueError("Les bordures ne peuvent pas être négatives")

    if top + bottom >= height or left + right >= width:
        raise ValueError("Les bordures sont plus grandes que l'image")

    # Création du masque
    mask = np.zeros(image.shape[:2], dtype=dtype)
    mask[top:height-bottom, left:width-right] = True

    return ~mask if invert else mask


@st.cache_data
def fwhm_fit(
    img: np.ndarray,
    fwhm: float,
    pixel_scale: float,
    mask: Optional[np.ndarray] = None,
    std_lo: float = 0.5,
    std_hi: float = 0.5
) -> Optional[float]:
    """
    Calculate the FWHM of an image using marginal sums and 1D Gaussian model fitting.
    
    This function first filters sources based on their flux, keeping those within a range
    defined by std_lo and std_hi around the median. For each filtered source, it extracts
    a sub-image, calculates marginal sums, and fits a 1D Gaussian model to estimate FWHM.
    
    Returns the median estimated FWHM in pixels.
    """
    def compute_fwhm_marginal_sums(image_data, center_row, center_col, box_size):
        """Compute FWHM using marginal sums and Gaussian fitting."""
        half_box = box_size // 2

        # Check if box is within image boundaries of the FULL IMAGE
        row_start = center_row - half_box
        row_end = center_row + half_box + 1
        col_start = center_col - half_box
        col_end = center_col + half_box + 1

        if row_start < 0 or row_end > img.shape[0] or col_start < 0 or col_end > img.shape[1]:
            return None  # Box extends beyond image boundaries

        # Extract box region from the COMPLETE IMAGE
        box_data = img[row_start:row_end, col_start:col_end]

        # Calculate marginal sums
        sum_rows = np.sum(box_data, axis=1)
        sum_cols = np.sum(box_data, axis=0)

        # Create axis data for fitting
        row_indices = np.arange(box_size)
        col_indices = np.arange(box_size)

        # Fit Gaussians
        fitter = fitting.LevMarLSQFitter()

        # Fit rows
        model_row = models.Gaussian1D()
        model_row.amplitude.value = np.max(sum_rows)
        model_row.mean.value = half_box
        model_row.stddev.value = half_box/3
        try:
            fitted_row = fitter(model_row, row_indices, sum_rows)
            center_row_fit = fitted_row.mean.value + row_start
            fwhm_row = 2 * np.sqrt(2 * np.log(2)) * fitted_row.stddev.value * pixel_scale
        except Exception as e:
            st.error(f"Error fitting row marginal sum: {e}")
            return None

        # Fit columns
        model_col = models.Gaussian1D()
        model_col.amplitude.value = np.max(sum_cols)
        model_col.mean.value = half_box
        model_col.stddev.value = half_box/3
        try:
            fitted_col = fitter(model_col, col_indices, sum_cols)
            center_col_fit = fitted_col.mean.value + col_start
            fwhm_col = 2 * np.sqrt(2 * np.log(2)) * fitted_col.stddev.value * pixel_scale
        except Exception as e:
            st.error(f"Error fitting column marginal sum: {e}")
            return None

        return fwhm_row, fwhm_col, center_row_fit, center_col_fit

    try:
        daofind = DAOStarFinder(fwhm=1.5*fwhm, threshold=6 * np.std(img))
        sources = daofind(img, mask=mask)
        if sources is None:
            st.warning("No sources found by DAOStarFinder!")
            return None

        st.write(f"Number of sources found by DAOStarFinder: {len(sources)}")

        # Filter sources by flux
        flux = sources['flux']
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        mask_flux = (flux > median_flux - std_lo * std_flux) & (flux < median_flux + std_hi * std_flux)
        filtered_sources = sources[mask_flux]

        # Remove sources with NaN flux values
        filtered_sources = filtered_sources[~np.isnan(filtered_sources['flux'])]

        st.write(f"Number of sources after flux filtering: {len(filtered_sources)}")

        if len(filtered_sources) == 0:
            msg = "No valid sources for fitting found after filtering."
            st.error(msg)
            raise ValueError(msg)

        # Define analysis box size (in pixels)
        box_size = int(6 * round(fwhm))
        if box_size % 2 == 0:  # Ensure box_size is odd
            box_size += 1

        fwhm_values = []

        # Fit model for each filtered source
        for source in filtered_sources:
            try:
                x_cen = int(source['xcentroid'])
                y_cen = int(source['ycentroid'])

                # Calculate FWHM using marginal sums
                fwhm_results = compute_fwhm_marginal_sums(img, y_cen, x_cen, box_size)
                if fwhm_results is None:
                    st.warning(f"FWHM calculation using marginal sums failed for source at ({x_cen}, {y_cen}). Skipping.")
                    continue

                fwhm_row, fwhm_col, _, _ = fwhm_results

                # Use average of FWHM_row and FWHM_col as FWHM estimate for this source
                fwhm_source = np.mean([fwhm_row, fwhm_col])
                fwhm_values.append(fwhm_source)

            except Exception as e:
                st.error(f"Error calculating FWHM for source at coordinates ({x_cen}, {y_cen}): {e}")
                continue

        if len(fwhm_values) == 0:
            msg = "No valid sources for FWHM fitting after marginal sums adjustment."
            st.error(msg)
            raise ValueError(msg)

        # Convert list to array to filter NaN and infinite values
        fwhm_values_arr = np.array(fwhm_values)
        valid = ~np.isnan(fwhm_values_arr) & ~np.isinf(fwhm_values_arr)
        if not np.any(valid):
            msg = "All FWHM values are NaN or infinite after marginal sums calculation."
            st.error(msg)
            raise ValueError(msg)

        mean_fwhm = np.median(fwhm_values_arr[valid])
        st.write(f"Median FWHM estimate based on marginal sums and Gaussian model: {round(mean_fwhm)} pixels")

        return round(mean_fwhm)
    except ValueError as e:
        raise e
    except Exception as e:
        st.error(f"Unexpected error in fwhm_fit: {e}")
        raise ValueError(f"Unexpected error in fwhm_fit: {e}")


def perform_epsf_photometry(
    img: np.ndarray,
    phot_table: Table,
    fwhm: float,
    daostarfind: Any,   
    mask: Optional[np.ndarray] = None
) -> Tuple[Table, Any]:
    """
    Perform PSF photometry using an EPSF model.
    
    Parameters
    ----------
    img : np.ndarray
        Image with sky background subtracted.
    phot_table : astropy.table.Table
        Table containing source positions.
    fwhm : float
        Full Width at Half Maximum used to define fitting size.
    daostarfind : callable
        Star detection function used as "finder" in PSF photometry.
    mask : np.ndarray, optional
        Mask to exclude certain image areas.
        
    Returns
    -------
    Tuple[astropy.table.Table, photutils.epsf.EPSF]
        - phot_epsf : PSF photometry results with EPSF model.
        - epsf : Fitted EPSF model.
    """
    try:
        # Prepare data: convert image to NDData object
        nddata = NDData(data=img)
        # st.write("NDData created successfully.")
    except Exception as e:
        st.error(f"Error creating NDData: {e}")
        raise

    try:
        # Build star table from phot_table
        stars_table = Table()
        stars_table['x'] = phot_table['xcenter']
        stars_table['y'] = phot_table['ycenter']
        st.write("Star positions table prepared.")
    except Exception as e:
        st.error(f"Error preparing star positions table: {e}")
        raise

    try:
        # Define fitting shape (box size for star extraction)
        fit_shape = 2 * round(fwhm) + 1
        st.write(f"Fitting shape defined: {fit_shape} pixels.")
    except Exception as e:
        st.error(f"Error calculating fitting shape: {e}")
        raise

    try:
        # Extract stars in sub-image
        stars = extract_stars(nddata, stars_table, size=fit_shape)
        st.write(f"{len(stars)} stars extracted.")
    except Exception as e:
        st.error(f"Error extracting stars: {e}")
        raise

    try:
        # Display extracted stars (optional)
        nrows, ncols = 5, 5
        fig_stars, ax_stars = plt.subplots(nrows=nrows, ncols=ncols, figsize=FIGURE_SIZES['stars_grid'], squeeze=False)
        ax_stars = ax_stars.ravel()
        n_disp = min(len(stars), nrows * ncols)
        for i in range(n_disp):
            norm = simple_norm(stars[i].data, 'log', percent=99.0)
            ax_stars[i].imshow(stars[i].data, norm=norm, origin='lower', cmap='viridis')
        plt.tight_layout()
        st.pyplot(fig_stars)
        st.write("Extracted stars display complete.")
    except Exception as e:
        st.warning(f"Error displaying extracted stars: {e}")

    try:
        # Build and fit EPSF model
        epsf_builder = EPSFBuilder(oversampling=2, maxiters=5, progress_bar=False)
        epsf, _ = epsf_builder(stars)
        st.write("EPSF model fitted successfully.")
        st.session_state['epsf_model'] = epsf
    except Exception as e:
        st.error(f"Error fitting EPSF model: {e}")
        raise

    try:
        # Display fitted EPSF model
        norm_epsf = simple_norm(epsf.data, 'log', percent=99.)
        fig_epsf_model, ax_epsf_model = plt.subplots(figsize=FIGURE_SIZES['medium'], dpi=100)
        ax_epsf_model.imshow(epsf.data, norm=norm_epsf, origin='lower', cmap='viridis', interpolation='nearest')
        # plt.colorbar(ax=ax_epsf_model)
        ax_epsf_model.set_title("Fitted EPSF Model")
        st.pyplot(fig_epsf_model)
        st.write("EPSF model display complete.")
    except Exception as e:
        st.warning(f"Error displaying EPSF model: {e}")

    try:
        # Perform PSF photometry with EPSF model
        psfphot = IterativePSFPhotometry(
            epsf,
            fit_shape,
            finder=daostarfind,
            aperture_radius=fit_shape / 2,
            maxiters=3,
            mode='new',
            progress_bar=False
        )
        # Specify source positions
        psfphot.x = phot_table['xcenter']
        psfphot.y = phot_table['ycenter']
        st.write("Source positions defined for PSF photometry.")
    except Exception as e:
        st.error(f"Error configuring PSF photometry: {e}")
        raise

    try:
        # Run PSF photometry with provided mask
        phot_epsf_result = psfphot(img, mask=mask)
        st.session_state['epsf_photometry_result'] = phot_epsf_result
        st.write("EPSF photometry completed successfully.")
    except Exception as e:
        st.error(f"Error executing EPSF photometry: {e}")
        raise

    return phot_epsf_result, epsf


@st.cache_data
def find_sources_and_photometry_streamlit(image_data, _science_header, mean_fwhm_pixel, threshold_sigma, detection_mask):
    """
    Find sources and perform photometry on astronomical images.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Science image data
    _science_header : dict
        Header information from FITS file (underscore prevents caching issues)
    mean_fwhm_pixel : float
        Estimated FWHM in pixels
    threshold_sigma : float
        Detection threshold in sigma
    detection_mask : int
        Border size to mask during detection
        
    Returns
    -------
    tuple
        (phot_table, epsf_table, daofind, bkg)
    """
    # Get pixel scale from header
    pixel_scale = _science_header.get('PIXSCALE', _science_header.get('PIXSIZE', _science_header.get('PIXELSCAL', 1.0)))
    
    # Estimate background and noise
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = SExtractorBackground()
    bkg = Background2D(
        data=image_data,
        box_size=100,
        filter_size=5,
        mask=None,
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator
    )
    
    # Create detection mask
    mask = make_border_mask(image_data, border=detection_mask)
    
    # Calculate total error
    total_error = np.sqrt(bkg.background_rms**2 + bkg.background_median)
    
    st.write("Estimating FWHM...")
    fwhm_estimate = fwhm_fit(image_data - bkg.background, mean_fwhm_pixel, pixel_scale, mask)
    
    if fwhm_estimate is None:
        st.warning("Failed to estimate FWHM. Using the initial estimate.")
        fwhm_estimate = mean_fwhm_pixel
    
    # Source detection using DAOStarFinder
    daofind = DAOStarFinder(fwhm=1.5*fwhm_estimate, threshold=threshold_sigma * np.std(image_data - bkg.background))
    sources = daofind(image_data - bkg.background, mask=mask)
    
    if sources is None or len(sources) == 0:
        st.warning("No sources found!")
        return None, None, daofind, bkg
    
    st.write(f"Found {len(sources)} sources")
    
    # Aperture photometry
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=1.5*fwhm_estimate)
    
    # Perform aperture photometry
    try:
        phot_table = aperture_photometry(
            image_data - bkg.background, 
            apertures, 
            error=total_error,
            wcs=WCS(_science_header) if 'CTYPE1' in _science_header else None
        )
    except Exception as e:
        st.error(f"Error performing aperture photometry: {e}")
        return None, None, daofind, bkg
    
    # Add source IDs and coordinates to photometry table
    phot_table['xcenter'] = sources['xcentroid']
    phot_table['ycenter'] = sources['ycentroid']
    
    # Perform PSF/EPSF photometry
    try:
        epsf_table, _ = perform_epsf_photometry(
            image_data - bkg.background, 
            phot_table, 
            fwhm_estimate, 
            daofind, 
            mask
        )
        
        # Calculate instrumental magnitudes for aperture photometry
        instrumental_mags = -2.5 * np.log10(phot_table['aperture_sum'])
        phot_table['instrumental_mag'] = instrumental_mags
        
        # Calculate instrumental magnitudes for PSF photometry
        epsf_instrumental_mags = -2.5 * np.log10(epsf_table['flux_fit'])
        epsf_table['instrumental_mag'] = epsf_instrumental_mags

        # Keep only valid sources
        valid_sources = (phot_table['aperture_sum'] > 0) & np.isfinite(phot_table['instrumental_mag'])
        phot_table = phot_table[valid_sources]

        epsf_valid_sources = (epsf_table['flux_fit'] > 0) & np.isfinite(epsf_table['instrumental_mag'])
        epsf_table = epsf_table[epsf_valid_sources]

        # Add RA and Dec if WCS is available once
        try:
            w = WCS(_science_header)
            # Process phot_table
            ra, dec = w.pixel_to_world_values(phot_table['xcenter'], phot_table['ycenter'])
            phot_table['ra'] = ra * u.deg
            phot_table['dec'] = dec * u.deg
            
            # Process epsf_table
            epsf_ra, epsf_dec = w.pixel_to_world_values(epsf_table['x_fit'], epsf_table['y_fit'])
            epsf_table['ra'] = epsf_ra * u.deg
            epsf_table['dec'] = epsf_dec * u.deg
        except Exception as e:
            st.warning(f"WCS transformation failed: {e}. RA and Dec not added to tables.")
            
        st.success(f"Found {len(phot_table)} sources and performed photometry.")
        return phot_table, epsf_table, daofind, bkg
    except Exception as e:
        st.error(f"Error performing PSF photometry: {e}")
        return phot_table, None, daofind, bkg


@st.cache_data
def cross_match_with_gaia_streamlit(_phot_table, _science_header, pixel_size_arcsec, mean_fwhm_pixel, gaia_band, gaia_min_mag, gaia_max_mag):
    """
    Cross-matches sources with Gaia catalog.
    
    Parameters
    ----------
    _phot_table : astropy.table.Table
        Table containing source positions (underscore prevents caching issues)
    _science_header : dict
        FITS header with WCS information (underscore prevents caching issues)
    pixel_size_arcsec : float
        Pixel scale in arcseconds per pixel
    mean_fwhm_pixel : float
        FWHM in pixels
    gaia_band : str
        Gaia magnitude band to use (e.g., 'phot_g_mean_mag')
    gaia_min_mag : float
        Minimum magnitude for filtering Gaia sources
    gaia_max_mag : float
        Maximum magnitude for filtering Gaia sources
        
    Returns
    -------
    pandas.DataFrame
        Matched sources table
    """
    st.write("Cross-matching with Gaia DR3...")

    if _science_header is None:
        st.warning("No header information available. Cannot cross-match with Gaia.")
        return None

    try:
        w = WCS(_science_header)
    except Exception as e:
        st.error(f"Error creating WCS: {e}")
        return None

    # Convert pixel positions to sky coordinates
    try:
        source_positions_pixel = np.transpose((_phot_table['xcenter'], _phot_table['ycenter']))
        source_positions_sky = w.pixel_to_world(source_positions_pixel[:,0], source_positions_pixel[:,1])
    except Exception as e:
        st.error(f"Error converting pixel positions to sky coordinates: {e}")
        return None

    # Query Gaia
    try:
        image_center_ra_dec = w.pixel_to_world(_science_header['NAXIS1']//2, _science_header['NAXIS2']//2)
        # Calculate search radius as half the image diagonal
        gaia_search_radius_arcsec = max(_science_header['NAXIS1'], _science_header['NAXIS2']) * pixel_size_arcsec / 2.0
        radius_query = gaia_search_radius_arcsec * u.arcsec

        st.write(f"Querying Gaia DR3 in a radius of {round(radius_query.value/60.,2)} arcmin.")
        job = Gaia.cone_search(image_center_ra_dec, radius=radius_query)
        gaia_table = job.get_results()
    except Exception as e:
        st.error(f"Error querying Gaia: {e}")
        return None

    if gaia_table is None or len(gaia_table) == 0:
        st.warning("No Gaia sources found within search radius.")
        return None

    # Apply filters to Gaia catalog
    try:
        # Apply magnitude filter
        mag_filter = (gaia_table[gaia_band] < gaia_max_mag) & (gaia_table[gaia_band] > gaia_min_mag)
        
        # Apply variable star filter if column exists
        if "phot_variable_flag" in gaia_table.colnames:
            var_filter = gaia_table["phot_variable_flag"] != "VARIABLE"
            combined_filter = mag_filter & var_filter
        else:
            combined_filter = mag_filter
            
        gaia_table_filtered = gaia_table[combined_filter]
        
        if len(gaia_table_filtered) == 0:
            st.warning(f"No Gaia sources found within magnitude range {gaia_min_mag} < {gaia_band} < {gaia_max_mag}.")
            return None
            
        st.write(f"Filtered Gaia catalog to {len(gaia_table_filtered)} sources.")
    except Exception as e:
        st.error(f"Error filtering Gaia catalog: {e}")
        return None

    # Cross-match
    try:
        gaia_skycoords = SkyCoord(ra=gaia_table_filtered['ra'], dec=gaia_table_filtered['dec'], unit='deg')
        idx, d2d, _ = source_positions_sky.match_to_catalog_sky(gaia_skycoords)
        
        # Set maximum separation constraint based on FWHM and pixel scale
        max_sep_constraint = 2 * mean_fwhm_pixel * pixel_size_arcsec * u.arcsec
        gaia_matches = (d2d <= max_sep_constraint)

        matched_indices_gaia = idx[gaia_matches]
        matched_indices_phot = np.where(gaia_matches)[0]

        if len(matched_indices_gaia) == 0:
            st.warning("No Gaia matches found within the separation constraint.")
            return None
            
        # Extract matched sources
        matched_table_qtable = _phot_table[matched_indices_phot]
        
        # Convert to pandas DataFrame
        matched_table = matched_table_qtable.to_pandas()
        matched_table['gaia_index'] = matched_indices_gaia
        matched_table['gaia_separation_arcsec'] = d2d[gaia_matches].arcsec
        matched_table[gaia_band] = gaia_table_filtered[gaia_band][matched_indices_gaia]
        
        # Filter invalid magnitudes
        valid_gaia_mags = np.isfinite(matched_table[gaia_band])
        matched_table = matched_table[valid_gaia_mags]
        
        st.success(f"Found {len(matched_table)} Gaia matches after filtering.")
        return matched_table
    except Exception as e:
        st.error(f"Error during cross-matching: {e}")
        return None


@st.cache_data
def calculate_zero_point_streamlit(_phot_table, _matched_table, gaia_band, air):
    """
    Calculates photometric zero point using matched sources.
    
    Parameters
    ----------
    _phot_table : astropy.table.Table or pandas.DataFrame
        Photometry results table (underscore prevents caching issues)
    _matched_table : pandas.DataFrame
        Table of Gaia cross-matched sources (underscore prevents caching issues)
    gaia_band : str
        Gaia magnitude band used
    air : float
        Airmass value for extinction correction
        
    Returns
    -------
    tuple
        (zero_point_value, zero_point_std, matplotlib_figure)
    """
    st.write("Calculating zero point...")

    if _matched_table is None or len(_matched_table) == 0:
        st.warning("No matched sources to calculate zero point.")
        return None, None, None

    try:
        # Calculate zero points
        zero_points = _matched_table[gaia_band] - _matched_table['instrumental_mag']
        _matched_table['zero_point'] = zero_points
        _matched_table['zero_point_error'] = np.std(zero_points)
        
        # Apply sigma clipping to remove outliers
        clipped_zero_points = sigma_clip(zero_points, sigma=3)
        zero_point_value = np.mean(clipped_zero_points)
        zero_point_std = np.std(clipped_zero_points)
        
        # Calculate calibrated magnitude for matched sources
        _matched_table['calib_mag'] = _matched_table['instrumental_mag'] + zero_point_value + 0.1*air
        
        # Convert _phot_table to DataFrame if needed
        if not isinstance(_phot_table, pd.DataFrame):
            _phot_table = _phot_table.to_pandas()
            
        # Calculate calibrated magnitudes for all sources
        _phot_table['calib_mag'] = _phot_table['instrumental_mag'] + zero_point_value + 0.1*air
        
        # Calculate errors
        if 'aperture_sum_err_0' in _phot_table.columns and 'aperture_sum_0' in _phot_table.columns:
            _phot_table['calib_mag_err'] = (2.5/np.log(10) * _phot_table['aperture_sum_err_0'] / 
                                          _phot_table['aperture_sum_0']) + zero_point_std
        
        # Store results in session state
        st.session_state['final_phot_table'] = _phot_table
        
        # Create plot
        fig, ax = plt.subplots(figsize=FIGURE_SIZES['medium'], dpi=100)
        ax.scatter(_matched_table[gaia_band], _matched_table['calib_mag'], alpha=0.75, label='Matched sources')
    
        ax.set_xlabel(f"Gaia {gaia_band}")
        ax.set_ylabel("Calibrated magnitude")
        ax.set_title("Gaia magnitude vs Calibrated magnitude")
        ax.legend()
        ax.grid(True, alpha=0.5)
        
        # Add annotation for zero point value
        # text_x = np.min(_matched_table[gaia_band]) + 0.1 * (np.max(_matched_table[gaia_band]) - np.min(_matched_table[gaia_band]))
        # text_y = np.max(_matched_table['calib_mag']) - 0.1 * (np.max(_matched_table['calib_mag']) - np.min(_matched_table['calib_mag']))
        # ax.text(text_x, text_y, f"Zero Point = {zero_point_value:.3f} ± {zero_point_std:.3f}", 
        #         bbox=dict(facecolor='white', alpha=0.8))
                
        st.success(f"Calculated Zero Point: {zero_point_value:.3f} ± {zero_point_std:.3f}")
        
        # Save plot but handle exceptions
        try:
            plt.savefig("zero_point_plot.png")
            st.write("Zero point plot saved as 'zero_point_plot.png'.")
        except Exception as e:
            st.warning(f"Could not save plot to file: {e}")
            
        return zero_point_value, zero_point_std, fig
    except Exception as e:
        st.error(f"Error calculating zero point: {e}")
        return None, None, None


def perform_epsf_photometry_streamlit(image_data, background_data, phot_table, fwhm, daofind, detection_mask):
    """
    Wrapper for PSF photometry with Streamlit UI integration.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Science image data
    background_data : numpy.ndarray
        Background data to subtract
    phot_table : astropy.table.Table
        Table with source positions
    fwhm : float
        FWHM estimate in pixels
    daofind : DAOStarFinder
        Configured DAOStarFinder object
    detection_mask : int
        Border size to mask
        
    Returns
    -------
    tuple
        (phot_epsf_result, epsf_model)
    """
    st.write("Starting PSF photometry...")
    
    # Create mask
    mask = make_border_mask(image_data, border=detection_mask)
    
    # Subtract background if provided
    image_bg_subtracted = image_data - background_data if background_data is not None else image_data
    
    try:
        # Perform PSF photometry
        phot_epsf_result, epsf_model = perform_epsf_photometry(
            image_bg_subtracted, phot_table, fwhm, daofind, mask
        )
        
        # Store results in session state
        st.session_state['epsf_photometry_result'] = phot_epsf_result
        st.session_state['epsf_model'] = epsf_model
        
        st.success("PSF-EPSF photometry completed successfully.")
        
        # Display EPSF model
        st.subheader("EPSF Model")
        norm_epsf = ImageNormalize(epsf_model.data, interval=ZScaleInterval())
        fig_epsf, ax_epsf = plt.subplots(figsize=FIGURE_SIZES['medium'], dpi=100)
        im_epsf = ax_epsf.imshow(epsf_model.data, norm=norm_epsf, origin='lower', cmap='viridis')
        fig_epsf.colorbar(im_epsf, ax=ax_epsf, label='EPSF Model Value')
        ax_epsf.set_title("EPSF Model (ZScale)")
        st.pyplot(fig_epsf)
        
        # Display photometry results preview
        st.subheader("PSF-EPSF Photometry Results (first 10 rows)")
        preview_df = phot_epsf_result[:10].to_pandas()
        st.dataframe(preview_df)
        
        return phot_epsf_result, epsf_model
    except Exception as e:
        st.error(f"Error performing PSF-EPSF photometry: {e}")
        return None, None


# Main workflow function for zero point calibration
def run_zero_point_calibration(image_data, header, pixel_size_arcsec, mean_fwhm_pixel, 
                              threshold_sigma, detection_mask, gaia_band, gaia_min_mag, gaia_max_mag, air):
    """
    Runs the complete zero point calibration workflow.
    
    Parameters
    ----------
    image_data : numpy.ndarray
        Image data to process
    header : dict
        FITS header
    pixel_size_arcsec : float
        Pixel scale in arcseconds
    mean_fwhm_pixel : float
        FWHM estimate in pixels
    threshold_sigma : float
        Detection threshold
    detection_mask : int
        Border mask size
    gaia_band : str
        Gaia magnitude band
    gaia_min_mag, gaia_max_mag : float
        Magnitude limits for Gaia sources
    air : float
        Airmass value
        
    Returns
    -------
    tuple
        (zero_point_value, zero_point_std, phot_table)
    """
    with st.spinner("Finding sources and performing photometry..."):
        phot_table_qtable, epsf_table, daofind, bkg = find_sources_and_photometry_streamlit(
            image_to_process, header_to_process, mean_fwhm_pixel, threshold_sigma, detection_mask
        )
        
        if phot_table_qtable is None:
            st.error("Failed to perform photometry - no sources found")
            return None, None, None
            
    with st.spinner("Cross-matching with Gaia..."):
        matched_table = cross_match_with_gaia_streamlit(
            phot_table_qtable, header, pixel_size_arcsec, mean_fwhm_pixel,
            gaia_band, gaia_min_mag, gaia_max_mag
        )
        
        if matched_table is None:
            st.error("Failed to cross-match with Gaia")
            return None, None, phot_table_qtable
    
    st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
    st.dataframe(matched_table.head(10))
    
    with st.spinner("Calculating zero point..."):
        zero_point_value, zero_point_std, zp_plot = calculate_zero_point_streamlit(
            phot_table_qtable, matched_table, gaia_band, air  # Use phot_table_qtable, not phot_table
        )
        
        if zero_point_value is not None:
            st.pyplot(zp_plot)
            
            # Prepare catalog for download
            if 'final_phot_table' in st.session_state:
                final_table = st.session_state['final_phot_table']
                
                try:
                    # Add EPSF photometry results if available
                    if 'epsf_photometry_result' in st.session_state and epsf_table is not None:
                        # Convert tables to pandas if needed
                        epsf_df = epsf_table.to_pandas() if not isinstance(epsf_table, pd.DataFrame) else epsf_table
                        
                        # Add a unique ID for matching (based on x,y position)
                        epsf_df['match_id'] = epsf_df['xcenter'].round(2).astype(str) + "_" + epsf_df['ycenter'].round(2).astype(str)
                        final_table['match_id'] = final_table['xcenter'].round(2).astype(str) + "_" + final_table['ycenter'].round(2).astype(str)
                        
                        # Select just the columns we need from EPSF results
                        epsf_cols = {
                            'match_id': 'match_id',
                            'flux_fit': 'epsf_flux_fit', 
                            'flux_unc': 'epsf_flux_unc',
                            'instrumental_mag': 'epsf_instrumental_mag'
                        }
                        epsf_subset = epsf_df[list(epsf_cols.keys())].rename(columns=epsf_cols)
                        
                        # Merge the EPSF results with the final table
                        final_table = pd.merge(final_table, epsf_subset, on='match_id', how='left')
                        
                        # Calculate calibrated magnitudes for EPSF photometry
                        if 'epsf_instrumental_mag' in final_table.columns:
                            final_table['epsf_calib_mag'] = final_table['epsf_instrumental_mag'] + zero_point_value + 0.1*air
                        
                        # Remove temporary match column
                        final_table.drop('match_id', axis=1, inplace=True)
                        
                        st.success("Added EPSF photometry results to the catalog")
                    
                    # Create a StringIO buffer for CSV data
                    csv_buffer = StringIO()
                    
                    # Check and remove problematic columns if they exist
                    cols_to_drop = []
                    for col_name in ['sky_center.ra', 'sky_center.dec']:
                        if col_name in final_table.columns:
                            cols_to_drop.append(col_name)
                    
                    if cols_to_drop:
                        final_table = final_table.drop(columns=cols_to_drop)
                    
                    # Write the filtered DataFrame to the buffer
                    final_table.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    # Ensure filename has .csv extension
                    filename = catalog_name if catalog_name.endswith('.csv') else f"{catalog_name}.csv"
                    
                    # Provide download button with better formatting
                    st.download_button(
                        label="📥 Download Photometry Catalog",
                        data=csv_data,
                        file_name=filename,
                        mime='text/csv',
                        help="Download the photometry results as a CSV file"
                    )
                    
                    # Also save locally if needed
                    with open(filename, 'w') as f:
                        f.write(csv_data)
                    # st.success(f"Catalog saved to {filename}")
                    
                except Exception as e:
                    st.error(f"Error preparing download: {e}")
                    st.exception(e)
        
        return zero_point_value, zero_point_std, final_table


# ------------------------------------------------------------------------------

# Main Script Execution
# ------------------------------------------------------------------------------

# Initialize session state variables if they don't exist
if 'calibrated_data' not in st.session_state:
    st.session_state['calibrated_data'] = None

if 'calibrated_header' not in st.session_state:
    st.session_state['calibrated_header'] = None

if 'final_phot_table' not in st.session_state:
    st.session_state['final_phot_table'] = None



st.title("_RAPAS Photometric Calibration_")

# Photometry parameters in sidebar
with st.sidebar:
    st.sidebar.header("Upload FITS Files")
    bias_file = st.file_uploader("Master Bias (optional)", type=['fits', 'fit', 'fts'])
    dark_file = st.file_uploader("Master Dark (optional)", type=['fits', 'fit', 'fts'])
    flat_file = st.file_uploader("Master Flat (optional)", type=['fits', 'fit', 'fts'])
    science_file = st.file_uploader("Science Image (required)", type=['fits', 'fit', 'fts'], key="science_upload")
     # Also move calibration options to sidebar
    st.header("Calibration Options")
    calibrate_bias = st.checkbox("Apply Bias", value=False,
                              help="Subtract bias frame from science image")
    calibrate_dark = st.checkbox("Apply Dark", value=False,
                              help="Subtract dark frame from science image")
    calibrate_flat = st.checkbox("Apply Flat Field", value=False,
                              help="Divide science image by flat field")
    
    st.header("Analysis Parameters")
    seeing = st.number_input("Seeing (arcsec)", value=3.5, 
                     help="Estimate of the atmospheric seeing in arcseconds")
    threshold_sigma = st.number_input("Detection Threshold (σ)", value=3.0,
                              help="Source detection threshold in sigma above background")
    detection_mask = st.slider("Border Mask (pixels)", 25, 200, 50, 25,
                           help="Size of border to exclude from source detection")
    
    st.header("Gaia Parameters")
    gaia_band = st.selectbox("Gaia Band", 
                          ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
                          index=0,
                          help="Gaia magnitude band to use for calibration")
    gaia_min_mag = st.slider("Gaia Min Magnitude", 7.0, 12.0, 10.0, 0.5,
                          help="Minimum magnitude for Gaia sources")
    gaia_max_mag = st.slider("Gaia Max Magnitude", 16.0, 20.0, 19.0, 0.5,
                          help="Maximum magnitude for Gaia sources")
    
   
    
    # Move catalog name to sidebar as well
    st.header("Output Options")
    catalog_name = st.text_input("Output Catalog Filename", "photometry_catalog.csv")

    st.link_button("GAIA Archive", "https://www.cosmos.esa.int/web/gaia/data-release-3")
    st.link_button("Simbad", "http://simbad.u-strasbg.fr/simbad/")
    st.link_button("VizieR", "http://vizier.u-strasbg.fr/viz-bin/VizieR")
    st.link_button("NED", "https://ned.ipac.caltech.edu/")
    st.link_button("ADS", "https://ui.adsabs.harvard.edu/")

# Main processing logic
if science_file is not None:
    science_data, science_header = load_fits_data(science_file)
    bias_data, _ = load_fits_data(bias_file)
    dark_data, dark_header = load_fits_data(dark_file)
    flat_data, _ = load_fits_data(flat_file)
    
    # If headers are None, create empty dictionaries to avoid errors
    if science_header is None:
        science_header = {}
    if dark_header is None:
        dark_header = {}

    st.header("Science Image", anchor="science-image")
    
    # Show the image with zscale stretching
    try:
        norm = ImageNormalize(science_data, interval=ZScaleInterval())
        fig_preview, ax_preview = plt.subplots(figsize=FIGURE_SIZES['medium'], dpi=100)  # Explicit figure size
        im = ax_preview.imshow(science_data, norm=norm, origin='lower', cmap="viridis")
        fig_preview.colorbar(im, ax=ax_preview, label='Pixel Value')
        ax_preview.set_title("(zscale stretch)")
        ax_preview.axis('off') 
        st.pyplot(fig_preview, clear_figure=True)
    except Exception as e:
        st.error(f"Error displaying image: {e}")

    # Show header information
    with st.expander("Science Image Header"):
        if science_header:
            st.text(repr(science_header))
        else:
            st.warning("No header information available for science image.")

    # Show basic statistics about the image
    st.subheader("Science Image Statistics")
    if science_data is not None:
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("Mean", f"{np.mean(science_data):.3f}")
        stats_col2.metric("Median", f"{np.median(science_data):.3f}")
        stats_col3.metric("Std Dev", f"{np.std(science_data):.3f}")

        # Get pixel scale and calculate estimated FWHM
        pixel_size_arcsec = None
        try:
            if 'PIXSIZE' in science_header:
                pixel_size_arcsec = science_header['PIXSIZE']
            elif 'PIXSCALE' in science_header:
                pixel_size_arcsec = science_header['PIXSCALE']
            elif 'PIXELSCAL' in science_header:
                pixel_size_arcsec = science_header['PIXELSCAL']
            elif 'CDELT2' in science_header:
                pixel_size_arcsec = abs(science_header['CDELT2']) * 3600.0
            elif 'CDELT1' in science_header:
                pixel_size_arcsec = abs(science_header['CDELT1']) * 3600.0

            if pixel_size_arcsec:
                st.metric("Pixel Size (arcsec)", f"{pixel_size_arcsec:.2f}")
                mean_fwhm_pixel = seeing / pixel_size_arcsec
                st.metric("Est. Mean FWHM (pixels)", f"{mean_fwhm_pixel:.2f} (from seeing)")
            else:
                st.warning("Pixel scale not found in header. Using default value of 1.0 arcsec/pixel")
                pixel_size_arcsec = 1.0
                mean_fwhm_pixel = seeing / pixel_size_arcsec
                st.metric("Est. Mean FWHM (pixels)", f"{mean_fwhm_pixel:.2f} (from seeing, with default pixel scale)")
        except Exception as e:
            st.warning(f"Error reading pixel scale from header: {e}")
            pixel_size_arcsec = 1.0
            mean_fwhm_pixel = seeing
            
        # Calculate airmass
        try:
            air = airmass(science_header)
            st.write(f"Airmass: {air:.2f}")
        except Exception as e:
            st.warning(f"Error calculating airmass: {e}")
            air = 1.0
            st.write(f"Using default airmass: {air:.2f}")

        # Image calibration button
        calibration_disabled = not (calibrate_bias or calibrate_dark or calibrate_flat)
        exposure_time_science = science_header.get('EXPOSURE', science_header.get('EXPTIME', 1.0))
        exposure_time_dark = dark_header.get('EXPOSURE', dark_header.get('EXPTIME', exposure_time_science))
        
        if st.button("Run Image Calibration", disabled=calibration_disabled):
            with st.spinner("Calibrating image..."):
                try:
                    calibrated_data, calibrated_header = calibrate_image_streamlit(
                        science_data, science_header, bias_data, dark_data, flat_data,
                        exposure_time_science, exposure_time_dark,
                        calibrate_bias, calibrate_dark, calibrate_flat
                    )
                    st.session_state['calibrated_data'] = calibrated_data
                    st.session_state['calibrated_header'] = calibrated_header
                    
                    # Show calibrated image
                    if calibrated_data is not None:
                        st.header("Calibrated Science Image")
                        norm_calibrated = ImageNormalize(calibrated_data, interval=ZScaleInterval())
                        fig_calibrated, ax_calibrated = plt.subplots(figsize=FIGURE_SIZES['medium'], dpi=100)
                        im_calibrated = ax_calibrated.imshow(calibrated_data, norm=norm_calibrated,
                                                           origin='lower', cmap="viridis")
                        fig_calibrated.colorbar(im_calibrated, ax=ax_calibrated, label='pixel value')
                        ax_calibrated.set_title("Calibrated Image (zscale)")
                        ax_calibrated.axis('off')
                        st.pyplot(fig_calibrated)
                except Exception as e:
                    st.error(f"Error during calibration: {e}")

        # Zero point calibration button
        zero_point_button_disabled = science_file is None
        if st.button("Run Zero Point Calibration", disabled=zero_point_button_disabled, key="run_zp"):
            # Determine which data to use
            image_to_process = science_data  # Default to raw science data
            header_to_process = science_header
            
            # Use calibrated data if available
            if st.session_state['calibrated_data'] is not None:
                image_to_process = st.session_state['calibrated_data']
                header_to_process = st.session_state['calibrated_header']
            
            st.write("Doing astrometry refinement with GAIA DR3...")
            # wcs = astrometry_script(image_to_process, header_to_process, catalog="GAIA", FWHM=mean_fwhm_pixel)
            # header_to_process.update(wcs.to_header())

            if image_to_process is not None:
                try:
                    with st.spinner("Background Extraction, Find Sources and Perform Photometry..."):
                        phot_table_qtable, epsf_table, daofind, bkg = find_sources_and_photometry_streamlit(  # Use QTable for caching
                            image_to_process, header_to_process, mean_fwhm_pixel, threshold_sigma, detection_mask
                        )
                        
                        # Create a deep copy to avoid modifying the cached data
                        if phot_table_qtable is not None:
                            phot_table_df = phot_table_qtable.to_pandas().copy(deep=True)
                        else:
                            st.error("No sources detected in the image.")
                            phot_table_df = None

                    if phot_table_df is not None:
                        with st.spinner("Cross-matching with Gaia..."):
                            matched_table = cross_match_with_gaia_streamlit(
                                phot_table_qtable, header_to_process, pixel_size_arcsec, 
                                mean_fwhm_pixel, gaia_band, gaia_min_mag, gaia_max_mag
                            )

                        if matched_table is not None:
                            st.subheader("Cross-matched Gaia Catalog (first 10 rows)")
                            st.dataframe(matched_table.head(10))

                            with st.spinner("Calculating zero point..."):
                                zero_point_value, zero_point_std, zp_plot = calculate_zero_point_streamlit(
                                    phot_table_qtable, matched_table, gaia_band, air
                                )
                                
                                if zero_point_value is not None:
                                    st.pyplot(zp_plot)
                                    
                                    # Prepare for download if final_phot_table exists in session state
                                    if 'final_phot_table' in st.session_state:
                                        final_table = st.session_state['final_phot_table']
                                        
                                        try:
                                            # Add EPSF photometry results if available
                                            if 'epsf_photometry_result' in st.session_state and epsf_table is not None:
                                                # Convert tables to pandas if needed
                                                epsf_df = epsf_table.to_pandas() if not isinstance(epsf_table, pd.DataFrame) else epsf_table
                                                
                                                # Check what column names are used for coordinates
                                                epsf_x_col = 'x_fit' if 'x_fit' in epsf_df.columns else 'xcenter'
                                                epsf_y_col = 'y_fit' if 'y_fit' in epsf_df.columns else 'ycenter'
                                                
                                                # Find coordinate columns in final table
                                                final_x_col = 'xcenter' if 'xcenter' in final_table.columns else 'x_0'
                                                final_y_col = 'ycenter' if 'ycenter' in final_table.columns else 'y_0'
                                                
                                                # Add a unique ID for matching (based on x,y position)
                                                if epsf_x_col in epsf_df.columns and epsf_y_col in epsf_df.columns:
                                                    epsf_df['match_id'] = epsf_df[epsf_x_col].round(2).astype(str) + "_" + epsf_df[epsf_y_col].round(2).astype(str)
                                                    
                                                if final_x_col in final_table.columns and final_y_col in final_table.columns:
                                                    final_table['match_id'] = final_table[final_x_col].round(2).astype(str) + "_" + final_table[final_y_col].round(2).astype(str)
                                                else:
                                                    st.warning(f"Coordinate columns not found in final table. Available columns: {final_table.columns.tolist()}")
                                                    
                                                # Select just the columns we need from EPSF results
                                                epsf_cols = {}
                                                epsf_cols['match_id'] = 'match_id'
                                                
                                                    
                                                # Only continue if we have the necessary columns
                                                if len(epsf_cols) > 1 and 'match_id' in epsf_df.columns and 'match_id' in final_table.columns:
                                                    # Select columns that actually exist
                                                    epsf_subset = epsf_df[[col for col in epsf_cols.keys() if col in epsf_df.columns]].rename(columns=epsf_cols)
                                                    
                                                    # Merge the EPSF results with the final table
                                                    final_table = pd.merge(final_table, epsf_subset, on='match_id', how='left')
                                                    
                                                    # Calculate calibrated magnitudes for EPSF photometry
                                                    if 'epsf_instrumental_mag' in final_table.columns:
                                                        final_table['psf_calib_mag'] = final_table['psf_instrumental_mag'] + zero_point_value + 0.1*air
                                                        st.success("Added EPSF photometry results to the catalog")
                                                    
                                                    # Ensure we have aperture photometry in the final table too
                                                    if 'instrumental_mag' in final_table.columns:
                                                        if 'calib_mag' not in final_table.columns:
                                                            final_table['aperture_instrumental_mag'] = final_table['instrumental_mag']
                                                            final_table['aperture_calib_mag'] = final_table['instrumental_mag'] + zero_point_value + 0.1*air
                                                        else:
                                                            final_table = final_table.rename(columns={
                                                                'instrumental_mag': 'aperture_instrumental_mag',
                                                                'calib_mag': 'aperture_calib_mag'
                                                            })
                                                        st.success("Added aperture photometry results to the catalog")
                                                    
                                                    # Remove temporary match column
                                                    final_table.drop('match_id', axis=1, inplace=True)
                                            
                                            # Create a StringIO buffer for CSV data
                                            csv_buffer = StringIO()
                                            
                                            # Check and remove problematic columns if they exist
                                            cols_to_drop = []
                                            for col_name in ['sky_center.ra', 'sky_center.dec']:
                                                if col_name in final_table.columns:
                                                    cols_to_drop.append(col_name)
                                            
                                            if cols_to_drop:
                                                final_table = final_table.drop(columns=cols_to_drop)
                                            
                                            # Before writing to CSV, ensure match_id column is removed
                                            if 'match_id' in final_table.columns:
                                                final_table.drop('match_id', axis=1, inplace=True)
                                            
                                            # Create a catalog summary in the UI
                                            st.subheader("Final Photometry Catalog")
                                            st.dataframe(final_table.head(10))
                                            
                                            # Show which columns are included
                                            st.success(f"Catalog includes {len(final_table)} sources with columns: {', '.join(final_table.columns.tolist())}")
                                            
                                            # Write the filtered DataFrame to the buffer
                                            final_table.to_csv(csv_buffer, index=False)
                                            csv_data = csv_buffer.getvalue()
                                            
                                            # Ensure filename has .csv extension
                                            filename = catalog_name if catalog_name.endswith('.csv') else f"{catalog_name}.csv"
                                            
                                            # Provide download button with better formatting
                                            st.download_button(
                                                label="📥 Download Photometry Catalog",
                                                data=csv_data,
                                                file_name=filename,
                                                mime='text/csv',
                                                help="Download the photometry results as a CSV file"
                                            )
                                            
                                            # Also save locally if needed
                                            with open(filename, 'w') as f:
                                                f.write(csv_data)
                                            # st.success(f"Catalog saved to {filename}")
                                            
                                        except Exception as e:
                                            st.error(f"Error preparing download: {e}")
                        else:
                            st.error("Failed to cross-match with Gaia catalog. Check WCS information in image header.")
                except Exception as e:
                    st.error(f"Error during zero point calibration: {str(e)}")
                    st.exception(e)  # This will show the full traceback for debugging

                # Display PanSTARRS color view with detected sources
                st.subheader("PanSTARRS Color View")

                # Extract RA/DEC from header or WCS
                ra_center = None
                dec_center = None

                if 'CRVAL1' in header_to_process and 'CRVAL2' in header_to_process:
                    ra_center = header_to_process['CRVAL1']
                    dec_center = header_to_process['CRVAL2']
                elif 'RA' in header_to_process and 'DEC' in header_to_process:
                    ra_center = header_to_process['RA']
                    dec_center = header_to_process['DEC']
                elif 'OBJRA' in header_to_process and 'OBJDEC' in header_to_process:
                    ra_center = header_to_process['OBJRA']
                    dec_center = header_to_process['OBJDEC']

                if ra_center is not None and dec_center is not None:
                    st.write(f"Creating Aladin view centered at RA={ra_center}, DEC={dec_center}")
    
                    # Create a direct URL to Aladin Lite with pre-configured parameters
                    aladin_url = f"https://aladin.u-strasbg.fr/AladinLite/?target={ra_center}%20{dec_center}&fov=0.3&survey=P/PanSTARRS/DR1/color"
                    
                    # Create an iframe to embed Aladin Lite
                    iframe_html = f"""
                    <iframe 
                        src="{aladin_url}" 
                        width="100%" 
                        height="550px" 
                        style="border: 1px solid #ddd; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);"
                        allowfullscreen>
                    </iframe>
                    """
                    
                    # Display the iframe
                    st.markdown(iframe_html, unsafe_allow_html=True)
                    st.info("PanSTARRS DR1 color image centered on target coordinates. Use Aladin controls to overlay catalogs.")
                    
                    # Add instructions for manual catalog overlay
                    with st.expander("How to overlay Gaia DR3 catalog"):
                        st.markdown("""
                        1. Click the "layers" icon in the upper right of the Aladin window
                        2. Select "Catalog → VizieR"
                        3. Search for "I/355" (Gaia DR3)
                        4. Click on "I/355/gaiadr3" to add the Gaia DR3 catalog
                        """)
                    
                    # Add ESA Sky button with target coordinates
                    st.link_button(
                        "ESA Sky", 
                        f"https://sky.esa.int/esasky/?target={ra_center}%20{dec_center}&hips=PanSTARRS+DR1+color+(i%2C+r%2C+g)&fov=1&projection=SIN&cooframe=J2000&sci=true&lang=en",
                        help="Open ESA Sky with the same target coordinates"
                    )
                else:
                    st.warning("Could not determine coordinates from image header. Cannot display PanSTARRS view.")
else:
    st.write("👆 Please upload a science image FITS file to start.")