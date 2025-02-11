import streamlit as st
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, SigmaClip
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
from astroquery.gaia import Gaia
from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.background import Background2D, SExtractorBackground
import matplotlib.pyplot as plt
import pandas as pd
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.wcs import WCS

from astrometry_lite import astrometry_script

from astropy.time import Time
from astropy.coordinates import get_sun
from typing import Union, Optional, Dict, Tuple
from astropy.modeling import models, fitting


st.title("Image Calibration and Zero Point", anchor="center")

# Initialize session state to store calibrated data
if 'calibrated_data' not in st.session_state:
    st.session_state['calibrated_data'] = None
if 'calibrated_header' not in st.session_state:
    st.session_state['calibrated_header'] = None
if 'final_phot_table' not in st.session_state:
    st.session_state['final_phot_table'] = None # To store the final phot_table

# ------------------------------------------------------------------------------
# Configuration - Streamlit UI Inputs
# ------------------------------------------------------------------------------

st.sidebar.header("File Uploads")
bias_file = st.sidebar.file_uploader("Upload Bias Frame (bias.fits)", type=['fits'])
dark_file = st.sidebar.file_uploader("Upload Dark Frame (dark.fits)", type=['fits'])
flat_file = st.sidebar.file_uploader("Upload Flat Frame (flat.fits)", type=['fits'])
science_file = st.sidebar.file_uploader("Upload Science Image (science.fits)", type=['fits'], key="science_upload")

st.sidebar.header("Calibration Options")
calibrate_bias = st.sidebar.checkbox("Apply Master Bias", value=True)
calibrate_dark = st.sidebar.checkbox("Apply Master Dark", value=True)
calibrate_flat = st.sidebar.checkbox("Apply Master Flat", value=True)

st.sidebar.header("Source Detection & Photometry")
seeing = st.sidebar.number_input("Estimated Seeing (arcsec)", value=3.5)
threshold_sigma = st.sidebar.number_input("Detection Threshold (sigma)", value=3.0)
detection_mask = st.sidebar.number_input("Border Mask (pixels)", value=100)
catalog_name = st.sidebar.text_input("Catalog Name", value="sources_cat.csv")

st.sidebar.header("Gaia DR3 Query")
gaia_band = st.sidebar.selectbox("Gaia Band", ['phot_g_mean_mag', 'phot_bp_mean_mag', 'phot_rp_mean_mag'], index=0)
gaia_max_mag = st.sidebar.number_input("Gaia Max Magnitude (filtering)", value=19.0)
gaia_min_mag = st.sidebar.number_input("Gaia Min Magnitude (filtering)", value=12.0)

# ------------------------------------------------------------------------------
# Streamlit Caching Functions
# ------------------------------------------------------------------------------

@st.cache_data
def load_fits_data(file):
    if file is not None:
        with fits.open(file) as hdu:
            return hdu[0].data, hdu[0].header
    return None, None

def calibrate_image_streamlit(science_data, science_header, bias_data, dark_data, flat_data,
                            exposure_time_science, dark_data_header,
                            apply_bias, apply_dark, apply_flat):
    """Calibrates a science image using bias, dark, and flat frames based on user selections."""

    if not apply_bias and not apply_dark and not apply_flat:
        st.info("Calibration steps are disabled. Returning raw science data.")
        return science_data, science_header  # Return original science data if no calibration

    calibrated_science = science_data.copy() # Operate on a copy to avoid modifying original loaded data
    steps_applied = []

    if apply_bias and bias_data is not None:
        st.write("Applying bias subtraction...")
        calibrated_science -= bias_data
        steps_applied.append("Bias Subtraction")
        if dark_data is not None: # Bias correct dark and flat only if bias is applied.
            dark_data_corrected = dark_data - bias_data
        if flat_data is not None:
            flat_data_corrected = flat_data - bias_data

    else:
        dark_data_corrected = dark_data # if bias is not applied, use raw dark and flat
        flat_data_corrected = flat_data

    if apply_dark and dark_data_corrected is not None:
        st.write("Applying dark subtraction...")
        # Scale dark frame if exposure times are different
        if exposure_time_science != dark_data_header['EXPTIME']:
            dark_scale_factor = exposure_time_science / dark_data_header['EXPTIME']
            scaled_dark = dark_data_corrected * dark_scale_factor
        else:
            scaled_dark = dark_data_corrected
        calibrated_science -= scaled_dark
        steps_applied.append("Dark Subtraction")

    if apply_flat and flat_data_corrected is not None:
        st.write("Applying flat field correction...")
        # Normalize the flat field (divide by its median)
        normalized_flat = flat_data_corrected / np.median(flat_data_corrected)
        calibrated_science /= normalized_flat
        steps_applied.append("Flat Field Correction")

    if not steps_applied:
        st.info("No calibration steps were applied as files are missing or options disabled.")
        return science_data, science_header # return original if no calibration done due to missing files.

    st.success(f"Calibration steps applied: {', '.join(steps_applied)}")
    return calibrated_science, science_header

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
        st.info("Observation details:")
        st.info(f"Date & Local Time: {obstime.iso}")
        st.info(f"Target position: RA={details['target_coords']['ra']}, "
              f"DEC={details['target_coords']['dec']}")
        st.info(f"Altitude: {details['altaz']['altitude']}°, "
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
def make_border_mask(
    image: np.ndarray,
    border: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 50,
    invert: bool = True,
    dtype: np.dtype = bool
) -> np.ndarray:
    """
    Crée un masque binaire pour une image en excluant une ou plusieurs bordures.

    Parameters
    ----------
    image : np.ndarray
        Image source pour laquelle le masque doit être créé
    border : int or tuple, optional
        Peut être :
        - int : même largeur pour toutes les bordures
        - tuple[int, int] : (vertical, horizontal)
        - tuple[int, int, int, int] : (haut, bas, gauche, droite)
        (default: 50)
    invert : bool, optional
        Si True, inverse le masque (bordure = True, centre = False)
        Si False, bordure = False, centre = True
        (default: True)
    dtype : np.dtype, optional
        Type de données du masque de sortie (default: bool)

    Returns
    -------
    np.ndarray
        Masque binaire de la même taille que l'image

    Raises
    ------
    ValueError
        Si les dimensions de la bordure sont invalides ou trop grandes
    TypeError
        Si les types d'entrée sont incorrects

    Examples
    --------
    >>> # Masque avec bordure uniforme de 50 pixels
    >>> mask = make_border_mask(image, 50)
    
    >>> # Masque avec bordures différentes (haut, bas, gauche, droite)
    >>> mask = make_border_mask(image, (100, 50, 75, 75))
    
    >>> # Masque non inversé (True au centre)
    >>> mask = make_border_mask(image, 50, invert=False)
    """
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
    mask: np.ndarray = None,
    std_lo: float = 1.,
    std_hi: float = 1.
) -> float:
    """
    Calcule la largeur à mi-hauteur (FWHM) d'une image en utilisant la méthode des sommes marginales
    et l'ajustement d'un modèle gaussien 1D aux sources détectées.

    Cette fonction filtre d'abord les sources en fonction de leur flux, en gardant celles dont le flux est
    situé dans une plage définie par les paramètres `std_lo` et `std_hi` autour de la médiane. Pour chaque
    source filtrée, une sous-image est extraite, les sommes marginales sont calculées et un ajustement
    d'un modèle gaussien 1D est réalisé sur ces sommes marginales pour estimer la FWHM en pixels.

    Retourne
    --------
    float
        FWHM médiane estimée en pixels, basée sur l'ajustement du modèle gaussien aux sommes marginales
        des sources filtrées.
    """

    def compute_fwhm_marginal_sums(center_row, center_col, box_size):
        """
        Computes the FWHM of a star using marginal sums and Gaussian fitting.

        Args:
            image_data (numpy.ndarray): 2D image data (sub-image around the star).
            center_row (int): Row index of the star's center within the *full* image.
            center_col (int): Column index of the star's center within the *full* image.
            box_size (int): Size of the NxN box around the center.

        Returns:
            tuple: (FWHM_row, FWHM_col, center_row_fit, center_col_fit) or None if error.
                   Returns None if the box extends beyond image boundaries or fitting fails.
        """
        half_box = box_size // 2

        # Check if box is within image boundaries of the FULL IMAGE
        row_start = center_row - half_box
        row_end = center_row + half_box + 1
        col_start = center_col - half_box
        col_end = center_col + half_box + 1

        if row_start < 0 or row_end > img.shape[0] or col_start < 0 or col_end > img.shape[1]: # Use 'img' (full image) shape
            return None  # Box extends beyond image

        # Extract the box region from the FULL IMAGE
        box_data = img[row_start:row_end, col_start:col_end] # Use 'img' (full image) to extract box

        # Compute marginal sums
        sum_rows = np.sum(box_data, axis=1)
        sum_cols = np.sum(box_data, axis=0)

        # Create x-axis data for fitting (just indices)
        row_indices = np.arange(box_size)
        col_indices = np.arange(box_size)

        # Fit Gaussians
        fitter = fitting.LevMarLSQFitter()  # Or other fitter

        # Row fit
        try:
            model_row = models.Gaussian1D()
            model_row.amplitude.value = np.max(sum_rows)
            model_row.mean.value = half_box
            model_row.stddev.value = half_box/3
            fitted_row = fitter(model_row, row_indices, sum_rows) # No keyword args here
            center_row_fit = fitted_row.mean.value + row_start  # Convert back to full image coordinates
            fwhm_row = 2 * np.sqrt(2 * np.log(2)) * fitted_row.stddev.value * pixel_scale # Apply pixel_scale here
        except Exception as e:
            st.error(f"Error fitting row marginal sum: {e}")
            return None

        # Column fit
        try:
            model_col = models.Gaussian1D()
            model_col.amplitude.value = np.max(sum_cols)
            model_col.mean.value = half_box
            model_col.stddev.value = half_box/3
            fitted_col = fitter(model_col, col_indices, sum_cols)
            center_col_fit = fitted_col.mean.value + col_start  # Convert back to full image coordinates
            fwhm_col = 2 * np.sqrt(2 * np.log(2)) * fitted_col.stddev.value * pixel_scale # Apply pixel_scale here
        except Exception as e:
            st.error(f"Error fitting column marginal sum: {e}")
            return None

        return fwhm_row, fwhm_col, center_row_fit, center_col_fit


    try:
        daofind = DAOStarFinder(fwhm=1.*fwhm, threshold=5 * np.std(img))
        sources = daofind(img, mask=mask)
        if sources is None:
            st.warning("No sources found by DAOStarFinder!")
            return None, daofind

        st.write(f"Number of sources found by DAOStarFinder: {len(sources)}") # Debugging output

        # Filtrage des sources en fonction du flux
        flux = sources['flux']
        median_flux = np.median(flux)
        std_flux = np.std(flux)
        mask_flux = ((flux > median_flux - std_lo * std_flux) & (flux < median_flux + std_hi * std_flux) & 
                    (sources['roundness1'] > -0.5) & (sources['roundness1'] < 0.5))
        filtered_sources = sources[mask_flux]

        # Suppression des sources contenant des valeurs NaN dans le flux
        filtered_sources = filtered_sources[~np.isnan(filtered_sources['flux'])]
        
        st.write(f"Number of sources after flux filtering: {len(filtered_sources)}") # Debugging output

        if len(filtered_sources) == 0: # Check again after NaN filtering
            msg = "Aucune source valide pour l'ajustement a été trouvée après filtrage." # More informative message
            st.error(msg)
            raise ValueError(msg)

        # Définition du rayon d'analyse (en pixels) pour la *boîte* (carrée NxN)
        box_size = int(6 * round(fwhm)) # Box size is ~ 2 * analysis_radius from previous method, make it larger for marginal sums maybe? Make box_size an even number for // 2
        if box_size % 2 == 0: # Ensure box_size is odd
            box_size += 1

        fwhm_values = []

        # Ajustement du modèle pour chaque source filtrée
        for source in filtered_sources:
            try:
                x_cen = int(source['xcentroid']) # Use integer center for box extraction
                y_cen = int(source['ycentroid']) # Use integer center for box extraction

                # Calcul de la FWHM en utilisant les sommes marginales
                fwhm_results = compute_fwhm_marginal_sums(y_cen, x_cen, box_size) # Pass full image 'img', integer y_cen, x_cen
                if fwhm_results is None:
                    st.warning(f"FWHM computation using marginal sums failed for source at ({x_cen}, {y_cen}). Skipping.")
                    continue

                fwhm_row, fwhm_col, _, _ = fwhm_results # Get FWHM_row and FWHM_col, ignore fitted centers

                # Using the average of FWHM_row and FWHM_col as the FWHM estimate for this source
                fwhm_source = np.mean([fwhm_row, fwhm_col])
                fwhm_values.append(fwhm_source)


            except Exception as e:
                st.error(f"Erreur lors du calcul de la FWHM pour la source aux coordonnées "
                         f"({x_cen}, {y_cen}): {e}") # More detailed error output with coordinates
                st.write(f"FWHM computation error details: {e}") # Even more details
                continue

        if len(fwhm_values) == 0: # Check one last time, after the loop
            msg = "Aucune source valide pour l'ajustement de la FWHM après la boucle de fitting (sommes marginales)." # More informative message
            st.error(msg)
            raise ValueError(msg)

        # Conversion de la liste en array pour filtrer les NaN et valeurs infinies
        fwhm_values_arr = np.array(fwhm_values)
        # Filtrer les valeurs non numériques ou infinies
        valid = ~np.isnan(fwhm_values_arr) & ~np.isinf(fwhm_values_arr)
        if not np.any(valid):
            msg = "Toutes les valeurs de FWHM sont NaN ou infinies après le calcul par sommes marginales." # More informative message
            st.error(msg)
            raise ValueError(msg)

        mean_fwhm = np.median(fwhm_values_arr[valid])
        st.info(f"Estimation médiane de la FWHM basée sur les sommes marginales et modèle Gaussien: {round(mean_fwhm, 2)} pixels")

        return round(mean_fwhm, 2)
    except ValueError as e: # Catch ValueError from DAOStarFinder if no sources are found initially
        raise e
    except Exception as e: # Catch any other exceptions during the process
        st.error(f"An unexpected error occurred in fwhm_fit: {e}")
        raise ValueError(f"Erreur inattendue dans fwhm_fit: {e}")

@st.cache_data
def find_sources_and_photometry_streamlit(image_data, _science_header, mean_fwhm_pixel, threshold_sigma, detection_mask):
    """Finds sources and performs photometry using Streamlit caching."""

    # Estimate background and noise
    sigma_clip = SigmaClip(sigma=3)
    bkg_estimator = SExtractorBackground()
    bkg = Background2D(data=image_data,
                       box_size=100,
                       filter_size=5,
                       mask=None,
                       sigma_clip=sigma_clip,
                       bkg_estimator=bkg_estimator,
                       exclude_percentile=25.0
        )
    
    mask = make_border_mask(image_data, border=detection_mask)
    
    total_error = np.sqrt(bkg.background_rms**2 + bkg.background_median)
    
    st.info("Estimating FWHM...")
    fwhm_estimate = fwhm_fit(image_data - bkg.background, mean_fwhm_pixel, pixel_size_arcsec, mask)
    
    # Source detection using DAOStarFinder
    daofind = DAOStarFinder(fwhm=2.5*fwhm_estimate, threshold=threshold_sigma * np.std(image_data - bkg.background))
    sources = daofind(image_data - bkg.background, mask=mask)
    if sources is None:
        st.warning("No sources found!")
        return None, daofind

    # Aperture photometry
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=2.5*fwhm_estimate)
    apers = [apertures]

    phot_table = aperture_photometry(image_data - bkg.background, apers, 
                                     wcs=WCS(_science_header), error=total_error)

    # Background subtraction using annulus
    final_flux = phot_table['aperture_sum_0']

    # Calculate instrumental magnitudes
    instrumental_mags = -2.5 * np.log10(final_flux)
    phot_table['instrumental_mag'] = instrumental_mags

    # Keep only valid sources
    valid_sources = (phot_table['aperture_sum_0'] > 0) & np.isfinite(phot_table['instrumental_mag'])
    phot_table = phot_table[valid_sources]

    # Add RA and Dec if WCS is available
    try:
        w = WCS(_science_header) # use _science_header to avoid hashing issues
        ra, dec = w.pixel_to_world_values(phot_table['xcenter'], phot_table['ycenter'])
        phot_table['ra'] = ra * u.deg
        phot_table['dec'] = dec * u.deg
    except Exception as e:
        st.warning(f"WCS transformation failed: {e}. RA and Dec not added to phot_table.")

    st.success(f"Found {len(phot_table)} sources and performed photometry.")
    return phot_table, daofind

@st.cache_data
def cross_match_with_gaia_streamlit(_phot_table, _science_header, pixel_size, gaia_band, gaia_min_mag, gaia_max_mag):
    """Cross-matches sources with Gaia using Streamlit caching.
    Using _phot_table and _science_header to avoid Streamlit hashing/serialization errors."""
    st.info("Cross-matching with Gaia DR3...")
    phot_table = _phot_table # Rename back to phot_table for readability within the function
    science_header = _science_header # Rename back to science_header for readability

    if science_header is None:
        st.warning("No WCS information in header. Cannot cross-match with Gaia.")
        return None

    try:
        w = WCS(science_header)
    except ImportError:
        st.error("astropy.wcs is not installed. Cannot cross-match with Gaia.")
        return None

    # Convert pixel positions to sky coordinates
    source_positions_pixel = np.transpose((phot_table['xcenter'], phot_table['ycenter']))
    source_positions_sky = w.pixel_to_world(source_positions_pixel[:,0], source_positions_pixel[:,1])

    # Query Gaia
    image_center_ra_dec = w.pixel_to_world(science_header['NAXIS1']//2, science_header['NAXIS2']//2)
    gaia_search_radius_arcsec = science_header['NAXIS1']*pixel_size/2.0 # Use half the image diagonal as search radius
    radius_query = gaia_search_radius_arcsec * u.arcsec

    try:
        st.info(f"Querying Gaia DR3...in a radius of {round(radius_query/60.,2)} arcmin.")
        job = Gaia.cone_search(image_center_ra_dec, radius=radius_query)
        gaia_table = job.get_results()
    except Exception as e:
        st.error(f"Error querying Gaia: {e}")
        return None

    if gaia_table is None or len(gaia_table) == 0:
        st.warning("No Gaia sources found within search radius.")
        return None

    # Apply magnitude filter to Gaia catalog BEFORE cross-matching
    gaia_table_filtered = gaia_table[
        (gaia_table[gaia_band] < gaia_max_mag) & (gaia_table[gaia_band] > gaia_min_mag)
    ]
    gaia_table_filtered = gaia_table[gaia_table["phot_variable_flag"] != "VARIABLE"]
    
    if len(gaia_table_filtered) == 0:
        st.warning(f"No Gaia sources found within magnitude range {gaia_min_mag} < {gaia_band} < {gaia_max_mag}.")
        return None
    else:
        st.info(f"Filtered Gaia catalog to {len(gaia_table_filtered)} sources within magnitude range.")
        gaia_table = gaia_table_filtered # Use filtered table for cross-matching

    # Cross-match
    gaia_skycoords = SkyCoord(ra=gaia_table['ra'], dec=gaia_table['dec'], unit='deg')
    idx, d2d, _ = source_positions_sky.match_to_catalog_sky(gaia_skycoords)
    max_sep_constraint = 2*mean_fwhm_pixel*pixel_size_arcsec * u.arcsec
    gaia_matches = (d2d <= max_sep_constraint)

    matched_indices_gaia = idx[gaia_matches]
    matched_indices_phot = np.where(gaia_matches)[0]

    if len(matched_indices_gaia) == 0:
        st.warning("No Gaia matches found within the search radius and magnitude filter.")
        return None

    matched_table_qtable = phot_table[matched_indices_phot] # Keep QTable for calculations

    # Convert QTable to Pandas DataFrame using to_pandas() method
    matched_table = matched_table_qtable.to_pandas()
    matched_table['gaia_index'] = matched_indices_gaia
    matched_table['gaia_separation_arcsec'] = d2d[gaia_matches].arcsec
    matched_table[gaia_band] = gaia_table[gaia_band][matched_indices_gaia]

    valid_gaia_mags = np.isfinite(matched_table[gaia_band])
    matched_table = matched_table[valid_gaia_mags]

    st.success(f"Found {len(matched_table)} Gaia matches after filtering.")
    return matched_table

@st.cache_data
def calculate_zero_point_streamlit(_phot_table, matched_table, gaia_band, air):
    """Calculates zero point and plots using Streamlit caching.
    Using _phot_table to avoid Streamlit hashing error."""
    st.info("Calculating Zero Point...")
    phot_table = _phot_table # Rename back to phot_table for readability


    if matched_table is None or len(matched_table) == 0:
        st.warning("No matched sources to calculate zero point.")
        return None, None, None

    zero_points = matched_table[gaia_band] - matched_table['instrumental_mag']
    matched_table['zero_point'] = zero_points
    matched_table['zero_point_error'] = np.std(zero_points)

    clipped_zero_points = sigma_clip(zero_points, sigma=3)
    zero_point_value = np.mean(clipped_zero_points)
    zero_point_std = np.std(clipped_zero_points)

    matched_table['calib_mag'] = matched_table['instrumental_mag'] + zero_point_value + 0.1*air

    # Convert phot_table to DataFrame if it's not already (for robust handling, though it should be QTable initially)
    if not isinstance(phot_table, pd.DataFrame):
        phot_table = phot_table.to_pandas()

    phot_table['calib_mag'] = phot_table['instrumental_mag'] + zero_point_value + 0.1*air
    phot_table['calib_mag_err'] = (2.5/np.log(10) * phot_table['aperture_sum_err_0'] / phot_table['aperture_sum_0']) * header_to_process['CVF']/10

    st.session_state['final_phot_table'] = phot_table # Store final phot_table in session state

    fig, ax = plt.subplots()
    ax.scatter(matched_table[gaia_band], matched_table['calib_mag'], alpha=0.75, label='Matched sources')
    ax.axhline(y=zero_point_value, color='blue', linestyle='dashed', linewidth=1, label=f'zero_point = {zero_point_value:.3f}')
    ax.set_xlabel("Calibrated mag")
    ax.set_ylabel(f"Gaia {gaia_band}")
    ax.set_title("Calibrated mag vs. Gaia mag") 
    ax.legend()
    ax.grid(True, alpha=0.5)
    st.success(f"Calculated Zero Point ({gaia_band}): {zero_point_value:.3f} +/- {zero_point_std:.3f}")
    plt.savefig("zero_point_plot.png")
    st.info("Zero Point plot saved as 'zero_point_plot.png'.")
    return zero_point_value, zero_point_std, fig

# ------------------------------------------------------------------------------
# Main Script Execution
# ------------------------------------------------------------------------------

if science_file is not None:
    science_data, science_header = load_fits_data(science_file)
    bias_data, _ = load_fits_data(bias_file)
    dark_data, dark_header = load_fits_data(dark_file)
    flat_data, _ = load_fits_data(flat_file)

    st.header("Science Image", anchor="center")

    norm = ImageNormalize(science_data, interval=ZScaleInterval())
    fig_preview, ax_preview = plt.subplots()
    im = ax_preview.imshow(science_data, norm=norm, origin='lower', cmap="viridis")
    fig_preview.colorbar(im, ax=ax_preview, label='pixel value')
    ax_preview.set_title("zscale_stretch")
    ax_preview.axis('off') 
    st.pyplot(fig_preview, clear_figure=True)

    with st.expander("Science Image Header"):
        if science_header:
            st.text(repr(science_header))
        else:
            st.warning("No header information available for science image.")

    st.subheader("Science Image Statistics", anchor="center")
    if science_data is not None:
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        stats_col1.metric("Mean", f"{np.mean(science_data):.2f}")
        stats_col2.metric("Median", f"{np.median(science_data):.2f}")
        stats_col3.metric("Std Dev", f"{np.std(science_data):.2f}")

        pixel_size_arcsec = None
        try:
            if 'PIXSIZE' in science_header:
                pixel_size_arcsec = science_header['PIXSIZE']  # Assuming PIXSIZE in arcsec
            elif 'PIXELSCAL' in science_header:
                pixel_size_arcsec = science_header['PIXELSCAL'] # or PIXELSCAL
            elif 'CDELT2' in science_header: # CDELT might be in degrees, convert to arcsec if necessary
                pixel_size_arcsec = abs(science_header['CDELT2']) * 3600.0 # Assuming CDELT2 in degrees
            elif 'CDELT1' in science_header: # Check CDELT1 if CDELT2 is missing
                pixel_size_arcsec = abs(science_header['CDELT1']) * 3600.0

            if pixel_size_arcsec:
                st.metric("Pixel Size (arcsec)", f"{pixel_size_arcsec:.2f}")
                mean_fwhm_pixel = seeing / pixel_size_arcsec
                st.metric("Est. Mean FWHM (pixels)", f"{mean_fwhm_pixel:.2f} (seeing estimation)")
            else:
                st.warning("Pixel size information not found in header. Cannot estimate the FWHM.")

        except Exception as e:
            st.warning(f"Error reading pixel size from header: {e}")
        
        air = airmass(science_header) # Calculate airmass using the header
        st.info(f"Airmass: {air:.2f}")

        calibration_disabled = not (calibrate_bias or calibrate_dark or calibrate_flat)
        if st.button("Run Image Calibration", disabled=calibration_disabled):
            if science_data is not None:
                with st.spinner("Calibrating image..."):
                    st.session_state['calibrated_data'], st.session_state['calibrated_header'] = calibrate_image_streamlit(
                        science_data, science_header, bias_data, dark_data, flat_data,
                        science_header['EXPOSURE'], dark_header,
                        calibrate_bias, calibrate_dark, calibrate_flat
                    )
                
            if st.session_state['calibrated_data'] is not None:
                st.header("Calibrated Science Image")
                norm_calibrated = ImageNormalize(st.session_state['calibrated_data'], interval=ZScaleInterval())
                fig_calibrated_preview, ax_calibrated_preview = plt.subplots()
                im_calibrated = ax_calibrated_preview.imshow(st.session_state['calibrated_data'], norm=norm_calibrated,
                                                             origin='lower', cmap="viridis")
                fig_calibrated_preview.colorbar(im_calibrated, ax=ax_calibrated_preview, label='pixel value')
                ax_calibrated_preview.set_title("zscale_stretch)")
                ax_calibrated_preview.axis('off')
                st.pyplot(fig_calibrated_preview)


    zero_point_button_disabled = science_file is None # Disable if no science image is uploaded yet
    if st.button("Run Zero Point Calibration", disabled=zero_point_button_disabled):
        image_to_process = science_data # Default to raw science data if calibration was not run or disabled
        header_to_process = science_header

        if st.session_state['calibrated_data'] is not None: # Use calibrated data if calibration was run
            image_to_process = st.session_state['calibrated_data']
            header_to_process = st.session_state['calibrated_header']
        
        st.info("Doing astrometry refinement with GAIA DR3...")
        wcs = astrometry_script(image_to_process, header_to_process, catalog="GAIA", FWHM=mean_fwhm_pixel)
        # header_to_process.update(wcs.to_header())

        if image_to_process is not None:
            with st.spinner("Background Extraction, Find Sources and Perform Photometry..."):
                phot_table_qtable, _ = find_sources_and_photometry_streamlit( # Use QTable for caching
                    image_to_process, header_to_process, mean_fwhm_pixel, threshold_sigma, detection_mask
                )
                phot_table_df = phot_table_qtable.to_pandas().copy(deep=True) # Convert phot_table to DataFrame and create deep copy

            if phot_table_df is not None: # Use DataFrame for subsequent steps
                with st.spinner("Cross-matching with Gaia..."):
                    matched_table = cross_match_with_gaia_streamlit(
                        phot_table_qtable, header_to_process, pixel_size_arcsec, gaia_band, gaia_min_mag, gaia_max_mag
                    )

                if matched_table is not None:
                    st.subheader("Cross-matched Gaia Catalog (head)")
                    st.dataframe(matched_table.head(10))

                    with st.spinner("Calculating zero point..."):
                        zero_point_value, zero_point_std, zp_plot = calculate_zero_point_streamlit(
                            phot_table_df, matched_table, gaia_band, air
                        )
                        if zero_point_value is not None:
                            st.pyplot(zp_plot)
                    
                            st.session_state['final_phot_table'].drop(columns=['sky_center.ra', 'sky_center.dec']).to_csv(catalog_name, index=False) # Save to file
                            
                            st.link_button("Open Aladin Lite",  "https://aladin.cds.unistra.fr/AladinLite/")
                            st.write("After Aladin Lite opens in a new tab, you can manually upload the photometry_table.csv file into Aladin Lite for visualization.")
                            
                            if st.button("Open in DS9 [NOT WORKING]"): # Add Open in DS9 Button here
                                if science_file and st.session_state['final_phot_table'] is not None:
                                    st.info("Opening DS9...")
                                    try:
                                        ds9_command_list = [
                                            "ds9",
                                            science_file.name, # Use filename as DS9 argument
                                            "-zscale",
                                            "-frame", "fit",
                                            "-wcs", "load",
                                            "-catalog", "csv", # Open CSV catalog
                                        ]

                                    except Exception as e:
                                        st.error(f"Error opening DS9: {e}")
                                else:
                                    st.warning("Science image and/or photometry table not available. Run Zero Point Calibration first and upload science image.")
        else:
            st.error("Please upload a science image and run calibration (or upload science image to skip calibration) to proceed with zero point calculation.")

else:
    st.info("Please upload a science image to start.")

# ------------------------------------------------------------------------------
# Exit Application Button
# ------------------------------------------------------------------------------

# exit_app = st.button("Shut Down")
# if exit_app:
#     # Give a bit of delay for user experience
#     time.sleep(1)
#     # Close streamlit browser tab
#     keyboard.press_and_release('ctrl+w')
#     # Terminate streamlit python process
#     pid = os.getpid()
#     p = psutil.Process(pid)
#     p.terminate()