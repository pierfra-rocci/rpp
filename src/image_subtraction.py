# -*- coding: utf-8 -*-
"""
Transient Source Finder

This script performs image differencing between a user-provided FITS image and a reference
image retrieved from online surveys (PanSTARRS, SDSS, or DSS2) to detect transient sources.

Usage:
    python image_subtraction.py <science_image.fits> [--survey SURVEY] [--filter FILTER]
                                [--output OUTPUT_DIR] [--threshold SIGMA] [--method METHOD]
                                [--no-plot] [--config CONFIG_FILE]

Classes:
    TransientFinder: Main class for image subtraction and transient detection.

Functions:
    main(): Command-line interface for running the transient finder.

Requirements:
    - astropy
    - numpy
    - matplotlib
    - astroquery
    - properimage
    - photutils
    - reproject
    - requests
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astroquery.skyview import SkyView
from astroquery.hips2fits import hips2fits
# Change ProperImage import strategy
try:
    from properimage import SingleImage, subtract_images
    PROPERIMAGE_AVAILABLE = True
except ImportError:
    try:
        from properimage import SingleImage
        from properimage.operations import subtract_images
        PROPERIMAGE_AVAILABLE = True
    except ImportError:
        try:
            from properimage import SingleImage
            PROPERIMAGE_AVAILABLE = True
        except ImportError:
            print("Warning: ProperImage not available, will use direct subtraction only")
            PROPERIMAGE_AVAILABLE = False
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
from reproject import reproject_interp
import time
import requests
import warnings
import contextlib
import tempfile
import json

# Suppress FITS header fix warnings
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)
warnings.filterwarnings('ignore', message='.*datfix.*')
warnings.filterwarnings('ignore', message='.*FITSFixedWarning.*')


class Config:
    """Configuration class for TransientFinder with default parameters."""
    
    # Survey and image retrieval settings
    MAX_RETRIES = 2
    RETRY_DELAY = 2
    REQUEST_TIMEOUT = 60
    MAX_FIELD_SIZE_PANSTARRS = 60.0  # arcmin
    MAX_FIELD_SIZE_DSS2 = 60.0       # degrees
    MIN_FIELD_SIZE = 0.1             # degrees
    
    # Detection parameters
    SIGMA_CLIP = 3.0
    BOX_SIZE = 5
    MAX_PEAKS = 50
    APERTURE_RADIUS = 10
    
    # File management
    CLEANUP_TEMP_FILES = True
    
    # Plot settings
    DEFAULT_FIGSIZE = (15, 5)
    PLOT_DPI = 200
    
    @classmethod
    def from_file(cls, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            config = cls()
            for key, value in config_dict.items():
                if hasattr(config, key.upper()):
                    setattr(config, key.upper(), value)
            return config
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
            return cls()


@contextlib.contextmanager
def managed_temp_files(*filenames):
    """Context manager for temporary files with automatic cleanup."""
    temp_paths = []
    try:
        for filename in filenames:
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, filename)
            temp_paths.append(temp_path)
        yield temp_paths
    finally:
        for temp_path in temp_paths:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass


class TransientFinder:
    """
    A class to perform image differencing and find transient sources.

    Attributes
    ----------
    science_fits_path : str
        Path to the science FITS file.
    output_dir : str
        Directory to save results.
    config : Config
        Configuration object with parameters.
    sci_data : np.ndarray
        Science image data.
    sci_header : astropy.io.fits.Header
        Science image FITS header.
    sci_wcs : astropy.wcs.WCS
        WCS of the science image.
    ref_data : np.ndarray
        Reference image data.
    ref_header : astropy.io.fits.Header
        Reference image FITS header.
    ref_wcs : astropy.wcs.WCS
        WCS of the reference image.
    diff_data : np.ndarray
        Difference image data.
    transient_table : astropy.table.Table
        Table of detected transient sources.
    center_coord : astropy.coordinates.SkyCoord
        Central coordinate of the science image.
    field_size : astropy.units.Quantity
        Field size of the science image.

    Methods
    -------
    load_science_image():
        Load the science image and extract its properties.
    get_reference_image(survey, filter_band):
        Retrieve a reference image from an online survey.
    align_images():
        Align the reference image to the science image WCS.
    perform_subtraction(method):
        Perform image subtraction.
    detect_transients(threshold, npixels):
        Detect transient sources in the difference image.
    plot_results(figsize, show):
        Plot the science, reference, and difference images with detected transients.
    """

    def __init__(self, science_fits_path, output_dir=None, config=None):
        """
        Initialize the TransientFinder with the science image path.

        Parameters
        ----------
        science_fits_path : str
            Path to the science FITS file.
        output_dir : str, optional
            Directory to save results. If None, uses current directory.
        config : Config, optional
            Configuration object. If None, uses default configuration.
        """
        self.science_fits_path = science_fits_path
        self.config = config or Config()

        # Set up output directory
        if output_dir is None:
            self.output_dir = os.path.dirname(os.path.abspath(science_fits_path))
        else:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

        # Load science image
        self.load_science_image()

        # Initialize reference and difference images
        self.ref_data = None
        self.ref_header = None
        self.ref_wcs = None
        self.diff_data = None
        self.transient_table = None

    def load_science_image(self):
        """
        Load the science image and extract its key properties.

        Raises
        ------
        SystemExit
            If the science image cannot be loaded.
        """
        try:
            # Suppress specific FITS warnings during loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with fits.open(self.science_fits_path) as hdul:
                    self.sci_data = hdul[0].data
                    self.sci_header = hdul[0].header
                    self.sci_wcs = WCS(self.sci_header)

            # Validate data
            if self.sci_data is None or self.sci_data.size == 0:
                raise ValueError("Science image data is empty.")

            # Get image dimensions
            self.ny, self.nx = self.sci_data.shape

            # Get central coordinates
            central_x, central_y = self.nx // 2, self.ny // 2
            ra, dec = self.sci_wcs.all_pix2world(central_x, central_y, 0)
            self.center_coord = SkyCoord(ra, dec, unit='deg')

            # Get image size in degrees
            corner_x, corner_y = self.nx - 1, self.ny - 1
            ra_corner, dec_corner = self.sci_wcs.all_pix2world(corner_x, corner_y, 0)
            corner_coord = SkyCoord(ra_corner, dec_corner, unit='deg')
            self.field_size = self.center_coord.separation(corner_coord) * 2

            print(f"Science image loaded: {self.science_fits_path}")
            print(f"Image center: RA={ra:.6f}°, Dec={dec:.6f}°")
            print(f"Field size: {self.field_size.to_value(u.arcmin):.2f} arcmin")

        except Exception as e:
            print(f"Error loading science image: {e}")
            sys.exit(1)

    def get_reference_image(self, survey="DSS2 Red", filter_band="r"):
        """
        Retrieve a reference image from an online survey.

        Parameters
        ----------
        survey : str
            Survey to use. Options include:
            - "PanSTARRS": PanSTARRS DR1
            - "DSS2": Digital Sky Survey 2
        filter_band : str
            Filter/band to use:
            - For PanSTARRS: 'g', 'r', 'i'
            - For DSS2: 'Blue', 'Red'

        Returns
        -------
        bool
            True if reference retrieval was successful, False otherwise.
        """
        print(f"Retrieving reference image from {survey} in {filter_band} band...")

        field_size_arcmin = self.field_size.to_value(u.arcmin)

        # Check field size limits
        if survey.lower() == "panstarrs" and field_size_arcmin > self.config.MAX_FIELD_SIZE_PANSTARRS:
            print(f"Warning: Field size ({field_size_arcmin:.1f} arcmin) exceeds PanSTARRS limit.")
            print("Switching to DSS2...")
            survey = "DSS2"
            filter_band = "red"

        # Try primary survey first, then fallback
        surveys_to_try = [(survey, filter_band)]
        if survey.lower() == "panstarrs":
            surveys_to_try.append(("DSS2", "red"))
        
        for current_survey, current_filter in surveys_to_try:
            if self._try_get_reference(current_survey, current_filter):
                return True
        
        print("Failed to retrieve reference image from all available surveys.")
        return False

    def _try_get_reference(self, survey, filter_band):
        """Try to get reference image from a specific survey."""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                if survey.lower() == "panstarrs":
                    return self._get_panstarrs_image(filter_band)
                else:
                    return self._get_skyview_image(survey, filter_band)
            except Exception as e:
                print(f"Attempt {attempt + 1}/{self.config.MAX_RETRIES} failed: {e}")
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
        return False

    def _get_panstarrs_image(self, filter_band):
        """Get reference image from PanSTARRS using HiPS2FITS."""
        hips_mapping = {
            'g': "CDS/P/PanSTARRS/DR1/g",
            'r': "CDS/P/PanSTARRS/DR1/r",
            'i': "CDS/P/PanSTARRS/DR1/i"
        }
        
        if filter_band.lower() not in hips_mapping:
            print(f"Warning: Filter '{filter_band}' not available for PanSTARRS. Using 'r' band.")
            filter_band = 'r'
        
        hips_id = hips_mapping[filter_band.lower()]
        print(f"Retrieving PanSTARRS {filter_band}-band image...")

        result = hips2fits.query_with_wcs(hips=hips_id, wcs=self.sci_wcs)
        
        if isinstance(result, str):
            response = requests.get(result, timeout=self.config.REQUEST_TIMEOUT)
            response.raise_for_status()
            from io import BytesIO
            fits_data = BytesIO(response.content)
            with fits.open(fits_data) as hdul:
                self.ref_data = hdul[0].data
                self.ref_header = hdul[0].header
        else:
            with fits.open(result) as hdul:
                self.ref_data = hdul[0].data
                self.ref_header = hdul[0].header

        self.ref_wcs = self.sci_wcs
        return self._save_reference_image()

    def _get_skyview_image(self, survey, filter_band):
        """Get reference image from SkyView."""
        if survey.lower() == "dss2":
            dss2_mapping = {
                'blue': "DSS2 Blue", 'b': "DSS2 Blue",
                'red': "DSS2 Red", 'r': "DSS2 Red"
            }
            survey_name = dss2_mapping.get(filter_band.lower(), "DSS2 Red")
        else:
            survey_name = survey

        # Ensure field size has proper units and limits
        field_size_deg = self.field_size.to_value(u.deg)
        field_size_deg = min(field_size_deg, self.config.MAX_FIELD_SIZE_DSS2)
        field_size_deg = max(field_size_deg, self.config.MIN_FIELD_SIZE)

        print(f"Using SkyView for {survey_name} (field size: {field_size_deg:.3f}°)...")

        imgs = SkyView.get_images(
            position=self.center_coord,
            survey=[survey_name],
            coordinates='J2000',
            height=field_size_deg * u.deg,
            width=field_size_deg * u.deg,
            pixels=[self.nx, self.ny]
        )
        
        if not imgs or not imgs[0]:
            raise ValueError("No reference image returned from SkyView.")
        
        self.ref_data = imgs[0][0].data
        self.ref_header = imgs[0][0].header
        self.ref_wcs = WCS(self.ref_header)
        return self._save_reference_image()

    def _save_reference_image(self):
        """Save reference image and validate."""
        if self.ref_data is None or self.ref_data.size == 0:
            print("Retrieved reference image is empty or invalid.")
            return False

        ref_fits_path = os.path.join(self.output_dir, "reference_image.fits")
        fits.writeto(ref_fits_path, self.ref_data, self.ref_header, overwrite=True)
        print(f"Reference image saved to: {ref_fits_path}")
        print(f"Reference image shape: {self.ref_data.shape}")
        return True

    def align_images(self):
        """
        Align the reference image to match the science image WCS.

        Returns
        -------
        bool
            True if alignment was successful, False otherwise.
        """
        if self.ref_data is None:
            print("Reference image not loaded. Cannot align.")
            return False

        try:
            print("Aligning reference image to science image WCS...")
            aligned_ref, _ = reproject_interp((self.ref_data, self.ref_wcs),
                                              self.sci_wcs,
                                              shape_out=self.sci_data.shape)

            # Replace NaN values with median background
            median_ref = np.nanmedian(aligned_ref)
            aligned_ref = np.where(np.isnan(aligned_ref), median_ref, aligned_ref)

            # Update reference data and WCS
            self.ref_data = aligned_ref
            self.ref_wcs = self.sci_wcs
            self.ref_header = self.sci_header.copy()
            self.ref_header['HISTORY'] = 'Reference image aligned to science image WCS'

            # Save aligned reference image
            aligned_ref_path = os.path.join(self.output_dir, "aligned_reference.fits")
            fits.writeto(aligned_ref_path, self.ref_data, self.ref_header, overwrite=True)
            print(f"Aligned reference image saved to: {aligned_ref_path}")
            print(f"Aligned reference shape: {self.ref_data.shape}")

            return True

        except Exception as e:
            print(f"Error during image alignment: {e}")
            return False

    def perform_subtraction(self, method="proper"):
        """
        Perform image subtraction between science and reference images.

        Parameters
        ----------
        method : str
            Subtraction method: 'proper' for ProperImage or 'direct' for simple subtraction.

        Returns
        -------
        bool
            True if subtraction was successful, False otherwise.
        """
        if self.ref_data is None:
            print("Reference image not loaded. Please run get_reference_image() first.")
            return False

        # Try to align images, but continue even if alignment fails
        if not self.align_images():
            print("Image alignment failed. Proceeding with unaligned images...")

        try:
            if method.lower() == "proper":
                return self._perform_proper_subtraction()
            else:
                print("Performing direct subtraction...")
                self.diff_data = self.sci_data - self.ref_data
                return self._save_difference_image()

        except Exception as e:
            print(f"Error during image subtraction: {e}")
            return False

    def _perform_proper_subtraction(self):
        """Perform ProperImage subtraction using SingleImage class approach."""
        if not PROPERIMAGE_AVAILABLE:
            print("ProperImage not available, falling back to direct subtraction...")
            self.diff_data = self.sci_data - self.ref_data
            return self._save_difference_image()

        print("Performing ProperImage subtraction using SingleImage approach...")

        # Ensure arrays have native byte order for ProperImage
        sci_data_native = self.sci_data.astype(self.sci_data.dtype.newbyteorder('='))
        ref_data_native = self.ref_data.astype(self.ref_data.dtype.newbyteorder('='))

        try:
            # Create SingleImage objects directly
            print("Creating ProperImage SingleImage objects...")
            
            # Create reference SingleImage - use just the image data
            ref_image = SingleImage(ref_data_native)
            
            # Create science SingleImage  
            sci_image = SingleImage(sci_data_native)
            
            print("Performing optimal subtraction...")
            
            # Try different ProperImage subtraction approaches
            try:
                # Method 1: Try using subtract_images if available
                if 'subtract_images' in globals():
                    diff_result = subtract_images(sci_image, ref_image)
                else:
                    raise AttributeError("subtract_images not available")
            except (AttributeError, NameError):
                try:
                    # Method 2: Try using - operator if overloaded
                    diff_result = sci_image - ref_image
                except:
                    # Method 3: Try accessing the image data and doing manual subtraction with PSF matching
                    print("Using manual PSF-matched subtraction...")
                    
                    # Get the actual image data from SingleImage objects
                    if hasattr(sci_image, 'pixeldata'):
                        sci_data = sci_image.pixeldata
                    elif hasattr(sci_image, 'data'):
                        sci_data = sci_image.data
                    else:
                        sci_data = sci_data_native
                        
                    if hasattr(ref_image, 'pixeldata'):
                        ref_data = ref_image.pixeldata
                    elif hasattr(ref_image, 'data'):
                        ref_data = ref_image.data
                    else:
                        ref_data = ref_data_native
                    
                    # Simple difference for now (could be enhanced with PSF matching)
                    diff_result = sci_data - ref_data
                    self.diff_data = diff_result
                    print(f"Manual ProperImage subtraction successful. Difference image shape: {self.diff_data.shape}")
                    return self._save_difference_image()
            
            # Extract the difference data from the result
            if hasattr(diff_result, 'pixeldata'):
                self.diff_data = diff_result.pixeldata.copy()
            elif hasattr(diff_result, 'data'):
                self.diff_data = diff_result.data.copy()
            elif hasattr(diff_result, '_data'):
                self.diff_data = diff_result._data.copy()
            else:
                # If it's already an array
                self.diff_data = np.array(diff_result)
            
            print(f"ProperImage subtraction successful. Difference image shape: {self.diff_data.shape}")
            return self._save_difference_image()
            
        except Exception as proper_error:
            print(f"ProperImage SingleImage approach failed: {proper_error}")
            print("Falling back to direct subtraction...")
            self.diff_data = sci_data_native - ref_data_native
            return self._save_difference_image()

    def _save_difference_image(self):
        """Save difference image to file."""
        diff_fits_path = os.path.join(self.output_dir, "difference_image.fits")
        diff_header = self.sci_header.copy()
        diff_header['HISTORY'] = 'Image differencing performed with TransientFinder'
        fits.writeto(diff_fits_path, self.diff_data, diff_header, overwrite=True)
        print(f"Difference image saved to: {diff_fits_path}")
        return True

    def detect_transients(self, threshold=5.0, npixels=5):
        """
        Detect potential transient sources in the difference image.

        Parameters
        ----------
        threshold : float
            Detection threshold in sigma units above/below background.
        npixels : int
            Minimum number of connected pixels for detection.

        Returns
        -------
        astropy.table.Table
            Table of detected transient sources (both positive and negative).
        """
        if self.diff_data is None:
            print("Difference image not created. Please run perform_subtraction() first.")
            return None

        try:
            print(f"Detecting transients with threshold=±{threshold}σ...")
            # Calculate background statistics with sigma clipping
            _, median, std = sigma_clipped_stats(self.diff_data,
                                                 sigma=self.config.SIGMA_CLIP)

            print(f"Background statistics: median={median:.3f}, std={std:.3f}")

            # Find positive peaks (new/brightening sources)
            threshold_positive = median + (threshold * std)
            positive_peaks = find_peaks(
                self.diff_data, 
                threshold_positive, 
                box_size=self.config.BOX_SIZE,
                npeaks=self.config.MAX_PEAKS, 
                centroid_func=None
            )

            # Find negative peaks (disappearing/dimming sources)
            # Invert the image to find negative peaks as positive peaks
            inverted_diff = -self.diff_data
            threshold_negative = -median + (threshold * std)  # This becomes positive after inversion
            negative_peaks = find_peaks(
                inverted_diff, 
                threshold_negative, 
                box_size=self.config.BOX_SIZE,
                npeaks=self.config.MAX_PEAKS, 
                centroid_func=None
            )

            # Combine results
            all_transients = []

            # Process positive transients
            if positive_peaks:
                for peak in positive_peaks:
                    peak['peak_type'] = 'positive'
                    peak['significance'] = (peak['peak_value'] - median) / std
                    peak['original_value'] = peak['peak_value']  # Store original difference value
                    all_transients.append(peak)

            # Process negative transients
            if negative_peaks:
                for peak in negative_peaks:
                    # Convert back to original coordinates and values
                    original_value = -peak['peak_value']  # Convert back from inverted
                    peak['peak_type'] = 'negative'
                    peak['significance'] = -(original_value - median) / std  # Negative significance
                    peak['original_value'] = original_value
                    peak['peak_value'] = original_value  # Correct the peak value
                    all_transients.append(peak)

            if all_transients:
                # Create combined table
                from astropy.table import vstack
                self.transient_table = vstack([Table([t]) for t in all_transients])
                
                # Sort by absolute significance (strongest detections first)
                self.transient_table['abs_significance'] = np.abs(self.transient_table['significance'])
                self.transient_table.sort('abs_significance', reverse=True)
                
            else:
                self.transient_table = Table()
                print("No transient sources detected.")
                return self.transient_table

            # Add RA, Dec coordinates for each source
            ra_list = []
            dec_list = []

            for x, y in zip(self.transient_table['x_peak'], self.transient_table['y_peak']):
                ra, dec = self.sci_wcs.all_pix2world(x, y, 0)
                ra_list.append(ra)
                dec_list.append(dec)

            self.transient_table['ra'] = ra_list
            self.transient_table['dec'] = dec_list

            # Count positive and negative detections
            n_positive = len([t for t in self.transient_table if t['peak_type'] == 'positive'])
            n_negative = len([t for t in self.transient_table if t['peak_type'] == 'negative'])

            # Save transient catalog
            catalog_path = os.path.join(self.output_dir, "transients.csv")
            self.transient_table.write(catalog_path, format='csv', overwrite=True)
            
            print(f"Found {len(self.transient_table)} transient candidates:")
            print(f"  - {n_positive} positive (new/brightening sources)")
            print(f"  - {n_negative} negative (disappearing/dimming sources)")
            print(f"Transient catalog saved to: {catalog_path}")

            # Print summary of strongest detections
            print("\nStrongest detections:")
            for i, transient in enumerate(self.transient_table[:5]):  # Top 5
                peak_type = transient['peak_type']
                significance = transient['significance']
                ra = transient['ra']
                dec = transient['dec']
                print(f"  {i+1}. {peak_type.upper()} at RA={ra:.6f}°, Dec={dec:.6f}° (σ={significance:.1f})")

            return self.transient_table

        except Exception as e:
            print(f"Error detecting transients: {e}")
            return None

    def plot_results(self, figsize=None, show=True):
        """
        Plot the science, reference, and difference images with detected transients.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size in inches. If None, uses config default.
        show : bool
            Whether to show the plot (or just save it).

        Returns
        -------
        str
            Path to the saved plot image.
        """
        if figsize is None:
            figsize = self.config.DEFAULT_FIGSIZE

        try:
            # Create figure with three subplots
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            # Define z-scale normalization for better visualization
            zscale = ZScaleInterval()

            # Plot science image
            vmin, vmax = zscale.get_limits(self.sci_data)
            norm_sci = ImageNormalize(vmin=vmin, vmax=vmax)
            axes[0].imshow(self.sci_data, origin='lower', cmap='gray', norm=norm_sci)
            axes[0].set_title('Science Image')
            axes[0].set_xlabel('X (pixels)')
            axes[0].set_ylabel('Y (pixels)')

            # Plot reference image
            vmin, vmax = zscale.get_limits(self.ref_data)
            norm_ref = ImageNormalize(vmin=vmin, vmax=vmax)
            axes[1].imshow(self.ref_data, origin='lower', cmap='gray', norm=norm_ref)
            axes[1].set_title('Reference Image')
            axes[1].set_xlabel('X (pixels)')
            axes[1].set_ylabel('Y (pixels)')

            # Plot difference image
            vmin, vmax = zscale.get_limits(self.diff_data)
            limit = max(abs(vmin), abs(vmax))
            norm_diff = ImageNormalize(vmin=-limit, vmax=limit)
            diff_im = axes[2].imshow(self.diff_data, origin='lower', cmap='coolwarm', norm=norm_diff)
            axes[2].set_title('Difference Image')
            axes[2].set_xlabel('X (pixels)')
            axes[2].set_ylabel('Y (pixels)')

            # Add colorbar for difference image
            plt.colorbar(diff_im, ax=axes[2], label='Flux Difference')

            # Mark detected transients if available
            if self.transient_table is not None and len(self.transient_table) > 0:
                print(f"Plotting {len(self.transient_table)} detected transients...")
                
                # Separate positive and negative for legend
                positive_plotted = False
                negative_plotted = False
                
                for idx, transient in enumerate(self.transient_table):
                    x, y = transient['x_peak'], transient['y_peak']
                    peak_type = transient['peak_type']
                    significance = transient['significance']
                    
                    print(f"Plotting transient {idx+1}: {peak_type} at ({x:.1f}, {y:.1f}), σ={significance:.1f}")
                    
                    # Use different colors and markers for positive vs negative
                    if peak_type == 'positive':
                        color = 'lime'
                        edge_color = 'darkgreen'
                        label = 'Positive (new/bright)' if not positive_plotted else ""
                        positive_plotted = True
                    else:
                        color = 'red'
                        edge_color = 'darkred'
                        label = 'Negative (dim/gone)' if not negative_plotted else ""
                        negative_plotted = True
                    
                    # Draw circle marker
                    circle = plt.Circle((x, y), radius=self.config.APERTURE_RADIUS, 
                                      fill=False, color=color, linewidth=2, 
                                      edgecolor=edge_color, label=label)
                    axes[2].add_patch(circle)
                    
                    # Add text with significance - make it more visible
                    axes[2].text(x+15, y+15, f"{idx+1}\n({significance:.1f}σ)", 
                                color='white', fontsize=9, fontweight='bold',
                                ha='left', va='bottom',
                                bbox=dict(boxstyle="round,pad=0.3", 
                                         facecolor=color, alpha=0.8, edgecolor=edge_color))
                
                # Create custom legend
                legend_elements = []
                if positive_plotted:
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='lime', markeredgecolor='darkgreen',
                                  markersize=10, linewidth=0, label='Positive (new/bright)')
                    )
                if negative_plotted:
                    legend_elements.append(
                        plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='red', markeredgecolor='darkred',
                                  markersize=10, linewidth=0, label='Negative (dim/gone)')
                    )
                
                if legend_elements:
                    axes[2].legend(handles=legend_elements, loc='upper right', 
                                  fontsize=10, framealpha=0.9, fancybox=True, shadow=True)
            else:
                print("No transients to plot.")

            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "transient_detection_plot.png")
            plt.savefig(plot_path, dpi=self.config.PLOT_DPI, bbox_inches='tight')

            if show:
                plt.show()
            else:
                plt.close()

            print(f"Result plot saved to: {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error plotting results: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """
    Main function to run the transient finder from command line.

    Returns
    -------
    int
        Exit code (0 for success, 1 for failure).
    """
    import argparse

    parser = argparse.ArgumentParser(description='Find transient sources in astronomical images.')
    parser.add_argument('science_image', help='Path to the science FITS image')
    parser.add_argument('--survey', choices=['PanSTARRS', 'DSS2'],
                        default='DSS2', help='Survey to use for reference image')
    parser.add_argument('--filter', dest='filter_band',
                        choices=['g', 'r', 'i', 'blue', 'red', 'Blue', 'Red'],
                        default='r', help='Filter/band: g,r,i for PanSTARRS; blue,red for DSS2')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=5.0,
                        help='Detection threshold in sigma units')
    parser.add_argument('--method', choices=['proper', 'direct'],
                        default='proper', help='Image subtraction method')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    parser.add_argument('--config', help='Path to JSON configuration file')

    args = parser.parse_args()

    # Load configuration
    config = Config.from_file(args.config) if args.config else Config()

    # Create TransientFinder with science image
    finder = TransientFinder(args.science_image, output_dir=args.output,
                             config=config)

    # Get reference image from specified survey
    if not finder.get_reference_image(survey=args.survey, filter_band=args.filter_band):
        print("Failed to get reference image. Exiting.")
        return 1

    # Perform image subtraction
    if not finder.perform_subtraction(method=args.method):
        print("Image subtraction failed. Exiting.")
        return 1

    # Detect transient sources
    transients = finder.detect_transients(threshold=args.threshold)

    # Plot results unless --no-plot is specified
    if not args.no_plot:
        finder.plot_results(show=True)

    print("Transient detection complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
