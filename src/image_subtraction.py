#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transient Source Finder
This script performs image differencing between a user-provided FITS image and reference
image retrieved from online surveys (PanSTARRS, SDSS, or DSS2) to detect transient sources.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import ZScaleInterval, ImageNormalize
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astroquery.skyview import SkyView
from astroquery.sdss import SDSS
from astroquery.hips2fits import hips2fits
from properimage.operations import subtract
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture
from reproject import reproject_interp
import time
import requests
import warnings

# Suppress FITS header fix warnings
warnings.filterwarnings('ignore', category=fits.verify.VerifyWarning)
warnings.filterwarnings('ignore', message='.*datfix.*')
warnings.filterwarnings('ignore', message='.*FITSFixedWarning.*')


class TransientFinder:
    """
    A class to perform image differencing and find transient sources.
    
    This class handles loading a science image, retrieving a reference image,
    aligning them, performing proper image subtraction, and detecting sources
    in the difference image.
    """
    def __init__(self, science_fits_path, output_dir=None):
        """
        Initialize the TransientFinder with the science image path.
        
        Parameters
        ----------
        science_fits_path : str
            Path to the science FITS file
        output_dir : str, optional
            Directory to save results. If None, uses current directory
        """
        self.science_fits_path = science_fits_path
        
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
        self.ref_wcs = None
        self.diff_data = None
        self.transient_table = None

    def load_science_image(self):
        """Load the science image and extract its key properties."""
        try:
            # Suppress specific FITS warnings during loading
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with fits.open(self.science_fits_path) as hdul:
                    self.sci_data = hdul[0].data
                    self.sci_header = hdul[0].header
                    self.sci_wcs = WCS(self.sci_header)

            # Get image dimensions
            self.ny, self.nx = self.sci_data.shape

            # Get central coordinates
            central_x, central_y = self.nx // 2, self.ny // 2
            ra, dec = self.sci_wcs.all_pix2world(central_x, central_y, 0)
            self.center_coord = SkyCoord(ra, dec, unit='deg')

            # Get image size in degrees
            corner_x, corner_y = self.nx - 1, self.ny - 1
            ra_corner, dec_corner = self.sci_wcs.all_pix2world(corner_x,
                                                               corner_y, 0)
            corner_coord = SkyCoord(ra_corner, dec_corner, unit='deg')
            self.field_size = self.center_coord.separation(corner_coord)*2
            
            print(f"Science image loaded: {self.science_fits_path}")
            print(f"Image center: RA={ra:.6f}°, Dec={dec:.6f}°")
            print(f"Field size: {self.field_size.to(u.arcmin):.2f}")
            
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
            True if reference retrieval was successful, False otherwise
        """
        print(f"Retrieving reference image from {survey} in {filter_band} band...")
        
        # Check field size compatibility with different surveys
        field_size_arcmin = self.field_size.to(u.arcmin).value
        
        # Define survey limitations
        if survey.lower() == "panstarrs" and field_size_arcmin > 60.0:
            print(f"Warning: Field size ({field_size_arcmin:.1f} arcmin) may exceed PanSTARRS practical limit.")
            print("Consider using DSS2 for very large fields.")
        
        max_retries = 2
        retry_delay = 3  # seconds
        
        for attempt in range(max_retries):
            try:
                if survey.lower() == "panstarrs":
                    # Use HiPS2FITS for PanSTARRS with retry logic
                    # Map filter to HiPS ID
                    hips_mapping = {
                        'g': "CDS/P/PanSTARRS/DR1/g",
                        'r': "CDS/P/PanSTARRS/DR1/r", 
                        'i': "CDS/P/PanSTARRS/DR1/i"
                    }
                    
                    if filter_band.lower() not in hips_mapping:
                        print(f"Warning: Filter '{filter_band}' not available for PanSTARRS. Using 'r' band.")
                        filter_band = 'r'
                    
                    hips_id = hips_mapping[filter_band.lower()]
                    print(f"Attempt {attempt + 1}/{max_retries} to retrieve PanSTARRS {filter_band}-band image...")
                    
                    # Try with increased timeout and smaller field if needed
                    try:
                        result = hips2fits.query_with_wcs(
                            hips=hips_id,
                            wcs=self.sci_wcs
                        )

                        if isinstance(result, str):
                            # If it's a URL, download it
                            response = requests.get(result, timeout=60)
                            response.raise_for_status()
                            from io import BytesIO
                            fits_data = BytesIO(response.content)
                            with fits.open(fits_data) as hdul:
                                self.ref_data = hdul[0].data
                                self.ref_header = hdul[0].header
                        else:
                            # If it's already a file-like object
                            with fits.open(result) as hdul:
                                self.ref_data = hdul[0].data
                                self.ref_header = hdul[0].header
                        
                        self.ref_wcs = self.sci_wcs
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if "timeout" in str(e).lower() or "connection" in str(e).lower():
                            print(f"Network timeout on attempt {attempt + 1}. Error: {e}")
                            if attempt < max_retries - 1:
                                print(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                continue
                            else:
                                print("Max retries reached for PanSTARRS. Falling back to DSS2...")
                                survey = "DSS2 Red"  # Fall back to DSS2
                        else:
                            raise e
                
                else:  # DSS2 or other SkyView surveys
                    # Use SkyView for DSS2 and others
                    # Map filter to DSS2 survey name
                    if survey.lower() == "dss2":
                        dss2_mapping = {
                            'blue': "DSS2 Blue",
                            'b': "DSS2 Blue",
                            'red': "DSS2 Red",
                            'r': "DSS2 Red"
                        }
                        
                        if filter_band.lower() not in dss2_mapping:
                            print(f"Warning: Filter '{filter_band}' not available for DSS2. Using Red band.")
                            survey_name = "DSS2 Red"
                        else:
                            survey_name = dss2_mapping[filter_band.lower()]
                    else:
                        survey_name = survey
                    
                    print(f"Using SkyView for {survey_name}...")
                    
                    # Ensure field size has proper units
                    if hasattr(self.field_size, 'unit'):
                        field_size_deg = self.field_size.to(u.deg).value
                    else:
                        # If no units, assume it's already in degrees
                        field_size_deg = float(self.field_size)
                    
                    # Limit field size to reasonable values for SkyView
                    field_size_deg = min(field_size_deg, 2.0)  # Max 2 degrees for DSS2
                    field_size_deg = max(field_size_deg, 0.1)  # Min 0.1 degrees
                    
                    print(f"Requesting field size: {field_size_deg:.3f} degrees")
                    
                    imgs = SkyView.get_images(
                        position=self.center_coord,
                        survey=[survey_name],
                        coordinates='J2000',
                        height=field_size_deg * u.deg,
                        width=field_size_deg * u.deg,
                        pixels=[self.nx, self.ny]
                    )
                    self.ref_data = imgs[0][0].data
                    self.ref_header = imgs[0][0].header
                    self.ref_wcs = WCS(self.ref_header)
                    break  # Success
                    
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to retrieve reference image after {max_retries} attempts.")
                    return False
        
        try:
            # Verify we got valid data
            if self.ref_data is None or self.ref_data.size == 0:
                print("Retrieved reference image is empty or invalid.")
                return False
            
            # Save reference image to FITS
            ref_fits_path = os.path.join(self.output_dir, "reference_image.fits")
            fits.writeto(ref_fits_path, self.ref_data, self.ref_header,
                         overwrite=True)
            print(f"Reference image saved to: {ref_fits_path}")
            print(f"Reference image shape: {self.ref_data.shape}")
            
            return True
            
        except Exception as e:
            print(f"Error saving reference image: {e}")
            return False

    def align_images(self):
        """
        Align the reference image to match the science image WCS.
        
        Returns
        -------
        bool
            True if alignment was successful, False otherwise
        """
        if self.ref_data is None:
            print("Reference image not loaded. Cannot align.")
            return False
        
        try:
            print("Aligning reference image to science image WCS...")
            # Use reproject to align reference image to science image WCS
            aligned_ref, _ = reproject_interp((self.ref_data, self.ref_wcs),
                                              self.sci_wcs,
                                              shape_out=self.sci_data.shape
                                              )

            # Replace NaN values with median background
            median_ref = np.nanmedian(aligned_ref)
            aligned_ref = np.where(np.isnan(aligned_ref), median_ref,
                                   aligned_ref)

            # Update reference data and WCS
            self.ref_data = aligned_ref
            self.ref_wcs = self.sci_wcs
            self.ref_header = self.sci_header.copy()
            self.ref_header['HISTORY'] = 'Reference image aligned to science image WCS'

            # Save aligned reference image
            aligned_ref_path = os.path.join(self.output_dir,
                                            "aligned_reference.fits")
            fits.writeto(aligned_ref_path, self.ref_data, self.ref_header,
                         overwrite=True)
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
            Subtraction method: 'proper' for ProperImage or 'direct' for simple subtraction

        Returns
        -------
        bool
            True if subtraction was successful, False otherwise
        """
        if self.ref_data is None:
            print("Reference image not loaded. Please run get_reference_image() first.")
            return False

        # Align images before subtraction
        if not self.align_images():
            print("Image alignment failed. Proceeding with unaligned images...")

        try:
            if method.lower() == "proper":
                print("Performing ProperImage subtraction...")

                # Ensure arrays have native byte order for ProperImage
                sci_data_native = self.sci_data.astype(self.sci_data.dtype.newbyteorder('='))
                ref_data_native = self.ref_data.astype(self.ref_data.dtype.newbyteorder('='))

                try:
                    # Save temporary FITS files for ProperImage
                    temp_sci_path = os.path.join(self.output_dir, "temp_science.fits")
                    temp_ref_path = os.path.join(self.output_dir, "temp_reference.fits")

                    fits.writeto(temp_sci_path, sci_data_native, self.sci_header, overwrite=True)
                    fits.writeto(temp_ref_path, ref_data_native, self.ref_header, overwrite=True)

                    # Perform subtraction using ProperImage operations.subtract
                    print("Performing difference imaging...")
                    result = subtract(
                        ref=temp_ref_path,
                        new=temp_sci_path,
                        smooth_psf=False,
                        fitted_psf=True,
                        align=True,  # Enable ProperImage's internal alignment
                        iterative=False,
                        beta=False,
                        shift=True   # Allow for small shifts
                    )

                    # Debug: print result type and structure
                    print(f"ProperImage result type: {type(result)}")
                    if hasattr(result, '__len__'):
                        print(f"ProperImage result length: {len(result)}")
                    
                    # Handle different possible return formats from ProperImage
                    if isinstance(result, (list, tuple)):
                        if len(result) >= 1:
                            # First element should be the difference image
                            diff_candidate = result[0]
                            
                            # Check if it's a file path string
                            if isinstance(diff_candidate, str):
                                print(f"ProperImage returned file path: {diff_candidate}")
                                # Try to load the difference image from file
                                try:
                                    with fits.open(diff_candidate) as hdul:
                                        self.diff_data = hdul[0].data
                                    print("Successfully loaded difference image from ProperImage output file")
                                except Exception as load_error:
                                    print(f"Failed to load ProperImage output file: {load_error}")
                                    raise load_error
                            else:
                                # Handle as array data
                                if hasattr(diff_candidate, 'real'):
                                    self.diff_data = diff_candidate.real
                                else:
                                    self.diff_data = diff_candidate
                                
                                # Apply mask if available and result has multiple elements
                                if len(result) >= 4:
                                    mask = result[3]
                                    if mask is not None:
                                        self.diff_data = np.ma.array(self.diff_data, mask=mask).filled(np.nan)
                        else:
                            raise ValueError("ProperImage returned empty result")
                    else:
                        # Single result - could be array or file path
                        if isinstance(result, str):
                            print(f"ProperImage returned single file path: {result}")
                            try:
                                with fits.open(result) as hdul:
                                    self.diff_data = hdul[0].data
                                print("Successfully loaded difference image from ProperImage output file")
                            except Exception as load_error:
                                print(f"Failed to load ProperImage output file: {load_error}")
                                raise load_error
                        else:
                            # Handle as direct array data
                            if hasattr(result, 'real'):
                                self.diff_data = result.real
                            else:
                                self.diff_data = result

                    print(f"ProperImage subtraction successful. Difference image shape: {self.diff_data.shape}")

                    # Clean up temporary files
                    try:
                        os.remove(temp_sci_path)
                        os.remove(temp_ref_path)
                    except OSError:
                        pass

                except Exception as proper_error:
                    print(f"ProperImage failed: {proper_error}")
                    print("Falling back to direct subtraction...")
                    # Fall back to direct subtraction
                    self.diff_data = sci_data_native - ref_data_native

                    # Clean up temporary files if they exist
                    for temp_filename in ['temp_science.fits', 'temp_reference.fits']:
                        try:
                            temp_path = os.path.join(self.output_dir, temp_filename)
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except OSError:
                            pass
            else:
                print("Performing direct subtraction...")
                # For direct subtraction, images should already be aligned
                self.diff_data = self.sci_data - self.ref_data

            # Save difference image
            diff_fits_path = os.path.join(self.output_dir, "difference_image.fits")
            diff_header = self.sci_header.copy()
            diff_header['HISTORY'] = 'Image differencing performed with TransientFinder'
            fits.writeto(diff_fits_path, self.diff_data, diff_header,
                         overwrite=True)
            print(f"Difference image saved to: {diff_fits_path}")

            return True

        except Exception as e:
            print(f"Error during image subtraction: {e}")
            return False

    def detect_transients(self, threshold=5.0, npixels=5):
        """
        Detect potential transient sources in the difference image.

        Parameters
        ----------
        threshold : float
            Detection threshold in sigma units above background
        npixels : int
            Minimum number of connected pixels for detection

        Returns
        -------
        astropy.table.Table
            Table of detected transient sources
        """
        if self.diff_data is None:
            print("Difference image not created. Please run perform_subtraction() first.")
            return None

        try:
            print(f"Detecting transients with threshold={threshold}σ...")
            # Calculate background statistics with sigma clipping
            _, median, std = sigma_clipped_stats(self.diff_data, sigma=3.0)

            # Find positive peaks (new sources)
            threshold_positive = median + (threshold * std)
            positive_peaks = find_peaks(self.diff_data, threshold_positive, box_size=5,
                                        npeaks=50, centroid_func=None)
            if positive_peaks:
                positive_peaks['peak_type'] = 'positive'
                positive_peaks['significance'] = (positive_peaks['peak_value'] - median) / std

            # Find negative peaks (disappeared sources)
            threshold_negative = median - (threshold * std)
            negative_peaks = find_peaks(-self.diff_data, -threshold_negative, box_size=5,
                                        npeaks=50, centroid_func=None)
            if negative_peaks:
                negative_peaks['peak_value'] = -negative_peaks['peak_value']
                negative_peaks['peak_type'] = 'negative'
                negative_peaks['significance'] = (median - negative_peaks['peak_value']) / std

            # Combine results
            if positive_peaks and negative_peaks:
                self.transient_table = Table(np.hstack([positive_peaks, negative_peaks]))
            elif positive_peaks:
                self.transient_table = positive_peaks
            elif negative_peaks:
                self.transient_table = negative_peaks
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

            # Save transient catalog
            catalog_path = os.path.join(self.output_dir, "transients.csv")
            self.transient_table.write(catalog_path, format='csv', overwrite=True)
            print(f"Found {len(self.transient_table)} transient candidates.")
            print(f"Transient catalog saved to: {catalog_path}")

            return self.transient_table

        except Exception as e:
            print(f"Error detecting transients: {e}")
            return None

    def plot_results(self, figsize=(15, 5), show=True):
        """
        Plot the science, reference, and difference images with detected transients.

        Parameters
        ----------
        figsize : tuple
            Figure size in inches
        show : bool
            Whether to show the plot (or just save it)

        Returns
        -------
        str
            Path to the saved plot image
        """
        try:
            # Create figure with three subplots
            _, axes = plt.subplots(1, 3, figsize=figsize)

            # Define z-scale normalization for better visualization
            zscale = ZScaleInterval()

            # Plot science image
            vmin, vmax = zscale.get_limits(self.sci_data)
            norm_sci = ImageNormalize(vmin=vmin, vmax=vmax)
            axes[0].imshow(self.sci_data, origin='lower', cmap='gray',
                           norm=norm_sci)
            axes[0].set_title('Science Image')

            # Plot reference image
            vmin, vmax = zscale.get_limits(self.ref_data)
            norm_ref = ImageNormalize(vmin=vmin, vmax=vmax)
            axes[1].imshow(self.ref_data, origin='lower', cmap='gray',
                           norm=norm_ref)
            axes[1].set_title('Reference Image')

            # Plot difference image
            vmin, vmax = zscale.get_limits(self.diff_data)
            # Use symmetric limits for difference image
            limit = max(abs(vmin), abs(vmax))
            norm_diff = ImageNormalize(vmin=-limit, vmax=limit)
            diff_im = axes[2].imshow(self.diff_data, origin='lower', cmap='coolwarm', norm=norm_diff)
            axes[2].set_title('Difference Image')

            # Add colorbar for difference image
            plt.colorbar(diff_im, ax=axes[2], label='Flux Difference')

            # Mark detected transients if available
            if self.transient_table is not None and len(self.transient_table) > 0:
                for idx, transient in enumerate(self.transient_table):
                    x, y = transient['x_peak'], transient['y_peak']
                    peak_type = transient['peak_type']

                    # Use different colors for positive/negative peaks
                    color = 'lime' if peak_type == 'positive' else 'red'

                    # Mark on difference image
                    circle = CircularAperture((x, y), r=10)
                    circle.plot(axes[2], color=color, lw=1.5)
                    axes[2].text(x+12, y+12, f"{idx+1}", color=color, fontsize=8,
                                 ha='left', va='bottom')

            # Adjust layout and save plot
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, "transient_detection_plot.png")
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')

            if show:
                plt.show()
            else:
                plt.close()

            print(f"Result plot saved to: {plot_path}")
            return plot_path

        except Exception as e:
            print(f"Error plotting results: {e}")
            return None


def main():
    """Main function to run the transient finder from command line."""
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
    parser.add_argument('--method', choices=['proper', 'direct'], default='proper', 
                        help='Image subtraction method')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')

    args = parser.parse_args()

    # Create TransientFinder with science image
    finder = TransientFinder(args.science_image, output_dir=args.output)

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
