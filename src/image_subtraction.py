#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Transient Source Finder
This script performs image differencing between a user-provided FITS image and a reference
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
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.table import Table
from astroquery.skyview import SkyView
from astroquery.sdss import SDSS
from astroquery.hips2fits import hips2fits
from properimage import single_image as si
from properimage import propersubtract as ps
from photutils.detection import find_peaks
from photutils.aperture import CircularAperture


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

    def get_reference_image(self, survey="DSS2 Red"):
        """
        Retrieve a reference image from an online survey.
        
        Parameters
        ----------
        survey : str
            Survey to use. Options include:
            - "PanSTARRS": PanSTARRS DR1 i-band
            - "SDSS": SDSS r-band
            - "DSS2 Red": Digital Sky Survey 2 Red band
        
        Returns
        -------
        bool
            True if reference retrieval was successful, False otherwise
        """
        print(f"Retrieving reference image from {survey}...")
        
        try:
            if survey.lower() == "panstarrs":
                # Use HiPS2FITS for PanSTARRS
                hips_id = "PanSTARRS/DR1/r"
                result = hips2fits.query_with_wcs(
                    hips=hips_id,
                    wcs=self.sci_wcs,
                    format="fits"
                )
                with fits.open(result) as hdul:
                    self.ref_data = hdul[0].data
                    self.ref_header = hdul[0].header
                self.ref_wcs = self.sci_wcs
                
            elif survey.lower() == "sdss":
                # Query SDSS with center coordinates and field size
                imgs = SDSS.get_images(
                    coordinates=self.center_coord,
                    band='r',
                    radius=self.field_size/2
                )
                self.ref_data = imgs[0][0].data
                self.ref_header = imgs[0][0].header
                self.ref_wcs = WCS(self.ref_header)
                
            else:  # Default to DSS2
                # Use SkyView for DSS2 and others
                imgs = SkyView.get_images(
                    position=self.center_coord,
                    survey=[survey],
                    coordinates='J2000',
                    height=self.field_size.to(u.deg).value,
                    width=self.field_size.to(u.deg).value,
                    grid=False
                )
                self.ref_data = imgs[0][0].data
                self.ref_header = imgs[0][0].header
                self.ref_wcs = WCS(self.ref_header)
            
            # Save reference image to FITS
            ref_fits_path = os.path.join(self.output_dir, "reference_image.fits")
            fits.writeto(ref_fits_path, self.ref_data, self.ref_header,
                         overwrite=True)
            print(f"Reference image saved to: {ref_fits_path}")
            
            return True
            
        except Exception as e:
            print(f"Error retrieving reference image: {e}")
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
        
        try:
            if method.lower() == "proper":
                print("Performing ProperImage subtraction...")
                
                # Create ProperImage SingleImage objects
                sci_img = si.SingleImage(self.sci_data, psf_basis_size=25)
                ref_img = si.SingleImage(self.ref_data, psf_basis_size=25)
                
                # Align images - proper image ensures they're aligned
                sci_img.coverage_mask
                ref_img.coverage_mask
                
                # Calculate PSF models
                sci_img.get_variable_psf()
                ref_img.get_variable_psf()
                
                # Perform subtraction
                D, P, S_zoomed, R_zoomed, mask, subtr = ps.diff(sci_img, ref_img)
                self.diff_data = D
                
                # Apply mask to exclude problematic regions in the difference image
                if mask is not None:
                    self.diff_data[~mask] = np.nan
            else:
                print("Performing direct subtraction...")
                # For direct subtraction, ensure images are aligned in WCS
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
            sigma_clip = SigmaClip(sigma=3.0)
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
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # Define z-scale normalization for better visualization
            zscale = ZScaleInterval()
            
            # Plot science image
            vmin, vmax = zscale.get_limits(self.sci_data)
            norm_sci = ImageNormalize(vmin=vmin, vmax=vmax)
            axes[0].imshow(self.sci_data, origin='lower', cmap='gray', norm=norm_sci)
            axes[0].set_title('Science Image')
            
            # Plot reference image
            vmin, vmax = zscale.get_limits(self.ref_data)
            norm_ref = ImageNormalize(vmin=vmin, vmax=vmax)
            axes[1].imshow(self.ref_data, origin='lower', cmap='gray', norm=norm_ref)
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
                    significance = transient['significance']
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
    parser.add_argument('--survey', choices=['PanSTARRS', 'SDSS', 'DSS2 Red'], 
                        default='DSS2 Red', help='Survey to use for reference image')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--threshold', type=float, default=5.0, 
                        help='Detection threshold in sigma units')
    parser.add_argument('--method', choices=['proper', 'direct'], default='proper', 
                        help='Image subtraction method')
    parser.add_argument('--no-plot', action='store_true', help='Skip plotting')
    
    args = parser.parse_args()
    
    # Initialize TransientFinder with science image
    finder = TransientFinder(args.science_image, output_dir=args.output)
    
    # Get reference image
    if not finder.get_reference_image(survey=args.survey):
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
