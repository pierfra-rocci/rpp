#!/usr/bin/env python3
import sys
import numpy as np
from astropy.io import fits

class ImageProcessor:
    def __init__(self):
        self.image = None
        
    def read_fits(self, filename):
        """
        Read FITS file and store the image data
        Equivalent to the C++ readFits function
        """
        with fits.open(filename) as hdul:
            # Read the primary HDU (Header Data Unit)
            image_data = hdul[0].data
            
            # In the C++ version, the image is flipped vertically
            # We'll do the same here for consistency
            self.image = np.flipud(image_data)
            
    def calculate_variance(self):
        """Calculate the variance of the image"""
        if self.image is None:
            raise ValueError("No image loaded")
        return np.var(self.image)
    
    def estimate_noise_variance(self):
        """
        Estimate noise variance using second moment method
        This is a simplified version of the CImg variance_noise method
        """
        if self.image is None:
            raise ValueError("No image loaded")
            
        # Calculate differences between adjacent pixels
        dx = np.diff(self.image, axis=1)
        dy = np.diff(self.image, axis=0)
        
        # Estimate noise variance (similar to CImg's method)
        variance_x = np.var(dx) / 2
        variance_y = np.var(dy) / 2
        
        # Return average of horizontal and vertical estimates
        return (variance_x + variance_y) / 2

def main():
    if len(sys.argv) != 2:
        print("Usage: python snr.py <my_fits_file.fits>", file=sys.stderr)
        sys.exit(1)
        
    filename = sys.argv[1]
    processor = ImageProcessor()
    
    try:
        # Read the FITS file
        processor.read_fits(filename)
        
        # Calculate variances
        variance_of_image = processor.calculate_variance()
        estimated_variance_of_noise = processor.estimate_noise_variance()
        
        # Calculate q ratio
        q = variance_of_image / estimated_variance_of_noise
        
        # Clip q to minimum of 1
        q_clip = max(q, 1)
        
        # Calculate SNR
        snr = np.sqrt(q_clip - 1)
        
        print(f"SNR: {snr}")
        
    except Exception as e:
        print(f"Error loading FITS file: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()