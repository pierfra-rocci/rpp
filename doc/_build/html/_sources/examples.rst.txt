Examples
========

This page provides examples of using the Photometry Factory for RAPAS for various astronomical analysis tasks.

Basic Photometry Example
---------------------

This example demonstrates how to perform basic photometry on a single FITS image:

1. Upload a science image
2. Apply a threshold of 3.0 sigma for detection
3. Run Zero Point Calibration
4. Download the resulting catalog

.. image:: _static/basic_example.png
   :width: 100%
   :alt: Basic photometry example

Variable Star Analysis
-------------------

This example shows how to analyze a variable star:

1. Upload a sequence of science images of the same field
2. Process each image independently
3. Compare the magnitudes across images
4. Generate a light curve

.. code-block:: python
   
   # Example code for processing multiple images and creating a light curve
   import pandas as pd
   import matplotlib.pyplot as plt
   import glob
   
   # Load all catalogs
   catalogs = []
   for file in glob.glob("pfr_results/*_phot.csv"):
       df = pd.read_csv(file)
       # Extract timestamp from filename
       timestamp = file.split('_')[-2]
       df['timestamp'] = timestamp
       catalogs.append(df)
   
   # Combine all catalogs
   all_data = pd.concat(catalogs)
   
   # Find a specific variable star
   var_star = all_data[all_data['aavso_Name'] == 'V* AB Aur']
   
   # Plot the light curve
   plt.figure(figsize=(10, 6))
   plt.errorbar(var_star['timestamp'], var_star['aperture_calib_mag'], 
                yerr=var_star['aperture_sum_err'], fmt='o')
   plt.gca().invert_yaxis()  # Astronomical magnitude convention
   plt.xlabel('Time')
   plt.ylabel('Calibrated Magnitude')
   plt.title('Light Curve of V* AB Aur')
   plt.grid(True, alpha=0.3)
   plt.savefig('light_curve.png')

Asteroids and Moving Objects
-------------------------

This example demonstrates how to detect and analyze moving objects:

1. Upload an image with potential solar system objects
2. Run the analysis with SkyBoT cross-matching enabled
3. Identify objects marked with "SkyBoT" in the catalog_matches column
4. Extract their positions and magnitudes

Tutorial: Complete Photometry Workflow
-----------------------------------

This step-by-step tutorial covers a complete workflow:

1. **Preparation**
   
   * Obtain calibration frames (bias, dark, flat)
   * Set up your observatory parameters
   
2. **Image Calibration**
   
   * Upload all calibration files
   * Enable all calibration steps
   * Run image calibration
   
3. **Source Detection and Photometry**
   
   * Adjust seeing based on current conditions
   * Set appropriate detection threshold
   * Run Zero Point Calibration
   
4. **Analysis**
   
   * Review the cross-matched catalog
   * Examine the zero point calibration plot
   * Explore objects in the Aladin viewer
   
5. **Results**
   
   * Download the complete results
   * Use the catalog for your scientific analysis

Advanced: Scripting with PFR Functions
-----------------------------------

Though PFR is primarily a web app, you can use its functions in your own Python scripts:

.. code-block:: python

   # Example of using PFR functions in a script
   from astropy.io import fits
   import numpy as np
   from pfr_app import find_sources_and_photometry_streamlit, cross_match_with_gaia_streamlit
   
   # Load your data
   with fits.open('my_image.fits') as hdul:
       data = hdul[0].data
       header = hdul[0].header
   
   # Set parameters
   fwhm_pixels = 5.0
   threshold = 3.0
   border_mask = 50
   
   # Find sources
   phot_table, epsf_table, daofind, bkg = find_sources_and_photometry_streamlit(
       data, header, fwhm_pixels, threshold, border_mask
   )
   
   # Cross-match with GAIA
   pixel_scale = 0.5  # arcsec/pixel
   matched_table = cross_match_with_gaia_streamlit(
       phot_table, header, pixel_scale, fwhm_pixels, 
       'phot_g_mean_mag', 11.0, 19.0
   )
   
   # Now work with the results
   print(f"Found {len(phot_table)} sources")
   print(f"Matched {len(matched_table)} sources with GAIA")