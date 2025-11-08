Examples
========

This page provides examples of using the Photometry Factory for RAPAS for various astronomical analysis tasks.

Basic Photometry Example
---------------------

This example demonstrates how to perform basic photometry on a single FITS image:

1. Log in to the application.
2. Upload a science image using the sidebar uploader.
3. Configure parameters in the sidebar (e.g., Seeing=3.0, Threshold=3.0 sigma, Border Mask=25).
4. Click "Run Zero Point Calibration".
5. Review the results (plots, tables, Aladin viewer).
6. Download the results using the "Download All Results (ZIP)" button.

.. image:: _static/basic_example.png
   :width: 100%
   :alt: Basic photometry example (Needs update if UI changed)

Variable Star Analysis
-------------------

This example shows how to analyze a variable star using PFR outputs:

1. Upload and process a sequence of science images of the same field individually through the PFR application. Ensure consistent parameters.
2. Download the results ZIP for each image.
3. Extract the `*_catalog.csv` files.
4. Use a script like the one below to combine catalogs and plot a light curve.

.. code-block:: python

   # Example script for processing multiple PFR catalogs
   import pandas as pd
   import matplotlib.pyplot as plt
   import glob
   import os
   from astropy.time import Time

   # Directory containing extracted CSV catalog files
   results_dir = "path/to/extracted/catalogs"
   catalog_files = glob.glob(os.path.join(results_dir, "*_catalog.csv"))

   # --- Option 1: Identify target by coordinates ---
   target_ra = 183.2641  # Example RA
   target_dec = 22.0145 # Example Dec
   search_radius_deg = 3.0 / 3600.0 # 3 arcseconds

   # --- Option 2: Identify target by a known ID (if available) ---
   # target_id_col = 'aavso_Name' # e.g., 'simbad_main_id', 'aavso_Name'
   # target_id_val = 'V* AB Aur'

   light_curve_data = []

   for file in catalog_files:
       try:
           df = pd.read_csv(file)
           # Extract observation time from header file (assuming it exists)
           header_file = file.replace("_catalog.csv", "_header.txt")
           obs_time_str = None
           if os.path.exists(header_file):
               with open(header_file, 'r') as hf:
                   for line in hf:
                       if line.startswith('DATE-OBS'):
                           obs_time_str = line.split('=')[1].strip().split('/')[0].strip() # Extract value
                           break
           if not obs_time_str:
               print(f"Warning: Could not find DATE-OBS in {header_file}")
               continue

           obs_time_jd = Time(obs_time_str).jd

           # Find the target
           target_row = None
           # Option 1: Match by coordinates
           if 'ra' in df.columns and 'dec' in df.columns:
               dist_deg = ((df['ra'] - target_ra)**2 + (df['dec'] - target_dec)**2)**0.5
               match_idx = dist_deg.idxmin() # Find closest match
               if dist_deg[match_idx] < search_radius_deg:
                   target_row = df.loc[match_idx]

           # Option 2: Match by ID (Uncomment and adapt if using)
           # elif target_id_col in df.columns:
           #     matches = df[df[target_id_col] == target_id_val]
           #     if not matches.empty:
           #         target_row = matches.iloc[0]

           if target_row is not None and 'aperture_calib_mag' in target_row and pd.notna(target_row['aperture_calib_mag']):
               mag = target_row['aperture_calib_mag']
               # Estimate magnitude error from aperture_sum_err (flux error)
               # mag_err = 1.0857 * flux_err / flux
               flux = target_row.get('aperture_sum', None)
               flux_err = target_row.get('aperture_sum_err', None)
               mag_err = None
               if flux is not None and flux_err is not None and flux > 0:
                   mag_err = 1.0857 * (flux_err / flux)

               light_curve_data.append({'jd': obs_time_jd, 'mag': mag, 'mag_err': mag_err})
           else:
               print(f"Warning: Target not found or missing magnitude in {file}")

       except Exception as e:
           print(f"Error processing {file}: {e}")

   if not light_curve_data:
       print("No data points collected for light curve.")
   else:
       # Create DataFrame and sort by time
       lc_df = pd.DataFrame(light_curve_data).sort_values('jd')

       # Plot the light curve
       plt.figure(figsize=(10, 6))
       plt.errorbar(lc_df['jd'], lc_df['mag'], yerr=lc_df['mag_err'], fmt='o', capsize=3, label='Aperture Mag')
       plt.gca().invert_yaxis()  # Astronomical magnitude convention
       plt.xlabel('Time (JD)')
       plt.ylabel('Calibrated Magnitude')
       plt.title(f'Light Curve (RA:{target_ra:.4f}, Dec:{target_dec:.4f})')
       plt.legend()
       plt.grid(True, alpha=0.3)
       plt.tight_layout()
       plt.savefig('light_curve.png')
       print("Light curve saved to light_curve.png")


Asteroids and Moving Objects
-------------------------

This example demonstrates how to identify potential moving objects:

1. Upload an image suspected to contain solar system objects.
2. Run the analysis ("Run Zero Point Calibration"). The pipeline automatically queries SkyBoT.
3. Review the "Matched Objects Summary" table or the full catalog (`*_catalog.csv`).
4. Look for sources with "SkyBoT" listed in the `catalog_matches` column.
5. The `skybot_NAME` column will contain the identified object's designation.

Tutorial: Complete Photometry Workflow
-----------------------------------

This step-by-step tutorial covers a complete workflow:

1. **Preparation**
   * Ensure the backend server is running.
   * Have your science image(s) ready.
   * Know your approximate observatory location.
   * Obtain an Astro-Colibri UID key if you want transient matching.

2. **Login & Configuration**
   * Launch the frontend (`streamlit run run_frontend.py`) and log in.
   * Upload your science image.
   * Configure Observatory, Process Options (CRR, Astrometry+), Analysis Parameters, Photometry Parameters, and Astro-Colibri key in the sidebar.
   * Click "Save Configuration" to store settings.

3. **Run Pipeline**
   * Click "Run Zero Point Calibration".
   * Monitor progress messages in the main panel.

4. **Analysis & Review**
   * Examine the image display and plots (Background, FWHM, PSF, Zero Point).
   * Review the matched sources tables (GAIA calibration, Catalog Matches Summary).
   * Explore objects in the Aladin viewer.
   * Check the log output for details or warnings.

5. **Results**
   * Download the complete results using the ZIP button.
   * Use the `*_catalog.csv` file for your scientific analysis.

Advanced: Scripting with PFR Functions
-----------------------------------

While PFR is primarily a web app, some core functions from `pages/app.py` and `tools.py` can potentially be used in standalone Python scripts if dependencies are managed correctly.

.. warning::
   Directly calling Streamlit-cached functions or functions relying heavily on `st.session_state` or `st.` calls outside the Streamlit environment might not work as expected or require significant adaptation.

.. code-block:: python

   # Example of potentially using some PFR functions in a script
   # NOTE: This requires careful dependency management and adaptation.
   # Functions relying on Streamlit state or UI elements will fail.

   import numpy as np
   from astropy.io import fits
   from astropy.table import Table
   # Adjust imports based on actual file structure and needs
   from tools import safe_wcs_create, extract_pixel_scale, airmass
   from pages.app import (estimate_background, fwhm_fit, make_border_mask,
                          detection_and_photometry, cross_match_with_gaia,
                          calculate_zero_point, enhance_catalog)

   # --- Configuration (Mimic Streamlit inputs) ---
   fits_filepath = 'my_image.fits'
   output_dir = 'script_results'
   # Analysis Params
   seeing_arcsec = 3.0
   threshold_sigma = 3.0
   detection_mask_px = 25
   # Gaia Params
   gaia_band = 'phot_g_mean_mag'
   gaia_max_mag = 20.0
Examples
========

This page contains a compact set of examples and a very small, self-contained
Python example that shows how to call core library functions. The production
app is a Streamlit frontend backed by a Flask backend; the example below is
meant to be runnable in a simple development environment (with appropriate
dependencies and a FITS file if you want real results).

Minimal Python example
----------------------

Save the following as `run_example.py` in the project root and run it with
`python run_example.py`. It demonstrates the call pattern used by the
application; if you don't have a FITS file handy it will print a friendly
message instead.

.. code-block:: python

    # run_example.py
    from src.pipeline import process_image_batch
    from src.tools import load_fits_image

    fits_path = 'example.fits'  # replace with a real .fits file to actually run

    try:
        image = load_fits_image(fits_path)
        results = process_image_batch([image])
        print('Processed', len(results), 'images')
        for r in results:
            print(r)
    except FileNotFoundError:
        print('example.fits not found. This example demonstrates the call pattern only.')

Notes and next steps
--------------------

- The repository includes higher-level examples in the `doc` folder. Use the
  `installation` and `usage` pages to set up your environment and run the
  frontend/backend stack.
- For batch/advanced workflows refer to the longer script examples that
  demonstrate combining multiple catalogs and plotting light curves.