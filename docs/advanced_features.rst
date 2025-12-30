Advanced Features
===============

This section covers advanced features and specialized use cases for RAPAS Photometry Pipeline.

Transient Detection (Beta)
-----------------------

The transient detection module enables automated search for new sources and variable objects in your astronomical images through comparison with reference survey templates.

**How It Works**:

1.  **Reference Survey Selection**: The pipeline automatically selects appropriate reference surveys based on your observation location:
    *   **Northern Hemisphere** (latitude > 0°): Uses PanSTARRS1 survey with g, r, i, z, y filter bands
    *   **Southern Hemisphere** (latitude < 0°): Uses SkyMapper survey with g, r, i filter bands

2.  **Template Comparison**: Using `stdpipe` image subtraction capabilities, your observation is compared against high-quality reference templates from the selected survey.

3.  **Candidate Identification**: Sources detected in the subtracted image represent potential transient candidates (new sources, variable objects, or anomalies).

4.  **Candidate Filtering**: Detected candidates are automatically cross-matched against known catalog sources to remove false positives:
    *   Filtered against GAIA DR3 for stellar objects
    *   Filtered against SIMBAD for known objects
    *   Filtered against SkyBoT for solar system objects
    *   Filtered against AAVSO VSX for variable stars

5.  **Classification**: Remaining candidates are flagged as potential transients with quality metrics indicating confidence level.

**Using the Transient Finder**:

*   Enable "Enable Transient Finder" in the sidebar under "Transient Candidates"
*   The reference survey and filter band are automatically selected based on your image coordinates
*   Transient candidates appear in the results catalog with a `transient_flag` column
*   Additional details include subtraction SNR, position offset from known sources, and magnitude estimate

**Interpreting Results**:

*   Candidates with high subtraction SNR are more likely to be genuine transients
*   Position matching helps identify whether a detection is truly new or a catalog miss
*   Review magnitude estimates and color information from cross-matching
*   Check the processing log for details on template selection and subtraction parameters

PSF Photometry
------------

While aperture photometry works well for isolated stars, PSF (Point Spread Function) photometry, performed automatically by RPP using `photutils`, offers advantages for:

*   Crowded fields where stars overlap.
*   Faint sources where aperture photometry is noise-limited.
*   Achieving potentially higher precision by modeling the star's profile.

The application performs these steps:

1.  **PSF Model Construction**: Builds an empirical PSF model (`EPSF`) from bright, isolated stars detected in the image using `photutils.psf.EPSFBuilder`. The model is saved as `*_psf.fits`.
2.  **PSF Fitting**: This model is then fitted to all detected sources using `photutils.psf.IterativePSFPhotometry` to measure their flux.

Results from both aperture and PSF photometry are included in the final catalog (`aperture_calib_mag`, `psf_calib_mag` if calculated) for comparison.

To optimize PSF photometry:

*   Ensure your **Seeing** estimate in the sidebar is reasonably accurate, as it influences the initial detection and PSF extraction box size.
*   Review the PSF model visualization in the results panel for quality (e.g., check for asymmetry or contamination).

.. code-block:: python

    # Example: Inspect the saved PSF model
    from astropy.io import fits
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    # Load the PSF model (adjust filename as needed)
    psf_filename = 'rppyour_image_base_name_psf.fits'
    try:
        psf_data = fits.getdata(psf_filename)

        # Plot it
        norm = simple_norm(psf_data, 'log', percent=99.)
        plt.figure(figsize=(8, 8))
        plt.imshow(psf_data, origin='lower', cmap='viridis', norm=norm)
        plt.colorbar(label='Normalized Intensity (log scale)')
        plt.title('Empirical PSF Model')
        plt.show()
    except FileNotFoundError:
        print(f"PSF file not found: {psf_filename}")
    except Exception as e:
        print(f"Error loading or plotting PSF: {e}")


Working with Time-Series Data
---------------------------

For variable stars, asteroids, or other time-variable objects, you can process multiple images taken over time and combine the results:

1.  Process each image individually through the RPP application, ensuring consistent analysis parameters.
2.  Download the results ZIP archive for each processed image.
3.  Extract the `*_catalog.csv` and `*_header.txt` files for each observation.
4.  Use a script (like the one in `examples.rst`) to:
    *   Read each catalog.
    *   Extract the observation time (e.g., `DATE-OBS` from the header file).
    *   Identify your target star(s) in each catalog (e.g., by matching coordinates).
    *   Extract the magnitude (`aperture_calib_mag` or `psf_calib_mag`) and its error.
    *   Combine the time-magnitude pairs to create and plot a light curve.

Differential Photometry
----------------------

For high-precision relative photometry, especially useful for detecting small variations:

1.  Process your image with RPP to get a calibrated catalog (`*_catalog.csv`).
2.  Identify your target star in the catalog.
3.  Select several (e.g., 3-10) suitable comparison stars:
    *   They should be close in brightness to your target.
    *   They should be nearby on the detector to minimize spatial variations.
    *   They should be confirmed as non-variable (check `catalog_matches` or `aavso_Name` columns; avoid stars flagged by AAVSO or SIMBAD as variable).
    *   They should have good SNR and low magnitude errors.
4.  Calculate the differential magnitude using a script.

Example workflow snippet:

.. code-block:: python

    import pandas as pd
    import numpy as np

    # Load the calibrated catalog
    catalog_file = 'rppyour_image_base_name_catalog.csv'
    catalog = pd.read_csv(catalog_file)

    # --- Identify Target and Potential Comparison Stars ---
    # Example: Target identified by its index in the DataFrame
    target_idx = 42 # Replace with your target star index or find via coordinates

    if target_idx not in catalog.index:
        print(f"Target index {target_idx} not found in catalog.")
        exit()

    target_mag = catalog.loc[target_idx, 'aperture_calib_mag'] # Or psf_calib_mag

    # Find potential comparison stars:
    # - Within 1 magnitude of the target
    # - Not the target itself
    # - Not known variables (checking AAVSO name as example)
    # - Having valid magnitude values
    potential_comps = catalog[
        (np.abs(catalog['aperture_calib_mag'] - target_mag) < 1.0) &
        (catalog.index != target_idx) &
        (catalog['aavso_Name'].isna()) & # Example: Exclude known AAVSO variables
        (catalog['aperture_calib_mag'].notna())
    ].copy()

    # Optional: Add proximity filter (calculate distance from target)
    # potential_comps['distance_pix'] = np.sqrt(
    #    (potential_comps['xcenter'] - catalog.loc[target_idx, 'xcenter'])**2 +
    #    (potential_comps['ycenter'] - catalog.loc[target_idx, 'ycenter'])**2
    # )
    # potential_comps = potential_comps[potential_comps['distance_pix'] < 500] # Example: within 500 pixels

    # Select top N comparison stars with lowest magnitude error
    # Estimate magnitude error from flux error (if available)
    if 'aperture_sum' in catalog.columns and 'aperture_sum_err' in catalog.columns:
         flux = potential_comps['aperture_sum']
         flux_err = potential_comps['aperture_sum_err']
         # Avoid division by zero or invalid values
         valid_err = (flux > 0) & (flux_err.notna())
         potential_comps.loc[valid_err, 'mag_err_est'] = 1.0857 * (flux_err[valid_err] / flux[valid_err])
         potential_comps.loc[~valid_err, 'mag_err_est'] = np.inf
    else:
        potential_comps['mag_err_est'] = np.inf # Cannot sort by error

    # Sort by estimated error and select the best N (e.g., 5)
    comp_stars = potential_comps.nsmallest(5, 'mag_err_est')

    if len(comp_stars) < 1:
        print("Not enough suitable comparison stars found.")
        exit()

    print(f"Using {len(comp_stars)} comparison stars.")

    # --- Calculate Differential Magnitude ---
    # Calculate the average instrumental magnitude of the comparison stars
    # It's often better to average fluxes, then convert back to magnitude
    comp_instrumental_mags = comp_stars['instrumental_mag'] # Use instrumental mag before ZP/airmass
    comp_flux_sum = np.sum(10**(-0.4 * comp_instrumental_mags))
    mean_comp_instrumental_mag = -2.5 * np.log10(comp_flux_sum / len(comp_stars))

    # Differential magnitude (Target Instrumental Mag - Mean Comparison Instrumental Mag)
    target_instrumental_mag = catalog.loc[target_idx, 'instrumental_mag']
    diff_mag = target_instrumental_mag - mean_comp_instrumental_mag

    print(f"Target Instrumental Mag: {target_instrumental_mag:.4f}")
    print(f"Mean Comparison Instrumental Mag: {mean_comp_instrumental_mag:.4f}")
    print(f"Differential Magnitude: {diff_mag:.4f}")

    # This differential magnitude is less sensitive to transparency variations
    # and airmass changes than the direct calibrated magnitude.


Custom Pipeline Integration
-------------------------

While RPP provides an integrated web UI, its outputs can be used as inputs for larger, custom analysis pipelines:

1.  **Process Images**: Use the RPP web application to process your FITS images individually or in batches.
2.  **Collect Outputs**: Download the results ZIP archives containing the calibrated catalogs (`*_catalog.csv`), logs, and other metadata.
3.  **Ingest Catalogs**: Write custom Python scripts (using `pandas`, `astropy`, etc.) to read these CSV catalogs.
4.  **Perform Further Analysis**: Implement specialized analysis not covered by PFR, such as:
    *   Detailed light curve modeling (e.g., using `gatspy`, `lightkurve`).
    *   Asteroid astrometry refinement and orbit determination.
    *   Stacking results from multiple observations.
    *   Generating publication-quality plots.
    *   Cross-matching with specialized or private catalogs.

Example integration concept:

.. code-block:: python

    import pandas as pd
    import glob
    import os

    # Directory where RPP results ZIPs were extracted
    rpp_output_dir = 'path/to/extracted/rpp_results'

    all_catalogs = []
    catalog_files = glob.glob(os.path.join(rpp_output_dir, '*_catalog.csv'))

    for cat_file in catalog_files:
        try:
            df = pd.read_csv(cat_file)
            # Add identifier based on filename
            df['source_image_base'] = os.path.basename(cat_file).replace('_catalog.csv', '')
            # Extract observation time from corresponding header file (add logic here)
            # df['obs_jd'] = ...
            all_catalogs.append(df)
        except Exception as e:
            print(f"Error reading {cat_file}: {e}")

    if all_catalogs:
        # Combine all catalogs into a single master DataFrame
        master_catalog = pd.concat(all_catalogs, ignore_index=True)
        print(f"Combined {len(master_catalog)} rows from {len(all_catalogs)} catalogs.")

        # --- Perform custom analysis on master_catalog ---
        # Example: Find all detections of a specific object across images
        # target_coords = (123.456, 45.678)
        # matched_target = master_catalog[
        #    (np.abs(master_catalog['ra'] - target_coords[0]) < tolerance) &
        #    (np.abs(master_catalog['dec'] - target_coords[1]) < tolerance)
        # ]
        # print(matched_target[['source_image_base', 'aperture_calib_mag']])
        #
        # Example: Save the combined catalog
        # master_catalog.to_csv('combined_rpp_catalog.csv', index=False)
    else:
        print("No catalogs found or loaded.")


Advanced Visualization
--------------------

PFR output catalogs (`*_catalog.csv`) can be visualized using advanced Python libraries like `matplotlib`, `seaborn`, or `plotly` for deeper insights.

Example using `matplotlib` for a density scatter plot:

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    # Load catalog
    catalog_file = 'rppyour_image_base_name_catalog.csv'
    catalog = pd.read_csv(catalog_file)

    # Filter out invalid magnitudes if necessary
    catalog = catalog[catalog['aperture_calib_mag'].notna()]

    if catalog.empty:
        print("No valid data for plotting.")
        exit()

    # Create density scatter plot of magnitude vs. position (X)
    x = catalog['xcenter']
    y = catalog['aperture_calib_mag'] # Use calibrated magnitude

    # Calculate point density
    try:
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)
        # Sort points by density, so dense points don't overplot sparse ones
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    except np.linalg.LinAlgError:
        print("Could not calculate density (singular matrix). Plotting without density.")
        z = None # Fallback: plot without density coloring

    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot
    scatter = ax.scatter(x, y, c=z, s=10, edgecolor='', cmap='viridis', alpha=0.7)

    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Calibrated Magnitude')
    ax.set_title('Magnitude vs. X Position (Density Colored)')
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis() # Magnitudes: fainter is larger number

    if z is not None:
        cbar = fig.colorbar(scatter)
        cbar.set_label('Point Density')

    plt.tight_layout()
    plt.savefig('magnitude_vs_x_density.png', dpi=150)
    print("Density plot saved to magnitude_vs_x_density.png")
