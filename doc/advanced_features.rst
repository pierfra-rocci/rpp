Advanced Features
===============

This section covers advanced features and specialized use cases for Photometry Factory for RAPAS.

PSF Photometry
------------

While aperture photometry works well for isolated stars, PSF (Point Spread Function) photometry offers advantages for:

* Crowded fields where stars overlap
* Faint sources where aperture photometry is noise-limited
* Variable seeing conditions across the image

The application automatically performs both aperture and PSF photometry, and you can compare the results:

1. **PSF Model Construction**: The application builds an empirical PSF model from bright, isolated stars
2. **PSF Fitting**: This model is then fitted to all detected sources

To optimize PSF photometry:

* Ensure your seeing estimate is accurate
* Verify the automatically selected PSF stars are good representatives
* Review the PSF model visualization for quality

.. code-block:: python

    # The PSF model is saved as a FITS file in your output directory
    from astropy.io import fits
    import matplotlib.pyplot as plt
    
    # Load the PSF model
    psf_data = fits.getdata('psf_model_epsf.fits')
    
    # Plot it with higher resolution
    plt.figure(figsize=(10, 8))
    plt.imshow(psf_data, origin='lower', cmap='viridis')
    plt.colorbar(label='Normalized Intensity')
    plt.title('PSF Model')
    plt.savefig('psf_model_highres.png', dpi=150)

Working with Time-Series Data
---------------------------

For variable stars, asteroids, or other time-variable objects, you can process multiple images and combine the results:

1. Process each image individually through the application
2. Collect the output catalogs
3. Match sources across catalogs and create light curves

This example script demonstrates combining multiple output catalogs:

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from astropy.time import Time
    
    # List of processed catalog files
    catalog_files = [
        'image1_phot.csv',
        'image2_phot.csv',
        'image3_phot.csv'
    ]
    
    # Observation times from FITS headers
    obs_times = [
        '2023-01-15T22:30:45',
        '2023-01-15T23:15:20',
        '2023-01-16T00:05:10'
    ]
    
    # Target coordinates (RA, Dec)
    target_coords = (183.2641, 22.0145)
    tolerance = 3.0/3600.0  # 3 arcseconds tolerance
    
    # Load catalogs and extract magnitude measurements
    times = []
    mags = []
    errs = []
    
    for i, (cat_file, obs_time) in enumerate(zip(catalog_files, obs_times)):
        catalog = pd.read_csv(cat_file)
        
        # Find the target in each catalog based on coordinates
        dist = np.sqrt((catalog['ra'] - target_coords[0])**2 + 
                       (catalog['dec'] - target_coords[1])**2)
        matches = dist < tolerance
        
        if np.any(matches):
            target_row = catalog[matches].iloc[0]
            times.append(Time(obs_time).jd)
            mags.append(target_row['aperture_calib_mag'])
            errs.append(target_row['aperture_sum_err'] * 1.0857)  # Convert to mag error
    
    # Plot light curve
    plt.figure(figsize=(12, 6))
    plt.errorbar(times, mags, yerr=errs, fmt='o-', capsize=4)
    plt.gca().invert_yaxis()  # Astronomical convention
    plt.xlabel('JD')
    plt.ylabel('Calibrated Magnitude')
    plt.title('Light Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('light_curve.png')

Differential Photometry
----------------------

For high-precision relative photometry:

1. Process your image with PFR to get a calibrated catalog
2. Select suitable comparison stars
3. Perform differential photometry relative to these stars

Example workflow:

.. code-block:: python

    import pandas as pd
    import numpy as np
    
    # Load the calibrated catalog
    catalog = pd.read_csv('output_phot.csv')
    
    # Target star index
    target_idx = 42  # Replace with your target star index
    
    # Select comparison stars with similar brightness
    target_mag = catalog.loc[target_idx, 'aperture_calib_mag']
    
    # Find stars within 1 magnitude of the target and not variable
    comp_stars = catalog[
        (abs(catalog['aperture_calib_mag'] - target_mag) < 1.0) &
        (catalog['aavso_Name'].isna()) &  # Not known variables
        (catalog.index != target_idx)
    ]
    
    # Use top 5 stars with lowest magnitude error as comparison
    comp_stars = comp_stars.nsmallest(5, 'aperture_sum_err')
    
    # Calculate mean magnitude of comparison stars
    comp_flux = np.sum(10**(-0.4 * comp_stars['aperture_calib_mag']))
    comp_mag = -2.5 * np.log10(comp_flux / len(comp_stars))
    
    # Differential magnitude
    diff_mag = catalog.loc[target_idx, 'aperture_calib_mag'] - comp_mag
    
    print(f"Target magnitude: {catalog.loc[target_idx, 'aperture_calib_mag']:.3f}")
    print(f"Comparison ensemble magnitude: {comp_mag:.3f}")
    print(f"Differential magnitude: {diff_mag:.3f}")

Custom Pipeline Integration
-------------------------

PFR can be integrated into larger pipelines by:

1. Using the application to generate calibrated catalogs
2. Importing these catalogs into your custom scripts
3. Performing additional specialized analysis

Example integration script:

.. code-block:: python

    import os
    import subprocess
    import pandas as pd
    import glob
    
    # Directory containing FITS files
    data_dir = '/path/to/data'
    fits_files = glob.glob(os.path.join(data_dir, '*.fits'))
    
    # Process each file with PFR
    for fits_file in fits_files:
        # Run PFR in headless mode
        cmd = [
            'streamlit', 'run', 'pfr_app.py', '--', 
            '--input', fits_file,
            '--no-calibration',
            '--output-dir', 'results',
            '--headless'
        ]
        subprocess.run(cmd, check=True)
    
    # Collect and combine results
    catalog_files = glob.glob('results/*_phot.csv')
    combined_data = []
    
    for cat_file in catalog_files:
        df = pd.read_csv(cat_file)
        # Add filename as reference
        df['source_file'] = os.path.basename(cat_file)
        combined_data.append(df)
    
    # Combine all catalogs
    master_catalog = pd.concat(combined_data, ignore_index=True)
    master_catalog.to_csv('master_catalog.csv', index=False)

Advanced Visualization
--------------------

PFR output can be visualized with advanced Python libraries:

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm
    from scipy.stats import gaussian_kde
    
    # Load catalog
    catalog = pd.read_csv('output_phot.csv')
    
    # Create density scatter plot of magnitude vs position
    x = catalog['xcenter']
    y = catalog['ycenter']
    z = catalog['aperture_calib_mag']
    
    # Set up plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    
    # Magnitude vs X position
    xy = np.vstack([x, z])
    density = gaussian_kde(xy)(xy)
    
    axs[0].scatter(x, z, c=density, s=5, edgecolor='')
    axs[0].set_xlabel('X Position (pixels)')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid(True, alpha=0.3)
    axs[0].invert_yaxis()
    
    # Magnitude vs Y position
    xy = np.vstack([y, z])
    density = gaussian_kde(xy)(xy)
    
    axs[1].scatter(y, z, c=density, s=5, edgecolor='')
    axs[1].set_xlabel('Y Position (pixels)')
    axs[1].set_ylabel('Magnitude')
    axs[1].grid(True, alpha=0.3)
    axs[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('position_magnitude.png', dpi=150)
