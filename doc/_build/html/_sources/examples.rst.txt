// filepath: c:\Users\pierf\P_F_R\doc\examples.rst
Examples
========

This section provides step-by-step examples demonstrating the capabilities of Photometry Factory for RAPAS with real astronomical data.

Example 1: Basic Photometry Workflow
----------------------------------

This example demonstrates the standard workflow using a sample image of the M67 open cluster.

Step 1: Data Preparation
^^^^^^^^^^^^^^^^^^^^^^^

Download the example data files:
- M67_science.fits (science image)
- M67_bias.fits (master bias)
- M67_dark.fits (master dark)
- M67_flat.fits (master flat)

Step 2: Upload and Calibrate
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Launch the application and upload the files using the sidebar uploaders
2. Enable all calibration options (bias, dark, flat)
3. Click "Run Image Calibration"

.. image:: _static/example1_calibration.png
   :width: 600px
   :alt: Image Calibration

Step 3: Run Photometry Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configure the analysis parameters:
- Seeing: 3.0 arcsec
- Detection Threshold: 4.0σ
- Border Mask: 50 pixels
- Gaia Band: phot_g_mean_mag
- Gaia Magnitude Range: 12-18

Click "Run Zero Point Calibration" to analyze the image.

Step 4: Review Results
^^^^^^^^^^^^^^^^^^^^

The application will generate:

1. A source catalog with ~150 stars
2. Zero point of approximately 22.4 ± 0.05 mag
3. Cross-matched identifications with SIMBAD and AAVSO

.. image:: _static/example1_results.png
   :width: 600px
   :alt: Photometry Results

Example 2: Analyzing a Variable Star Field
----------------------------------------

This example focuses on analyzing a field containing known variable stars.

Step 1: Upload Data
^^^^^^^^^^^^^^^^^

Use the included example file `variable_field.fits`.

Step 2: Configure Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Enable automatic WCS determination through astrometry.net
- Set the detection threshold to 3.0σ
- Adjust Gaia magnitude range to 10-17

Step 3: Interpret Results
^^^^^^^^^^^^^^^^^^^^^^^

After processing, examine the cross-matches with the AAVSO VSX catalog:

.. code-block:: python
    
    # Sample code to filter the output catalog for variable stars
    import pandas as pd
    
    catalog = pd.read_csv('variable_field_phot.csv')
    variables = catalog[catalog['aavso_Name'].notna()]
    
    print(f"Found {len(variables)} variable stars in the field")
    print(variables[['aavso_Name', 'aavso_Type', 'aperture_calib_mag']])

Example 3: Asteroid Photometry
----------------------------

This example demonstrates how to use PFR for asteroid brightness measurements.

Step 1: Data Setup
^^^^^^^^^^^^^^^^

Upload a series of images containing a moving asteroid.

Step 2: Special Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Use the SkyBoT integration to automatically identify solar system objects
- Adjust the detection threshold to 2.5σ to catch fainter objects
- Set the border mask to 25 pixels to utilize more of the image area

Step 3: Analyze Results
^^^^^^^^^^^^^^^^^^^^^

The application will identify the asteroid in each image and provide:

1. Calibrated magnitude measurements
2. Cross-match with the SkyBoT database providing asteroid identification
3. Optional light curve generation by combining multiple measurements

.. image:: _static/example3_asteroid.png
   :width: 600px
   :alt: Asteroid Photometry

Advanced Example: Crowded Field Photometry
----------------------------------------

This example demonstrates how to analyze images of dense star fields like globular clusters.

Step 1: Configuration
^^^^^^^^^^^^^^^^^^^

- Upload a high-resolution image of a globular cluster
- Enable PSF photometry which works better in crowded fields
- Increase the detection threshold to reduce false positives

Step 2: Specialized Settings
^^^^^^^^^^^^^^^^^^^^^^^^^^

Adjust the following parameters for crowded field analysis:

.. code-block:: python
    
    # Recommended parameters for crowded fields
    threshold_sigma = 5.0  # Higher threshold to avoid noise detections
    detection_mask = 30    # Smaller mask to utilize more of the image
    seeing = 2.5           # Typical seeing in arcseconds

Step 3: Interpretation
^^^^^^^^^^^^^^^^^^^^

Compare the results of aperture vs. PSF photometry:

1. PSF photometry typically detects 20-30% more stars in crowded regions
2. Magnitudes from PSF fitting are less affected by crowding
3. Review the PSF model quality to ensure proper fitting

Using the exported data with other tools:

.. code-block:: python
    
    # Example of loading results into Python for further analysis
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Load the catalog
    cat = pd.read_csv('globular_cluster_phot.csv')
    
    # Create a color-magnitude diagram if multiple filters were analyzed
    plt.figure(figsize=(8, 10))
    plt.scatter(cat['b_mag'] - cat['v_mag'], cat['v_mag'], s=3, alpha=0.7)
    plt.gca().invert_yaxis()  # Astronomical convention
    plt.xlabel('B-V Color')
    plt.ylabel('V Magnitude')
    plt.title('Color-Magnitude Diagram')
    plt.grid(True, alpha=0.3)
    plt.savefig('cmd_diagram.png')