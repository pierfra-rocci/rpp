User Guide
=========

Getting Started
--------------

Photometry Factory for RAPAS provides a streamlined workflow for astronomical image analysis through a user-friendly web interface. This guide walks you through the complete process from image upload to final photometric catalog generation.

Application Layout
-----------------

.. image:: _static/app_layout.png
   :width: 700px
   :alt: Application Layout

The interface is divided into two main sections:

* **Left Sidebar**: Contains file uploaders, configuration options, and analysis parameters
* **Main Panel**: Displays results, visualizations, and interactive elements

Step 1: Uploading Files
----------------------

Begin by uploading your astronomical images using the file uploaders in the sidebar:

1. **Science Image** (required): Your main astronomical image in FITS format
2. **Master Bias** (optional): Bias calibration frame
3. **Master Dark** (optional): Dark current calibration frame
4. **Master Flat** (optional): Flat field calibration frame

.. note::
   The application accepts standard FITS formats (.fits, .fit, .fts) including compressed variants.

Step 2: Image Calibration
------------------------

If you've uploaded calibration frames, you can apply them to your science image:

1. Select the calibration steps you wish to apply (bias, dark, flat field)
2. Click the "Run Image Calibration" button
3. View the calibrated image displayed in the main panel

The application automatically handles exposure time scaling for dark frames and normalization of flat fields.

Step 3: WCS Determination
-----------------------

Accurate World Coordinate System (WCS) information is essential for photometry and catalog matching:

1. The application first attempts to read WCS from the FITS header
2. If WCS is missing or invalid, you can:
   * Enter RA/Dec coordinates manually
   * Use the astrometry.net service for plate solving (requires an API key)

.. tip::
   For plate solving with astrometry.net, enter your API key in the sidebar field.
   If you don't have one, you can register for free at nova.astrometry.net.

Step 4: Photometry and Analysis
-----------------------------

Once your image is calibrated and WCS is determined:

1. Click "Run Zero Point Calibration" to start the analysis pipeline
2. The application will:
   * Detect sources in your image
   * Measure their positions and fluxes
   * Cross-match with the Gaia DR3 catalog
   * Calculate photometric zero point
   * Generate calibrated magnitudes for all sources

Advanced configuration options in the sidebar allow you to adjust:

* Source detection threshold
* Border mask size
* Seeing estimate
* Gaia magnitude range for calibration

Step 5: Reviewing Results
-----------------------

After processing, the application provides comprehensive results:

* **Calibrated Science Image**: View the processed image
* **Source Catalog**: Interactive table of detected sources
* **Calibration Plot**: Zero point determination visualization
* **Cross-matched Catalog**: Sources identified in other catalogs
* **Aladin Viewer**: Interactive sky map with your catalog overlaid

Step 6: Exporting Data
--------------------

Your analysis results are automatically saved to the `pfr_results` directory and can be downloaded directly:

* **CSV Catalog**: Complete photometry results with calibrated magnitudes
* **Metadata File**: Analysis parameters and observation details
* **PSF Model**: The fitted PSF model as a FITS file
* **Log File**: Complete processing log

Using the Interactive Aladin Viewer
----------------------------------

The embedded Aladin Lite viewer allows you to explore your field with detected sources marked:

* Pan and zoom the image using mouse controls
* Click on marked sources to view detailed information
* Toggle between different sky surveys using the layer control
* Search for specific objects using the search box

Best Practices and Tips
----------------------

* **Image Quality**: Higher quality science images yield better photometric results
* **Calibration Frames**: Using proper calibration frames significantly improves accuracy
* **Detection Parameters**: Adjust threshold and FWHM according to your image characteristics
* **Gaia Magnitude Range**: Set appropriate magnitude limits to get good calibration stars
* **Border Mask**: Increase the border mask if your image has edge artifacts

Additional Resources
------------------

* See the Examples section for walkthrough tutorials
* Check Advanced Features for specialized use cases
* Refer to Troubleshooting for common issues and solutions