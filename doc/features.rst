Features
========

Image Processing
--------------

**FITS File Support**
   PFR supports standard FITS file formats used in astronomy. It can handle multi-extension FITS files, data cubes, and properly extracts header information.

**Image Calibration**
   - Bias subtraction
   - Dark frame correction with exposure time scaling
   - Flat field correction with normalization

**Background Estimation**
   Sophisticated background modeling using photutils.Background2D with sigma-clipping and SExtractor algorithms.

Source Detection
--------------

**DAOStarFinder Implementation**
   Uses the DAOPhot algorithm to detect point sources in astronomical images based on PSF shapes.

**FWHM Estimation**
   Automatically estimates the Full Width at Half Maximum of stars using Gaussian fitting.

**PSF Modeling**
   Creates an empirical PSF model from bright, unsaturated stars in the image.

Photometry
---------

**Aperture Photometry**
   Measures source brightness using circular apertures scaled to the seeing conditions.

**PSF Photometry**
   Measures source brightness by fitting the PSF model to each detected source.

**Error Estimation**
   Provides uncertainty estimates for photometric measurements.

Astrometry
---------

**WCS Handling**
   Extracts and validates World Coordinate System information from FITS headers.

**Plate Solving**
   Integration with astrometry.net for automatic plate solving when WCS is missing.

**Coordinate Transformations**
   Converts between pixel coordinates and celestial coordinates (RA/Dec).

Calibration
----------

**Zero Point Calculation**
   Determines photometric zero point by cross-matching with GAIA catalog.

**Extinction Correction**
   Applies atmospheric extinction correction based on airmass.

**Magnitude Systems**
   Converts between instrumental and calibrated magnitude systems.

Catalog Integration
-----------------

**GAIA DR3**
   Accesses the GAIA Data Release 3 catalog for star reference and calibration.

**SIMBAD**
   Cross-matches with SIMBAD database for object identification and classification.

**SkyBoT**
   Identifies solar system objects in the field of view.

**AAVSO VSX**
   Checks for known variable stars in the field of view.

**VizieR**
   Access to the VizieR VII/294 catalog for quasar identification.

Visualization
-----------

**Interactive Plots**
   Dynamic plots of image data, detected sources, and calibration results.

**Aladin Integration**
   Interactive sky visualization with source overlay and catalog information.

**Data Tables**
   Tabular display of photometry results with sorting and filtering options.

Output and Results
----------------

**Catalog Generation**
   Creates CSV catalogs with comprehensive source information.

**Metadata Logging**
   Records all processing parameters and results.

**Result Downloads**
   Provides downloadable ZIP archives of all generated outputs.
