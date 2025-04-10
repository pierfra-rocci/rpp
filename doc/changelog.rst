Changelog
=========

This document records all notable changes to Photometry Factory for RAPAS.

Version 1.1.0 (Upcoming)
----------------------

**New Features**
- Cross-matching with multiple astronomical catalogs (SIMBAD, SkyBot, AAVSO VSX)
- Interactive Aladin Lite sky viewer integration
- Automatic PSF model construction and PSF photometry
- Enhanced zero-point calibration with sigma clipping

**Improvements**
- Improved WCS refinement using Gaia DR3
- More robust source detection in crowded fields
- Better error handling for network services
- Performance optimizations for large images

**Bug Fixes**
- Fixed memory leak when processing multiple large images
- Resolved issue with coordinate transformations near celestial poles
- Corrected zero point calculation for images with small number of reference stars
- Fixed display errors in magnitude histograms

Version 1.0.0 (2023-09-01)
----------------------

**Features**
- FITS image calibration (bias, dark, flat)
- Automatic plate solving with astrometry.net
- Source detection and aperture photometry
- Zero-point calibration with Gaia DR3
- CSV catalog export
- Image visualization with matplotlib
- User-friendly Streamlit interface

**Requirements**
- Python 3.8+
- Required packages: astropy, photutils, astroquery, streamlit, numpy, pandas, matplotlib

Version 0.9.0 (2023-07-15)
----------------------

**Initial Beta Release**
- Core functionality implemented
- Basic calibration and photometry workflow
- Preliminary documentation
