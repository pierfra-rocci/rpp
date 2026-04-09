API Reference
=============

This section summarizes the main Python modules and HTTP endpoints that make up
RAPAS Photometry Pipeline. The project is primarily used through the Streamlit
frontend, but several modules and backend routes are also useful for
programmatic access.

Application Modules
-------------------

Frontend and UI
^^^^^^^^^^^^^^^

``pages/app.py``
   Main Streamlit workflow for FITS upload, parameter selection, astrometry,
   photometry, cross-matching, result export, and transient finding.

``pages/login.py``
   Login, registration, password recovery, and backend selection workflow.

``pages/api_client.py``
   Thin HTTP client used by the Streamlit frontend to talk to the FastAPI
   backend. Includes backend auto-detection logic.

Science and Processing Modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``src/pipeline.py``
   Core photometry helpers including masking, cosmic-ray handling, and other
   lower-level processing helpers.

``src/astrometry.py``
   Plate solving and WCS-related utilities using ``stdpipe`` and local
   Astrometry.net tooling.

``src/psf.py``
   PSF photometry helpers and PSF model construction.

``src/xmatch_catalogs.py``
   Catalog cross-matching and catalog enhancement, including GAIA, SIMBAD,
   SkyBoT, and related services.

``src/transient.py``
   Transient candidate search, template comparison, and extra filtering such as
   Solar System object rejection.

``src/tools_pipeline.py``
   Shared photometry and WCS utilities such as filter mapping, pixel scale
   extraction, background estimation, and calibrated magnitude helpers.

``src/tools_app.py``
   Streamlit-side helper functions including FITS loading, cache reset, and UI
   support routines.

Database and API Modules
^^^^^^^^^^^^^^^^^^^^^^^^

``api/main.py``
   FastAPI application with health, authentication, config persistence, and
   FITS upload/listing endpoints.

``api/models.py``
   SQLAlchemy ORM models for users, FITS storage, recovery codes, and tracking
   tables.

``api/schemas.py``
   Pydantic request and response models for the FastAPI service.

``api/storage.py``
   Storage path generation for uploaded FITS files and related backend storage
   conventions.

Useful Functions
----------------

``src.tools_app.load_fits_data(file)``
   Load image data and headers from a FITS file with robust handling for empty,
   invalid, multi-extension, or multi-dimensional FITS content.

``src.tools_pipeline.safe_wcs_create(header)``
   Attempt to build a WCS object from a FITS header while returning structured
   error information and log messages.

``src.tools_pipeline.extract_pixel_scale(header)``
   Extract or estimate the pixel scale used by later photometric and
   astrometric steps.

``src.pipeline.make_border_mask(image, border=50, invert=True, dtype=bool)``
   Build a border mask for detection workflows.

``src.astrometry.solve_with_astrometrynet(file_path)``
   Run local astrometric solving with the configured Astrometry.net/stdpipe
   stack.

``src.xmatch_catalogs.cross_match_with_gaia(...)``
   Cross-match measured sources with calibration catalogs and prepare data for
   zero-point estimation and later enrichment.

``src.xmatch_catalogs.enhance_catalog(...)``
   Enrich the photometry catalog with additional catalog matches such as
   SIMBAD, SkyBoT, and Astro-Colibri.

``src.transient.find_candidates(...)``
   Run the transient candidate workflow after photometry and catalog
   preparation.

Photometric Formula Conventions
-------------------------------

The project documentation and code use the following formulas consistently:

- ``S/N = flux / flux_error``
- ``σ_mag = 1.0857 × (σ_flux / flux)``
- ``σ_mag_calib = √(σ_mag_inst² + σ_zp²)``

HTTP API Endpoints
------------------

The FastAPI backend in ``api/main.py`` currently exposes these primary routes:

- ``GET /health``
- ``POST /api/register``
- ``POST /api/login``
- ``POST /api/recovery/request``
- ``POST /api/recovery/confirm``
- ``GET /api/config``
- ``POST /api/config``
- ``POST /api/upload/fits``
- ``GET /api/fits``

Maintenance and Migration Scripts
---------------------------------

The repository also contains command-line helpers in ``scripts/``:

- ``scripts/migrate_add_wcs_zip_tables.py``
- ``scripts/migrate_legacy_db.py``
- ``scripts/satellite_trail_detector.py``

These are maintenance or standalone utilities and are not all part of the main
interactive Streamlit workflow.
