Advanced Features
=================

This section collects the current advanced and developer-oriented workflows in
RAPAS Photometry Pipeline.

Database Tracking System
------------------------

RPP stores analysis-history metadata in SQLite so uploaded FITS files and
generated ZIP archives can be tracked per user.

The tracking layer uses three linked tables:

.. code-block:: text

    wcs_fits_files
        Uploaded or generated FITS records tied to a user

    zip_archives
        Result ZIP archives tied to a user

    wcs_fits_zip_assoc
        Association table linking one FITS record to one or more ZIP results

This supports:

- One FITS file associated with multiple processing runs
- Per-user isolation of stored files and metadata
- Automatic recording after successful result creation
- Safe re-use of existing records when the same stored filename or archive is encountered again

Migrating Existing Databases
----------------------------

If you are upgrading an older deployment, use the migration scripts in
``scripts/``.

.. code-block:: bash

    python scripts/migrate_add_wcs_zip_tables.py --db-path /path/to/users.db
    python scripts/migrate_legacy_db.py

The tracking-table migration creates a backup unless told otherwise and is
designed to be rerunnable.

Inspecting Analysis History
---------------------------

The helper functions in ``src/db_tracking.py`` expose the recorded history:

.. code-block:: python

    from src.db_tracking import (
        get_fits_files_for_user,
        get_zip_archives_for_user,
        get_zips_for_fits,
    )

    fits_files = get_fits_files_for_user("alice")
    archives = get_zip_archives_for_user("alice")
    related_zips = get_zips_for_fits(wcs_fits_id=1)

Transient Detection
-------------------

The transient workflow is optional and runs after the main photometry and
catalog-enhancement steps.

Current behavior:

- The user enables it from the **Transient Candidates** sidebar section.
- The reference filter is selected in the UI.
- The code chooses PanSTARRS or SkyMapper based on field declination.
- Additional filtering removes known sources and Solar System objects.
- SkyBoT filtering is handled explicitly in ``src/transient.py`` with a 120-second timeout.

This workflow should be treated as best-effort. Reference data, remote catalog
services, and transient filtering may degrade gracefully when an external
dependency is unavailable.

Astrometric Workflows
---------------------

RPP supports several WCS-related paths:

- Validation and reuse of an existing WCS already present in the header
- Forced astrometry checks from the sidebar when the user wants a fresh solve attempt
- Local blind solving through Astrometry.net and ``stdpipe``
- Optional refinement workflows when the local toolchain is available

If a new forced solve fails but the file already had a valid WCS, the frontend
falls back to that valid solution so the rest of the pipeline can continue.

PSF Photometry
--------------

PSF photometry complements the aperture measurements and is useful when:

- sources are crowded,
- aperture measurements become unstable, or
- a PSF-based comparison is desirable.

The pipeline builds an empirical PSF model from suitable stars and then reports
PSF-derived fluxes and magnitudes alongside the aperture results when the fit is
successful.

Custom Post-Processing
----------------------

The generated CSV catalogs and ZIP archives can be reused in external scripts or
notebooks for tasks such as:

- light-curve construction,
- differential photometry,
- cross-run source aggregation,
- custom filtering, or
- publication-ready plotting.

For most programmatic entry points, prefer the modules summarized in
``api_reference.rst`` and the concrete pipeline helpers under ``src/``.

Standalone Utilities
--------------------

The ``scripts/`` directory contains maintenance or utility scripts that are not
part of the normal Streamlit button flow:

- ``migrate_add_wcs_zip_tables.py``
- ``migrate_legacy_db.py``
- ``satellite_trail_detector.py``

These are intended for administration, migration, or offline workflows rather
than day-to-day frontend use.