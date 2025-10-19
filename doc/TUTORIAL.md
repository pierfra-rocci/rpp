RPP Quick Tutorial
==================

This very short tutorial walks through getting the project running locally
and executing the minimal example from the documentation.

1) Clone and install

   - Open PowerShell (Windows) or a terminal (macOS/Linux)

   - Run the steps in :doc:`installation` (create virtualenv and install

   `requirements.txt`).

2. Start the backend

  RPP Quick Tutorial
  ==================

  This short tutorial shows the minimal steps to run the RAPAS Photometry
  Pipeline locally and to run the minimal example included in the docs.

  1. Clone and install

  - Open PowerShell (Windows) or a terminal (macOS/Linux).
  - Clone the repository, create a virtual environment and install dependencies:

  ```powershell
  git clone https://github.com/your-repo/rpp.git
  cd rpp
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1   # PowerShell
  pip install -r requirements.txt
  ```

  On Unix/macOS use:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

  2. Start the backend

  In one terminal run:

  ```powershell
  python backend.py
  ```

  The backend listens on port 5000 by default.

  3. Start the frontend

  In another terminal run:

  ```powershell
  streamlit run frontend.py
  ```

  Open the URL printed by Streamlit (usually http://localhost:8501).

  4. Try the minimal Python example

  - Create `run_example.py` in the project root as shown in `doc/examples.rst`.
  - Run it with:

  ```powershell
  python run_example.py
  ```

  If you do not have a FITS file, the example script will show the call
  pattern and print a friendly message.

  What's next

  - To perform real photometry, obtain one or more FITS images and re-run the
    example or use the web frontend to upload and process images.
  - For plate solving and advanced catalog cross-matching, install Astrometry.net
    and (optionally) SCAMP; obtain appropriate astrometry index files.

  Additional short guide
  ----------------------

  Common UI steps and important options:

  - Launch the Streamlit app and open the provided localhost URL.
  - Log in or register using the login page.
  - Configure observatory data in the sidebar (name, latitude, longitude,
    elevation). The app attempts to read header coordinates if present.
  - Upload a FITS file using the main uploader (supported extensions: .fits,
    .fit, .fts, .fits.gz).
  - Main analysis parameters include seeing (FWHM), detection threshold,
    border mask, calibration band, and max calibration magnitude.
  - Optional: enable "Remove Cosmic Rays" (configure camera gain/read-noise)
    or "Refine Astrometry" (requires Astrometry.net/SCAMP installed).

  Outputs and logs
  ---------------

  - Results are saved under a per-user results folder: `<username>_rpp_results/`.
  - Typical outputs: `*_catalog.csv`, `*_bkg.fits`, `*_psf.fits`, `*_image.png`,
    `*.log`, and a `*.zip` archive of the run.
  - Check the per-run log file (`<base_filename>.log`) for diagnostic details.

  Support
  -------

  - If you encounter issues, collect logs and open a repository issue with a
    short reproducible example.
  - For Astrometry.net and SCAMP installation consult their official docs.


