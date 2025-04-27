# RAPAS Photometry Pipeline (RPP)

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/gallery)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive astronomical image processing and photometry tool designed specifically for [RAPAS](https://rapas.imcce.fr/) project data.

![RPP Screenshot](doc/_static/logo.png)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Workflow](#workflow)
- [Authentication & Backend](#authentication--backend)
- [Documentation](#documentation)
- [Example Output](#example-output)
- [Key Updates](#key-updates)
- [Contributing](#contributing)
- [Reporting Issues](#reporting-issues)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Features

- **Authentication**: Secure login, registration, and password reset with session state management.
- **User Configuration**: Save and load user-specific analysis parameters, observatory settings, and API keys.
- **Pre-processing Options**: Toggle "Astrometry++" (stdpipe refinement) and "Remove Cosmic Rays" (L.A.Cosmic algorithm) before analysis.
- **Interactive Analysis**: All parameters (seeing, detection threshold, border mask, Gaia band/mag range, etc.) are adjustable via sidebar widgets.
- **Catalog Cross-matching**: Automatic cross-match with Gaia DR3, SIMBAD, SkyBoT, AAVSO VSX, and Milliquas catalogs. Astro-Colibri API integration for transient events.
- **Results Download**: All output files for a session can be downloaded as a ZIP archive. Detailed logs are generated for each analysis.
- **Session State**: All parameters and results are managed via Streamlit session state for persistence and reproducibility.
- **FWHM Distribution Plot**: Automatically generates and saves a histogram of star FWHM values (`[filename]_fwhm.png`) for quality assessment of image seeing and focus.


## Installation

### Requirements

- Python 3.11 or later
- Key dependencies:
  - astropy
  - photutils
  - astroquery
  - astroscrappy
  - matplotlib
  - numpy
  - pandas
  - streamlit
  - stdpipe
  - Flask (for backend)
  - SIRIL 1.4.0+ (for plate solving)

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/pierfra-rocci/pfr.git
   cd pfr
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the backend server (use backend_prod.py for production):
   ```bash
   python backend_dev.py
   ```

4. In a new terminal, start the frontend Streamlit app:
   ```bash
   streamlit run run_frontend.py
   ```

   This will redirect you to the login page. Register or log in to access the main app.

## Usage


1. **Login/Register**
   - On first use, register a new account. Credentials are stored in `users.db` (SQLite).
   - Login to access the main photometry app.

2. **Upload your FITS files**
   - Required: Science image

3. **Configure parameters**
   - Set seeing estimate, detection threshold, border mask size, observatory info, Gaia calibration parameters, and API keys as needed.

4. **Run the analysis pipeline**
   - Apply cosmic ray removal, plate solving, source detection, photometry, zero-point calibration, and catalog cross-matching.
   - View results and interactive visualizations (including Aladin Lite and ESA Sky links).
   - **New:** Inspect the FWHM distribution plot to evaluate image quality and star sharpness.

5. **Export and analyze results**
   - Download photometry catalog, metadata, and all results as a ZIP archive from the `rpp_results` directory.
   - Session parameters can be saved to both the backend and as a JSON file.


## Workflow

1. **Start the backend**: `python backend_dev.py` (required for authentication and config saving).
2. **Start the frontend**: `streamlit run run_frontend.py` (always redirects to login page).
3. **Authenticate**: Register or log in. User/session parameters are loaded from the backend if available.
4. **Upload and analyze**: Upload your science FITS file (and optional calibration frames), set parameters, and run the photometry pipeline.
5. **Results**: All outputs (catalogs, plots, logs, config) are saved in `pfr_results` and can be downloaded as a ZIP archive. Cross-matching with Gaia, SIMBAD, SkyBoT, AAVSO VSX, Milliquas, and Astro-Colibri is supported.
6. **Save configuration**: Save your analysis parameters and observatory info to the backend and as a JSON file for reproducibility.

## Authentication & Backend

- User authentication: login, registration, and password recovery are handled via the Streamlit frontend, with user data stored in `users.db` (SQLite).
- User-specific configuration: analysis parameters, observatory info, and catalog settings are saved and restored per user.
- The frontend is built with [Streamlit](https://streamlit.io/), providing interactive widgets for all analysis parameters, observatory location, and catalog settings.
- The frontend communicates with the backend via HTTPS (default: `https://localhost:5000`).
- Utility scripts: `tools.py` (misc helpers).

## Documentation

Comprehensive documentation is available in the `doc/` directory:

- [Installation Guide](doc/installation.rst)
- [User Guide](doc/user_guide.rst)

To build the documentation:
```bash
cd doc
make html
```

Then open `doc/_build/html/index.html` in your browser.

## Example Output

The application generates several output files in the pfr_results directory:

- `[filename]_phot.csv` - Photometry catalog with calibrated magnitudes
- `[filename]_header.txt` - FITS header information
- `[filename].log` - Processing log with parameter details and analysis steps
- `[filename]_image.png` - Preview image of the science image
- `[filename]_psf.fits` - PSF model file
- `[filename]_metadata.txt` - Analysis parameters and results
- `[filename]_wcs_header.txt` - WCS solution from plate solving
- `[filename]_zero_point_plot.png` - Visualization of zero-point calibration
- `[filename]_results.zip` - Downloadable archive of all output files for the session
- `[filename]_bkg.fits` - Background model FITS file
- `[filename]_image_hist.png` - Histogram of image pixel values
- `[filename]_fwhm.png` - FWHM distribution histogram (new): shows the distribution of star FWHM values for image quality assessment

## Key Updates

- The Gaia minimum magnitude parameter has been removed. Now only a maximum magnitude ("Gaia Max Magnitude") is used for Gaia source filtering in the photometric calibration workflow.
- The output includes a FWHM distribution histogram for image quality assessment.
- All results for a session can be downloaded as a single ZIP archive from the interface.
- The workflow and sidebar options have been streamlined for clarity and ease of use.

## Contributing

Contributions are welcome! Here's how you can contribute:

1. **Fork the repository** - Create your own copy of the project
2. **Create a branch** - `git checkout -b feature/amazing-feature`
3. **Make your changes** - Implement your feature or bug fix
4. **Run tests** - Ensure your changes don't break existing functionality
5. **Commit your changes** - `git commit -m 'Add some amazing feature'`
6. **Push to your branch** - `git push origin feature/amazing-feature`
7. **Open a Pull Request** - Submit your changes for review

Please make sure your code follows the project's coding style and includes appropriate documentation.

## Reporting Issues

Found a bug or have a feature request? Please submit an issue through the GitHub issue tracker:

1. Go to the [Issues page](https://github.com/pierfra-rocci/pfr/issues)
2. Click "New Issue"
3. Provide a clear title and detailed description

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE).

The MIT License allows you to use, modify, and distribute this software, provided you include the original copyright notice and disclaimer.

## Acknowledgements

- This tool relies on numerous open-source astronomy packages including:
  - [Astropy](https://www.astropy.org/) for core astronomical functionality
  - [Photutils](https://photutils.readthedocs.io/) for photometry algorithms
  - [Astroquery](https://astroquery.readthedocs.io/) for accessing astronomical databases
  - [Streamlit](https://streamlit.io/) for the web interface
  - [stdpipe](https://github.com/karpov-sv/stdpipe) for astrometric calibration
  - [astroscrappy](https://github.com/astropy/astroscrappy) for cosmic ray removal
- Thanks to the [RAPAS](https://rapas.imcce.fr/) team for support and feedback
- Thanks to the [Astro-Colibri](https://astro-colibri.science/#/) for API access to transient event data
- Thanks to [SIRIL](https://siril.org/) for the plate-solving functionality used in this project
- Thanks to the various catalog services that power this tool:
  - [Gaia DR3](https://www.cosmos.esa.int/web/gaia/dr3)
  - [SIMBAD](http://simbad.u-strasbg.fr/simbad/)
  - [VizieR](https://vizier.u-strasbg.fr/)
  - [SkyBoT](https://ssp.imcce.fr/webservices/skybot/)
  - [AAVSO VSX](https://www.aavso.org/vsx/)

---
Created with ❤️ by Pier-Francesco Rocci and helped by GitHub Copilot
