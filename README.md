# RAPAS Photometry Factory (RPF)

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/gallery)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive astronomical image processing and photometry tool designed specifically for [RAPAS](https://rapas.imcce.fr/) project data.

![PFR Screenshot](doc/_static/logo.png)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Authentication & Backend](#authentication--backend)
- [Documentation](#documentation)
- [Example Output](#example-output)
- [Contributing](#contributing)
- [Reporting Issues](#reporting-issues)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Features

- **Complete Image Calibration Pipeline**
  - Bias, dark and flat field correction with exposure time scaling
  - Background estimation and subtraction with robust outlier detection
  - Cosmic ray removal using L.A.Cosmic algorithm with configurable parameters
    
- **Advanced Astrometry**
  - WCS coordinate determination from image headers
  - Automatic plate solving via SIRIL script integration
  - Manual coordinate entry for challenging fields
    
- **Comprehensive Photometry Tools**
  - Aperture photometry with configurable parameters
  - PSF photometry with automatic PSF modeling and visualization
  - Zero-point calibration with Gaia DR3
  - Automatic airmass calculation and correction
  - Observatory information management and customization
    
- **Extensive Catalog Cross-matching**
  - Gaia DR3 source matching and calibration
  - SIMBAD object identification with object types
  - SkyBoT solar system object detection
  - AAVSO Variable Star cross-matching
  - Quasar identification from VizieR VII/294 catalog
  - Astro-Colibri source cross-matching for transient events
    
- **Interactive Visualization**
  - Image display with multiple scaling options
  - Embedded Aladin Lite for DSS2 color overlays and catalog visualization
  - Interactive tables for data exploration
  - Direct links to ESA Sky and other astronomy resources
  - Real-time photometry updates with live plots during pipeline execution
  - One-click access to online astronomical databases and services
    
- **Analysis and Export**
  - Comprehensive photometry catalog output
  - Detailed logging of all processing steps
  - Export of PSF models as FITS files
  - Metadata files with analysis parameters
  - Unified download of all results as a ZIP archive

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
  - SIRIL 1.2.6+ (for plate solving)

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

3. Start the backend server (for authentication and config storage):
   ```bash
   python backend.py
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
   - Optional: Bias, dark, and flat field calibration frames

3. **Configure parameters**
   - Set seeing estimate, detection threshold, border mask size, observatory info, Gaia calibration parameters, and API keys as needed.

4. **Run the analysis pipeline**
   - Apply calibration, cosmic ray removal, plate solving, source detection, photometry, zero-point calibration, and catalog cross-matching.
   - View results and interactive visualizations (including Aladin Lite and ESA Sky links).

5. **Export and analyze results**
   - Download photometry catalog, metadata, and all results as a ZIP archive from the `pfr_results` directory.
   - Session parameters can be saved to both the backend and as a JSON file.

## Authentication & Backend

- The backend (`backend.py`) is a Flask server handling user registration, login, password recovery, and config storage.
- User data is stored in `users.db` (SQLite).
- The frontend communicates with the backend via HTTP (default: `http://localhost:5000`).
- Utility scripts: `hash_passwords.py` (password hashing), `tools.py` (misc helpers).

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
- `[filename]_image.png` - Preview image
- `[filename]_psf.fits` - PSF model file
- `[filename]_metadata.txt` - Analysis parameters and results
- `[filename]_wcs_header.txt` - WCS solution from plate solving
- `[filename]_zero_point_plot.png` - Visualization of zero-point calibration

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License allows you to use, modify, and distribute this software for both private and commercial purposes, provided you include the original copyright notice and disclaimer.

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
