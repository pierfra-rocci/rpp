# Photometry Factory for RAPAS (PFR)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/gallery)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive astronomical image processing and photometry tool designed specifically for RAPAS data.

![PFR Screenshot](doc/_static/pfr_screenshot.png)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
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
    
- **Advanced Astrometry**
  - WCS coordinate determination from image headers
  - Automatic plate solving via astrometry.net API integration
  - Manual coordinate entry for challenging fields
    
- **Comprehensive Photometry Tools**
  - Aperture photometry with configurable parameters
  - PSF photometry with automatic PSF modeling and visualization
  - Zero-point calibration with Gaia DR3
  - Automatic airmass calculation and correction
    
- **Extensive Catalog Cross-matching**
  - Gaia DR3 source matching
  - SIMBAD object identification with object types
  - SkyBoT solar system object detection
  - AAVSO Variable Star cross-matching
  - Quasar identification from VizieR VII/294 catalog
  - Astro-Colibri source cross-matching
    
- **Interactive Visualization**
  - Image display with adjustable scaling
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
  - **New:** Unified download of all results as a ZIP archive

## Installation

### Requirements

- Python 3.8 or later
- Key dependencies:
  - astropy
  - photutils
  - astroquery
  - matplotlib
  - numpy
  - pandas
  - streamlit

### Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pfr.git
   cd pfr
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run pfr_app.py
   ```

## Usage

1. **Upload your FITS files**
   - Required: Science image
   - Optional: Bias, dark, and flat field calibration frames

2. **Configure parameters**
   - Set seeing estimate
   - Adjust detection threshold
   - Configure border mask size
   - Select Gaia parameters for calibration

3. **Run the analysis pipeline**
   - Apply calibration frames if needed
   - Run zero-point calibration
   - View results and interactive visualizations

4. **Export and analyze results**
   - Download photometry catalog
   - Examine cross-matched objects
   - Link directly to online resources for further analysis

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
- `[filename].log` - Processing log
- `[filename]_image.png` - Preview image
- `[filename]_epsf.fits` - PSF model file

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
4. For bugs, include:
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - System information (OS, Python version, etc.)
   - Screenshots if applicable

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This tool relies on numerous open-source astronomy packages including Astropy, Photutils, and Astroquery
- Special thanks to the RAPAS team for their support and feedback

## Contact

Pier-Francesco Rocci - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/pfr](https://github.com/yourusername/pfr)

---

Created with ❤️ by Pier-Francesco Rocci
