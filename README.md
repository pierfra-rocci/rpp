# Photometry Factory for RAPAS (PFR)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/gallery)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive astronomical image processing and photometry tool designed specifically for RAPAS (Remote Astronomical Photometry & Astrometry System) data.

![PFR Screenshot](doc/_static/pfr_screenshot.png)

## Features

- **Complete Image Calibration Pipeline**
  - Bias, dark and flat field correction
  - Background estimation and subtraction
  
- **Advanced Astrometry**
  - WCS coordinate determination from image headers
  - Automatic plate solving via astrometry.net integration
  - Manual coordinate entry for challenging fields
  
- **Comprehensive Photometry Tools**
  - Aperture photometry with configurable parameters
  - PSF photometry with automatic PSF modeling
  - Zero-point calibration with Gaia DR3
  
- **Extensive Catalog Cross-matching**
  - Gaia DR3 source matching
  - SIMBAD object identification
  - SkyBoT solar system object detection
  - AAVSO Variable Star cross-matching
  
- **Interactive Visualization**
  - Image display with adjustable scaling
  - Embedded Aladin Lite for catalog overlays
  - Interactive tables for data exploration
  
- **Analysis and Export**
  - Comprehensive photometry catalog output
  - Detailed logging of all processing steps
  - Direct links to online astronomy resources

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
   git clone https://github.com/yourusername/P_F_R.git
   cd P_F_R
   ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:
    ```bash
    stramlit run prf_app.py
    ```

# Photometry Factory for RAPAS (PFR)

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/gallery)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive astronomical image processing and photometry tool designed specifically for RAPAS (Remote Astronomical Photometry & Astrometry System) data.

![PFR Screenshot](doc/_static/pfr_screenshot.png)

## Features

- **Complete Image Calibration Pipeline**
  - Bias, dark and flat field correction
  - Background estimation and subtraction
  
- **Advanced Astrometry**
  - WCS coordinate determination from image headers
  - Automatic plate solving via astrometry.net integration
  - Manual coordinate entry for challenging fields
  
- **Comprehensive Photometry Tools**
  - Aperture photometry with configurable parameters
  - PSF photometry with automatic PSF modeling
  - Zero-point calibration with Gaia DR3
  
- **Extensive Catalog Cross-matching**
  - Gaia DR3 source matching
  - SIMBAD object identification
  - SkyBoT solar system object detection
  - AAVSO Variable Star cross-matching
  
- **Interactive Visualization**
  - Image display with adjustable scaling
  - Embedded Aladin Lite for catalog overlays
  - Interactive tables for data exploration
  
- **Analysis and Export**
  - Comprehensive photometry catalog output
  - Detailed logging of all processing steps
  - Direct links to online astronomy resources

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
   git clone https://github.com/yourusername/P_F_R.git
   cd P_F_R
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run pfr_app.py
   ```

### Alternative Installation

A standalone Windows executable is also available:
1. Download the latest release from the releases page
2. Extract all files to a directory of your choice
3. Run `PfrApp.bat`

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
- [API Reference](doc/api_reference.rst)

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- This tool relies on numerous open-source astronomy packages including Astropy, Photutils, and Astroquery
- Special thanks to the RAPAS team for their support and feedback

---

Created with ❤️ by [Pier-Francesco Rocci]
