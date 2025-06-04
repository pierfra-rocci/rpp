# RAPAS Photometry Pipeline (RPP)

A comprehensive web-based astronomical photometry pipeline built with Streamlit, designed for automated stellar photometry and catalog cross-matching.

## Features

### Core Photometry Capabilities
- **Automated Source Detection**: Uses DAOStarFinder for robust stellar source detection
- **Multi-Aperture Photometry**: Performs aperture photometry with multiple radii (1.5×, 2.0×, 2.5×, 3.0× FWHM)
- **PSF Photometry**: Builds empirical PSF models using EPSFBuilder for precise photometry
- **Background Estimation**: 2D background modeling using SExtractor algorithm
- **FWHM Estimation**: Automatic stellar FWHM determination through Gaussian fitting

### Astrometric Solutions
- **Plate Solving**: Integration with SIRIL for automatic astrometric calibration
- **WCS Refinement**: Optional astrometry refinement using stdpipe and Gaia DR3
- **Header Validation**: Automatic FITS header fixing and WCS validation
- **Coordinate Systems**: Support for multiple coordinate reference systems

### Photometric Calibration
- **Gaia DR3 Integration**: Automatic cross-matching with Gaia DR3 catalog
- **Zero Point Calculation**: Robust photometric calibration with outlier rejection
- **Atmospheric Extinction**: Automatic airmass calculation and extinction correction
- **Multiple Filter Bands**: Support for Gaia G, BP, RP bands and synthetic photometry

### Advanced Processing
- **Cosmic Ray Removal**: L.A.Cosmic algorithm implementation using astroscrappy
- **Data Quality Filtering**: Multiple quality filters for reliable photometry
- **Multi-Format Support**: Handles various FITS formats including multi-extension files
- **Image Enhancement**: ZScale visualization and histogram equalization

### Catalog Cross-Matching
- **SIMBAD Integration**: Object identification and classification
- **SkyBoT**: Solar system object detection
- **AAVSO VSX**: Variable star cross-matching
- **Astro-Colibri**: Transient event database queries
- **VizieR Catalogs**: Quasar and specialized catalog access

### Interactive Visualization
- **Aladin Lite Integration**: Interactive sky viewer with catalog overlays
- **Statistical Plots**: Magnitude distributions and error analysis
- **ESA Sky Links**: Direct links to external sky surveys
- **Real-time Progress**: Live processing updates and status monitoring

### User Management & Configuration
- **Multi-User Support**: Individual user accounts with isolated workspaces
- **Configuration Persistence**: Save and restore analysis parameters
- **Observatory Profiles**: Custom observatory location settings
- **API Key Management**: Secure storage of external service credentials

### Data Management
- **Automated Archiving**: ZIP archive creation for result downloads
- **File Organization**: Structured output with timestamps and metadata
- **Log Generation**: Comprehensive processing logs
- **Cleanup Automation**: Automatic cleanup of temporary and old files

## Installation

### Prerequisites
- Python 3.10 or higher
- SIRIL (for plate solving functionality)
- Modern web browser (for Streamlit interface)

### Required Python Packages
```bash
pip install streamlit astropy photutils astroquery numpy pandas matplotlib
pip install astroscrappy stdpipe requests flask flask-cors werkzeug
```

### Optional Dependencies
```bash
pip install importlib-metadata  # For frozen/packaged applications
```

### External Tools
- **SIRIL**: Required for plate solving functionality
  - Install from [https://siril.org/](https://siril.org/)
  - Ensure `siril-cli` is available in system PATH

## Usage

### Starting the Application
```bash
# Start the backend server
python backend.py

# Start the frontend (in a separate terminal)
streamlit run frontend.py
```

### Basic Workflow
1. **Login/Registration**: Create account or login with existing credentials
2. **Upload FITS File**: Select astronomical image for processing
3. **Configure Parameters**: Set observatory location and analysis parameters
4. **Run Analysis**: Execute automated photometry pipeline
5. **Review Results**: Examine catalogs, plots, and cross-matches
6. **Download Data**: Export results as ZIP archives

### Configuration Options

#### Observatory Settings
- **Name**: Observatory identifier
- **Latitude/Longitude**: Geographic coordinates (decimal degrees)
- **Elevation**: Height above sea level (meters)

#### Analysis Parameters
- **Seeing (FWHM)**: Initial stellar seeing estimate (1.0-6.0 arcsec)
- **Detection Threshold**: Source detection sigma threshold (0.5-4.5)
- **Border Mask**: Edge exclusion zone (pixels)
- **Filter Band**: Gaia photometric band for calibration
- **Maximum Magnitude**: Faint limit for calibration stars

#### Processing Options
- **Refine Astrometry**: Use stdpipe for WCS improvement
- **Remove Cosmic Rays**: Apply L.A.Cosmic algorithm
- **Cosmic Ray Parameters**: Gain, read noise, sigma clipping settings

## Technical Architecture

### Backend Components
- **Flask Server**: RESTful API for user management and configuration
- **Database**: User credentials and settings storage
- **File Management**: Secure file handling and workspace isolation

### Frontend Components
- **Streamlit Interface**: Interactive web application
- **Session Management**: Stateful user sessions and data persistence
- **Real-time Updates**: Live progress monitoring and error handling

### Processing Pipeline
1. **Image Loading**: FITS file parsing with multi-extension support
2. **Header Processing**: WCS validation and coordinate extraction
3. **Preprocessing**: Optional cosmic ray removal and background estimation
4. **Source Detection**: DAOStarFinder with configurable parameters
5. **Photometry**: Multi-aperture and PSF photometry
6. **Astrometry**: Optional plate solving and WCS refinement
7. **Calibration**: Gaia cross-matching and zero point calculation
8. **Enhancement**: Multi-catalog cross-matching and object identification
9. **Output**: Structured results with logs and visualizations

### Data Flow
```
FITS Image → Header Analysis → Preprocessing → Source Detection
     ↓
Photometry ← Astrometry ← Background Estimation ← Quality Filtering
     ↓
Gaia Cross-match → Zero Point → Calibrated Magnitudes
     ↓
Multi-Catalog Cross-match → Final Catalog → Results Export
```

## File Structure
```
rpp/
├── src/
│   ├── __version__.py         # Version information
│   ├── tools.py              # Utility functions
│   ├── pipeline.py           # Core processing pipeline
│   ├── plate_solve.ps1       # Windows plate solving script
│   └── plate_solve.sh        # Linux/macOS plate solving script
├── pages/
│   ├── app.py               # Main application interface
│   └── login.py             # User authentication
├── backend.py               # Flask server
├── frontend.py             # Streamlit entry point
├── README.md               # This file
└── LICENSE                 # MIT License
```

## Output Files

### Generated Results
- **Photometry Catalog**: CSV file with source measurements
- **Processing Log**: Detailed execution log
- **Plots**: Magnitude distributions and error analysis
- **Headers**: FITS header dumps and WCS solutions
- **Archives**: ZIP files containing all results

### Catalog Columns
- **Coordinates**: RA, Dec (decimal degrees)
- **Photometry**: Multi-aperture and PSF magnitudes
- **Uncertainties**: Photometric errors and SNR
- **Astrometry**: Pixel coordinates and WCS information
- **Cross-matches**: SIMBAD, Gaia, variable star identifications
- **Metadata**: Zero point, airmass, processing parameters

## API Integration

### Supported Services
- **Gaia DR3**: ESA's stellar catalog
- **SIMBAD**: CDS astronomical database
- **VizieR**: CDS catalog service
- **SkyBoT**: IMCCE solar system service
- **AAVSO VSX**: Variable star database
- **Astro-Colibri**: Transient event platform

### Rate Limiting
- Automatic query throttling to respect service limits
- Batch processing for multiple object queries
- Error handling and retry mechanisms

## Performance Considerations

### Optimization Features
- **Caching**: Streamlit caching for expensive operations
- **Memory Management**: Efficient handling of large FITS files
- **Parallel Processing**: Multi-threaded operations where applicable
- **Resource Cleanup**: Automatic temporary file management

### Scalability
- **Multi-User**: Isolated user workspaces
- **Concurrent Sessions**: Support for multiple simultaneous users
- **Storage Management**: Automatic cleanup of old results

## Error Handling

### Robust Processing
- **Input Validation**: Comprehensive parameter checking
- **Graceful Degradation**: Fallback options for failed operations
- **Detailed Logging**: Complete processing audit trails
- **User Feedback**: Clear error messages and recovery suggestions

### Common Issues
- **WCS Problems**: Automatic header fixing and plate solving fallback
- **Network Timeouts**: Retry mechanisms for catalog queries
- **Memory Limitations**: Efficient image processing algorithms
- **File Permissions**: Secure temporary file handling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License allows you to use, modify, and distribute this software, provided you include the original copyright notice and disclaimer.

## Acknowledgements

### Core Dependencies
- [Astropy](https://www.astropy.org/) - Core astronomical functionality
- [Photutils](https://photutils.readthedocs.io/) - Photometry algorithms
- [Astroquery](https://astroquery.readthedocs.io/) - Astronomical database access
- [Streamlit](https://streamlit.io/) - Web interface framework
- [stdpipe](https://github.com/karpov-sv/stdpipe) - Astrometric calibration
- [astroscrappy](https://github.com/astropy/astroscrappy) - Cosmic ray removal

### External Services
- [RAPAS](https://rapas.imcce.fr/) - Project support and feedback
- [Astro-Colibri](https://astro-colibri.science/) - Transient event data API
- [SIRIL](https://siril.org/) - Plate solving functionality
- [Gaia DR3](https://www.cosmos.esa.int/web/gaia/dr3) - Stellar catalog
- [SIMBAD](http://simbad.u-strasbg.fr/simbad/) - Astronomical database
- [VizieR](https://vizier.u-strasbg.fr/) - Catalog service
- [SkyBoT](https://ssp.imcce.fr/webservices/skybot/) - Solar system objects
- [AAVSO VSX](https://www.aavso.org/vsx/) - Variable star catalog

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the pipeline.

## Support

For questions, issues, or feature requests, please contact the development team or create an issue in the project repository.
