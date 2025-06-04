Installation Guide
=================

System Requirements
------------------

**Python Requirements**:
   - Python 3.8 or higher
   - pip package manager
   - Virtual environment (recommended)

**External Dependencies**:
   - SIRIL astronomical image processing software
   - Modern web browser (Chrome, Firefox, Safari, Edge)
   - Internet connection for catalog queries

**Operating System Support**:
   - Windows 10/11 (with PowerShell)
   - macOS 10.15 or later
   - Linux (Ubuntu 18.04+ or equivalent)

Installation Steps
-----------------

1. **Clone or Download the Repository**:
   
   .. code-block:: bash

      git clone https://github.com/your-repo/rpp.git
      cd rpp

   Or download and extract the ZIP archive.

2. **Create Virtual Environment** (recommended):
   
   .. code-block:: bash

      python -m venv rpp-env
      
      # On Windows:
      rpp-env\Scripts\activate
      
      # On macOS/Linux:
      source rpp-env/bin/activate

3. **Install Python Dependencies**:
   
   .. code-block:: bash

      pip install -r requirements.txt

   **Core Dependencies**:
   - streamlit >= 1.28.0 (web interface)
   - astropy >= 5.0 (astronomical computations)
   - photutils >= 1.5.0 (photometry algorithms)
   - astroquery >= 0.4.6 (catalog access)
   - numpy >= 1.21.0 (numerical computations)
   - pandas >= 1.3.0 (data manipulation)
   - matplotlib >= 3.5.0 (plotting)
   - flask >= 2.0.0 (backend server)
   - flask-cors (cross-origin resource sharing)
   - werkzeug (WSGI utilities)

   **Optional Dependencies**:
   - astroscrappy (cosmic ray removal)
   - stdpipe (astrometry refinement)
   - requests (HTTP client)

4. **Install SIRIL** (required for plate solving):
   
   **Windows**:
   - Download from https://siril.org/download/
   - Install using the provided installer
   - Ensure `siril-cli` is available in system PATH

   **macOS**:
   - Install via Homebrew: `brew install siril`
   - Or download from https://siril.org/download/

   **Linux**:
   - Ubuntu/Debian: `sudo apt install siril`
   - Fedora: `sudo dnf install siril`
   - Or compile from source: https://siril.org/download/

5. **Verify SIRIL Installation**:
   
   .. code-block:: bash

      siril-cli --version

   If this command fails, ensure SIRIL is properly installed and in your system PATH.

6. **Initialize User Database**:
   The backend will automatically create `users.db` on first run. No manual initialization required.

Configuration
------------

**Environment Variables** (optional):

For email-based password recovery, set these environment variables:

.. code-block:: bash

   # Email server configuration
   export SMTP_SERVER=smtp.gmail.com
   export SMTP_PORT=587
   export SMTP_USER=your-email@gmail.com
   export SMTP_PASS=your-app-password

**File Structure**:

After installation, your directory should contain:

.. code-block:: text

   rpp/
   ├── src/
   │   ├── __version__.py      # Version information
   │   ├── tools.py           # Utility functions
   │   ├── pipeline.py        # Core processing pipeline
   │   ├── plate_solve.ps1    # Windows plate solving script
   │   └── plate_solve.sh     # Linux/macOS plate solving script
   ├── pages/
   │   ├── app.py            # Main application interface
   │   └── login.py          # User authentication
   ├── doc/                  # Documentation (Sphinx)
   ├── backend.py           # Flask server
   ├── frontend.py         # Streamlit entry point
   ├── requirements.txt    # Python dependencies
   └── README.md          # Project overview

Running the Application
----------------------

**1. Start the Backend Server**:
   
   Open a terminal and run:

   .. code-block:: bash

      python backend.py

   The server will start on http://localhost:5000 by default.

**2. Start the Frontend Application**:
   
   In a separate terminal, run:

   .. code-block:: bash

      streamlit run frontend.py

   The application will open in your default browser at http://localhost:8501.

**3. First-Time Setup**:
   - Register a new user account
   - Configure observatory settings
   - Set analysis parameters
   - Enter API keys (optional)
   - Save configuration for future sessions

API Keys and External Services
-----------------------------

Some features require API keys from external services:

**Astro-Colibri** (optional):
   - Register at https://astro-colibri.science/
   - Obtain your UID key
   - Enter in the application's API Keys section
   - Enables transient event cross-matching

**Gaia Archive** (automatic):
   - No registration required
   - Accessed via astroquery.gaia
   - Used for photometric calibration

**SIMBAD/VizieR** (automatic):
   - No registration required
   - Accessed via astroquery
   - Used for object identification

Troubleshooting Installation
---------------------------

**Common Issues**:

1. **ImportError for Python packages**:
   - Ensure virtual environment is activated
   - Reinstall requirements: `pip install -r requirements.txt --force-reinstall`
   - Check Python version compatibility

2. **SIRIL not found**:
   - Verify SIRIL installation: `siril-cli --version`
   - Check system PATH includes SIRIL directory
   - Restart terminal after SIRIL installation

3. **Permission errors**:
   - Use virtual environment to avoid system-wide installations
   - On Linux/macOS: Check file permissions for script execution
   - On Windows: Run as administrator if needed

4. **Network connectivity issues**:
   - Check internet connection for catalog queries
   - Verify firewall settings allow HTTP/HTTPS traffic
   - Some institutional networks may block astronomical databases

5. **Browser compatibility**:
   - Use modern browsers (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)
   - Enable JavaScript and allow local connections
   - Clear browser cache if interface doesn't load properly

**Platform-Specific Notes**:

**Windows**:
   - PowerShell execution policy may need adjustment
   - Windows Defender may quarantine downloaded files
   - Use Command Prompt or PowerShell for terminal commands

**macOS**:
   - May need to allow unsigned applications in Security preferences
   - Homebrew installation recommended for dependencies
   - Command Line Tools for Xcode required for some packages

**Linux**:
   - Package manager installation preferred for system dependencies
   - May need development headers for compilation
   - Check distribution-specific package names

Performance Optimization
-----------------------

**System Resources**:
   - Minimum 4GB RAM recommended
   - SSD storage preferred for temporary file handling
   - Multi-core processor beneficial for image processing

**Configuration Tuning**:
   - Adjust detection parameters for your typical image sizes
   - Configure appropriate timeout values for slow networks
   - Set reasonable maximum catalog query limits

**Data Management**:
   - Regular cleanup of old result directories
   - Monitor disk space usage for large datasets
   - Consider archival strategies for important results

Updating the Application
-----------------------

**Update Process**:
   1. Backup your configuration files and important results
   2. Pull latest changes from repository or download new version
   3. Update Python dependencies: `pip install -r requirements.txt --upgrade`
   4. Restart both backend and frontend services
   5. Test with a simple image to verify functionality

**Version Compatibility**:
   - Configuration files are generally forward-compatible
   - User databases automatically migrate when needed
   - Previous result archives remain accessible