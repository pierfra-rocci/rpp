Installation Guide
=================

System Requirements
------------------

**Python Requirements**:
   - Python 3.8 or higher
   - pip package manager
   - Virtual environment (recommended)

**External Dependencies**:
   - Astrometry.net with solve-field binary
   - SCAMP for astrometric refinement (optional)
   - Modern web browser (Chrome, Firefox, Safari, Edge)
   - Internet connection for catalog queries

**Operating System Support**:
   - Windows 10/11
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

   **Advanced Dependencies**:
   - astroscrappy (cosmic ray removal)
   - stdpipe (astrometry refinement and plate solving)
   - requests (HTTP client for catalog queries)
   - scikit-image (image processing utilities)

4. **Install Astrometry.net** (required for plate solving):
   
   **Ubuntu/Debian**:
   
   .. code-block:: bash

      sudo apt-get update
      sudo apt-get install astrometry.net astrometry-data-tycho2
      
   **macOS with Homebrew**:
   
   .. code-block:: bash

      brew install astrometry-net
      
   **Windows**:
   - Download from http://astrometry.net/downloads/
   - Install using provided installer
   - Ensure solve-field is available in system PATH
   - Download appropriate index files for your field scale

5. **Install SCAMP** (optional, for astrometric refinement):
   
   **Ubuntu/Debian**:
   
   .. code-block:: bash

      sudo apt-get install scamp
      
   **macOS with Homebrew**:
   
   .. code-block:: bash

      brew install scamp
      
   **Windows**:
   - Download from https://www.astromatic.net/software/scamp
   - Follow installation instructions
   - Ensure scamp is available in system PATH

Configuration
------------

1. **Astrometry.net Index Files**:
   Download appropriate index files for your typical field scale:
   
   - For wide-field images (>30 arcmin): index-4200 series
   - For medium fields (1-30 arcmin): index-4100 series  
   - For narrow fields (<1 arcmin): index-5000 series

2. **Environment Variables** (optional):
   
   .. code-block:: bash

      # For email functionality
      export SMTP_SERVER="smtp.gmail.com"
      export SMTP_PORT="587"
      export SMTP_USER="your-email@gmail.com"
      export SMTP_PASS="your-app-password"
      
      # For Astro-Colibri API
      export ASTROCOLIBRI_API="your-api-key"

3. **Directory Structure**:
   The application will create user-specific directories:
   
   .. code-block::

      rpp/
      ├── frontend.py
      ├── backend.py
      ├── pages/
      │   ├── app.py
      │   └── login.py
      ├── src/
      │   ├── tools.py
      │   ├── pipeline.py
      │   └── __version__.py
      └── [username]_rpp_results/

Verification
-----------

1. **Test Installation**:
   
   .. code-block:: bash

      # Test Astrometry.net
      solve-field --help
      
      # Test Python dependencies
      python -c "import streamlit, astropy, photutils; print('All core dependencies available')"
      
      # Test SCAMP (if installed)
      scamp -v

2. **Launch Application**:
   
   .. code-block:: bash

      # Start backend (in one terminal)
      python backend.py
      
      # Start frontend (in another terminal)
      streamlit run frontend.py

3. **Access Web Interface**:
   Open your browser to http://localhost:8501 and verify the interface loads correctly.

Troubleshooting
--------------

**Common Issues**:

- **Astrometry.net not found**: Ensure solve-field is in your system PATH
- **SCAMP not found**: Install SCAMP or disable astrometry refinement
- **Permission errors**: Run with appropriate user permissions
- **Port conflicts**: Change Streamlit port with --server.port option
- **Index file errors**: Download appropriate astrometry.net index files

**Performance Tips**:

- Use SSD storage for better I/O performance
- Ensure adequate RAM (8GB+ recommended)
- Close other applications during processing
- Use virtual environment to avoid dependency conflicts

**Getting Help**:

- Check the troubleshooting section
- Review application logs for error details
- Verify all dependencies are properly installed
- Test with sample FITS files first