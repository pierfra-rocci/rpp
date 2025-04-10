Installation
===========

System Requirements
------------------

Photometry Factory for RAPAS has the following system requirements:

* **Operating Systems**: Windows 10/11, macOS 10.14+, Linux (most modern distributions)
* **Python**: Version 3.8 or later
* **RAM**: Minimum 4GB, 8GB+ recommended for larger images
* **Disk Space**: 500MB for installation + space for your astronomical images

Dependencies
-----------

PFR depends on several powerful astronomical Python packages:

* **astropy** (v4.0+): Core astronomy library for coordinates, WCS, and FITS handling
* **photutils** (v1.0+): Specialized package for astronomical photometry
* **astroquery** (v0.4+): Tools for querying astronomical web services and databases
* **matplotlib** (v3.3+): Visualization library
* **numpy** (v1.20+): Numerical computing foundation
* **pandas** (v1.2+): Data analysis and manipulation tools
* **streamlit** (v1.10+): Interactive web application framework
* **stdpipe** (v0.4+): Pipeline tools for standard astronomical data processing

Standard Installation
--------------------

The recommended installation method is using pip with the provided requirements file:

.. code-block:: bash

   # Clone the repository (if you haven't already)
   git clone https://github.com/yourusername/P_F_R.git
   cd P_F_R
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the application
   streamlit run pfr_app.py

Using Virtual Environments
-------------------------

For a clean, isolated installation, using a virtual environment is recommended:

.. code-block:: bash

   # Create a virtual environment
   python -m venv pfr_env
   
   # Activate the environment
   # On Windows:
   pfr_env\\Scripts\\activate
   # On macOS/Linux:
   source pfr_env/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run the application
   streamlit run pfr_app.py

Standalone Executable (Windows)
------------------------------

A pre-packaged executable version is available for Windows users who prefer not to install Python:

1. Download the latest release from the project's release page
2. Extract all files to a directory of your choice
3. Run ``PfrApp.bat``

The executable includes all necessary dependencies and Python runtime.

Docker Installation
------------------

For containerized deployment, a Docker image is available:

.. code-block:: bash

   # Pull the image
   docker pull username/photometry-factory-rapas:latest
   
   # Run the container
   docker run -p 8501:8501 username/photometry-factory-rapas:latest

Then access the application by navigating to ``http://localhost:8501`` in your web browser.

Troubleshooting Installation
---------------------------

Common installation issues:

1. **Missing compiler**: Some dependencies may require a C compiler. Install the appropriate compiler for your system:
   
   * Windows: Microsoft Visual C++ Build Tools
   * macOS: Xcode Command Line Tools
   * Linux: gcc and development libraries

2. **SSL Certificate errors**: If you're behind a corporate firewall or proxy, you may need to configure pip to use an alternative certificate:

   .. code-block:: bash
   
      pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

3. **Version conflicts**: If you encounter dependency conflicts, try installing in a fresh virtual environment.

For additional help, please see the troubleshooting section or open an issue on the project repository.