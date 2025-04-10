Troubleshooting
=============

This section addresses common issues and provides solutions to help you get the most out of Photometry Factory for RAPAS.

Installation Issues
-----------------

**Error: ImportError: No module named 'astropy'**

* **Problem**: Required Python packages are not installed.
* **Solution**: Run `pip install -r requirements.txt` from the application directory.

**Error: DLL load failed while importing...**

* **Problem**: Binary dependencies are missing (Windows).
* **Solution**: Install Visual C++ Redistributable packages, which are required by some astronomical libraries.

**Error: File "streamlit/...", line X, in \<module\>**

* **Problem**: Incompatible streamlit version.
* **Solution**: Install the specific version with `pip install streamlit==1.15.0`.

Image Loading Issues
------------------

**Error: Could not read FITS file**

* **Problem**: The file format is not recognized or the file is corrupted.
* **Solution**: 
  - Verify the file is a valid FITS file using another tool like DS9
  - Try re-saving the file from your astronomy software
  - Ensure proper file extension (.fits, .fit, or .fts)

**Error: FITS header corruption detected**

* **Problem**: The FITS header contains invalid values or formatting.
* **Solution**: 
  - Use a FITS utility like `fitsverify` to check for header problems
  - Repair the header with a tool like `fitsheader` or by resaving from your astronomy software

**Warning: Multiple HDUs detected...**

* **Problem**: Multi-extension FITS file has complex structure.
* **Solution**: 
  - The application will try to use the primary HDU or first valid image HDU
  - If this fails, convert your FITS file to a single-extension format using a tool like FITSSPLIT

WCS and Astrometry Issues
-----------------------

**Error: Missing required WCS keywords**

* **Problem**: The image lacks proper WCS information.
* **Solution**:
  - Enter coordinates manually in the fields provided
  - Use astrometry.net for plate solving by entering your API key

**Error: Error solving with astrometry.net**

* **Problem**: The online plate solving service couldn't identify the field.
* **Solution**:
  - Verify your API key is correct
  - Check your internet connection
  - If your field is sparse, try the advanced options at nova.astrometry.net directly
  - Provide constraints like approximate coordinates or field size if known

**Warning: Large WCS errors detected**

* **Problem**: The WCS solution has high residuals.
* **Solution**:
  - This may be due to optical distortions in your system
  - The application will attempt to refine the solution with Gaia DR3

Source Detection Issues
---------------------

**Warning: No sources found!**

* **Problem**: No astronomical sources were detected in the image.
* **Solution**:
  - Decrease the detection threshold (try 2.0-2.5 sigma)
  - Verify the image has sufficient exposure (not too short or faint)
  - Check if the image is properly calibrated (not too noisy)
  - Adjust the seeing estimate to better match the actual seeing in your image

**Warning: Too many spurious detections**

* **Problem**: Many false positives detected.
* **Solution**:
  - Increase the detection threshold (try 4.0-5.0 sigma)
  - Increase the border mask size if edge artifacts are present
  - Apply proper calibration to reduce noise
  - Check for hot pixels or cosmic rays and apply appropriate filtering

Photometry Issues
---------------

**Error: Background estimation failed**

* **Problem**: Cannot estimate sky background properly.
* **Solution**:
  - Try processing again with a different Border Mask setting
  - If your image has large extended objects, try increasing the box size parameter
  - Ensure your image has sufficient sky area without sources

**Error: FWHM estimation failed**

* **Problem**: Unable to determine star sizes accurately.
* **Solution**:
  - Manually enter a reasonable FWHM estimate based on your seeing
  - Typical values range from 2-5 pixels for most amateur telescopes
  - Ensure stars in your image are not saturated or severely distorted

**Error: PSF fitting failed**

* **Problem**: Cannot generate a proper PSF model.
* **Solution**:
  - PSF modeling requires multiple well-exposed, non-saturated stars
  - Try using only aperture photometry results instead
  - Check that your image has good enough focus/seeing for PSF fitting

Zero-Point Calibration Issues
---------------------------

**Error: No Gaia sources found within search radius**

* **Problem**: Cannot find Gaia stars in the field.
* **Solution**:
  - Verify your WCS solution is correct
  - Adjust the Gaia magnitude range to better match your image depth
  - Check if your coordinates are pointing to a very sparse field

**Error: Zero point calculation failed**

* **Problem**: Cannot determine the photometric zero point.
* **Solution**:
  - Ensure you have enough Gaia matched stars (at least 5-10)
  - Adjust the Gaia magnitude range to include more calibration stars
  - Check that your aperture size is appropriate for the seeing
  - Verify that your image is properly calibrated

Performance Issues
----------------

**Warning: Application running slowly**

* **Problem**: Processing takes too long.
* **Solution**:
  - For large images, try cropping or binning them before processing
  - Adjust Border Mask to process only the central portion of very large images
  - Close other resource-intensive applications
  - Process smaller areas or fewer sources for initial testing

**Error: Memory error during processing**

* **Problem**: Application runs out of memory.
* **Solution**:
  - Use a system with more RAM
  - Process smaller images or reduce the image size through binning
  - Close other applications to free up memory
  - If using a virtual environment, make sure it has access to sufficient system resources

Output Issues
-----------

**Error: Permission denied when saving files**

* **Problem**: Cannot write to output directory.
* **Solution**:
  - Run the application with appropriate permissions
  - Change the output directory to one where you have write access
  - Close any open files that might be locked

**Warning: Catalog file contains NaN values**

* **Problem**: Some measurements failed or couldn't be calculated.
* **Solution**:
  - This is normal for some sources, especially at image edges
  - Filter out rows with NaN values in your subsequent analysis
  - Try adjusting detection parameters to improve measurement quality

Getting Help
-----------

If you encounter issues not covered here:

1. Check the log file generated during processing for detailed error messages
2. Search for similar issues in the project repository
3. Contact the development team with:
   - A clear description of the problem
   - Steps to reproduce the issue
   - Relevant error messages
   - Sample data if possible (or a description if data cannot be shared)
