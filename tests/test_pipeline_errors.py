"""
Unit tests for pipeline photometry error propagation.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from astropy.table import Table
from src.pipeline import detection_and_photometry

# Mock required dependencies to avoid complex setup
@pytest.fixture
def mock_dependencies():
    with patch('src.pipeline.safe_wcs_create') as mock_wcs, \
         patch('src.pipeline.estimate_background') as mock_bkg, \
         patch('src.pipeline.make_border_mask') as mock_mask, \
         patch('src.pipeline.mask_and_remove_cosmic_rays') as mock_cr, \
         patch('src.pipeline.fwhm_fit') as mock_fwhm, \
         patch('src.pipeline.DAOStarFinder') as mock_dao, \
         patch('src.pipeline.aperture_photometry') as mock_phot, \
         patch('src.pipeline.perform_psf_photometry') as mock_psf, \
         patch('src.pipeline.is_streamlit_context') as mock_st_ctx:
        
        # Setup common return values
        mock_wcs.return_value = (None, "No WCS", [])
        mock_mask.return_value = np.zeros((100, 100), dtype=bool)
        mock_cr.return_value = np.zeros((100, 100), dtype=bool)
        mock_fwhm.return_value = (3.0, 10.0)  # fwhm, std
        mock_st_ctx.return_value = False
        
        # Setup background mock
        bkg_obj = MagicMock()
        bkg_obj.background = np.zeros((100, 100)) + 100
        bkg_obj.background_rms = np.zeros((100, 100)) + 5
        mock_bkg.return_value = (bkg_obj, None, None)
        
        yield {
            'wcs': mock_wcs,
            'bkg': mock_bkg,
            'mask': mock_mask,
            'cr': mock_cr,
            'fwhm': mock_fwhm,
            'dao': mock_dao,
            'phot': mock_phot,
            'psf': mock_psf
        }

def test_background_error_propagation(mock_dependencies):
    """Test that background subtraction error is propagated to total error."""
    
    # Setup image and header
    image = np.zeros((100, 100))
    header = {"GAIN": 1.0, "PIXSCALE": 1.0}
    
    # Setup mock sources found by DAOStarFinder
    mock_dao_instance = mock_dependencies['dao'].return_value
    sources = Table()
    sources['xcentroid'] = [50.0]
    sources['ycentroid'] = [50.0]
    sources['id'] = [1]
    mock_dao_instance.return_value = sources
    
    # Setup mock aperture photometry results
    # We need to simulate two calls: one for aperture, one for annulus
    
    # Call 1: Source Aperture
    aperture_table = Table()
    aperture_table['id'] = [1]
    aperture_table['aperture_sum'] = [1000.0]
    aperture_table['aperture_sum_err'] = [10.0]  # Poisson error
    
    # Call 2: Annulus Background
    annulus_table = Table()
    annulus_table['id'] = [1]
    annulus_table['aperture_sum'] = [500.0]
    annulus_table['aperture_sum_err'] = [20.0]  # Background error (sum in annulus)
    
    # Configure mock side_effect to return these tables in sequence
    # Since there are multiple aperture radii (1.1, 1.3), the loop runs multiple times.
    # The code creates lists of apertures.
    # detection_and_photometry loop structure:
    # for i in range(len(radii)):
    #    phot_result = aperture_photometry(...)
    #    bkg_result = aperture_photometry(...)
    
    # We expect 2 iterations (radii 1.1 and 1.3) -> 4 calls
    # Let's just return the same tables for simplicity, checking the first one is enough
    mock_dependencies['phot'].side_effect = [
        aperture_table.copy(), annulus_table.copy(),  # Radius 1
        aperture_table.copy(), annulus_table.copy()   # Radius 2
    ]
    
    # Mock PSF return to None to skip
    mock_dependencies['psf'].return_value = (None, None)
    
    # Run the function
    result = detection_and_photometry(
        image, header, 
        mean_fwhm_pixel=3.0, 
        threshold_sigma=3.0, 
        detection_mask=5
    )
    
    phot_table = result[0]
    
    # Verify result
    assert phot_table is not None
    
    # Check if the error column exists for radius 1.1
    err_col = 'aperture_sum_bkg_corr_err_1_1'
    assert err_col in phot_table.colnames
    
    # Calculate expected error
    # We need to know aperture and annulus areas to verify exact value
    # But checking if it's > aperture_sum_err is a good first step
    # bkg_sub_err = bkg_sum_err * (A_ap / A_ann)
    # total = sqrt(ap_err^2 + bkg_sub_err^2)
    
    original_err = 10.0
    propagated_err = phot_table[err_col][0]
    
    assert propagated_err > original_err
    print(f"Original Err: {original_err}, Propagated Err: {propagated_err}")

def test_snr_calculation_uses_propagated_error(mock_dependencies):
    """Test that S/N calculation uses the propagated error column."""
    
    image = np.zeros((100, 100))
    header = {"GAIN": 1.0}
    
    mock_dao_instance = mock_dependencies['dao'].return_value
    sources = Table()
    sources['xcentroid'] = [50.0]
    sources['ycentroid'] = [50.0]
    mock_dao_instance.return_value = sources
    
    # Setup tables
    aperture_table = Table()
    aperture_table['aperture_sum'] = [1000.0]
    aperture_table['aperture_sum_err'] = [1.0] # Small error
    
    annulus_table = Table()
    annulus_table['aperture_sum'] = [100.0]
    annulus_table['aperture_sum_err'] = [100.0] # Large error
    
    mock_dependencies['phot'].side_effect = [
        aperture_table.copy(), annulus_table.copy(),
        aperture_table.copy(), annulus_table.copy()
    ]
    mock_dependencies['psf'].return_value = (None, None)
    
    result = detection_and_photometry(
        image, header, mean_fwhm_pixel=3.0, threshold_sigma=3.0, detection_mask=5
    )
    phot_table = result[0]
    
    # S/N should be roughly flux / large_error, not flux / small_error
    # Flux ~ 1000, Small err = 1 -> SNR 1000
    # Large err (propagated) will be significant.
    # A_ap / A_ann is usually < 1, let's say 0.2.
    # bkg_err contrib = 100 * 0.2 = 20.
    # Total err ~ sqrt(1^2 + 20^2) ~ 20.
    # Expected SNR ~ 1000/20 = 50.
    
    snr = phot_table['snr_1_1'][0]
    
    # If it used only aperture error (1.0), SNR would be ~1000 (minus bkg subtraction)
    # If it uses propagated error, it should be much lower
    
    assert snr < 500  # Should be significantly impacted by the large background error
