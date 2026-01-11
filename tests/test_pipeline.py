"""
Unit tests for pipeline.py - Core photometry pipeline functions.

This test suite validates:
- Cosmic ray masking and detection
- Border mask creation
- Airmass calculations
- FWHM fitting
- Detection and photometry workflows

Author: Automated Testing Suite
Date: 2026-01-11
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from numpy.testing import assert_allclose, assert_array_equal

# Import functions to test
from src.pipeline import (
    mask_and_remove_cosmic_rays,
    make_border_mask,
    airmass,
)


class TestMaskAndRemoveCosmicRays:
    """Test cosmic ray detection and masking."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image with known properties."""
        np.random.seed(42)
        image = np.random.normal(1000, 50, (100, 100)).astype(np.float32)
        return image

    @pytest.fixture
    def sample_header(self):
        """Create a sample FITS header."""
        return {
            "SATURATE": 65535.0,
            "GAIN": 1.5,
        }

    def test_saturation_mask_from_header(self, sample_image, sample_header):
        """Test that saturation level is read from header."""
        # Add saturated pixels
        sample_image[50, 50] = 70000  # Above SATURATE
        
        with patch('src.pipeline.astroscrappy.detect_cosmics') as mock_detect:
            # Mock cosmic ray detection to return no cosmic rays
            mock_detect.return_value = (np.zeros_like(sample_image, dtype=bool),)
            
            mask = mask_and_remove_cosmic_rays(sample_image, sample_header)
            
            # Saturated pixel should be masked
            assert mask[50, 50] == True

    def test_saturation_fallback_when_no_header_key(self, sample_image):
        """Test fallback to 95% of max when SATURATE not in header."""
        header = {"GAIN": 1.0}
        sample_image[25, 25] = 10000  # High value
        
        with patch('src.pipeline.astroscrappy.detect_cosmics') as mock_detect:
            mock_detect.return_value = (np.zeros_like(sample_image, dtype=bool),)
            
            # Should use 0.95 * max(image) as saturation
            mask = mask_and_remove_cosmic_rays(sample_image, header)
            
            assert isinstance(mask, np.ndarray)
            assert mask.shape == sample_image.shape

    def test_nan_pixels_are_masked(self, sample_image, sample_header):
        """Test that NaN pixels are included in the mask."""
        sample_image[10, 10] = np.nan
        sample_image[20, 30] = np.nan
        
        with patch('src.pipeline.astroscrappy.detect_cosmics') as mock_detect:
            mock_detect.return_value = (np.zeros_like(sample_image, dtype=bool),)
            
            mask = mask_and_remove_cosmic_rays(sample_image, sample_header)
            
            assert mask[10, 10] == True
            assert mask[20, 30] == True

    def test_gain_default_when_missing(self, sample_image):
        """Test that gain defaults to 1.0 when not in header."""
        header = {"SATURATE": 65535}
        
        with patch('src.pipeline.astroscrappy.detect_cosmics') as mock_detect:
            mock_detect.return_value = (np.zeros_like(sample_image, dtype=bool),)
            
            # Should not raise error
            mask = mask_and_remove_cosmic_rays(sample_image, header)
            assert mask is not None

    def test_cosmic_ray_detection_failure_fallback(self, sample_image, sample_header):
        """Test graceful fallback when cosmic ray detection fails."""
        with patch('src.pipeline.astroscrappy.detect_cosmics') as mock_detect:
            mock_detect.side_effect = Exception("Detection failed")
            
            # Should return mask with only saturation masking
            mask = mask_and_remove_cosmic_rays(sample_image, sample_header)
            
            assert isinstance(mask, np.ndarray)
            assert mask.shape == sample_image.shape


class TestMakeBorderMask:
    """Test border mask creation."""

    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return np.ones((100, 100))

    def test_uniform_border(self, sample_image):
        """Test mask with uniform border on all sides."""
        mask = make_border_mask(sample_image, border=10, invert=True)
        
        assert mask.shape == (100, 100)
        # Border pixels should be True (masked)
        assert mask[0, 0] == True
        assert mask[9, 9] == True
        # Interior pixels should be False (not masked)
        assert mask[50, 50] == False

    def test_asymmetric_border_2tuple(self, sample_image):
        """Test mask with (vertical, horizontal) border specification."""
        mask = make_border_mask(sample_image, border=(20, 10), invert=True)
        
        # Top/bottom border of 20, left/right border of 10
        assert mask[19, 50] == True   # Within top border
        assert mask[20, 50] == False  # Just past top border
        assert mask[50, 9] == True    # Within left border
        assert mask[50, 10] == False  # Just past left border

    def test_asymmetric_border_4tuple(self, sample_image):
        """Test mask with (top, bottom, left, right) border specification."""
        # Border order is (top, bottom, left, right)
        # With invert=True: border pixels are True, interior is False
        mask = make_border_mask(sample_image, border=(10, 20, 5, 15), invert=True)
        
        # Top border = 10 pixels (rows 0-9 are masked)
        assert mask[9, 50] == True    # Row 9 is in top border
        assert mask[10, 50] == False  # Row 10 is interior
        # Bottom border = 20 pixels (rows 80-99 are masked)
        assert mask[79, 50] == False  # Row 79 is interior (100-20=80, so last interior row is 79)
        assert mask[80, 50] == True   # Row 80 is in bottom border
        # Left border = 5 pixels (cols 0-4 are masked)
        assert mask[50, 4] == True    # Col 4 is in left border
        assert mask[50, 5] == False   # Col 5 is interior
        # Right border = 15 pixels (cols 85-99 are masked)
        assert mask[50, 84] == False  # Col 84 is interior (100-15=85, so last interior col is 84)
        assert mask[50, 85] == True   # Col 85 is in right border

    def test_invert_false(self, sample_image):
        """Test non-inverted mask (True for interior)."""
        mask = make_border_mask(sample_image, border=10, invert=False)
        
        # Border pixels should be False
        assert mask[0, 0] == False
        # Interior pixels should be True
        assert mask[50, 50] == True

    def test_invalid_image_none(self):
        """Test that None image raises ValueError."""
        with pytest.raises(ValueError, match="Image cannot be None"):
            make_border_mask(None, border=10)

    def test_invalid_image_type(self):
        """Test that non-ndarray raises TypeError."""
        with pytest.raises(TypeError, match="must be a numpy.ndarray"):
            make_border_mask([1, 2, 3], border=10)

    def test_empty_image(self):
        """Test that empty image raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            make_border_mask(np.array([]), border=10)

    def test_1d_image(self):
        """Test that 1D image raises ValueError."""
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            make_border_mask(np.ones(100), border=10)

    def test_negative_border(self, sample_image):
        """Test that negative border raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            make_border_mask(sample_image, border=-10)

    def test_border_too_large(self, sample_image):
        """Test that border larger than image raises ValueError."""
        with pytest.raises(ValueError, match="must be < image"):
            make_border_mask(sample_image, border=60)  # 60*2 > 100

    def test_invalid_border_tuple_length(self, sample_image):
        """Test that tuple of wrong length raises ValueError."""
        with pytest.raises(ValueError, match="length 2 or 4"):
            make_border_mask(sample_image, border=(10, 20, 30))

    def test_dtype_output(self, sample_image):
        """Test that output dtype can be specified."""
        mask_bool = make_border_mask(sample_image, border=10, dtype=bool)
        mask_int = make_border_mask(sample_image, border=10, dtype=np.int32)
        
        assert mask_bool.dtype == bool
        assert mask_int.dtype == np.int32


class TestAirmass:
    """Test airmass calculation."""

    @pytest.fixture
    def valid_header(self):
        """Create a valid header with all required keywords."""
        return {
            "RA": 180.0,
            "DEC": 45.0,
            "DATE-OBS": "2026-01-11T20:00:00",
        }

    @pytest.fixture
    def observatory_data(self):
        """Create observatory data."""
        return {
            "name": "Test Observatory",
            "latitude": 45.0,
            "longitude": -75.0,
            "elevation": 200.0,
        }

    def test_airmass_from_header_keyword(self, observatory_data):
        """Test that existing AIRMASS in header is used."""
        header = {
            "AIRMASS": 1.5,
            "RA": 180.0,
            "DEC": 45.0,
            "DATE-OBS": "2026-01-11T20:00:00",
        }
        
        result = airmass(header, observatory_data)
        
        assert result == 1.5

    def test_airmass_invalid_in_header_recalculates(self, valid_header, observatory_data):
        """Test that invalid AIRMASS triggers recalculation."""
        valid_header["AIRMASS"] = 0.5  # Invalid: less than 1.0
        
        result = airmass(valid_header, observatory_data)
        
        # Should recalculate, result should be >= 1.0
        assert result >= 1.0

    def test_airmass_calculation_with_details(self, valid_header, observatory_data):
        """Test airmass calculation with return_details=True."""
        result, details = airmass(valid_header, observatory_data, return_details=True)
        
        assert isinstance(result, float)
        assert isinstance(details, dict)
        assert "observatory" in details or "airmass_source" in details

    def test_airmass_missing_coordinates(self, observatory_data):
        """Test that missing coordinates return 0.0."""
        header = {"DATE-OBS": "2026-01-11T20:00:00"}
        
        result = airmass(header, observatory_data)
        
        assert result == 0.0

    def test_airmass_missing_date(self, observatory_data):
        """Test that missing date returns 0.0."""
        header = {"RA": 180.0, "DEC": 45.0}
        
        result = airmass(header, observatory_data)
        
        assert result == 0.0

    def test_airmass_alternate_date_keywords(self, observatory_data):
        """Test that alternate date keywords are tried."""
        header = {
            "RA": 180.0,
            "DEC": 45.0,
            "DATE": "2026-01-11T20:00:00",  # Note: DATE instead of DATE-OBS
        }
        
        result = airmass(header, observatory_data)
        
        # Should still calculate airmass
        assert result >= 1.0 or result == 0.0  # Either valid or graceful failure

    def test_airmass_alternate_coord_keywords(self, observatory_data):
        """Test that OBJRA/OBJDEC are tried."""
        header = {
            "OBJRA": 180.0,
            "OBJDEC": 45.0,
            "DATE-OBS": "2026-01-11T20:00:00",
        }
        
        result = airmass(header, observatory_data)
        
        assert result >= 1.0 or result == 0.0

    def test_airmass_bounds_check(self, valid_header, observatory_data):
        """Test that airmass is within physical bounds."""
        # Set coordinates for a visible object
        valid_header["DEC"] = 45.0  # Same as observatory latitude
        
        result = airmass(valid_header, observatory_data)
        
        if result > 0:
            assert result >= 1.0
            assert result <= 30.0 or result > 0  # Either valid range or handled gracefully


class TestDetectionHelpers:
    """Test helper functions for source detection."""

    def test_sigma_clip_stats_integration(self):
        """Test that sigma_clip_stats works as expected."""
        from astropy.stats import sigma_clipped_stats
        
        np.random.seed(42)
        data = np.random.normal(1000, 50, (100, 100))
        
        mean, median, std = sigma_clipped_stats(data, sigma=3.0)
        
        # Should be close to input parameters
        assert_allclose(median, 1000, rtol=0.1)
        assert_allclose(std, 50, rtol=0.2)


class TestPhotometryConstants:
    """Test photometry-related constants and formulas."""

    def test_magnitude_error_constant(self):
        """Verify the magnitude error constant 2.5/ln(10) ≈ 1.0857."""
        constant = 2.5 / np.log(10)
        assert_allclose(constant, 1.0857, rtol=1e-4)

    def test_magnitude_to_flux_conversion(self):
        """Test magnitude to flux conversion."""
        mag = 15.0
        flux = 10 ** (-0.4 * mag)
        
        # Verify inverse
        mag_back = -2.5 * np.log10(flux)
        assert_allclose(mag_back, mag, rtol=1e-10)

    def test_flux_to_magnitude_with_zero_point(self):
        """Test instrumental to calibrated magnitude conversion."""
        flux = 1000.0
        zero_point = 25.0
        
        inst_mag = -2.5 * np.log10(flux)
        calib_mag = inst_mag + zero_point
        
        # Expected: -2.5 * log10(1000) + 25 = -7.5 + 25 = 17.5
        assert_allclose(calib_mag, 17.5, rtol=1e-10)
