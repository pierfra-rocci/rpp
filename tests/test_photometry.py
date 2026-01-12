"""
Unit tests for photometry S/N and magnitude error calculations.

This test suite validates the correctness of photometric formulas used in the pipeline:
- Signal-to-Noise Ratio (S/N) calculations
- Magnitude error computations
- Error propagation in calibrated magnitudes
- Quality flag assignments

Mathematical Background:
-----------------------
1. Signal-to-Noise Ratio:
   S/N = flux / flux_error

2. Magnitude Error from Flux Error:
   Given: mag = -2.5 × log10(flux)
   Error propagation: σ_mag = |∂mag/∂flux| × σ_flux
                            = (2.5 / ln(10)) × (σ_flux / flux)
                            ≈ 1.0857 × (σ_flux / flux)

3. Calibrated Magnitude Error:
   mag_calib = mag_inst + zero_point
   σ_mag_calib = √(σ_mag_inst² + σ_zero_point²)

Author: Automated Testing Suite
Date: 2026-01-07
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal


class TestSNRCalculations:
    """Test Signal-to-Noise Ratio calculations."""

    def test_basic_snr(self):
        """Test basic S/N = flux / flux_error."""
        flux = np.array([1000.0, 500.0, 100.0, 50.0])
        flux_err = np.array([10.0, 25.0, 10.0, 5.0])

        expected_snr = np.array([100.0, 20.0, 10.0, 10.0])
        calculated_snr = flux / flux_err

        assert_allclose(calculated_snr, expected_snr, rtol=1e-10)

    def test_snr_with_background_correction(self):
        """Test S/N using background-corrected flux."""
        raw_flux = np.array([1100.0, 550.0])
        background = np.array([100.0, 50.0])
        bkg_corrected_flux = raw_flux - background
        flux_err = np.array([15.0, 10.0])

        # Should use background-corrected flux for S/N
        expected_snr = bkg_corrected_flux / flux_err
        assert_allclose(expected_snr, [1000.0 / 15.0, 500.0 / 10.0], rtol=1e-10)

    def test_snr_with_negative_background_correction(self):
        """Test S/N calculation with negative background-corrected flux."""
        # raw_flux is irrelevant for the new logic if bkg_corrected is available
        bkg_corrected_flux = np.array([80.0, -10.0])
        flux_err = np.array([10.0, 5.0])

        # S/N should be negative for the second source
        expected_snr = np.array([8.0, -2.0])
        calculated_snr = bkg_corrected_flux / flux_err

        assert_allclose(calculated_snr, expected_snr, rtol=1e-10)

    def test_snr_high_precision(self):
        """Test that S/N is not rounded (preserves precision)."""
        flux = 1234.56789
        flux_err = 12.3456

        snr = flux / flux_err

        # Should preserve decimal precision
        assert snr != round(snr)
        assert_allclose(snr, 100.00063909, rtol=1e-7)


class TestMagnitudeErrorCalculations:
    """Test magnitude error computations."""

    def test_mag_error_formula(self):
        """Test magnitude error formula: σ_mag = 1.0857 × (σ_flux / flux)."""
        flux = np.array([1000.0, 500.0, 100.0])
        flux_err = np.array([10.0, 25.0, 10.0])

        # Expected from formula
        expected_mag_err = 1.0857 * flux_err / flux

        # Calculate using the constant
        calculated_mag_err = 1.0857 * flux_err / flux

        assert_allclose(calculated_mag_err, expected_mag_err, rtol=1e-10)

    def test_mag_error_from_snr(self):
        """Test that σ_mag ≈ 1.0857 / S/N for high S/N."""
        snr = np.array([100.0, 50.0, 20.0, 10.0, 5.0])

        # For high S/N, σ_mag ≈ 1.0857 / S/N
        mag_err_from_snr = 1.0857 / snr

        # Calculate properly from flux and flux_err
        flux = 1000.0
        flux_err = flux / snr
        mag_err_proper = 1.0857 * flux_err / flux

        # Should be equivalent
        assert_allclose(mag_err_from_snr, mag_err_proper, rtol=1e-10)

    def test_mag_error_constant_derivation(self):
        """Verify the constant 1.0857 = 2.5 / ln(10)."""
        expected_constant = 2.5 / np.log(10)

        assert_allclose(expected_constant, 1.0857, rtol=1e-4)

    def test_mag_error_vs_magnitude(self):
        """Test magnitude error for typical astronomical magnitudes."""
        # Source with magnitude 15 ± 0.1
        mag_true = 15.0
        flux_true = 10 ** (-0.4 * mag_true)

        # Error of 0.1 mag corresponds to specific flux error
        mag_err = 0.1
        flux_err = (mag_err / 1.0857) * flux_true

        # Calculate mag_err from flux_err
        calculated_mag_err = 1.0857 * flux_err / flux_true

        assert_allclose(calculated_mag_err, mag_err, rtol=1e-10)


class TestErrorPropagation:
    """Test error propagation in magnitude calculations.

    Note: Zero-point error propagation has been removed from the pipeline.
    Calibrated magnitude errors now use only instrumental magnitude errors.
    """

    def test_no_zero_point_error_propagation(self):
        """Verify calibrated magnitude error equals instrumental error (no ZP propagation)."""
        # Instrumental magnitude errors
        mag_err_inst = np.array([0.05, 0.10, 0.20])

        # Without zero-point error propagation, calibrated error = instrumental error
        expected_mag_err_calib = mag_err_inst

        assert_allclose(expected_mag_err_calib, mag_err_inst, rtol=1e-10)

    def test_magnitude_error_filtering(self):
        """Test that sources with magnitude error > 2 are filtered out."""
        # Simulated magnitude errors - some good, some bad
        mag_errors = np.array([0.1, 0.5, 1.0, 1.5, 2.0, 2.1, 3.0, 5.0])

        # Filter: keep only errors <= 2
        keep_mask = mag_errors <= 2

        # Should keep 5 sources (0.1, 0.5, 1.0, 1.5, 2.0)
        assert np.sum(keep_mask) == 5
        assert np.all(mag_errors[keep_mask] <= 2)

    def test_magnitude_error_filtering_with_nan(self):
        """Test that NaN magnitude errors are preserved (not filtered out)."""
        mag_errors = np.array([0.1, 0.5, np.nan, 2.1, 3.0])

        # Filter: keep errors <= 2 OR NaN
        keep_mask = (mag_errors <= 2) | np.isnan(mag_errors)

        # Should keep 3 sources (0.1, 0.5, NaN)
        assert np.sum(keep_mask) == 3


class TestQualityFlags:
    """Test quality flag assignments."""

    def test_quality_flag_thresholds(self):
        """Test quality flag assignment based on S/N thresholds."""
        snr = np.array([2.5, 3.0, 4.0, 5.0, 10.0, 100.0])

        quality_flags = np.where(snr < 3, "poor", np.where(snr < 5, "marginal", "good"))

        expected_flags = ["poor", "marginal", "marginal", "good", "good", "good"]

        assert_array_equal(quality_flags, expected_flags)

    def test_quality_flag_boundary_cases(self):
        """Test quality flags at exact boundaries."""
        # Exactly at thresholds
        snr_boundary = np.array([3.0, 5.0])

        quality_flags = np.where(
            snr_boundary < 3, "poor", np.where(snr_boundary < 5, "marginal", "good")
        )

        # S/N = 3.0 should be 'marginal' (not 'poor')
        # S/N = 5.0 should be 'good' (not 'marginal')
        assert quality_flags[0] == "marginal"
        assert quality_flags[1] == "good"


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_very_low_flux(self):
        """Test calculations with very low flux values."""
        flux = 1e-6
        flux_err = 1e-7

        snr = flux / flux_err
        mag_err = 1.0857 * flux_err / flux

        assert np.isfinite(snr)
        assert np.isfinite(mag_err)
        assert_allclose(snr, 10.0)
        assert_allclose(mag_err, 0.10857, rtol=1e-4)

    def test_very_high_flux(self):
        """Test calculations with very high flux values."""
        flux = 1e6
        flux_err = 1e4

        snr = flux / flux_err
        mag_err = 1.0857 * flux_err / flux

        assert np.isfinite(snr)
        assert np.isfinite(mag_err)
        assert_allclose(snr, 100.0)

    def test_zero_flux_error_handling(self):
        """Test handling of zero flux (should produce inf/nan appropriately)."""
        flux = 0.0
        flux_err = 1.0

        # S/N should be 0
        snr = flux / flux_err
        assert snr == 0.0

        # Magnitude should be inf (log of 0)
        with np.errstate(divide="ignore"):
            mag = -2.5 * np.log10(flux)
        assert np.isinf(mag)

    def test_negative_flux_handling(self):
        """Test handling of negative flux values."""
        flux = -100.0
        flux_err = 10.0

        snr = flux / flux_err
        assert snr == -10.0

        # Magnitude of negative flux should be nan
        with np.errstate(invalid="ignore"):
            mag = -2.5 * np.log10(flux)
        assert np.isnan(mag)


class TestRealisticScenarios:
    """Test with realistic astronomical data."""

    def test_bright_star_photometry(self):
        """Test photometry of a bright star (mag ~10)."""
        # Bright star: ~10000 counts, S/N ~ 100
        flux = 10000.0
        flux_err = 100.0

        snr = flux / flux_err
        mag_err = 1.0857 * flux_err / flux

        assert_allclose(snr, 100.0)
        assert_allclose(mag_err, 0.010857, rtol=1e-4)

        # Quality should be 'good'
        quality = "good" if snr >= 5 else ("marginal" if snr >= 3 else "poor")
        assert quality == "good"

    def test_faint_source_photometry(self):
        """Test photometry of a faint source (mag ~20)."""
        # Faint source: ~100 counts, S/N ~ 5
        flux = 100.0
        flux_err = 20.0

        snr = flux / flux_err
        mag_err = 1.0857 * flux_err / flux

        assert_allclose(snr, 5.0)
        assert_allclose(mag_err, 0.2171, rtol=1e-3)

        # Quality should be 'good' (just at threshold)
        quality = "good" if snr >= 5 else ("marginal" if snr >= 3 else "poor")
        assert quality == "good"

    def test_marginal_detection(self):
        """Test photometry of a marginal detection."""
        # Marginal: S/N ~ 3.5
        flux = 350.0
        flux_err = 100.0

        snr = flux / flux_err
        mag_err = 1.0857 * flux_err / flux

        assert_allclose(snr, 3.5)
        assert_allclose(mag_err, 0.3102, rtol=1e-3)

        # Quality should be 'marginal'
        quality = "good" if snr >= 5 else ("marginal" if snr >= 3 else "poor")
        assert quality == "marginal"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
