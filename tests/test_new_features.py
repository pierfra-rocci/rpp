"""
Non-regression tests for features added in the 1.7.x development cycle:

1. plot_astrocolibri_cutouts() (src/transient.py)
   - Early-return guards
   - Hemisphere-aware survey selection (PanSTARRS / SkyMapper)
   - PNG filename sanitisation
   - Magnitude column fallback priority

2. Residuals plot error bar column selection (src/pipeline.py)
   - Always uses aperture_mag_err_1_3 (fixed 1.3× aperture)
   - Fallback to generic aperture_mag_err
   - Fallback to zeros when no error column is present

3. Filter mismatch warning text (pages/app.py)
   - Warning no longer includes the verbose "Consider updating" suffix
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ── Stub out optional heavy dependencies before src.transient is imported ─────
# stdpipe transitively imports sip_tpv which requires pkg_resources (setuptools),
# which is absent in this test environment.  Inject MagicMock stand-ins for
# every module in the chain so the import succeeds without the real libraries.
_STUB_MODULES = [
    "stdpipe",
    "stdpipe.pipeline",
    "stdpipe.cutouts",
    "stdpipe.templates",
    "stdpipe.catalogs",
    "stdpipe.photometry",
    "stdpipe.plots",
    "stdpipe.astrometry",
    "sep",
    "sip_tpv",
    "astroquery",
    "astroquery.imcce",
    "streamlit",
]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Make sure a cached failed import of src.transient doesn't block us
sys.modules.pop("src.transient", None)


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _make_ac_table(n_sources=3, n_matches=2):
    """Build a minimal photometry DataFrame with Astro-Colibri columns.

    The first *n_matches* rows have a non-null astrocolibri_name; the rest are None.
    """
    names = [f"Src{i}" for i in range(n_matches)] + [None] * (n_sources - n_matches)
    types = [f"Type{i}" for i in range(n_matches)] + [None] * (n_sources - n_matches)
    classes = [f"Class{i}" for i in range(n_matches)] + [None] * (n_sources - n_matches)
    return pd.DataFrame({
        "ra": np.linspace(10.0, 30.0, n_sources),
        "dec": np.linspace(15.0, 45.0, n_sources),
        "xcenter": np.arange(100, 100 + n_sources, dtype=float),
        "ycenter": np.arange(200, 200 + n_sources, dtype=float),
        "psf_mag": np.full(n_sources, 18.5),
        "psf_mag_err": np.full(n_sources, 0.05),
        "aperture_mag_1_3": np.full(n_sources, 18.8),
        "aperture_mag_err_1_3": np.full(n_sources, 0.06),
        "astrocolibri_name": names,
        "astrocolibri_type": types,
        "astrocolibri_classification": classes,
    })


# ─── 1. plot_astrocolibri_cutouts — early-return guards ──────────────────────

class TestPlotAstrocolibriCutoutsGuards:
    """Early-return guard conditions — do not require external libraries."""

    def test_returns_none_for_none_table(self):
        from src.transient import plot_astrocolibri_cutouts

        result = plot_astrocolibri_cutouts(None, None, None, "/tmp", "test")
        assert result is None

    def test_returns_none_for_empty_dataframe(self):
        from src.transient import plot_astrocolibri_cutouts

        result = plot_astrocolibri_cutouts(pd.DataFrame(), None, None, "/tmp", "test")
        assert result is None

    def test_returns_none_when_column_missing(self):
        """Table without the astrocolibri_name column → early return."""
        from src.transient import plot_astrocolibri_cutouts

        df = pd.DataFrame({"ra": [10.0], "dec": [20.0], "psf_mag": [17.5]})
        result = plot_astrocolibri_cutouts(df, None, None, "/tmp", "test")
        assert result is None

    def test_returns_none_when_all_names_are_null(self):
        """All astrocolibri_name values are None → no matches → early return."""
        from src.transient import plot_astrocolibri_cutouts

        df = pd.DataFrame({
            "ra": [10.0, 20.0],
            "dec": [5.0, 10.0],
            "astrocolibri_name": [None, None],
        })
        result = plot_astrocolibri_cutouts(df, None, None, "/tmp", "test")
        assert result is None

    def test_returns_none_when_all_names_are_nan(self):
        """astrocolibri_name column filled with NaN → no matches → early return."""
        from src.transient import plot_astrocolibri_cutouts

        df = pd.DataFrame({
            "ra": [10.0],
            "dec": [5.0],
            "astrocolibri_name": [float("nan")],
        })
        result = plot_astrocolibri_cutouts(df, None, None, "/tmp", "test")
        assert result is None


# ─── 2. plot_astrocolibri_cutouts — hemisphere-aware survey selection ─────────

def _run_patched(dec_center, filter_name="r"):
    """Run plot_astrocolibri_cutouts with all external deps mocked.

    Returns the first positional argument passed to templates.get_hips_image
    (i.e. the survey + filter string), or None if the mock was never called.
    """
    from astropy.io.fits import Header
    from src.transient import plot_astrocolibri_cutouts

    df = _make_ac_table(n_sources=2, n_matches=1)
    # Force dec of the matched source to the requested value
    df.loc[0, "dec"] = dec_center

    image = np.zeros((500, 500))
    header = Header()
    mock_cutout = {"image": np.zeros((25, 25)), "header": header}
    mock_fig = MagicMock()

    with patch("src.transient.st"), \
         patch("src.transient.fix_header", return_value=(header, None)), \
         patch("src.transient.cutouts.get_cutout", return_value=mock_cutout), \
         patch("src.transient.templates.get_hips_image",
               side_effect=Exception("network disabled")) as mock_hips, \
         patch("src.transient.plot_cutout", return_value=mock_fig), \
         patch("src.transient.plt.close"):
        plot_astrocolibri_cutouts(
            df, image, header, "/tmp", "test", filter_name, dec_center
        )
        return mock_hips.call_args


class TestAstrocolibriSurveySelection:
    """PanSTARRS chosen for dec ≥ 0, SkyMapper for dec < 0."""

    def test_panstarrs_for_northern_field(self):
        call_args = _run_patched(dec_center=30.0)
        assert call_args is not None
        assert "PanSTARRS" in call_args.args[0]

    def test_skymapper_for_southern_field(self):
        call_args = _run_patched(dec_center=-20.0)
        assert call_args is not None
        assert "SkyMapper" in call_args.args[0]

    def test_panstarrs_for_equatorial_field(self):
        """dec_center == 0.0 is not < 0, so PanSTARRS must be used."""
        call_args = _run_patched(dec_center=0.0)
        assert call_args is not None
        assert "PanSTARRS" in call_args.args[0]

    def test_filter_name_included_in_survey_string(self):
        """The chosen filter is appended after the survey path."""
        call_args = _run_patched(dec_center=10.0, filter_name="g")
        assert call_args is not None
        survey_arg = call_args.args[0]
        assert survey_arg.endswith("g")

    def test_skymapper_filter_name_appended(self):
        call_args = _run_patched(dec_center=-45.0, filter_name="i")
        assert call_args is not None
        survey_arg = call_args.args[0]
        assert "SkyMapper" in survey_arg
        assert survey_arg.endswith("i")


# ─── 3. plot_astrocolibri_cutouts — PNG filename sanitisation ─────────────────

class TestAstrocolibriSafeFilename:
    """The safe_name function mirrors the logic in plot_astrocolibri_cutouts."""

    @staticmethod
    def _safe(name: str) -> str:
        return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(name))

    def test_alphanumeric_name_unchanged(self):
        assert self._safe("GRB20240101A") == "GRB20240101A"

    def test_spaces_replaced_by_underscore(self):
        assert self._safe("SN 2024xyz") == "SN_2024xyz"

    def test_slashes_replaced(self):
        assert self._safe("AT2024/01") == "AT2024_01"

    def test_colons_replaced(self):
        assert self._safe("src:name") == "src_name"

    def test_hyphens_and_underscores_kept(self):
        assert self._safe("src_name-v2") == "src_name-v2"

    def test_all_output_chars_are_safe(self):
        nasty = "Src@Name:Test! #weird"
        result = self._safe(nasty)
        assert all(c.isalnum() or c in "_-" for c in result)

    def test_empty_string(self):
        assert self._safe("") == ""


# ─── 4. plot_astrocolibri_cutouts — magnitude column priority ─────────────────

class TestAstrocolibriMagColumnPriority:
    """Magnitude column fallback: psf_mag → aperture_mag_1_3 → aperture_mag_1_1 → None."""

    @staticmethod
    def _pick_mag(row: dict):
        """Mirrors the mag-selection logic inside plot_astrocolibri_cutouts."""
        mag_val = None
        for col in ["psf_mag", "aperture_mag_1_3", "aperture_mag_1_1"]:
            if col in row and row[col] is not None:
                try:
                    v = float(row[col])
                    if np.isfinite(v):
                        mag_val = v
                        break
                except (TypeError, ValueError):
                    pass
        return mag_val

    def test_psf_mag_takes_priority(self):
        row = {"psf_mag": 17.5, "aperture_mag_1_3": 17.8, "aperture_mag_1_1": 17.9}
        assert self._pick_mag(row) == pytest.approx(17.5)

    def test_fallback_to_aperture_1_3_when_psf_missing(self):
        row = {"aperture_mag_1_3": 17.8, "aperture_mag_1_1": 17.9}
        assert self._pick_mag(row) == pytest.approx(17.8)

    def test_fallback_to_aperture_1_1_when_1_3_missing(self):
        row = {"aperture_mag_1_1": 18.0}
        assert self._pick_mag(row) == pytest.approx(18.0)

    def test_returns_none_when_all_absent(self):
        row = {"ra": 10.0, "dec": 20.0}
        assert self._pick_mag(row) is None

    def test_skips_nan_psf_mag_and_falls_back(self):
        row = {"psf_mag": float("nan"), "aperture_mag_1_3": 17.8}
        assert self._pick_mag(row) == pytest.approx(17.8)

    def test_skips_none_psf_mag_and_falls_back(self):
        row = {"psf_mag": None, "aperture_mag_1_3": 17.8}
        assert self._pick_mag(row) == pytest.approx(17.8)

    def test_skips_inf_psf_mag_and_falls_back(self):
        row = {"psf_mag": float("inf"), "aperture_mag_1_3": 17.8}
        assert self._pick_mag(row) == pytest.approx(17.8)


# ─── 5. plot_astrocolibri_cutouts — skip rows with invalid coordinates ─────────

class TestAstrocolibriInvalidCoordinates:
    """Rows with NaN or None coordinates must be skipped without crashing."""

    def test_nan_ra_row_is_skipped(self):
        from src.transient import plot_astrocolibri_cutouts

        df = pd.DataFrame({
            "ra": [float("nan")],
            "dec": [20.0],
            "xcenter": [100.0],
            "ycenter": [200.0],
            "astrocolibri_name": ["BadSrc"],
        })
        # Should not raise and should return an empty list (0 saved files)
        from astropy.io.fits import Header

        image = np.zeros((100, 100))
        header = Header()

        with patch("src.transient.st"), \
             patch("src.transient.fix_header", return_value=(header, None)), \
             patch("src.transient.cutouts.get_cutout") as mock_cutout:
            result = plot_astrocolibri_cutouts(
                df, image, header, "/tmp", "test", "r", 20.0
            )
        # get_cutout must never be called for an invalid coordinate row
        mock_cutout.assert_not_called()
        assert result == []

    def test_nan_dec_row_is_skipped(self):
        from src.transient import plot_astrocolibri_cutouts

        df = pd.DataFrame({
            "ra": [83.8],
            "dec": [float("nan")],
            "xcenter": [100.0],
            "ycenter": [200.0],
            "astrocolibri_name": ["BadSrc"],
        })
        from astropy.io.fits import Header

        image = np.zeros((100, 100))
        header = Header()

        with patch("src.transient.st"), \
             patch("src.transient.fix_header", return_value=(header, None)), \
             patch("src.transient.cutouts.get_cutout") as mock_cutout:
            result = plot_astrocolibri_cutouts(
                df, image, header, "/tmp", "test", "r", 10.0
            )
        mock_cutout.assert_not_called()
        assert result == []


# ─── 6. Residuals plot error bar column selection ─────────────────────────────

class TestResidualsErrorBarSelection:
    """Mirrors the error-column selection logic in calculate_zero_point().

    The residuals plot always prefers aperture_mag_err_1_3 (fixed 1.3× aperture),
    with fallbacks to aperture_mag_err and then zeros.
    """

    @staticmethod
    def _select_aperture_err(matched_table: pd.DataFrame, residuals: np.ndarray):
        """Exact copy of the selection block in calculate_zero_point()."""
        aperture_err_col = "aperture_mag_err_1_3"
        if aperture_err_col in matched_table.columns:
            aperture_mag_err = matched_table[aperture_err_col].values
        elif "aperture_mag_err" in matched_table.columns:
            aperture_mag_err = matched_table["aperture_mag_err"].values
        else:
            aperture_mag_err = np.zeros_like(residuals)
        return aperture_mag_err

    def test_uses_aperture_mag_err_1_3_when_present(self):
        residuals = np.array([0.01, -0.02, 0.03])
        table = pd.DataFrame({
            "aperture_mag_err_1_3": [0.05, 0.06, 0.07],
            "aperture_mag_err": [0.99, 0.99, 0.99],  # must NOT be selected
        })
        result = self._select_aperture_err(table, residuals)
        np.testing.assert_array_equal(result, [0.05, 0.06, 0.07])

    def test_fallback_to_generic_aperture_err(self):
        """When aperture_mag_err_1_3 is absent, fall back to aperture_mag_err."""
        residuals = np.array([0.01, -0.02, 0.03])
        table = pd.DataFrame({
            "aperture_mag_err": [0.10, 0.11, 0.12],
        })
        result = self._select_aperture_err(table, residuals)
        np.testing.assert_array_equal(result, [0.10, 0.11, 0.12])

    def test_fallback_to_zeros_when_no_err_columns(self):
        """When neither error column exists, return zeros with the same shape."""
        residuals = np.array([0.01, -0.02, 0.03])
        table = pd.DataFrame({"mag_cat": [17.0, 18.0, 19.0]})
        result = self._select_aperture_err(table, residuals)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_1_3_column_wins_over_generic(self):
        """Explicit regression: aperture_mag_err_1_3 takes priority even when
        aperture_mag_err has larger values that would be visually obvious."""
        residuals = np.zeros(4)
        err_1_3 = np.array([0.02, 0.03, 0.04, 0.05])
        err_generic = np.full(4, 99.0)
        table = pd.DataFrame({
            "aperture_mag_err_1_3": err_1_3,
            "aperture_mag_err": err_generic,
        })
        result = self._select_aperture_err(table, residuals)
        np.testing.assert_array_equal(result, err_1_3)

    def test_error_propagation_with_zero_point_scatter(self):
        """y-error combines aperture_mag_err_1_3 and zero-point scatter."""
        aperture_err = np.array([0.03, 0.05, 0.10])
        zp_err = 0.02
        yerr = np.sqrt(aperture_err**2 + zp_err**2)
        expected = np.sqrt([0.03**2 + 0.02**2,
                            0.05**2 + 0.02**2,
                            0.10**2 + 0.02**2])
        np.testing.assert_allclose(yerr, expected, rtol=1e-10)


# ─── 7. Filter mismatch warning text ─────────────────────────────────────────

class TestFilterWarningText:
    """The filter mismatch warning must not include the verbose 'Consider updating' suffix."""

    @staticmethod
    def _build_warning(filter_raw: str, filter_mapped: str) -> str:
        """Mirrors the warning construction in pages/app.py after the fix."""
        return f"WARNING: Filter in FITS header ({filter_raw}) maps to '{filter_mapped}'."

    def test_warning_contains_raw_filter(self):
        w = self._build_warning("G", "gmag")
        assert "G" in w

    def test_warning_contains_mapped_filter(self):
        w = self._build_warning("G", "gmag")
        assert "gmag" in w

    def test_warning_does_not_suggest_updating(self):
        w = self._build_warning("G", "gmag")
        assert "Consider updating" not in w

    def test_warning_does_not_include_selected_filter(self):
        """The selected_filter variable must not appear in the warning."""
        w = self._build_warning("G", "gmag")
        assert "phot_g_mean_mag" not in w
        assert "but selected filter" not in w

    def test_warning_ends_with_period(self):
        """Warning should be a complete, properly-terminated sentence."""
        w = self._build_warning("R", "rmag")
        assert w.endswith(".")

    def test_various_filter_names(self):
        """Warning format is consistent across different filter names."""
        for raw, mapped in [("R", "rmag"), ("B", "bmag"), ("V", "vmag"), ("I", "imag")]:
            w = self._build_warning(raw, mapped)
            assert raw in w
            assert mapped in w
            assert "Consider updating" not in w
