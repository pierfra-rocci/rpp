# PANSTARRS DR1 and SkyMapper Integration

## Overview
Modified the `cross_match_with_gaia()` function in `src/xmatch_catalogs.py` to support PANSTARRS DR1 (northern hemisphere) and SkyMapper DR2 (southern hemisphere) for Sloan filter photometry (g, r, i, z), while maintaining GAIA DR3 as the reference catalog for other filter bands.

## Changes Made

### 1. **Import Addition** (Line 20)
Added astroquery Catalogs module for PANSTARRS and SkyMapper queries:
```python
from astroquery.mast import Catalogs
```

### 2. **Function Docstring Update** (Lines 33-65)
Updated documentation to reflect dual catalog support:
- Clarifies that Sloan filters (gmag, rmag, imag, zmag) use PANSTARRS DR1 or SkyMapper
- GAIA bands (phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag) and synthetic photometry bands use GAIA DR3
- Notes hemisphere-based catalog selection

### 3. **Coordinate and Radius Calculation** (Lines 110-127)
- Added `radius_query_deg` for degree-based radius (required by Catalogs API)
- Maintained existing `radius_query` for GAIA (arcsec units)
- Radius calculation consistent with GAIA approach: `max(NAXIS1, NAXIS2) * pixel_size_arcsec / 1.5`

### 4. **Catalog Selection Logic** (Lines 129-160)
```python
# Determine if we should use PANSTARRS or GAIA based on filter_band
sloan_bands = ["gmag", "rmag", "imag", "zmag"]
is_sloan_filter = filter_band in sloan_bands

if is_sloan_filter:
    # For Sloan filters (g, r, i, z), use PANSTARRS DR1 or SkyMapper
    # Check if we're in southern hemisphere
    is_southern = image_center_ra_dec[1] < 0
    
    if is_southern:
        # Use SkyMapper DR2 (southern hemisphere)
        catalog_table = Catalogs.query_region(
            coord_string,
            radius=radius_query_deg,
            catalog="Skymapper",
            data_release="dr2",
            table="mean"
        )
    else:
        # Use PANSTARRS DR1 (northern hemisphere)
        catalog_table = Catalogs.query_region(
            coord_string,
            radius=radius_query_deg,
            catalog="Panstarrs",
            data_release="dr1",
            table="mean"
        )
else:
    # Continue with existing GAIA logic
```

**Hemisphere Detection:**
- Southern Hemisphere: `dec < 0` → SkyMapper DR2
- Northern Hemisphere: `dec ≥ 0` → PANSTARRS DR1

### 5. **Quality Filters** (Lines 244-284)
- **PANSTARRS/SkyMapper:** Basic detection quality filter (`nDetections > 1`)
- **GAIA:** Original quality filters maintained:
  - Variability flag check
  - Color index filter (BP-RP: -1 to 2)
  - Astrometric quality (RUWE < 1.6)

### 6. **Coordinate Matching** (Lines 291-340)
Flexible RA/Dec column name handling:

**PANSTARRS:**
- Primary columns: `raMean`, `decMean`
- Fallback: `ra`, `dec`

**SkyMapper:**
- Primary columns: `ra`, `dec`
- Smart fallback to any available RA/Dec columns

**GAIA:**
- Standard: `ra`, `dec`

### 7. **Source Identification** (Lines 343-360)
Output format remains consistent:
- Uses generic `catalog_source_id` column (catalog-agnostic)
- Automatically detects appropriate ID column:
  - **SkyMapper:** `object_id`
  - **PANSTARRS:** `objID`
  - **GAIA:** `designation`

### 8. **Cross-matching Algorithm** (Lines 369-385)
- Separation constraint: `2.5 * mean_fwhm_pixel * pixel_size_arcsec` arcsec (same as before)
- Magnitude validity check: `np.isfinite()`
- SNR filtering: Remove sources with `snr < 1`
- Output: Single unified DataFrame with catalog-agnostic column names

## Output Consistency

The matched table structure remains uniform across all catalogs:
- `catalog_separation_arcsec`: Separation between detected and catalog source
- `catalog_source_id`: Source identifier from reference catalog
- `{filter_band}`: Magnitude in selected band
- All original photometry columns preserved

## GAIA_BANDS Integration

From `src/tools_pipeline.py`:
```python
GAIA_BANDS = [
    ("G", "phot_g_mean_mag"),           # GAIA
    ("BP", "phot_bp_mean_mag"),         # GAIA
    ("RP", "phot_rp_mean_mag"),         # GAIA
    ("U", "u_jkc_mag"),                 # GAIA synthetic
    ("V", "v_jkc_mag"),                 # GAIA synthetic
    ("B", "b_jkc_mag"),                 # GAIA synthetic
    ("R", "r_jkc_mag"),                 # GAIA synthetic
    ("I", "i_jkc_mag"),                 # GAIA synthetic
    ("u", "u_sdss_mag"),                # GAIA synthetic
    ("g", "gmag"),                      # PANSTARRS/SkyMapper
    ("r", "rmag"),                      # PANSTARRS/SkyMapper
    ("i", "imag"),                      # PANSTARRS/SkyMapper
    ("z", "zmag"),                      # PANSTARRS/SkyMapper
]
```

The function now uses these 4 Sloan bands (gmag, rmag, imag, zmag) as markers to trigger PANSTARRS/SkyMapper queries, while all other bands use GAIA DR3 with optional synthetic photometry.

## Usage

No API changes required - existing code continues to work:
```python
matched_table, log_messages = cross_match_with_gaia(
    phot_table,
    science_header,
    pixel_size_arcsec,
    mean_fwhm_pixel,
    filter_band="gmag",           # Triggers PANSTARRS/SkyMapper
    filter_max_mag=20.0,
    refined_wcs=None
)
```

## Error Handling

- Graceful fallback if catalog query fails
- Missing RA/Dec columns detection for both PANSTARRS and SkyMapper
- Comprehensive log messages indicating catalog selection and processing steps
- Original GAIA pipeline remains fully functional for non-Sloan bands

## Testing Recommendations

1. **Northern Hemisphere Image:**
   - Dec > 0
   - Filter: gmag, rmag, imag, or zmag
   - Verify PANSTARRS DR1 catalog is used

2. **Southern Hemisphere Image:**
   - Dec < 0
   - Filter: gmag, rmag, imag, or zmag
   - Verify SkyMapper DR2 catalog is used

3. **GAIA Bands:**
   - Filter: phot_g_mean_mag or synthetic photometry bands
   - Verify GAIA DR3 is used regardless of hemisphere

4. **Fallback Behavior:**
   - Test with invalid coordinates
   - Verify error messages and log output
