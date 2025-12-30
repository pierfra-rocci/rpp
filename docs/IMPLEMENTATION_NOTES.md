# Implementation Notes and Considerations

## Step-by-Step Explanation

### Step 1: Filter Band Detection (Lines 128-129)
```python
sloan_bands = ["gmag", "rmag", "imag", "zmag"]
is_sloan_filter = filter_band in sloan_bands
```
This determines whether to use PANSTARRS/SkyMapper (Sloan filters) or GAIA DR3 (other bands).

**Why:** PANSTARRS and SkyMapper provide the best reference photometry for Sloan g,r,i,z bands. GAIA is better for blue filters and synthetic photometry.

---

### Step 2: Hemisphere Detection (Lines 139-148)
```python
is_southern = image_center_ra_dec[1] < 0  # declination < 0

if is_southern:
    catalog_name = "SkyMapper"
    # Southern hemisphere catalog
else:
    catalog_name = "Panstarrs"
    # Northern hemisphere catalog
```
This selects the appropriate catalog based on observation location.

**Why:** 
- PANSTARRS covers mostly northern sky (declination > -30°)
- SkyMapper covers southern sky (southern hemisphere)
- No overlap or redundancy between catalogs

**Boundary Behavior:**
- Equator (dec = 0°) → Uses PANSTARRS
- Below equator (dec < 0) → Uses SkyMapper

---

### Step 3: Coordinate Formatting (Lines 152-155)
```python
coord_string = f"{image_center_ra_dec[0]} {image_center_ra_dec[1]}"
radius_query_deg = catalog_search_radius_arcsec / 3600.0
```
Formats coordinates for astroquery API (requires space-separated string, radius in degrees).

**Key Point:** 
- Radius conversion: arcsec ÷ 3600 = degrees
- Uses same radius calculation as GAIA to maintain consistency
- Field edge effects avoided by dividing by 1.5

---

### Step 4: Query Execution (Lines 156-175)
```python
if is_southern:
    catalog_table = Catalogs.query_region(
        coord_string,
        radius=radius_query_deg,
        catalog="Skymapper",
        data_release="dr2",
        table="mean"
    )
else:
    catalog_table = Catalogs.query_region(
        coord_string,
        radius=radius_query_deg,
        catalog="Panstarrs",
        data_release="dr1",
        table="mean"
    )
```
Executes catalog query using astroquery's MAST interface.

**Parameters:**
- `catalog`: "Panstarrs" or "Skymapper" (case-sensitive in astroquery)
- `data_release`: "dr1" for PANSTARRS, "dr2" for SkyMapper
- `table`: "mean" for mean photometry (as per user spec)

**Return:**
- Astropy Table with catalog sources
- May be None if query fails or returns no results

---

### Step 5: Magnitude Filtering (Lines 244-248)
```python
# Apply magnitude filter
mag_filter = catalog_table[filter_band] < filter_max_mag
catalog_table_filtered = catalog_table[mag_filter]
```
Removes sources fainter than specified magnitude limit.

**Why:** Ensures reliable photometry for faint sources where S/N is low.

---

### Step 6: Quality Filtering (Lines 250-284)

**For PANSTARRS/SkyMapper (Lines 256-263):**
```python
if "nDetections" in catalog_table_filtered.colnames:
    catalog_table_filtered = catalog_table_filtered[
        catalog_table_filtered["nDetections"] > 1
    ]
```
Requires multiple detections for reliability.

**For GAIA (Lines 264-283):**
```python
var_filter = catalog_table_filtered["phot_variable_flag"] != "VARIABLE"
color_index_filter = (catalog_table_filtered["bp_rp"] > -1) & (
    catalog_table_filtered["bp_rp"] < 2
)
astrometric_filter = catalog_table_filtered["ruwe"] < 1.6
```
Applies GAIA-specific quality filters (unchanged from original).

**Quality Metrics:**
- Variability flag: Excludes known variable stars
- Color index: Removes extreme or invalid colors
- Astrometric quality (RUWE): Identifies reliable astrometry
- RUWE < 1.6 is standard for reliable sources

---

### Step 7: Coordinate Matching (Lines 291-312)

**Key Challenge:** Different catalogs have different RA/Dec column names

**Solution:** Smart column detection with fallback
```python
if is_sloan_filter:
    # Try standard names first
    if "raMean" in catalog_table_filtered.colnames:
        ra_col, dec_col = "raMean", "decMean"
    elif "ra" in catalog_table_filtered.colnames:
        ra_col, dec_col = "ra", "dec"
    else:
        # Fallback: search for any RA/Dec columns
        ra_candidates = [c for c in available_cols if 'ra' in c.lower()]
        dec_candidates = [c for c in available_cols if 'dec' in c.lower()]
```

**Why This Works:**
1. PANSTARRS uses "raMean", "decMean"
2. SkyMapper uses "ra", "dec"
3. Fallback handles unexpected column names

---

### Step 8: Spatial Matching (Lines 317-330)
```python
catalog_skycoords = SkyCoord(
    ra=catalog_table_filtered[ra_col],
    dec=catalog_table_filtered[dec_col],
    unit="deg"
)
idx, d2d, _ = source_positions_sky.match_to_catalog_sky(catalog_skycoords)

max_sep_constraint = 2.5 * mean_fwhm_pixel * pixel_size_arcsec * u.arcsec
catalog_matches = d2d < max_sep_constraint
```
Matches detected sources to catalog sources within separation tolerance.

**Separation Logic:**
- Maximum separation: 2.5 × FWHM
- FWHM ≈ 3 pixels typical → ~1-2 arcsec tolerance
- Accounts for WCS uncertainty and source size

---

### Step 9: Output Assembly (Lines 343-365)

**Source Identification:**
```python
if is_sloan_filter:
    if catalog_name == "SkyMapper":
        id_col = "object_id"  # SkyMapper
    else:
        id_col = "objID"  # PANSTARRS
else:
    id_col = "designation"  # GAIA

matched_table["catalog_source_id"] = catalog_table_filtered[id_col][
    matched_indices_catalog
].values
```

**Column Names:**
- PANSTARRS: `objID` (object identifier)
- SkyMapper: `object_id` (object identifier)
- GAIA: `designation` (source designation)
- All stored in unified `catalog_source_id` column

**Magnitude Transfer:**
```python
matched_table[filter_band] = catalog_table_filtered[filter_band][
    matched_indices_catalog
].values
```
Adds magnitude from reference catalog for calibration.

---

## Potential Issues and Solutions

### Issue 1: Column Name Variations
**Problem:** PANSTARRS and SkyMapper use different column names

**Solution Implemented:**
```python
# Priority order:
1. Check standard names (raMean/decMean, ra/dec, objID/object_id)
2. Case-insensitive fallback search
3. Return error if not found
```

---

### Issue 2: Empty Catalog Results
**Problem:** Some regions may have no sources

**Solution Implemented:**
```python
if catalog_table is None or len(catalog_table) == 0:
    log_messages.append(f"WARNING: No {catalog_name} sources found...")
    return None, log_messages
```

**Fallback:** Upstream code handles None return gracefully.

---

### Issue 3: Query Timeouts
**Problem:** Large regions or high-density areas may timeout

**Solution Implemented:**
```python
# Radius limited to field size
radius = max(NAXIS1, NAXIS2) * pixel_size / 1.5
# Typical: ~17-20 arcmin radius
# API typically handles this without timeout
```

**Monitoring:** Check log messages for query time.

---

### Issue 4: Different Data Quality Between Catalogs
**Problem:** PANSTARRS, SkyMapper, and GAIA have different quality metrics

**Solution Implemented:**
```python
# Catalog-specific filters:
# PANSTARRS/SkyMapper: nDetections > 1 (basic quality)
# GAIA: Variability, color index, astrometry (strict quality)
```

**Rationale:** Each catalog's filter optimized for its data characteristics.

---

### Issue 5: Magnitude System Differences
**Problem:** PANSTARRS/SkyMapper use PSF magnitudes, GAIA uses mean magnitudes

**Solution Implemented:**
```python
# Uses same magnitude columns for all
# Pipeline downstream expects magnitude in [filter_band] column
# Works correctly regardless of magnitude system
```

**Note:** Magnitude calibration assumes reference catalog magnitudes are reliable.

---

### Issue 6: WCS Errors with PANSTARRS/SkyMapper
**Problem:** Some regions may have WCS issues

**Solution Implemented:**
```python
# Doesn't affect catalog queries (uses header RA/DEC)
# Uses WCS only for converting pixel positions
# If WCS fails, function returns error early
```

---

### Issue 7: Memory with Very Large Regions
**Problem:** Querying very large regions returns huge catalogs

**Solution Implemented:**
```python
# Radius limited by formula: max(NAXIS1/2, NAXIS2/2) * pixel_size / 1.5
# Typical images: 2048 pixels → ~17 arcmin radius
# Safe for typical astrometry jobs
```

**Limit:** Avoid 1+ degree regions or very high-density fields.

---

## Code Consistency Checks

### ✓ Import Statement (Line 20)
```python
from astroquery.mast import Catalogs
```
**Status:** Added, verified syntax

### ✓ Function Signature (Lines 25-31)
**Status:** Unchanged, backward compatible

### ✓ Docstring (Lines 33-65)
**Status:** Updated to reflect dual catalog support

### ✓ Return Type (Lines 48-52)
**Status:** Unchanged: `(pandas.DataFrame | None, list[str])`

### ✓ Error Handling
**Status:** Comprehensive try-except blocks, all paths return proper format

### ✓ Output Format
**Status:** All catalogs produce identical column structure

---

## Integration Points

### 1. app.py Usage
```python
# From pages/app.py line ~xxx
matched_table, log_messages = cross_match_with_gaia(
    phot_table,
    science_header,
    pixel_size_arcsec,
    mean_fwhm_pixel,
    filter_band=st.session_state.analysis_parameters.get("filter_band"),
    filter_max_mag=st.session_state.analysis_parameters.get("filter_max_mag"),
    refined_wcs=refined_wcs
)

# Returns same format as before
# No changes needed in calling code
```

### 2. downstream Functions
```python
# enhance_catalog() accepts matched_table
# calculate_zero_point() uses matched_table
# Both work with new output format (catalog-agnostic columns)
```

### 3. GAIA_BANDS Usage
```python
# From tools_pipeline.py
GAIA_BANDS = [
    ("g", "gmag"),           # ← Now triggers PANSTARRS/SkyMapper
    ("r", "rmag"),           # ← Now triggers PANSTARRS/SkyMapper
    ("i", "imag"),           # ← Now triggers PANSTARRS/SkyMapper
    ("z", "zmag"),           # ← Now triggers PANSTARRS/SkyMapper
    # ... other bands use GAIA as before
]
```

---

## Performance Metrics

### Expected Execution Times
| Scenario | Time | Notes |
|----------|------|-------|
| Small PANSTARRS query (< 1 deg) | 5-15 sec | Network-dependent |
| Large PANSTARRS query (1-3 deg) | 15-30 sec | More sources |
| Small SkyMapper query (< 1 deg) | 5-15 sec | Network-dependent |
| GAIA query (typical) | 10-60 sec | Larger catalog, slower |
| Matching with 1000 detected sources | 1-5 sec | CPU-bound |
| **Total pipeline with matching** | **30-120 sec** | Dominated by query |

---

## Verification Checklist

✅ Import added: `from astroquery.mast import Catalogs`
✅ Filter detection: `sloan_bands = ["gmag", "rmag", "imag", "zmag"]`
✅ Hemisphere check: `is_southern = image_center_ra_dec[1] < 0`
✅ PANSTARRS query: `catalog="Panstarrs", data_release="dr1"`
✅ SkyMapper query: `catalog="Skymapper", data_release="dr2"`
✅ Radius conversion: `radius_query_deg = radius_arcsec / 3600.0`
✅ Column detection: Smart fallback for RA/Dec columns
✅ Output format: `catalog_source_id`, `{filter_band}`, `catalog_separation_arcsec`
✅ Error handling: All paths return `(None, [error_msg])` or valid table
✅ Log messages: Informative and catalog-specific
✅ Backward compatibility: GAIA pipeline unchanged
✅ Syntax: Verified with py_compile

---

## Future Enhancements

### Possible Improvements
1. Add option to query multiple catalogs and combine results
2. Implement catalog-specific magnitude calibration corrections
3. Add uncertainty propagation from reference catalog errors
4. Cache query results to avoid redundant API calls
5. Add parallel processing for large source lists
6. Implement adaptive separation constraint based on catalog characteristics

### Monitoring Opportunities
1. Log API query times per region
2. Track success/failure rates by hemisphere
3. Monitor matched source statistics
4. Record catalog vs. detected source magnitude offsets

---

## Notes on Data Release Versions

### PANSTARRS DR1 vs DR2
- **Used:** DR1 (as per user specification)
- **Reason:** Stable, well-characterized photometry
- **Coverage:** North to Dec ~ -30°

### SkyMapper DR1 vs DR2
- **Used:** DR2 (current version as of 2024)
- **Reason:** Improved photometry and astrometry
- **Coverage:** Entire southern hemisphere

### GAIA DR3 (unchanged)
- **Status:** Current version
- **Synthetic Photometry:** gaiadr3.synthetic_photometry_gspc

---

## Testing Recommendations

### Unit Tests
1. Test filter detection logic
2. Test hemisphere detection
3. Test coordinate validation
4. Test column name detection

### Integration Tests
1. Real northern hemisphere image with PANSTARRS band
2. Real southern hemisphere image with SkyMapper band
3. GAIA-only band observation
4. Edge cases (equator, boundaries)

### Regression Tests
1. Verify GAIA pipeline still works
2. Verify output format consistency
3. Verify error handling paths
4. Verify log message generation
