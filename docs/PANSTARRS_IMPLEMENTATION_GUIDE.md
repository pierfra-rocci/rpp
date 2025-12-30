# Quick Reference: PANSTARRS/SkyMapper Integration

## What Changed?

The `cross_match_with_gaia()` function now intelligently selects which reference catalog to use based on:
1. The filter band selected
2. The hemisphere of observation (declination)

## Decision Tree

```
Is filter_band in [gmag, rmag, imag, zmag]?
│
├─ YES (Sloan filter)
│  │
│  └─ Is declination < 0?
│     │
│     ├─ YES → Use SkyMapper DR2 (southern hemisphere)
│     └─ NO  → Use PANSTARRS DR1 (northern hemisphere)
│
└─ NO (GAIA band)
   └─ Use GAIA DR3 (with synthetic photometry if needed)
```

## Supported Filters by Catalog

| Filter | Field Name | Catalog (North) | Catalog (South) |
|--------|-----------|-----------------|-----------------|
| **g** | gmag | PANSTARRS DR1 | SkyMapper DR2 |
| **r** | rmag | PANSTARRS DR1 | SkyMapper DR2 |
| **i** | imag | PANSTARRS DR1 | SkyMapper DR2 |
| **z** | zmag | PANSTARRS DR1 | SkyMapper DR2 |
| **G** | phot_g_mean_mag | GAIA DR3 | GAIA DR3 |
| **BP** | phot_bp_mean_mag | GAIA DR3 | GAIA DR3 |
| **RP** | phot_rp_mean_mag | GAIA DR3 | GAIA DR3 |
| **U,V,B,R,I** | u_jkc_mag, etc. | GAIA DR3 Synthetic | GAIA DR3 Synthetic |

## Code Sections to Know

### 1. Filter Detection (Line 129)
```python
sloan_bands = ["gmag", "rmag", "imag", "zmag"]
is_sloan_filter = filter_band in sloan_bands
```

### 2. Hemisphere Check (Line 134)
```python
is_southern = image_center_ra_dec[1] < 0  # dec < 0 = Southern
```

### 3. Query Execution (Lines 157-175)
- **PANSTARRS:** `Catalogs.query_region(..., catalog="Panstarrs", data_release="dr1", ...)`
- **SkyMapper:** `Catalogs.query_region(..., catalog="Skymapper", data_release="dr2", ...)`

### 4. Output Consistency
All catalogs produce identical output structure:
- `catalog_source_id`: Unique identifier
- `catalog_separation_arcsec`: Positional error
- `{filter_band}`: Magnitude value
- All original photometry columns preserved

## No Breaking Changes

✅ Existing code continues to work unchanged
✅ Return values have same format
✅ Log messages enhanced but compatible
✅ Error handling improved
✅ GAIA pipeline fully functional for non-Sloan bands

## Key Implementation Details

### Radius Calculation
Both PANSTARRS and SkyMapper use the same radius as GAIA:
```
radius = max(NAXIS1, NAXIS2) × pixel_size_arcsec / 1.5
```
This avoids field edge effects common in astronomy.

### Quality Filters
- **PANSTARRS/SkyMapper:** Detection quality (nDetections > 1)
- **GAIA:** Variability, color index, astrometric quality

### Matching Tolerance
Separation constraint: `2.5 × FWHM (pixels) × pixel_size_arcsec`
Same for all catalogs to maintain consistency.

## Error Handling

The function now handles:
✓ Missing coordinate keywords
✓ Invalid coordinates
✓ Missing RA/Dec columns in PANSTARRS/SkyMapper
✓ Empty catalog results
✓ Query failures
✓ Column name variations

All errors return `(None, [error_messages])` for graceful handling.
