# Test Scenarios for PANSTARRS/SkyMapper Integration

## Test Case 1: Northern Hemisphere with PANSTARRS (Sloan g-band)

### Setup
- **Image Header:**
  - RA = 120.5°
  - Dec = 45.0° (North)
  - NAXIS1 = 2048
  - NAXIS2 = 2048

- **Parameters:**
  - filter_band = "gmag"
  - filter_max_mag = 20.0
  - pixel_size_arcsec = 0.5
  - mean_fwhm_pixel = 3.0

### Expected Behavior
```
1. ✓ Detects filter_band "gmag" in sloan_bands
2. ✓ Calculates is_sloan_filter = True
3. ✓ Checks is_southern = (45.0 < 0) = False
4. ✓ Selects PANSTARRS DR1 catalog
5. ✓ Queries region at 120.5, 45.0 with radius ~17 arcmin
6. ✓ Applies magnitude filter (gmag < 20.0)
7. ✓ Applies quality filter (nDetections > 1)
8. ✓ Matches sources within separation constraint
9. ✓ Returns matched_table with catalog_source_id column
10. ✓ Log: "SUCCESS: Found X PANSTARRS matches after filtering"
```

### Assertion Points
```python
assert matched_table is not None
assert "catalog_source_id" in matched_table.columns
assert "gmag" in matched_table.columns
assert "catalog_separation_arcsec" in matched_table.columns
assert len(log_messages) > 0
assert any("PANSTARRS" in msg for msg in log_messages)
```

---

## Test Case 2: Southern Hemisphere with SkyMapper (Sloan i-band)

### Setup
- **Image Header:**
  - RA = 250.0°
  - Dec = -30.5° (South)
  - NAXIS1 = 2048
  - NAXIS2 = 2048

- **Parameters:**
  - filter_band = "imag"
  - filter_max_mag = 19.5
  - pixel_size_arcsec = 0.55
  - mean_fwhm_pixel = 2.8

### Expected Behavior
```
1. ✓ Detects filter_band "imag" in sloan_bands
2. ✓ Calculates is_sloan_filter = True
3. ✓ Checks is_southern = (-30.5 < 0) = True
4. ✓ Selects SkyMapper DR2 catalog
5. ✓ Queries region at 250.0, -30.5 with radius ~17 arcmin
6. ✓ Applies magnitude filter (imag < 19.5)
7. ✓ Applies quality filter (nDetections > 1)
8. ✓ Matches sources within separation constraint
9. ✓ Returns matched_table with catalog_source_id column
10. ✓ Log: "SUCCESS: Found X SkyMapper matches after filtering"
```

### Assertion Points
```python
assert matched_table is not None
assert "catalog_source_id" in matched_table.columns
assert "imag" in matched_table.columns
assert len(log_messages) > 0
assert any("SkyMapper" in msg for msg in log_messages)
assert any("southern hemisphere" in msg.lower() for msg in log_messages)
```

---

## Test Case 3: GAIA Band at Northern Hemisphere

### Setup
- **Image Header:**
  - RA = 150.0°
  - Dec = 60.0° (North)
  - NAXIS1 = 2048
  - NAXIS2 = 2048

- **Parameters:**
  - filter_band = "phot_g_mean_mag"
  - filter_max_mag = 18.0
  - pixel_size_arcsec = 0.5
  - mean_fwhm_pixel = 3.0

### Expected Behavior
```
1. ✓ Detects filter_band "phot_g_mean_mag" NOT in sloan_bands
2. ✓ Calculates is_sloan_filter = False
3. ✓ Skips hemisphere check
4. ✓ Uses GAIA DR3 catalog
5. ✓ Executes cone_search on GAIA
6. ✓ Applies GAIA quality filters (variability, color, astrometry)
7. ✓ Matches sources within separation constraint
8. ✓ Returns matched_table with catalog_source_id column
9. ✓ Log: "SUCCESS: Found X Gaia matches after filtering"
```

### Assertion Points
```python
assert matched_table is not None
assert "catalog_source_id" in matched_table.columns
assert "phot_g_mean_mag" in matched_table.columns
assert any("Gaia" in msg for msg in log_messages)
assert any("GAIA" in msg or "Gaia" in msg for msg in log_messages)
```

---

## Test Case 4: GAIA Synthetic Photometry Band at Southern Hemisphere

### Setup
- **Image Header:**
  - RA = 200.0°
  - Dec = -45.0° (South)
  - NAXIS1 = 2048
  - NAXIS2 = 2048

- **Parameters:**
  - filter_band = "u_sdss_mag"
  - filter_max_mag = 20.0
  - pixel_size_arcsec = 0.5
  - mean_fwhm_pixel = 3.0

### Expected Behavior
```
1. ✓ Detects filter_band "u_sdss_mag" NOT in sloan_bands
2. ✓ Calculates is_sloan_filter = False
3. ✓ Uses GAIA DR3 catalog (not affected by hemisphere)
4. ✓ Queries synthetic photometry from gaiadr3.synthetic_photometry_gspc
5. ✓ Applies GAIA quality filters
6. ✓ Matches sources within separation constraint
7. ✓ Returns matched_table with u_sdss_mag column
8. ✓ Log includes synthetic photometry query info
```

### Assertion Points
```python
assert matched_table is not None
assert "u_sdss_mag" in matched_table.columns
assert any("synthetic" in msg.lower() for msg in log_messages)
```

---

## Test Case 5: Error Handling - Missing Coordinates

### Setup
- **Image Header:** (missing RA/DEC)
  - RA = None
  - DEC = None
  - NAXIS1 = 2048
  - NAXIS2 = 2048

- **Parameters:**
  - filter_band = "gmag"
  - Other parameters valid

### Expected Behavior
```
1. ✓ Detects missing RA/DEC in header
2. ✓ Returns (None, ["ERROR: Missing RA/DEC coordinates in header"])
3. ✓ No catalog query attempted
```

### Assertion Points
```python
assert matched_table is None
assert any("Missing RA/DEC" in msg for msg in log_messages)
```

---

## Test Case 6: Error Handling - Invalid Coordinates

### Setup
- **Image Header:**
  - RA = 450.0° (Invalid, > 360)
  - Dec = 45.0°
  - NAXIS1 = 2048
  - NAXIS2 = 2048

- **Parameters:**
  - filter_band = "gmag"
  - Other parameters valid

### Expected Behavior
```
1. ✓ Detects invalid RA value
2. ✓ Returns (None, ["ERROR: Invalid coordinates: RA=450.0, DEC=45.0"])
3. ✓ No catalog query attempted
```

### Assertion Points
```python
assert matched_table is None
assert any("Invalid coordinates" in msg for msg in log_messages)
```

---

## Test Case 7: Empty Catalog Results

### Setup
- **Image Header:**
  - RA = 5.0° (Region with potentially few/no sources)
  - Dec = -89.0° (Near south pole)
  - NAXIS1 = 2048
  - NAXIS2 = 2048

- **Parameters:**
  - filter_band = "imag"
  - filter_max_mag = 25.0 (Very faint limit)
  - Other parameters valid

### Expected Behavior
```
1. ✓ Detects Sloan filter → SkyMapper
2. ✓ Executes query_region
3. ✓ Gets empty or None result
4. ✓ Returns (None, ["WARNING: No SkyMapper sources found within search radius."])
```

### Assertion Points
```python
assert matched_table is None
assert any("sources found" in msg for msg in log_messages)
```

---

## Test Case 8: No Matches Within Separation Constraint

### Setup
- **Image Header:**
  - RA = 120.0°
  - Dec = 30.0°
  - NAXIS1 = 2048
  - NAXIS2 = 2048

- **Parameters:**
  - filter_band = "rmag"
  - filter_max_mag = 20.0
  - mean_fwhm_pixel = 0.5 (Very small FWHM)
  - pixel_size_arcsec = 0.1 (Very small pixels)
  - **Result:** Very tight separation constraint ~0.125 arcsec

### Expected Behavior
```
1. ✓ Queries PANSTARRS successfully
2. ✓ Gets catalog sources
3. ✓ Applies magnitude filter
4. ✓ Attempts matching with very tight constraint
5. ✓ No sources meet constraint
6. ✓ Returns (None, ["WARNING: No ... matches found within the separation constraint."])
```

### Assertion Points
```python
assert matched_table is None
assert any("separation constraint" in msg for msg in log_messages)
```

---

## Integration Test: Full Pipeline

### Scenario
1. Load real FITS image from northern hemisphere with PANSTARRS coverage
2. Extract photometry (sources with xcenter, ycenter)
3. Call cross_match_with_gaia with gmag filter
4. Verify matched_table contains all expected columns
5. Pass matched_table to subsequent pipeline steps
6. Verify no breaking changes in downstream functions

### Expected Result
```
✓ Pipeline executes without errors
✓ Catalog sources properly matched
✓ Magnitudes correctly transferred
✓ Downstream functions receive expected data format
✓ Final photometry table includes catalog calibration data
```

---

## Column Name Verification

### PANSTARRS DR1 Expected Columns
```python
required_cols = ["raMean", "decMean", "gmag", "rmag", "imag", "zmag", "objID"]
optional_cols = ["nDetections", "gMeanPSFMag", "rMeanPSFMag", ...]
```

### SkyMapper DR2 Expected Columns
```python
required_cols = ["ra", "dec", "g_psf", "r_psf", "i_psf", "z_psf", "object_id"]
optional_cols = ["n_detections", ...]
```

### Test: Column Detection
```python
# The function should handle column name variations
# Primary: raMean/decMean for PANSTARRS, ra/dec for SkyMapper
# Fallback: Any column with 'ra'/'dec' in name (case-insensitive)

assert ra_col in ["raMean", "ra"] or "ra" in ra_col.lower()
assert dec_col in ["decMean", "dec"] or "dec" in dec_col.lower()
```

---

## Performance Considerations

### Expected Query Times
- **PANSTARRS:** 5-30 seconds (depends on region density)
- **SkyMapper:** 5-30 seconds (depends on region density)
- **GAIA:** 10-60 seconds (larger catalog)

### Memory Usage
- **Small region (< 1 degree):** ~ 50-100 MB
- **Large region (1-5 degrees):** ~ 100-500 MB
- **Very large region (> 5 degrees):** May exceed API limits

### Recommended Monitoring
- Query execution time (log message timestamps)
- Number of retrieved sources
- Number of filtered sources
- Final match count
