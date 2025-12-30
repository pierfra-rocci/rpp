# Visual Flow Diagrams

## Decision Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  cross_match_with_gaia() called                                      │
│  Parameters: filter_band, header with RA/DEC, pixel_size, FWHM     │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │ Validate header          │
        │ Extract RA/DEC           │
        │ Check coordinates valid  │
        └────────┬─────────────────┘
                 │
                 ▼
     ┌───────────────────────────────┐
     │ Calculate search radius:      │
     │ radius = max(NAXIS1,NAXIS2)  │
     │          × pixel_size / 1.5   │
     └────────┬──────────────────────┘
              │
              ▼
     ┌─────────────────────────┐
     │ Create WCS object       │
     │ pixel → sky conversion  │
     └────────┬────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │  Is filter_band in               │
   │  ["gmag", "rmag", "imag", "zmag"]?  │
   └──────┬──────────────────┬────────┘
      YES │                  │ NO
          │                  │
    ┌─────▼─────┐      ┌────▼──────────┐
    │ Sloan      │      │ GAIA/Synthetic│
    │ Filter     │      │ Filter        │
    │ Detected   │      └─────┬─────────┘
    └─────┬─────┘              │
          │                    ▼
          │          ┌─────────────────────┐
          │          │ Query GAIA DR3      │
          │          │ (cone_search)       │
          │          └────────┬────────────┘
          │                   │
          │                   ▼
          │          ┌──────────────────────┐
          │          │ Is synthetic photo   │
          │          │ band (not G/BP/RP)?  │
          │          │ (YES)                │
          │          │ Query synthetic      │
          │          │ photometry table     │
          │          │ Join with results    │
          │          └────────┬─────────────┘
          │                   │
          │                   ▼
          │         ┌─────────────────────┐
          │         │ GAIA Path Ready     │
          │         │ for filtering       │
          │         └────────┬────────────┘
          │                  │
    ┌─────▼──────────────────┘
    │
    ▼
┌────────────────────────────────────────┐
│  Hemisphere Check (Sloan only)        │
│  is_southern = (Dec < 0)?             │
└──┬────────────────────────────────┬───┘
   │ NO (northern)                  │ YES (southern)
   │                                │
┌──▼─────────────────┐  ┌───────────▼────────────┐
│ Query PANSTARRS    │  │ Query SkyMapper        │
│ DR1                │  │ DR2                    │
│ (northern coverage)│  │ (southern coverage)    │
└──┬─────────────────┘  └───────────┬────────────┘
   │                                │
   └──────────────┬─────────────────┘
                  │
                  ▼
      ┌───────────────────────────┐
      │ Catalog retrieved         │
      │ Apply magnitude filter    │
      │ mag < filter_max_mag      │
      └───────────┬───────────────┘
                  │
                  ▼
      ┌────────────────────────────┐
      │ Apply quality filters      │
      │ (catalog-specific)         │
      │                            │
      │ PANSTARRS/SkyMapper:       │
      │ - nDetections > 1          │
      │                            │
      │ GAIA:                      │
      │ - No variability           │
      │ - Color index OK           │
      │ - Astrometry OK (RUWE<1.6) │
      └───────────┬────────────────┘
                  │
                  ▼
      ┌────────────────────────────┐
      │ Find RA/Dec columns        │
      │ (smart detection)          │
      │                            │
      │ Priority:                  │
      │ 1. Try standard names      │
      │ 2. Try fallback names      │
      │ 3. Search available cols   │
      └───────────┬────────────────┘
                  │
                  ▼
      ┌────────────────────────────┐
      │ Create SkyCoord from       │
      │ catalog sources            │
      │ Match to detected sources  │
      │ Separation < 2.5×FWHM      │
      └───────────┬────────────────┘
                  │
                  ▼
      ┌────────────────────────────┐
      │ Extract matched sources    │
      │ Add magnitude from catalog │
      │ Remove invalid magnitudes  │
      │ Filter SNR > 1             │
      └───────────┬────────────────┘
                  │
                  ▼
      ┌────────────────────────────┐
      │ Assemble output table      │
      │ - catalog_source_id        │
      │ - catalog_separation_arcsec│
      │ - {filter_band} magnitude  │
      │ - original phot columns    │
      │                            │
      │ Log success message with   │
      │ catalog name and count     │
      └───────────┬────────────────┘
                  │
                  ▼
      ┌────────────────────────────┐
      │ RETURN                     │
      │ (matched_table,            │
      │  log_messages)             │
      └────────────────────────────┘
```

## Catalog Selection Matrix

```
╔════════════════════════════════════════════════════════════════════╗
║                  CATALOG SELECTION DECISION MATRIX                 ║
╠═════════════════════════════╦═══════════════════╦═════════════════╣
║ Filter Band                 ║ Northern (Dec≥0)  ║ Southern (Dec<0)║
╠═════════════════════════════╬═══════════════════╬═════════════════╣
║ gmag (Sloan g)              ║ PANSTARRS DR1     ║ SkyMapper DR2   ║
║ rmag (Sloan r)              ║ PANSTARRS DR1     ║ SkyMapper DR2   ║
║ imag (Sloan i)              ║ PANSTARRS DR1     ║ SkyMapper DR2   ║
║ zmag (Sloan z)              ║ PANSTARRS DR1     ║ SkyMapper DR2   ║
╠═════════════════════════════╬═══════════════════╬═════════════════╣
║ phot_g_mean_mag (GAIA G)    ║ GAIA DR3          ║ GAIA DR3        ║
║ phot_bp_mean_mag (GAIA BP)  ║ GAIA DR3          ║ GAIA DR3        ║
║ phot_rp_mean_mag (GAIA RP)  ║ GAIA DR3          ║ GAIA DR3        ║
║ u_jkc_mag (synth U)         ║ GAIA DR3 synth    ║ GAIA DR3 synth  ║
║ v_jkc_mag (synth V)         ║ GAIA DR3 synth    ║ GAIA DR3 synth  ║
║ b_jkc_mag (synth B)         ║ GAIA DR3 synth    ║ GAIA DR3 synth  ║
║ r_jkc_mag (synth R)         ║ GAIA DR3 synth    ║ GAIA DR3 synth  ║
║ i_jkc_mag (synth I)         ║ GAIA DR3 synth    ║ GAIA DR3 synth  ║
║ u_sdss_mag (synth u SDSS)   ║ GAIA DR3 synth    ║ GAIA DR3 synth  ║
╚═════════════════════════════╩═══════════════════╩═════════════════╝
```

## Data Flow Diagram

```
┌──────────────────────────────────┐
│  Input Data                      │
├──────────────────────────────────┤
│  _phot_table (photometry)        │
│  - xcenter, ycenter (pixels)     │
│  - flux, magnitude, snr          │
│                                  │
│  science_header (FITS header)    │
│  - RA, DEC (image center)        │
│  - NAXIS1, NAXIS2 (image size)   │
│  - WCS keywords                  │
│                                  │
│  pixel_size_arcsec               │
│  mean_fwhm_pixel                 │
│  filter_band (e.g., "gmag")      │
│  filter_max_mag (e.g., 20.0)     │
└──────────────────┬───────────────┘
                   │
        ┌──────────▼──────────┐
        │  Parameter Parsing  │
        │  & Validation       │
        └──────────┬──────────┘
                   │
        ┌──────────▼──────────────────────┐
        │  WCS Conversion                 │
        │  pixel (xcenter,ycenter) →      │
        │  sky (RA, DEC) coordinates      │
        └──────────┬──────────────────────┘
                   │
     ┌─────────────┴──────────────┐
     │                            │
  ┌──▼──────────┐           ┌────▼──────┐
  │ PANSTARRS   │           │ SkyMapper  │
  │ / SkyMapper │           │   or       │
  │ Query       │           │  GAIA      │
  │             │           │ Query      │
  │ Input:      │           │            │
  │ - RA, DEC   │           │ Input:     │
  │ - radius    │           │ - RA, DEC  │
  │ - filter    │           │ - radius   │
  │ - data_rel  │           │ - filter   │
  │             │           │            │
  │ Output:     │           │ Output:    │
  │ - Table     │           │ - Table    │
  │ - Sources   │           │ - Sources  │
  │   with:     │           │   with:    │
  │   RA, DEC   │           │   RA, DEC  │
  │   mags      │           │   mags     │
  └──┬──────────┘           └────┬───────┘
     │                           │
     └───────────┬───────────────┘
                 │
        ┌────────▼────────┐
        │ Filter Sources  │
        │                 │
        │ Magnitude cut   │
        │ Quality cuts    │
        │ (catalog-spec.) │
        │                 │
        │ Output: Table   │
        │ with filtered   │
        │ sources         │
        └────────┬────────┘
                 │
        ┌────────▼────────────────┐
        │ Spatial Matching        │
        │                         │
        │ Input:                  │
        │ - Detected source sky   │
        │   positions             │
        │ - Catalog sky positions │
        │ - Max separation        │
        │                         │
        │ Output:                 │
        │ - Matched indices       │
        │ - Separations           │
        └────────┬────────────────┘
                 │
        ┌────────▼──────────────────┐
        │ Assemble Output Table     │
        │                           │
        │ Input:                    │
        │ - Detected photometry     │
        │ - Catalog photometry      │
        │ - Match indices           │
        │                           │
        │ Output:                   │
        │ - pandas.DataFrame with:  │
        │   - All phot columns      │
        │   - catalog_source_id     │
        │   - {filter_band} mag     │
        │   - separation_arcsec     │
        └────────┬──────────────────┘
                 │
        ┌────────▼──────────────────┐
        │ Return Results            │
        │                           │
        │ (matched_table,           │
        │  log_messages)            │
        │                           │
        │ SUCCESS or FAILURE msg    │
        │ with detailed logging     │
        └───────────────────────────┘
```

## Quality Filter Comparison

```
╔═══════════════════════════════════════════════════════════════════════╗
║                      QUALITY FILTER COMPARISON                        ║
╠════════════════════════╦═════════════════════╦═════════════════════════╣
║ Filter Type            ║ PANSTARRS/SkyMapper ║ GAIA                    ║
╠════════════════════════╬═════════════════════╬═════════════════════════╣
║ Detection Quality      ║ nDetections > 1     ║ (N/A)                   ║
║                        ║ (multi-epoch)       ║                         ║
├────────────────────────┼─────────────────────┼─────────────────────────┤
║ Variability Flag       ║ (N/A)               ║ phot_variable_flag !=   ║
║                        ║                     ║ "VARIABLE"              ║
├────────────────────────┼─────────────────────┼─────────────────────────┤
║ Color Index            ║ (N/A)               ║ bp_rp in (-1, 2)        ║
║                        ║                     ║ (typical stars)         ║
├────────────────────────┼─────────────────────┼─────────────────────────┤
║ Astrometric Quality    ║ (N/A)               ║ ruwe < 1.6              ║
║                        ║                     ║ (well-measured positions)│
├────────────────────────┼─────────────────────┼─────────────────────────┤
║ SNR Threshold          ║ (Applied later)     ║ (Applied later)         ║
║                        ║ snr > 1             ║ snr > 1                 ║
╚════════════════════════╩═════════════════════╩═════════════════════════╝
```

## Column Name Resolution Flow

```
┌──────────────────────────────────┐
│  Detected catalog_table           │
│  (e.g., PANSTARRS or SkyMapper)   │
└────────────┬─────────────────────┘
             │
   ┌─────────▼────────┐
   │ Need RA/DEC cols │
   └─────────┬────────┘
             │
        ┌────▼──────────────────────┐
        │ Check RA column name       │
        │                            │
        │ Priority order:            │
        │ 1. Try "raMean" (PANSTARRS)│
        │ 2. Try "ra" (SkyMapper)    │
        │ 3. Search for "ra" (case  │
        │    insensitive in any col) │
        │ 4. ERROR if not found      │
        └────┬───────────────────────┘
             │
        ┌────▼──────────────────────┐
        │ Check DEC column name      │
        │                            │
        │ Priority order:            │
        │ 1. Try "decMean" (PANSTARRS│
        │ 2. Try "dec" (SkyMapper)   │
        │ 3. Search for "dec" (case │
        │    insensitive in any col) │
        │ 4. ERROR if not found      │
        └────┬───────────────────────┘
             │
        ┌────▼───────────────────┐
        │ Use detected RA/DEC    │
        │ columns for matching   │
        └────┬───────────────────┘
             │
        ┌────▼──────────────────┐
        │ SUCCESS: Proceed to   │
        │ source matching       │
        └───────────────────────┘
```

## Error Path Diagram

```
┌─────────────────────────────────┐
│  Error Conditions               │
└──────────┬──────────────────────┘
           │
   ┌───────┼───────┬─────────┬──────────────┬─────────────┐
   │       │       │         │              │             │
┌──▼─┐ ┌──▼──┐ ┌──▼──┐ ┌────▼──┐ ┌────────▼──┐ ┌──────▼──┐
│No  │ │Bad  │ │Bad  │ │Query  │ │No columns │ │No       │
│header  │RA/  │ │WCS  │ │Fails  │ │Found     │ │Matches  │
│   │ │DEC  │ │     │ │       │ │          │ │Found    │
└──┬─┘ └──┬──┘ └──┬──┘ └────┬──┘ └────────┬──┘ └──────┬──┘
   │      │      │         │             │            │
   │      │      │         │             │            │
   └──────┴──────┴─────┬────┴─────────────┴────────────┘
                       │
              ┌────────▼─────────┐
              │ RETURN:          │
              │ (None,           │
              │  [error_message])│
              │                  │
              │ Log:             │
              │ - Error type     │
              │ - Details        │
              │ - Suggestion (?) │
              └──────────────────┘
```

## Matching Tolerance Illustration

```
                        Separation tolerance
                    (2.5 × FWHM × pixel_size)
                                │
                                ▼
                        
                    ╔═══════════════════════╗
                    ║   Catalog Source      ║
                    ║   (exact position)    ║
                    ║                       ║
                    ║      *  ← center      ║
                    │     /│\               │
                    │    / │ \              │
                    │   /  │  \ ← tolerance │
                    │      │                │
                    ║   ◆ Detected Source   ║
                    ║   Must be within      ║
                    ║   this circle to      ║
                    ║   be considered a     ║
                    ║   match!              ║
                    ║                       ║
                    ╚═══════════════════════╝
                    
Typical values:
- FWHM: 2-4 pixels
- pixel_size: 0.3-1.0 arcsec/pixel
- Tolerance: 1.5-10 arcsec
- Most matches: 0.1-2 arcsec
```

---

## Summary of Diagrams

1. **Decision Flow:** Shows complete logical path from input to output
2. **Catalog Selection:** Matrix showing which catalog to use
3. **Data Flow:** Transformation of data through processing pipeline
4. **Quality Filters:** Comparison of filtering approaches
5. **Column Detection:** Smart fallback mechanism for column names
6. **Error Handling:** All error paths and recovery
7. **Matching Tolerance:** Visual representation of spatial matching constraint

These diagrams help understand both the high-level logic and implementation details.
