# Pipeline Migration Summary: Potassium → Nitrate (NO3)

## Overview

This document summarizes the migration from the potassium (K2O) prediction pipeline to the nitrate (NO3) prediction pipeline. All source files have been copied and adapted with nitrate-specific configurations based on LIBS spectroscopy literature.

## Date

2025-10-24

## Key Changes

### 1. Spectral Regions (`src/config/pipeline_config.py`)

#### Primary Detection Region
- **Potassium**: K I doublet at 766.49, 769.90 nm
- **Nitrate**: N I triplet at 742.36, 744.23, 746.83 nm (742-747 nm range)
  - Most sensitive region for soil nitrogen detection

#### Context Regions
**Nitrate-specific additions:**
- **Secondary nitrogen lines**:
  - N I 821: 821.63, 824.21 nm (near-IR doublet for verification)
  - N I 868: 868.03, 871.17 nm (additional verification)
- **Supporting elements** (critical for nitrogen speciation):
  - O_I: 777.4 nm (for N/O ratio - nitrate indicator)
  - H_alpha: 656.3 nm (for N/H ratio - NH4 vs NO3 discrimination)
  - C_I: 833.5 nm (for N/C ratio - baseline normalization)
  - P_I, K_I: NPK nutrient context
  - CA_II_393: Soil chemistry context

#### Molecular Bands
**ENABLED for nitrate** (CRITICAL for nitrogen speciation):
- CN_violet_1, CN_violet_2, CN_red: Organic nitrogen indicators
- NH_band: Ammonium (NH4) indicator
- NO_band, NO_gamma: Direct nitrate/nitrite indicators

**Configuration changes:**
```python
enable_molecular_bands: bool = True   # CRITICAL for NO3
enable_oxygen_hydrogen: bool = True   # CRITICAL for NO3 vs NH4
```

### 2. Feature Engineering (`src/features/`)

#### Strategy Changes
- **K_only** → **N_only**: Focus on nitrogen spectral regions
- Feature extraction now uses `nitrate_region` instead of `potassium_region`
- Added N/O and N/H ratios alongside N/C ratio

#### New Feature Functions (`feature_helpers.py`)
Added nitrate-specific feature generators:

**`generate_high_nitrate_features()`**:
- NC_ratio_squared, NC_ratio_cubic, NC_ratio_log
- NC_height_ratio, NC_height_ratio_squared
- N_signal_baseline_ratio, N_signal_baseline_ratio_squared

**`generate_focused_nitrate_features()`**:
- Core N/C transformations
- N/O area ratio (nitrate indicator)
- N/H area ratio (NH4 vs NO3 discrimination)
- N/P and N/K ratios (NPK agricultural context)

#### Parallel Processing (`parallel_feature_engineering.py`)
- Updated imports to use `generate_high_nitrate_features` and `generate_focused_nitrate_features`
- Changed instance variable `_high_k_names` → `_high_no3_names`
- Modified `_get_strategy_regions()` for N_only strategy
- Updated N/C ratio calculation (N_I_primary_peak_0 / C_I_peak_0)
- Feature name selection now filters for N_I and nitrogen/nitrate terms

### 3. Target and Concentration Ranges

#### Target Column
- **Potassium**: `"K 766.490\n(wt%)"`
- **Nitrate**: `"NO3 (wt%)"`

#### Concentration Thresholds
Based on typical soil NO3 levels (0-500 ppm or 0-0.05%):
```python
target_value_min: 0.0001  # ~1 ppm NO3 - minimum detectable
target_value_max: 0.1     # ~1000 ppm NO3 - upper range for agricultural soils
```

**Potassium reference** (for comparison):
- target_value_min: 2.0 (wt%)
- target_value_max: 15.0 (wt%)

### 4. Documentation Updates

#### README.md
- Updated project description for NO3 prediction
- Listed nitrogen spectral regions (742-747nm primary, 821-824nm, 868-871nm secondary)
- Documented molecular bands (CN, NH, NO) critical for NO3
- Explained feature strategies (N_only, simple_only, full_context)
- Key differences from potassium pipeline

#### CLAUDE.md
- Updated all references to nitrate prediction
- Modified command examples
- Documented nitrogen-specific spectral regions
- Updated architecture overview

#### pyproject.toml
- Changed project name to `"nitrate-pipeline"`
- Updated description for NO3 concentration prediction

## Scientific Rationale

### Why These Spectral Lines?

**Primary N I triplet (742-747 nm)**:
- Most sensitive for soil nitrogen detection
- Strong atomic emission lines
- Minimal spectral interference

**Secondary N I lines (821-824 nm, 868-871 nm)**:
- Cross-validation and verification
- Different excitation conditions
- Enhances model robustness

**Molecular bands (CN, NH, NO)**:
- **CN bands**: Indicate organic nitrogen compounds
- **NH band**: Signature of ammonium (NH4+)
- **NO bands**: Direct indicators of nitrate/nitrite (NO3-/NO2-)
- Essential for distinguishing nitrogen forms

### Why N/O and N/H Ratios?

**N/O ratio**:
- Nitrate (NO3-) is nitrogen + oxygen
- Higher N/O suggests nitrate presence
- Discriminates from ammonium

**N/H ratio**:
- Ammonium (NH4+) is nitrogen + hydrogen
- Lower N/H suggests ammonium presence
- Critical for NH4 vs NO3 discrimination

**N/C ratio**:
- Baseline normalization
- Accounts for organic matter variation
- Standard spectroscopic practice

## Files Modified

### Core Configuration
- `src/config/pipeline_config.py` ✓

### Feature Engineering
- `src/features/parallel_feature_engineering.py` ✓
- `src/features/feature_helpers.py` ✓ (new functions added)

### Documentation
- `README.md` ✓
- `CLAUDE.md` ✓
- `pyproject.toml` ✓

## Validation

All modified Python files passed syntax validation:
```bash
python -m py_compile src/config/pipeline_config.py
python -m py_compile src/features/parallel_feature_engineering.py
python -m py_compile src/features/feature_helpers.py
```

## Next Steps

1. **Data Preparation**:
   - Verify NO3 reference data format matches `target_column: "NO3 (wt%)"`
   - Ensure raw spectral files contain 742-747 nm range
   - Check for molecular band coverage (226-280 nm, 335-425 nm, 606-620 nm)

2. **Initial Testing**:
   ```bash
   # Train with N_only strategy
   uv run python main.py train --models lightgbm --strategy N_only --gpu --data-parallel --feature-parallel

   # Train with all strategies
   uv run python main.py train --gpu --data-parallel --feature-parallel
   ```

3. **Feature Analysis**:
   - Run SHAP analysis on initial models
   - Verify N/C, N/O, N/H ratios are calculated correctly
   - Check molecular band features are extracted

4. **Model Optimization**:
   ```bash
   # Optimize for best model
   uv run python main.py optimize-models --models xgboost lightgbm catboost \
     --strategy N_only --trials 200 --gpu --data-parallel --feature-parallel
   ```

## References

### LIBS Literature for Nitrogen Detection
- Primary N I lines: 742.36, 744.23, 746.83 nm
- Secondary N I lines: 821.63, 824.21, 868.03, 871.17 nm
- Molecular bands: CN (387, 420, 612 nm), NH (336 nm), NO (237, 257 nm)
- Source: NIST Atomic Spectra Database, LIBS soil analysis literature

### Key Differences from Potassium Pipeline
1. **Spectral regions**: 742-747 nm (N) vs 766-770 nm (K)
2. **Molecular bands enabled**: Critical for N speciation, not used for K
3. **Atmospheric correction**: N2 interference from air vs minimal for K
4. **Feature ratios**: N/C, N/O, N/H vs K/Ca, K/Mg
5. **Concentration ranges**: 0-0.1 wt% (NO3) vs 2-15 wt% (K2O)

## Status

✅ **Migration Complete**
- All source files copied and adapted
- Nitrate-specific configurations applied
- Python syntax validated
- Documentation updated
- Ready for data preparation and testing

---

**Generated**: 2025-10-24
**Pipeline**: nitrates_pipeline
**Base**: potassium_pipeline
**Target**: NO3 (wt%) concentration prediction
