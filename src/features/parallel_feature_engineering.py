"""
Parallel Feature Engineering Module: Optimized version using multiprocessing
for faster feature generation from spectral data.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.feature_helpers import (
    extract_full_simple_features,
    generate_high_nitrate_features,
    generate_focused_nitrate_features,
)
from src.config.pipeline_config import Config, PeakRegion
from src.spectral_extraction.extractor import SpectralFeatureExtractor
from src.spectral_extraction.results import PeakRegion as ResultsPeakRegion
from src.features.feature_helpers import PeakRegion as HelperPeakRegion
from src.features.enhanced_features import EnhancedSpectralFeatures

logger = logging.getLogger(__name__)


def _convert_to_results_regions(
    config_regions: List[PeakRegion],
) -> List[ResultsPeakRegion]:
    """Convert config PeakRegion objects to extractor.results PeakRegion objects."""
    return [
        ResultsPeakRegion(
            element=r.element,
            lower_wavelength=r.lower_wavelength,
            upper_wavelength=r.upper_wavelength,
            center_wavelengths=list(r.center_wavelengths),
        )
        for r in config_regions
    ]


def _extract_features_for_row(args: Tuple) -> Dict[str, Any]:
    """
    Extract features for a single row. This function is designed to be pickle-able
    for multiprocessing.

    Args:
        args: Tuple containing (idx, wavelengths, intensities, config_dict, strategy, use_enhanced, regions_list)

    Returns:
        Dictionary with 'idx' and 'features' keys
    """
    idx, wavelengths, intensities, config_dict, strategy, use_enhanced, regions_list = args

    try:
        # Reconstruct config from dict (configs aren't pickle-able directly)
        from src.config.pipeline_config import Config, PeakRegion
        config = Config(**config_dict)

        # Reconstruct regions from list of dicts
        regions = [PeakRegion(**r) for r in regions_list]

        features = {}
        wavelengths = np.asarray(wavelengths)
        intensities = np.asarray(intensities)

        if intensities.size == 0:
            logger.warning(f"Empty intensities found for sample {idx}")
            return {'idx': idx, 'features': {}}

        # Extract simple features for strategy-specific regions
        for region in regions:
            helper_region = HelperPeakRegion(
                region.element,
                region.lower_wavelength,
                region.upper_wavelength,
                region.center_wavelengths,
            )
            features.update(
                extract_full_simple_features(helper_region, wavelengths, intensities)
            )
        
        # Extract complex features using SpectralFeatureExtractor
        extractor = SpectralFeatureExtractor(
            enable_preprocessing=config.use_spectral_preprocessing,
            preprocessing_method=config.spectral_preprocessing_method
        )
        spectra_2d = intensities.reshape(-1, 1) if intensities.ndim == 1 else intensities

        fit_results = extractor.extract_features(
            wavelengths=wavelengths,
            spectra=spectra_2d,
            regions=_convert_to_results_regions(regions),
            peak_shapes=config.peak_shapes * len(regions),
        )
        
        for res in fit_results.fitting_results:
            element = res.region_result.region.element

            # Extract peak areas (original feature)
            for i, area in enumerate(res.peak_areas):
                features[f"{element}_peak_{i}"] = area

            # === EXTRACT PHYSICS-INFORMED FEATURES (NEW) ===
            for i, peak_fit in enumerate(res.fit_results):
                # FWHM - Full Width at Half Maximum
                features[f"{element}_fwhm_{i}"] = peak_fit.fwhm

                # Gamma (Stark broadening parameter)
                features[f"{element}_gamma_{i}"] = peak_fit.gamma

                # Fit quality (R²)
                features[f"{element}_fit_quality_{i}"] = peak_fit.fit_quality

                # Peak asymmetry (self-absorption indicator)
                features[f"{element}_asymmetry_{i}"] = peak_fit.peak_asymmetry

                # Amplitude (peak height for Lorentzian)
                features[f"{element}_amplitude_{i}"] = peak_fit.amplitude

                # Kurtosis (tailedness of peak distribution)
                features[f"{element}_kurtosis_{i}"] = peak_fit.kurtosis

                # Derived feature: FWHM × Asymmetry (absorption strength)
                features[f"{element}_absorption_index_{i}"] = peak_fit.fwhm * abs(peak_fit.peak_asymmetry)
        
        # Add enhanced features if enabled
        if use_enhanced:
            enhanced_features = EnhancedSpectralFeatures(config)
            enhanced_feats = enhanced_features.transform(
                features, wavelengths, intensities
            )
            features.update(enhanced_feats)
        
        return {'idx': idx, 'features': features}
        
    except Exception as e:
        logger.warning(f"Error processing sample {idx}: {e}")
        return {'idx': idx, 'features': {}}


class ParallelSpectralFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Parallel version of SpectralFeatureGenerator that uses multiprocessing
    to extract features from multiple samples simultaneously.
    """
    
    def __init__(self, config: Config, strategy: str = "simple_only", n_jobs: int = -1):
        """
        Initialize parallel feature generator.

        Args:
            config: Pipeline configuration
            strategy: Feature strategy ('N_only', 'simple_only', 'full_context')
            n_jobs: Number of parallel jobs. -1 uses all CPU cores, -2 uses all but one
        """
        self.config = config
        self.strategy = strategy
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count() + 1 + n_jobs
        self.feature_names_out_: List[str] = []
        self._all_simple_names = []
        self._high_no3_names = []

        # Strategy-optimized regions to avoid extracting unused features
        self._regions = self._get_strategy_regions()

        # Debug: Check for duplicate regions
        region_elements = [r.element for r in self._regions]
        from collections import Counter
        region_counts = Counter(region_elements)
        region_dups = {elem: count for elem, count in region_counts.items() if count > 1}
        if region_dups:
            logger.warning(f"[DEBUG INIT] DUPLICATE REGIONS in self._regions: {region_dups}")

        # Initialize enhanced features flag
        self._use_enhanced = any([
            config.enable_molecular_bands,
            config.enable_advanced_ratios,
            config.enable_spectral_patterns,
            config.enable_interference_correction,
            config.enable_plasma_indicators,
        ])

        logger.info(f"Initialized ParallelSpectralFeatureGenerator with {self.n_jobs} workers")

    def _get_strategy_regions(self) -> List[PeakRegion]:
        """Get regions optimized for current strategy."""
        if self.strategy == "N_only":
            # N regions + essential context elements for N/C, N/O, N/H ratios and NPK context
            n_regions = [self.config.nitrate_region]
            # Add secondary nitrogen lines
            n_regions.extend([r for r in self.config.context_regions if r.element.startswith("N_I")])

            # Add C_I for N_C_ratio calculation (baseline normalization for nitrogen)
            c_region = next((r for r in self.config.context_regions if r.element == "C_I"), None)
            if c_region:
                n_regions.append(c_region)

            # Add O and H for N/O and N/H ratios (CRITICAL for NO3 vs NH4 discrimination)
            o_region = next((r for r in self.config.context_regions if r.element == "O_I"), None)
            if o_region:
                n_regions.append(o_region)
            h_region = next((r for r in self.config.context_regions if r.element == "H_alpha"), None)
            if h_region:
                n_regions.append(h_region)

            # Add P and K for NPK nutrient context (agricultural interpretation)
            p_region = next((r for r in self.config.context_regions if r.element == "P_I"), None)
            if p_region:
                n_regions.append(p_region)
            k_region = next((r for r in self.config.context_regions if r.element == "K_I"), None)
            if k_region:
                n_regions.append(k_region)

            # Add Ca for soil chemistry context
            ca_region = next((r for r in self.config.context_regions if r.element == "CA_II_393"), None)
            if ca_region:
                n_regions.append(ca_region)

            logger.info(f"N_only strategy: using {len(n_regions)} regions (vs {len(self.config.all_regions)} total)")
            logger.info(f"Regions: {[r.element for r in n_regions]}")
            return n_regions
        else:
            # Full regions for simple_only and full_context
            return self.config.all_regions
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits the transformer by determining the canonical feature names."""
        self._set_feature_names(X.iloc[0:1])
        logger.info(f"ParallelSpectralFeatureGenerator fitted for strategy '{self.strategy}'.")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw spectral data into the final feature matrix using parallel processing.
        """
        # Handle numpy arrays from SHAP or other tools
        if isinstance(X, np.ndarray):
            logger.debug("Converting numpy array to DataFrame for parallel feature extraction")
            # Check if this is a structured array with object dtype (array-valued columns)
            if X.dtype == object and X.shape[1] == 2:
                # Array-valued columns (wavelengths and intensities as arrays)
                X = pd.DataFrame({
                    "wavelengths": X[:, 0],
                    "intensities": X[:, 1]
                })
            elif X.shape[1] == 2:
                # Regular 2-column array
                X = pd.DataFrame(X, columns=["wavelengths", "intensities"])
            else:
                raise ValueError(f"Expected 2 columns (wavelengths, intensities), got {X.shape[1]}")

        logger.info(f"Starting parallel feature extraction for {len(X)} samples with {self.n_jobs} workers")

        # Prepare arguments for parallel processing
        # Convert config to dict for pickling
        config_dict = self.config.model_dump()

        # Convert regions to list of dicts for pickling
        regions_list = [r.model_dump() for r in self._regions]

        args_list = [
            (
                idx,
                row["wavelengths"],
                row["intensities"],
                config_dict,
                self.strategy,
                self._use_enhanced,
                regions_list
            )
            for idx, row in X.iterrows()
        ]
        
        # Process samples in parallel
        results = []
        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(_extract_features_for_row, args): args[0] 
                for args in args_list
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_args):
                idx = future_to_args[future]
                try:
                    result = future.result()
                    sample_features = result.get('features', {})

                    if not sample_features:
                        logger.warning("No features extracted for sample %s in parallel processing", idx)
                    else:
                        logger.debug("Extracted %d features for sample %s in parallel processing",
                                   len(sample_features), idx)

                    results.append(result)
                except Exception as e:
                    logger.error("Failed to process sample %s in parallel: %s", idx, e)
                    logger.error("Creating empty feature set for sample %s", idx)
                    results.append({'idx': idx, 'features': {}})
        
        # Sort results by index to maintain original order
        if hasattr(X, 'index'):
            try:
                results.sort(key=lambda x: X.index.get_loc(x['idx']))
            except (KeyError, ValueError):
                # If index sorting fails, maintain original order
                pass
        
        # Extract features from results
        base_features_list = [r['features'] for r in results]
        index_to_use = X.index if hasattr(X, 'index') else None

        # CRITICAL FIX: Ensure all feature dicts have the same keys to prevent NaN creation
        # Get all unique feature names across all results
        all_feature_names = set()
        for features_dict in base_features_list:
            all_feature_names.update(features_dict.keys())

        # Fill missing keys with 0.0 in each feature dictionary
        for features_dict in base_features_list:
            for feature_name in all_feature_names:
                if feature_name not in features_dict:
                    features_dict[feature_name] = 0.0

        base_features_df = pd.DataFrame(base_features_list, index=index_to_use)
        
        # Calculate N/C ratio (baseline normalization for nitrogen)
        n_area = (
            base_features_df["N_I_primary_peak_0"].fillna(0.0)
            if "N_I_primary_peak_0" in base_features_df
            else pd.Series(0.0, index=base_features_df.index)
        )
        c_area = (
            base_features_df["C_I_peak_0"].fillna(1e-6)
            if "C_I_peak_0" in base_features_df
            else pd.Series(1e-6, index=base_features_df.index)
        )

        nc_ratio_raw = n_area / c_area.replace(0, 1e-6)
        base_features_df["N_C_ratio"] = np.clip(nc_ratio_raw, -50.0, 50.0)

        # Calculate K/C ratio (nutrient balance context)
        k_area = (
            base_features_df["K_I_peak_0"].fillna(0.0)
            if "K_I_peak_0" in base_features_df
            else pd.Series(0.0, index=base_features_df.index)
        )
        kc_ratio_raw = k_area / c_area.replace(0, 1e-6)
        base_features_df["K_C_ratio"] = np.clip(kc_ratio_raw, -50.0, 50.0)

        # Generate additional features based on config
        if self.config.use_focused_nitrate_features:
            full_features_df, _ = generate_focused_nitrate_features(
                base_features_df, self._all_simple_names
            )
        else:
            full_features_df, _ = generate_high_nitrate_features(
                base_features_df, self._all_simple_names
            )
        
        # Validate feature names
        expected_features = self.get_feature_names_out()
        if len(expected_features) != len(set(expected_features)):
            from collections import Counter
            feature_counts = Counter(expected_features)
            duplicates = {
                name: count for name, count in feature_counts.items() if count > 1
            }
            raise ValueError(f"Duplicate feature names: {list(duplicates.keys())[:10]}")
        
        # Check for missing features before reindexing
        missing_features = [f for f in expected_features if f not in full_features_df.columns]
        if missing_features:
            logger.warning("Missing %d features during parallel processing reindexing, filling with zeros: %s",
                         len(missing_features), missing_features[:10])

        # Reindex to ensure all expected features are present
        final_df = full_features_df.reindex(
            columns=expected_features, fill_value=0.0
        )

        # Final NaN check and cleanup for parallel processing
        total_values = final_df.shape[0] * final_df.shape[1]
        nan_count = final_df.isnull().sum().sum()
        nan_percentage = (nan_count / total_values * 100) if total_values > 0 else 0.0

        logger.debug(
            "[PARALLEL FEATURE ENGINEERING] NaN Statistics: %d total NaNs (%.2f%% of %d total values)",
            nan_count, nan_percentage, total_values
        )

        if final_df.isnull().any().any():
            nan_cols = final_df.columns[final_df.isnull().any()].tolist()

            # CRITICAL ERROR: NaN values should not occur after parallel feature engineering
            error_msg = (
                f"[PARALLEL FEATURE ENGINEERING] CRITICAL: Found {nan_count} NaN values ({nan_percentage:.2f}%) "
                f"across {len(nan_cols)} features.\n"
                f"Sample NaN features: {nan_cols[:20]}\n"
                f"Strategy: {self.strategy}\n"
                f"Workers: {self.n_jobs}\n"
                f"This indicates a problem with parallel feature extraction or worker synchronization."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Ensure the final dataframe maintains the original index
        if hasattr(X, 'index'):
            final_df.index = X.index

        logger.info(
            "[PARALLEL FEATURE ENGINEERING] Transformation complete: %d features for strategy '%s'.",
            final_df.shape[1], self.strategy
        )
        return final_df
    
    def _set_feature_names(self, X_sample):
        """
        Defines the canonical list of feature names for each strategy.
        This is identical to the original SpectralFeatureGenerator method.
        """
        # Define all possible base feature names
        all_complex_names = []
        physics_informed_names = []  # NEW: Physics-informed features
        for region in self._regions:
            for i in range(region.n_peaks):
                # Original peak area feature
                all_complex_names.append(f"{region.element}_peak_{i}")

                # NEW: Physics-informed features from Lorentzian fits
                physics_informed_names.append(f"{region.element}_fwhm_{i}")
                physics_informed_names.append(f"{region.element}_gamma_{i}")
                physics_informed_names.append(f"{region.element}_fit_quality_{i}")
                physics_informed_names.append(f"{region.element}_asymmetry_{i}")
                physics_informed_names.append(f"{region.element}_amplitude_{i}")
                physics_informed_names.append(f"{region.element}_kurtosis_{i}")
                physics_informed_names.append(f"{region.element}_absorption_index_{i}")

        # Simple features (8 per region)
        self._all_simple_names = []
        for region in self._regions:
            prefix = f"{region.element}_simple"
            self._all_simple_names.extend([
                f"{prefix}_peak_area",
                f"{prefix}_peak_height",
                f"{prefix}_peak_center_intensity",
                f"{prefix}_baseline_avg",
                f"{prefix}_signal_range",
                f"{prefix}_total_intensity",
                f"{prefix}_height_to_baseline",
                f"{prefix}_normalized_area",
            ])
        
        # Extract a sample to determine high_p_names dynamically
        sample_idx = X_sample.index[0] if hasattr(X_sample, 'index') else 0
        regions_list = [r.model_dump() for r in self._regions]
        sample_features = _extract_features_for_row((
            sample_idx,
            X_sample.iloc[0]["wavelengths"],
            X_sample.iloc[0]["intensities"],
            self.config.model_dump(),
            self.strategy,
            False,  # Don't use enhanced for name determination
            regions_list
        ))
        
        sample_base_df = pd.DataFrame([sample_features['features']])
        sample_base_df["N_C_ratio"] = 0.0
        sample_base_df["K_C_ratio"] = 0.0

        # Get high nitrate feature names
        if self.config.use_focused_nitrate_features:
            _, self._high_no3_names = generate_focused_nitrate_features(
                sample_base_df, self._all_simple_names
            )
        else:
            _, self._high_no3_names = generate_high_nitrate_features(
                sample_base_df, self._all_simple_names
            )
        
        # Get enhanced feature names if enabled
        enhanced_names = []
        if self._use_enhanced:
            enhanced_features = EnhancedSpectralFeatures(self.config)
            sample_wavelengths = np.asarray(X_sample.iloc[0]["wavelengths"])
            sample_intensities = np.asarray(X_sample.iloc[0]["intensities"])
            enhanced_sample = enhanced_features.transform(
                sample_features['features'], sample_wavelengths, sample_intensities
            )
            enhanced_names = list(enhanced_sample.keys())
        
        # Set feature names based on strategy
        if self.strategy == "N_only":
            # Include all N_I features (N_I_primary_, N_I_821_, N_I_868_) and N_C_ratio
            n_complex = [name for name in all_complex_names if name.startswith("N_I")]
            n_simple = [
                name for name in self._all_simple_names if name.startswith("N_I")
            ]
            # Include all enhanced N features (N/C, N/O, N/H ratios, nitrate indicators)
            n_enhanced = [
                name for name in enhanced_names
                if "N" in name or "nitrogen" in name.lower() or "nitrate" in name.lower()
            ]
            # NEW: Include physics-informed features for N
            n_physics = [name for name in physics_informed_names if name.startswith("N_I")]

            # Include context features (K, C, P for nutrient balance) - simple features only
            context_features = [name for name in self._all_simple_names if any(name.startswith(elem) for elem in ["K_I", "C_I", "P_I", "CA_"])]

            # Always add N_C_ratio and K_C_ratio (critical nutrient indicators, computed separately)
            self.feature_names_out_ = n_complex + n_physics + n_simple + n_enhanced + context_features + ["N_C_ratio", "K_C_ratio"] + self._high_no3_names
        elif self.strategy == "simple_only":
            self.feature_names_out_ = (
                self._all_simple_names
                + physics_informed_names  # NEW: Add physics-informed features
                + ["N_C_ratio", "K_C_ratio"]
                + self._high_no3_names
                + enhanced_names
            )
        elif self.strategy == "full_context":
            self.feature_names_out_ = (
                all_complex_names
                + physics_informed_names  # NEW: Add physics-informed features
                + self._all_simple_names
                + ["N_C_ratio", "K_C_ratio"]
                + self._high_no3_names
                + enhanced_names
            )
        else:
            raise ValueError(f"Unknown feature strategy: {self.strategy}")
        
        # Validate no duplicates
        from collections import Counter

        # Debug: Log feature counts for each component
        logger.warning(f"[DEBUG] Strategy: {self.strategy}")
        if self.strategy == "simple_only":
            logger.warning(f"[DEBUG]   _all_simple_names: {len(self._all_simple_names)} features")
            logger.warning(f"[DEBUG]   physics_informed_names: {len(physics_informed_names)} features")
            logger.warning(f"[DEBUG]   _high_no3_names: {len(self._high_no3_names)} features")
            logger.warning(f"[DEBUG]   enhanced_names: {len(enhanced_names)} features")

            # Check for duplicates in each list BEFORE combining
            from collections import Counter
            simple_counts = Counter(self._all_simple_names)
            simple_dups = {name: count for name, count in simple_counts.items() if count > 1}
            if simple_dups:
                logger.warning(f"[DEBUG] DUPLICATES IN _all_simple_names: {list(simple_dups.keys())[:10]}")

            physics_counts = Counter(physics_informed_names)
            physics_dups = {name: count for name, count in physics_counts.items() if count > 1}
            if physics_dups:
                logger.warning(f"[DEBUG] DUPLICATES IN physics_informed_names: {list(physics_dups.keys())[:10]}")

            # Check for O_I, H_alpha, P_I in high_no3_names
            o_in_high = [n for n in self._high_no3_names if 'O_I' in n]
            h_in_high = [n for n in self._high_no3_names if 'H_alpha' in n]
            p_in_high = [n for n in self._high_no3_names if 'P_I' in n]
            if o_in_high or h_in_high or p_in_high:
                logger.warning(f"Found O_I/H_alpha/P_I in _high_no3_names: O_I={o_in_high[:3]}, H_alpha={h_in_high[:3]}, P_I={p_in_high[:3]}")

            # Check for O_I, H_alpha, P_I in enhanced_names
            o_in_enh = [n for n in enhanced_names if 'O_I' in n]
            h_in_enh = [n for n in enhanced_names if 'H_alpha' in n]
            p_in_enh = [n for n in enhanced_names if 'P_I' in n]
            if o_in_enh or h_in_enh or p_in_enh:
                logger.warning(f"Found O_I/H_alpha/P_I in enhanced_names: O_I={o_in_enh[:3]}, H_alpha={h_in_enh[:3]}, P_I={p_in_enh[:3]}")

        counts = Counter(self.feature_names_out_)
        duplicates = {name: count for name, count in counts.items() if count > 1}
        if duplicates:
            raise ValueError(f"Duplicate feature names: {list(duplicates.keys())}")
    
    def get_feature_names_out(self, input_features=None):
        """Return feature names for output features."""
        return self.feature_names_out_