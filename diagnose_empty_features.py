#!/usr/bin/env python3
"""
Diagnostic script to identify which samples produce empty features in parallel processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.config.pipeline_config import Config
from src.features.parallel_feature_engineering import _extract_features_for_row

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sample_feature_extraction():
    """Test feature extraction on actual sample files."""

    # Load config with required timestamp
    from datetime import datetime
    config = Config(run_timestamp=datetime.now().strftime("%Y%m%d_%H%M%S"))
    config_dict = config.model_dump()

    # Get all regions for simple_only strategy
    regions = config.all_regions
    regions_list = [r.model_dump() for r in regions]

    # Test on a few sample files
    data_dir = Path("/home/payanico/nitrates_pipeline/data/cleansed_files_per_sample")
    files = list(data_dir.glob("*.txt"))[:50]  # Test first 50 files

    logger.info(f"Testing feature extraction on {len(files)} files...")

    empty_count = 0
    error_samples = []

    for file_path in files:
        try:
            # Read file
            df = pd.read_csv(file_path)
            wavelengths = df['Wavelength'].values

            # Get intensities - average across all columns
            intensity_cols = [col for col in df.columns if col != 'Wavelength']
            intensities = df[intensity_cols].mean(axis=1).values

            # Test extraction
            result = _extract_features_for_row((
                file_path.name,
                wavelengths,
                intensities,
                config_dict,
                "simple_only",
                False,  # use_enhanced
                regions_list
            ))

            if not result['features']:
                empty_count += 1
                error_samples.append({
                    'file': file_path.name,
                    'reason': 'Empty features returned',
                    'wavelength_range': f"{wavelengths.min():.2f}-{wavelengths.max():.2f}",
                    'intensity_shape': intensities.shape,
                    'has_nan_intensities': np.isnan(intensities).any()
                })
                logger.warning(f"Empty features for {file_path.name}")
            else:
                logger.debug(f"✓ {file_path.name}: {len(result['features'])} features extracted")

        except Exception as e:
            empty_count += 1
            error_samples.append({
                'file': file_path.name,
                'reason': f'Exception: {str(e)}',
                'wavelength_range': 'N/A',
                'intensity_shape': 'N/A',
                'has_nan_intensities': 'N/A'
            })
            logger.error(f"Error processing {file_path.name}: {e}")

    logger.info(f"\n=== Results ===")
    logger.info(f"Total files tested: {len(files)}")
    logger.info(f"Files with empty/error features: {empty_count}")
    logger.info(f"Success rate: {(len(files) - empty_count) / len(files) * 100:.1f}%")

    if error_samples:
        logger.warning(f"\n=== Problematic Samples ===")
        error_df = pd.DataFrame(error_samples)
        print(error_df.to_string())
        error_df.to_csv("empty_features_diagnostic.csv", index=False)
        logger.info("\nDetailed report saved to: empty_features_diagnostic.csv")

    return empty_count, error_samples

if __name__ == "__main__":
    test_sample_feature_extraction()
