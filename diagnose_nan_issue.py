#!/usr/bin/env python3
"""
Diagnostic script to investigate NaN generation in parallel feature engineering.
Checks if spectral data has sufficient coverage for nitrogen regions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define nitrogen regions
nitrogen_regions = {
    "N_I_primary": (741.0, 748.0),
    "N_I_821": (820.0, 826.0),
    "N_I_868": (867.0, 872.0)
}

def check_file_coverage(file_path: Path) -> dict:
    """Check if a file has sufficient wavelength coverage for nitrogen regions."""
    try:
        df = pd.read_csv(file_path)
        wavelengths = df['Wavelength'].values

        results = {
            'file': file_path.name,
            'min_wavelength': wavelengths.min(),
            'max_wavelength': wavelengths.max(),
            'total_points': len(wavelengths)
        }

        for region_name, (lower, upper) in nitrogen_regions.items():
            mask = (wavelengths >= lower) & (wavelengths <= upper)
            points_in_region = np.sum(mask)
            results[f'{region_name}_points'] = points_in_region
            results[f'{region_name}_has_data'] = points_in_region >= 2

        return results

    except Exception as e:
        logger.error(f"Error reading {file_path.name}: {e}")
        return None

def main():
    # Check cleansed files
    data_dir = Path("/home/payanico/pipeline_nitrates/data/cleansed_files_per_sample")
    files = list(data_dir.glob("*.txt"))[:20]  # Check first 20 files

    logger.info(f"Checking {len(files)} files for nitrogen region coverage...")

    results_list = []
    for file_path in files:
        result = check_file_coverage(file_path)
        if result:
            results_list.append(result)

    # Create summary DataFrame
    summary_df = pd.DataFrame(results_list)

    # Print summary statistics
    logger.info("\n=== Wavelength Coverage Summary ===")
    logger.info(f"Min wavelength across all files: {summary_df['min_wavelength'].min():.2f} nm")
    logger.info(f"Max wavelength across all files: {summary_df['max_wavelength'].max():.2f} nm")
    logger.info(f"Avg total points per file: {summary_df['total_points'].mean():.0f}")

    logger.info("\n=== Nitrogen Region Coverage ===")
    for region_name in nitrogen_regions.keys():
        points_col = f'{region_name}_points'
        has_data_col = f'{region_name}_has_data'

        avg_points = summary_df[points_col].mean()
        files_with_data = summary_df[has_data_col].sum()
        files_without_data = len(summary_df) - files_with_data

        logger.info(f"\n{region_name} ({nitrogen_regions[region_name][0]}-{nitrogen_regions[region_name][1]} nm):")
        logger.info(f"  Avg points in region: {avg_points:.1f}")
        logger.info(f"  Files with sufficient data (>=2 points): {files_with_data}/{len(summary_df)}")
        logger.info(f"  Files with insufficient data (<2 points): {files_without_data}/{len(summary_df)}")

        if files_without_data > 0:
            problematic = summary_df[~summary_df[has_data_col]]
            logger.warning(f"  Problematic files: {list(problematic['file'].head(5))}")

    # Save detailed report
    output_file = "/home/payanico/pipeline_nitrates/nitrogen_region_coverage_report.csv"
    summary_df.to_csv(output_file, index=False)
    logger.info(f"\nDetailed report saved to: {output_file}")

if __name__ == "__main__":
    main()
