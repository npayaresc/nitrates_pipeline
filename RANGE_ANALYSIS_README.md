# Range Performance Analysis Tools

Tools for analyzing model performance across different NO3 concentration ranges to identify where models perform best.

## Overview

During optimization, validation predictions are saved. These tools analyze those predictions to show:
- Which concentration ranges have the best/worst performance
- How performance metrics (R², RMSE, MAPE, Within 20%) vary across ranges
- Which models excel in which concentration ranges
- Visual comparisons across multiple models

## Tools

### 1. `analyze_range_performance.py` - Single Model Analysis

Analyzes a single model's predictions across concentration ranges.

#### Usage

```bash
# Analyze with auto-generated ranges (based on data quantiles)
uv run python analyze_range_performance.py \
    --predictions reports/predictions_simple_only_optimized_xgboost_20251111_205228.csv

# Specify custom ranges
uv run python analyze_range_performance.py \
    --predictions reports/predictions_simple_only_optimized_xgboost_20251111_205228.csv \
    --ranges "8000-12000,12000-16000,16000-20000,20000-24000,24000-29000"

# Custom output directory
uv run python analyze_range_performance.py \
    --predictions reports/predictions_simple_only_optimized_xgboost_20251111_205228.csv \
    --output-dir reports/my_custom_analysis
```

#### Output Files

For each model, generates:
- `{model_name}_range_performance.csv` - Detailed metrics table
- `{model_name}_range_performance.png` - 6-panel visualization showing:
  - R² by range
  - RMSE by range
  - MAPE by range
  - Within 10%/20% tolerance
  - Sample distribution
  - Combined performance score
- `{model_name}_predictions_by_range.png` - Actual vs predicted scatter plots for each range
- `{model_name}_range_analysis_report.txt` - Text summary with recommendations

#### Metrics Calculated

For each concentration range:
- **R²**: Coefficient of determination
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **RRMSE**: Relative RMSE (RMSE / mean)
- **Within 10%**: Percentage of predictions within ±10.5% of actual
- **Within 20%**: Percentage of predictions within ±20.5% of actual
- **N_Samples**: Number of samples in range
- **Mean_Actual**: Average actual concentration in range

---

### 2. `analyze_all_models_ranges.sh` - Batch Analysis

Analyzes multiple optimization runs in batch mode.

#### Usage

```bash
# Analyze all optimization prediction files
./analyze_all_models_ranges.sh
```

This script:
1. Finds all `predictions_*optimized*.csv` files in reports/
2. Runs `analyze_range_performance.py` on each
3. Saves all results to `reports/range_analysis/`

#### What it Processes

Automatically finds and analyzes:
- `predictions_simple_only_optimized_xgboost_*.csv`
- `predictions_simple_only_optimized_lightgbm_*.csv`
- `predictions_full_context_optimized_lightgbm_*.csv`
- Any other optimization prediction files

---

### 3. `compare_model_ranges.py` - Multi-Model Comparison

Compares range performance across all analyzed models.

#### Usage

```bash
# Compare all models (run after analyze_all_models_ranges.sh)
uv run python compare_model_ranges.py

# Custom analysis directory
uv run python compare_model_ranges.py --analysis-dir reports/my_custom_analysis
```

#### Output Files

Generates comparison across all models:
- `all_models_range_comparison.csv` - Combined results from all models
- `model_comparison_r2_heatmap.png` - R² heatmap across models and ranges
- `model_comparison_within20_heatmap.png` - Within 20% tolerance heatmap
- `model_comparison_mape_heatmap.png` - MAPE heatmap
- `model_comparison_best_per_range.png` - 4-panel visualization showing:
  - Best R² model per range
  - Best Within 20% model per range
  - Best MAPE model per range
  - Frequency: which models are best most often
- `model_range_comparison_report.txt` - Comprehensive comparison report

#### Key Insights Provided

- **Best models overall**: Averaged across all ranges
- **Best model per range**: Identifies specialists for specific concentration ranges
- **Most consistent model**: Lowest variance in R² across ranges
- **Best all-around model**: Combined score considering R², MAPE, and Within 20%

---

## Complete Workflow

### Step-by-Step Example

```bash
# 1. Run optimization (already done)
uv run python main.py optimize-models --models xgboost lightgbm --strategy simple_only --trials 200 --gpu

# 2. Analyze all optimization results by concentration range
./analyze_all_models_ranges.sh

# 3. Compare models to find best performers
uv run python compare_model_ranges.py

# 4. Review results
ls -lh reports/range_analysis/
cat reports/range_analysis/model_range_comparison_report.txt
```

### Quick Single Model Analysis

```bash
# Find latest optimization prediction file
LATEST=$(ls -t reports/predictions_*optimized*.csv | head -1)

# Analyze just that model
uv run python analyze_range_performance.py --predictions "$LATEST"

# View results
xdg-open reports/range_analysis/$(basename ${LATEST%.csv})_range_performance.png
```

---

## Interpreting Results

### Good Performance Indicators

- **R² > 0.6**: Strong correlation between predictions and actuals
- **MAPE < 10%**: High accuracy, predictions within 10% on average
- **Within 20% > 80%**: Most predictions fall within acceptable tolerance
- **Low RMSE relative to concentration**: Errors are small relative to signal

### Warning Signs

- **R² < 0.0**: Model performs worse than predicting the mean (very poor)
- **R² < 0.3**: Weak predictive power
- **MAPE > 20%**: Large relative errors
- **Within 20% < 60%**: Many predictions outside acceptable range

### Range-Specific Insights

Models often perform differently across concentration ranges:
- **Low ranges** (8000-14000): Often hardest to predict due to noise
- **Mid ranges** (15000-20000): Typically best performance
- **High ranges** (24000-29000): May have fewer samples, more variability

### Using Results for Model Selection

1. **For general use**: Choose model with best **average R²** or **combined score**
2. **For specific range**: Choose model that excels in **that range** (from heatmaps)
3. **For reliability**: Choose **most consistent model** (lowest R² variance)
4. **For practical tolerance**: Choose model with best **Within 20%** performance

---

## Advanced Usage

### Custom Range Definitions

Define ranges that match your application needs:

```bash
# Agronomic ranges (example)
uv run python analyze_range_performance.py \
    --predictions reports/predictions_*.csv \
    --ranges "8000-12000,12000-15000,15000-18000,18000-22000,22000-30000"

# Fine-grained low-range analysis
uv run python analyze_range_performance.py \
    --predictions reports/predictions_*.csv \
    --ranges "8000-10000,10000-12000,12000-14000,14000-16000,16000-18000,18000-29000"
```

### Programmatic Access

The CSV files can be loaded for custom analysis:

```python
import pandas as pd

# Load single model results
df = pd.read_csv('reports/range_analysis/simple_only_optimized_xgboost_20251111_205228_range_performance.csv')

# Load comparison across all models
all_models = pd.read_csv('reports/range_analysis/all_models_range_comparison.csv')

# Find best model for 15000-18000 range
range_subset = all_models[all_models['Range'].str.contains('15000-18000')]
best_model = range_subset.loc[range_subset['R²'].idxmax(), 'Model']
print(f"Best model for 15000-18000 range: {best_model}")
```

---

## Troubleshooting

### No prediction files found

```bash
# Check if optimization was run
ls reports/predictions_*optimized*.csv

# If empty, run optimization first
uv run python main.py optimize-models --models xgboost --strategy simple_only --trials 100 --gpu
```

### Script permission denied

```bash
# Make scripts executable
chmod +x analyze_all_models_ranges.sh
chmod +x analyze_range_performance.py
chmod +x compare_model_ranges.py
```

### Missing dependencies

```bash
# Install required packages
uv sync
```

### Empty or corrupt visualizations

- Check that prediction CSV files have both `ElementValue` and `PredictedValue` columns
- Ensure there are at least 10 samples in the dataset
- Verify the concentration range covers expected values (8000-30000)

---

## Example Output Interpretation

### Scenario: XGBoost Optimization Results

```
Best performing range (by R²):
  17597-20022: R²=-8.8861

Worst performing range (by R²):
  14924-17597: R²=-11.6520
```

**Interpretation:**
- All ranges have negative R², indicating the model is underfitting
- Predictions lack variance (model predicting too narrow a range)
- However, check **Within 20%** and **MAPE** for practical usability
- Range 17597-20022 shows 93.42% within 20% tolerance - practically useful!

**Action:**
- Model has poor R² but acceptable practical accuracy
- Consider increasing model complexity or adjusting hyperparameters
- Focus training on the 14924-17597 range which performs worst
- Current model may still be usable for 17597-20022 range (93% accuracy)

---

## Tips

1. **Run batch analysis regularly** after optimization to track improvements
2. **Compare different strategies** (simple_only vs full_context) to see which works best in each range
3. **Use heatmaps** to quickly identify which models excel in which ranges
4. **Check sample distribution** - ranges with few samples may have unreliable metrics
5. **Focus on Within 20%** if R² is poor - it's often more practically relevant
6. **Look for consistent performers** - models with low variance across ranges are more reliable

---

## Output Directory Structure

```
reports/range_analysis/
├── simple_only_optimized_xgboost_20251111_205228_range_performance.csv
├── simple_only_optimized_xgboost_20251111_205228_range_performance.png
├── simple_only_optimized_xgboost_20251111_205228_predictions_by_range.png
├── simple_only_optimized_xgboost_20251111_205228_range_analysis_report.txt
├── simple_only_optimized_lightgbm_20251111_181451_range_performance.csv
├── simple_only_optimized_lightgbm_20251111_181451_range_performance.png
├── ... (similar files for each model)
├── all_models_range_comparison.csv
├── model_comparison_r2_heatmap.png
├── model_comparison_within20_heatmap.png
├── model_comparison_mape_heatmap.png
├── model_comparison_best_per_range.png
└── model_range_comparison_report.txt
```

---

## Integration with Existing Pipeline

These tools complement existing pipeline features:

- **After optimization**: Use range analysis to understand where optimized models excel
- **Before feature selection**: Identify problematic ranges that need better features
- **For model ensembling**: Combine models that excel in different ranges
- **With SHAP analysis**: Compare feature importance across concentration ranges
- **For data quality**: Identify ranges with poor performance that may have data issues

---

## Future Enhancements

Potential additions:
- Automatic ensemble creation (best model per range)
- Cross-validation analysis across ranges
- Time-series analysis of range performance across optimization runs
- Integration with MLflow for tracking
- Automated recommendations for hyperparameter adjustments per range
