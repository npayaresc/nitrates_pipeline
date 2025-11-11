"""
Centralized Configuration Management for the Nitrate (NO3) Prediction ML Pipeline.

Uses Pydantic for data validation and clear structure.
"""

from typing import List, Dict, Any, Optional, Literal, Tuple, Union
from pathlib import Path
import os
from pydantic import BaseModel, field_validator
import numpy as np


class ModelParamsConfig(BaseModel):
    """Default parameters for the standard training models."""

    # Linear models - strengthened regularization for high-dimensional features
    ridge: Dict[str, Any] = {"alpha": 10.0}  # Increased from 1.0
    lasso: Dict[str, Any] = {"alpha": 0.5}  # Increased from 0.1
    elastic_net: Dict[str, Any] = {"alpha": 0.5, "l1_ratio": 0.5}  # Increased alpha

    # Tree-based models - added regularization parameters
    random_forest: Dict[str, Any] = {
        "n_estimators": 100,
        "min_samples_leaf": 3,  # Added regularization
        "max_features": "sqrt",  # Limit feature sampling
        "min_samples_split": 5,  # Added regularization
    }
    gradient_boost: Dict[str, Any] = {
        "n_estimators": 100,
        "min_samples_leaf": 3,  # Added regularization
        "min_samples_split": 5,  # Added regularization
        "subsample": 0.8,  # Added subsample for regularization
    }

    # Gradient boosting models - optimized for better R²
    xgboost: Dict[str, Any] = {
        "n_estimators": 500,  # Increased for complex patterns
        "learning_rate": 0.03,  # Lower for stable learning
        "max_depth": 6,  # Increased to capture interactions
        "min_child_weight": 5,  # Increased to prevent overfitting on 720 samples
        "subsample": 0.9,  # Higher sampling
        "colsample_bytree": 0.9,  # Higher feature sampling
        "colsample_bylevel": 0.9,  # Add level sampling
        "colsample_bynode": 0.9,  # Add node sampling
        "reg_alpha": 0.1,  # L1 regularization
        "reg_lambda": 1.0,  # L2 regularization
        "gamma": 0.1,  # Minimum split loss
        "max_delta_step": 1,  # Handle imbalanced data
        "tree_method": "hist",  # Required for GPU
        "validate_parameters": True,
        "random_state": 42,
    }
    lightgbm: Dict[str, Any] = {
        "n_estimators": 100,
        "learning_rate": 0.05,  # Reduced from 0.1
        "max_depth": 4,  # Reduced from 6
        "min_child_samples": 3,  # Added regularization (LightGBM equivalent)
        "subsample": 0.8,  # Added subsample
        "feature_fraction": 0.8,  # Added feature subsampling
    }
    catboost: Dict[str, Any] = {
        "n_estimators": 100,
        "learning_rate": 0.05,  # Reduced from 0.1
        "depth": 4,  # Reduced from 6
        "min_data_in_leaf": 3,  # Added regularization (CatBoost equivalent)
        "subsample": 0.8,  # Added subsample
        "bootstrap_type": "Bernoulli",  # Required to use subsample parameter
    }

    # SVR - kept same but can adjust C for more regularization if needed
    svr: Dict[str, Any] = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}

    # ExtraTrees - added regularization
    extratrees: Dict[str, Any] = {
        "n_estimators": 100,
        "min_samples_leaf": 3,  # Added regularization
        "max_features": "sqrt",  # Limit feature sampling
        "min_samples_split": 5,  # Added regularization
    }

    # Neural Network parameters - optimized for small spectroscopic datasets
    neural_network: Dict[str, Any] = {
        "model_type": "light",  # Use lighter architecture to prevent overfitting
        "epochs": 400,  # Moderate epochs with early stopping
        "batch_size": 32,  # Balanced batch size for 444 training samples
        "learning_rate": 0.001,  # Standard learning rate
        "weight_decay": 0.005,  # Moderate regularization
        "dropout_rate": 0.3,  # Moderate dropout
        "early_stopping_patience": 50,
        "verbose": True,
        "use_sample_weights": False,  # Use custom loss weighting only (avoid double-weighting)
        "target_range": (
            12000,
            24000,
        ),  # Target concentration range for loss weighting (ppm NO3, ~p20-p80)
        "range_focused": False,  # Disable range specialist mode (use standard weighting)
    }
    neural_network_light: Dict[str, Any] = {
        "model_type": "light",  # Lightweight version
        "epochs": 200,  # More epochs for better convergence
        "batch_size": 32,  # Stable batch size
        "learning_rate": 0.001,  # Standard learning rate
        "weight_decay": 0.01,  # Moderate regularization
        "dropout_rate": 0.3,  # Moderate dropout (0.5 was too high)
        "early_stopping_patience": 40,  # More patience for convergence
        "verbose": True,
        "use_sample_weights": False,  # Use custom loss weighting only (avoid double-weighting)
        "target_range": (
            12000,
            24000,
        ),  # Target concentration range for loss weighting (ppm NO3, ~p20-p80)
        "range_focused": False,  # Disable range specialist mode (use standard weighting)
    }
    neural_network_autogluon: Dict[str, Any] = {
        "model_type": "autogluon",  # AutoGluon-optimized architecture
        "epochs": 150,  # Based on AutoGluon training
        "batch_size": 64,  # Larger batch size (within max_batch_size: 512)
        "learning_rate": 0.001,  # Standard learning rate
        "weight_decay": 0.01,  # Moderate regularization
        "dropout_rate": 0.3,  # Lower dropout for larger network
        "early_stopping_patience": 30,  # More patience for larger model
        "verbose": True,
        "use_sample_weights": False,  # Use custom loss weighting only (avoid double-weighting)
        "target_range": (
            12000,
            24000,
        ),  # Target concentration range for loss weighting (ppm NO3, ~p20-p80)
        "range_focused": False,  # Disable range specialist mode (use standard weighting)
    }

    # AutoGluon parameters for model trainer integration
    # autogluon: Dict[str, Any] = {
    #     "time_limit": 3600,  # 30 minutes
    #     "presets": 'best_quality',
    #     "verbosity": 2  # Moderate logging
    # }


# Tuner Config
class ObjectiveConfig(BaseModel):
    """Configuration for objective function parameters and sample weighting."""

    # === Robust Objective Parameters ===
    stability_penalty_factor: float = 0.3
    overfitting_penalty_factor: float = 0.4
    cv_stability_cap: float = 0.2  # Cap CV standard deviation at this value
    generalization_gap_cap: float = 0.2  # Cap train-test gap at this value

    # Weighting factors for robust_v2 objective
    cv_weight: float = 0.7
    stability_weight: float = 0.15
    generalization_weight: float = 0.15

    # === Concentration-based Weighting Parameters ===
    # Use data-driven percentiles instead of fixed thresholds
    use_data_driven_thresholds: bool = True

    # Fixed thresholds (used when use_data_driven_thresholds=False)
    # Based on nitrate concentration ranges (typical agricultural soil values in wt%)
    low_concentration_threshold: float = 0.01  # 0.01% NO3 (~100 ppm) - low end
    medium_concentration_threshold: float = 0.05  # 0.05% NO3 (~500 ppm) - medium range
    high_concentration_threshold: float = 0.10  # 0.10% NO3 (~1000 ppm) - high end

    # Weight values for different concentration ranges
    low_concentration_weight: float = 4.0  # Higher weight for rare low concentrations
    medium_concentration_weight: float = 1.5  # Moderate weight for mid-range
    high_concentration_weight: float = 3.5  # Higher weight for rare high concentrations
    default_weight: float = 1.0

    # === Sample Weighting Configuration ===
    # Weight clipping bounds to prevent extreme weights
    min_sample_weight: float = 0.2
    max_sample_weight: float = 5.0

    # Percentile-based weighting
    quartile_low_weight: float = 2.0  # Weight for bottom quartile
    quartile_high_weight: float = 1.5  # Weight for top quartile

    # Distribution-based weighting
    distribution_bins: List[int] = [0, 20, 40, 60, 80, 100]  # Percentile bins
    low_range_modifier: float = 1.3  # Modifier for low concentrations
    high_range_modifier: float = 1.2  # Modifier for high concentrations

    # === MAPE-focused Objective Parameters ===
    # Concentration ranges for MAPE calculation (percentile-based)
    mape_low_percentile: float = 33.0  # Lower 33% for "low" range
    mape_high_percentile: float = 66.0  # Upper 33% for "high" range

    # MAPE weighting factors
    low_mape_weight: float = 0.7
    medium_mape_weight: float = 0.3
    mape_r2_weight: float = 0.5
    mape_score_weight: float = 0.5

    # High penalty for empty ranges
    empty_range_penalty: float = 100.0

    # === Advanced Weighting Parameters ===
    # Quantile-based evaluation
    n_quantiles: int = 5
    min_quantile_samples: int = 2  # Minimum samples needed per quantile

    # KDE smoothing parameters
    kde_epsilon: float = 1e-8  # Small value to prevent division by zero

    # Legacy weighting method concentration ranges and weights (in wt% NO3)
    legacy_ranges: List[Tuple[float, float, float]] = [
        (0.0, 0.01, 2.5),  # (min, max, weight) for each range - very low NO3
        (0.01, 0.03, 2.0),  # low NO3
        (0.03, 0.05, 1.2),  # medium NO3
        (0.05, 0.08, 1.5),  # high NO3
        (0.08, 0.15, 2.5),  # very high NO3
    ]

    # Improved method percentiles and weights
    improved_percentiles: List[float] = [10, 25, 50, 75, 90]
    improved_weights: List[float] = [3.0, 2.2, 1.8, 1.0, 1.5, 2.5]

    def validate_concentration_ranges(self, y_data: np.ndarray) -> Dict[str, Any]:
        """
        Validates that the configured concentration ranges make sense for the actual data.

        Args:
            y_data: Array of actual concentration values

        Returns:
            Dictionary with validation results and recommendations
        """
        import numpy as np

        data_min, data_max = np.min(y_data), np.max(y_data)
        data_mean, data_std = np.mean(y_data), np.std(y_data)

        warnings_list: List[str] = []
        recommendations_list: List[str] = []

        validation_results: Dict[str, Any] = {
            "data_stats": {
                "min": data_min,
                "max": data_max,
                "mean": data_mean,
                "std": data_std,
                "range": data_max - data_min,
            },
            "fixed_thresholds_valid": True,
            "warnings": warnings_list,
            "recommendations": recommendations_list,
        }

        # Check if fixed thresholds are within data range
        if self.low_concentration_threshold < data_min:
            validation_results["fixed_thresholds_valid"] = False
            warnings_list.append(
                f"Low threshold ({self.low_concentration_threshold}) is below data minimum ({data_min:.3f})"
            )

        if self.high_concentration_threshold > data_max:
            validation_results["fixed_thresholds_valid"] = False
            warnings_list.append(
                f"High threshold ({self.high_concentration_threshold}) is above data maximum ({data_max:.3f})"
            )

        # Check legacy ranges validity
        for i, (min_val, max_val, _) in enumerate(self.legacy_ranges):
            samples_in_range = np.sum((y_data >= min_val) & (y_data <= max_val))
            if samples_in_range == 0:
                warnings_list.append(
                    f"Legacy range {i + 1} ({min_val}-{max_val}) contains no samples"
                )

        # Generate recommendations
        if data_max - data_min < 0.1:
            recommendations_list.append(
                "Data range is narrow (<0.1). Consider using fewer concentration bins."
            )

        if not validation_results["fixed_thresholds_valid"]:
            low_percentile = np.percentile(y_data, 33)
            high_percentile = np.percentile(y_data, 67)
            recommendations_list.append(
                f"Consider using data-driven thresholds: low={low_percentile:.3f}, high={high_percentile:.3f}"
            )
            recommendations_list.append(
                "Set use_data_driven_thresholds=True for automatic threshold selection"
            )

        return validation_results


class DimensionReductionConfig(BaseModel):
    """Configuration for dimensionality reduction strategies."""

    method: Literal[
        "pca",
        "pls",
        "autoencoder",
        "feature_clustering",
        "vae",
        "denoising_ae",
        "sparse_ae",
    ] = "pca"

    # Common parameters
    n_components: Union[int, float, str] = (
        0.95  # int for fixed, float for variance (PCA), 'auto' for clustering
    )

    # PCA-specific parameters
    pca_params: Dict[str, Any] = {}

    # PLS-specific parameters
    pls_params: Dict[str, Any] = {"scale": True, "max_iter": 500}

    # Autoencoder-specific parameters
    autoencoder_params: Dict[str, Any] = {
        "hidden_layers": [64, 32],
        "epochs": 400,
        "batch_size": 32,
        "learning_rate": 0.001,
        "device": "auto",
    }

    # Feature clustering-specific parameters
    clustering_params: Dict[str, Any] = {
        "method": "kmeans",  # Clustering algorithm: 'kmeans' (more methods can be added)
        "n_init": 10,  # Number of initializations for KMeans
        "random_state": 42,  # For reproducibility
        "max_iter": 300,  # Maximum iterations for clustering
        "tol": 1e-4,  # Tolerance for convergence
        "algorithm": "lloyd",  # KMeans algorithm: 'lloyd', 'elkan', or 'auto'
        "verbose": 0,  # Verbosity level
    }

    # Advanced autoencoder parameters
    vae_params: Dict[str, Any] = {
        "hidden_layers": [128, 64],
        "epochs": 400,
        "batch_size": 32,
        "learning_rate": 0.001,
        "beta": 1.0,  # Standard VAE
        "device": "auto",
        "auto_components_method": "elbow",  # 'elbow' or 'reconstruction_threshold'
        "auto_components_range": (5, 25),  # Range to search for optimal components
    }

    denoising_ae_params: Dict[str, Any] = {
        "hidden_layers": [128, 64],
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "noise_factor": 0.2,
        "device": "auto",
    }

    sparse_ae_params: Dict[str, Any] = {
        "hidden_layers": [128, 64],
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "sparsity_param": 0.05,
        "beta": 3.0,
        "device": "auto",
    }

    def get_params(self) -> Dict[str, Any]:
        """Get parameters for the selected method."""
        params = {"n_components": self.n_components}

        if self.method == "pca":
            params.update(self.pca_params)
        elif self.method == "pls":
            params.update(self.pls_params)
        elif self.method == "autoencoder":
            params.update(self.autoencoder_params)
        elif self.method == "feature_clustering":
            params.update(self.clustering_params)
        elif self.method == "vae":
            params.update(self.vae_params)
        elif self.method == "denoising_ae":
            params.update(self.denoising_ae_params)
        elif self.method == "sparse_ae":
            params.update(self.sparse_ae_params)

        return params


class ParallelConfig(BaseModel):
    """Configuration for parallel processing and resource utilization."""

    # Feature engineering parallelization
    use_feature_parallel: bool = (
        True  # Enable parallel feature extraction (--feature-parallel)
    )
    use_data_parallel: bool = True  # Enable parallel data processing (--data-parallel)

    # Number of jobs for different operations
    feature_n_jobs: int = -1  # Jobs for feature engineering (-1 = use all cores)
    data_n_jobs: int = -1  # Jobs for data processing operations (-1 = use all cores)
    model_n_jobs: int = 1  # Jobs for model training/optimization

    # Override for specific operations (can be set per operation)
    override_single_threaded: bool = False  # Force single-threaded for debugging


class TunerConfig(BaseModel):
    """Configuration for the Optuna hyperparameter tuner."""

    n_trials: int = 400
    timeout: int = 120
    # models_to_tune: List[str] = ["random_forest", "xgboost", "lightgbm", "catboost", "svr", "extratrees"]
    models_to_tune: List[str] = ["extratrees"]
    # Use this to select the tuning goal.
    objective_function_name: Literal[
        "r2",
        "robust",
        "concentration_weighted",
        "mape_focused",
        "robust_v2",
        "weighted_r2",
        "balanced_mae",
        "quantile_weighted",
        "distribution_based",
        "hybrid_weighted",
    ] = "distribution_based"

    # Objective function configuration
    objectives: ObjectiveConfig = ObjectiveConfig()


# This placeholder class is needed for Pydantic validation.
# In a real project, it would be imported from src.spectral_extraction.results
class PeakRegion(BaseModel):
    element: str
    lower_wavelength: float
    upper_wavelength: float
    center_wavelengths: List[float]

    @property
    def n_peaks(self) -> int:
        return len(self.center_wavelengths)


class AutoGluonConfig(BaseModel):
    """
    Configuration specific to the AutoGluon pipeline.

    BASED ON BEST AutoGluon RUN: R² = 0.6035 (training_summary_simple_only_autogluon_20250812_221647.csv)
    Key settings from best run:
    - Strategy: simple_only
    - Time limit: 9800s
    - Presets: best_quality
    - Bag folds: 3, Bag sets: 1, Stack levels: 0 (NO STACKING - critical!)
    - Excluded models: ['KNN', 'FASTAI', 'TABPFN', 'FASTTEXT', 'CAT']
    - Sample weights: NOW ENABLED (was disabled in best run, but we want to test with weights)
    """

    # ENABLED: sample weights for extreme concentration ranges (from best run but with weights enabled)
    use_improved_config: bool = True

    # Sample weighting configuration - NOW ENABLED (from best run but with sample weights)
    weight_method: Literal["legacy", "improved"] = "improved"

    time_limit: int = 12000  # 3 hours for thorough optimization
    presets: str = "good_quality"  # From best AutoGluon run configuration

    # GPU-safe preset switching - automatically use good_quality when GPU is enabled
    gpu_safe_preset: str = "extreme_quality"  # Fallback preset for GP
    model_subdirectory: str = "autogluon"
    num_trials: int = 100  # Increased trials for better hyperparameter search

    # Optimized training arguments - ENABLED STACKING for better performance
    ag_args_fit: Dict[str, Any] = {
        "num_bag_folds": 5,  # Increased for better cross-validation
        "num_bag_sets": 2,  # Increased for more diverse models
        "num_stack_levels": 2,  # ENABLE STACKING - critical for R² > 0.75
        "auto_stack": True,  # Enable AutoGluon's automatic stacking decisions
        "dynamic_stacking": True,  # Enable dynamic stacking override
        #'num_gpus': 0,          # GPU usage
    }
    ag_args_ensemble: Dict[str, Any] = {
        "fold_fitting_strategy": "sequential_local",
    }
    excluded_model_types: List[str] = [
        "FASTAI",
        "FASTTEXT",
    ]  # From best AutoGluon run configuration

    # GPU-specific exclusions - models that commonly fail with GPU + best_quality
    gpu_excluded_models: List[
        str
    ] = []  # Add models here if they consistently fail with GPU
    # Common problematic combinations:
    # ['CAT'] - Exclude CatBoost if it consistently throws GPU exceptions
    # ['GBM'] - Exclude LightGBM if it has GPU memory issues
    # ['CAT', 'GBM'] - Exclude both gradient boosting models, keep XGBoost and neural networks

    # Aggressive hyperparameters for spectral regression - focus on performance over speed
    hyperparameters: Dict[str, Any] = {
        "GBM": [
            # LightGBM - more aggressive configurations (GPU settings added automatically by autogluon_trainer.py)
            {
                "num_boost_round": 1500,
                "learning_rate": 0.01,
                "num_leaves": 63,
                "min_data_in_leaf": 2,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
            },
            {
                "num_boost_round": 1000,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "min_data_in_leaf": 3,
                "feature_fraction": 0.8,
                "lambda_l1": 0.1,
                "lambda_l2": 0.1,
            },
            {
                "num_boost_round": 500,
                "learning_rate": 0.03,
                "num_leaves": 127,
                "min_data_in_leaf": 1,
                "feature_fraction": 0.7,
                "extra_trees": True,
            },
            {
                "num_boost_round": 200,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 5,
            },
            {
                "num_boost_round": 100,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "min_data_in_leaf": 8,
            },
            # {'num_boost_round': 1000, 'learning_rate': 0.02, 'num_leaves': 31, 'feature_fraction': 0.9, 'min_data_in_leaf': 2, 'reg_alpha': 0.1, 'reg_lambda': 0.1},
            # {'num_boost_round': 800, 'learning_rate': 0.03, 'num_leaves': 63, 'feature_fraction': 0.8, 'min_data_in_leaf': 3, 'reg_alpha': 0.0, 'reg_lambda': 0.0},
            # {'num_boost_round': 600, 'learning_rate': 0.05, 'num_leaves': 127, 'feature_fraction': 0.7, 'min_data_in_leaf': 1, 'reg_alpha': 0.5, 'reg_lambda': 0.5},
        ],
        "CAT": [
            # CatBoost - optimized for small datasets (GPU settings added automatically by autogluon_trainer.py)
            {
                "iterations": 1000,
                "learning_rate": 0.02,
                "depth": 8,
                "min_data_in_leaf": 1,
                "l2_leaf_reg": 1.0,
                "subsample": 0.8,
            },
            {
                "iterations": 800,
                "learning_rate": 0.03,
                "depth": 6,
                "min_data_in_leaf": 2,
                "l2_leaf_reg": 3.0,
                "subsample": 0.9,
            },
            {
                "iterations": 600,
                "learning_rate": 0.05,
                "depth": 10,
                "min_data_in_leaf": 1,
                "l2_leaf_reg": 0.5,
                "subsample": 0.7,
            },
        ],
        "XGB": [
            # XGBoost - high performance configurations with explicit device control
            {
                "n_estimators": 1000,
                "max_depth": 8,
                "learning_rate": 0.02,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "tree_method": "hist",
                "device": "cpu",
            },
            {
                "n_estimators": 800,
                "max_depth": 6,
                "learning_rate": 0.03,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "reg_alpha": 0.0,
                "reg_lambda": 0.5,
                "tree_method": "hist",
                "device": "cpu",
            },
            {
                "n_estimators": 600,
                "max_depth": 10,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "reg_alpha": 0.5,
                "reg_lambda": 2.0,
                "tree_method": "hist",
                "device": "cpu",
            },
        ],
        "RF": [
            # Random Forest - more trees, less regularization for better fit
            {
                "n_estimators": 500,
                "max_features": "sqrt",
                "max_depth": None,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
            },
            {
                "n_estimators": 300,
                "max_features": 0.8,
                "max_depth": 25,
                "min_samples_leaf": 2,
                "min_samples_split": 5,
            },
            {
                "n_estimators": 200,
                "max_features": 0.6,
                "max_depth": 15,
                "min_samples_leaf": 3,
                "min_samples_split": 8,
            },
        ],
        "XT": [
            # ExtraTrees - similar to RF but with more randomness
            {
                "n_estimators": 500,
                "max_features": "sqrt",
                "max_depth": None,
                "min_samples_leaf": 1,
                "min_samples_split": 2,
            },
            {
                "n_estimators": 300,
                "max_features": 0.8,
                "max_depth": 25,
                "min_samples_leaf": 2,
                "min_samples_split": 5,
            },
            {
                "n_estimators": 200,
                "max_features": 0.6,
                "max_depth": 15,
                "min_samples_leaf": 3,
                "min_samples_split": 8,
            },
        ],
        "NN_TORCH": [
            # Neural networks - optimized for spectral data with proper regularization
            {
                "num_epochs": 500,
                "learning_rate": 0.0003,
                "activation": "elu",
                "dropout_prob": 0.2,
                "weight_decay": 0.01,
            },
            {
                "num_epochs": 400,
                "learning_rate": 0.0005,
                "activation": "relu",
                "dropout_prob": 0.3,
                "weight_decay": 0.005,
            },
            {
                "num_epochs": 300,
                "learning_rate": 0.001,
                "activation": "leaky_relu",
                "dropout_prob": 0.4,
                "weight_decay": 0.001,
            },
            {
                "num_epochs": 200,
                "learning_rate": 0.001,
                "dropout_prob": 0.3,
                "weight_decay": 0.01,
            },
        ],
        "LR": [
            # Linear models - use sklearn-compatible parameters
            {"penalty": "L2"},
            {"penalty": "L1"},
        ],
    }

    # GPU-safe hyperparameters - less aggressive configurations for stability
    gpu_safe_hyperparameters: Dict[str, Any] = {
        "GBM": [
            # LightGBM - conservative GPU-safe configurations with explicit GPU device settings
            {
                "num_boost_round": 500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "min_data_in_leaf": 5,
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
            },
            {
                "num_boost_round": 300,
                "learning_rate": 0.1,
                "num_leaves": 15,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "device": "gpu",
                "gpu_platform_id": 0,
                "gpu_device_id": 0,
            },
        ],
        "CAT": [
            # CatBoost - conservative GPU-safe configurations with explicit GPU task type
            {
                "iterations": 500,
                "learning_rate": 0.05,
                "depth": 6,
                "min_data_in_leaf": 3,
                "l2_leaf_reg": 3.0,
                "task_type": "GPU",
                "devices": "0",
            },
            {
                "iterations": 300,
                "learning_rate": 0.1,
                "depth": 4,
                "min_data_in_leaf": 5,
                "l2_leaf_reg": 1.0,
                "task_type": "GPU",
                "devices": "0",
            },
        ],
        "XGB": [
            # XGBoost - conservative GPU-safe configurations with explicit CUDA device control
            {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "device": "cuda",
            },
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "tree_method": "hist",
                "device": "cuda",
            },
        ],
        "RF": [
            # Random Forest - same as aggressive (doesn't use GPU anyway)
            {
                "n_estimators": 300,
                "max_features": "sqrt",
                "max_depth": 20,
                "min_samples_leaf": 2,
            },
            {
                "n_estimators": 200,
                "max_features": 0.8,
                "max_depth": 15,
                "min_samples_leaf": 3,
            },
        ],
        "XT": [
            # ExtraTrees - same as aggressive (doesn't use GPU anyway)
            {
                "n_estimators": 300,
                "max_features": "sqrt",
                "max_depth": 20,
                "min_samples_leaf": 2,
            },
            {
                "n_estimators": 200,
                "max_features": 0.8,
                "max_depth": 15,
                "min_samples_leaf": 3,
            },
        ],
        "NN_TORCH": [
            # Neural networks - more conservative for GPU stability
            {
                "num_epochs": 100,
                "learning_rate": 0.001,
                "dropout_prob": 0.3,
                "weight_decay": 0.01,
            },
            {
                "num_epochs": 80,
                "learning_rate": 0.005,
                "dropout_prob": 0.4,
                "weight_decay": 0.001,
            },
        ],
        "LR": [
            # Linear models - same as aggressive
            {"penalty": "L2"},
            {"penalty": "L1"},
        ],
    }


class Config(BaseModel):
    """Main configuration class for the entire pipeline."""

    project_name: str = "NitratePrediction"
    run_timestamp: str
    random_state: int = 42
    use_gpu: bool = False  # Global GPU flag

    _data_dir: str
    _raw_data_dir: str
    _processed_data_dir: str
    _model_dir: str
    _reports_dir: str
    _log_dir: str
    _bad_files_dir: str
    _averaged_files_dir: str
    _cleansed_files_dir: str
    _bad_prediction_files_dir: str

    _reference_data_path: str

    # --- Data Management ---
    target_column: str = "NO3"
    sample_id_column: str = "Sample ID"
    exclude_pot_samples: bool = False
    test_split_size: float = 0.20
    max_samples: Optional[int] = None

    # ADDED: Configuration for target value filtering - adjusted for nitrate concentration ranges
    # Typical soil NO3 levels: 0-500 ppm (0-0.05%), agricultural soils: 10-200 ppm (0.001-0.02%)
    # Values in wt% NO3
    target_value_min: Optional[float] = 8000  # 0 wt% NO3 - minimum detectable
    target_value_max: Optional[float] = 30000  # 0.20 wt% (~2000 ppm) - upper range for agricultural soils



    # Custom validation set directory - if provided, will process raw files from this directory for validation
    # custom_validation_dir: Optional[str] = "/home/payanico/pipeline_nitrates/data/raw/validation_set"
    custom_validation_dir: Optional[str] = None

    # Wavelength standardization configuration
    enable_wavelength_standardization: bool = False
    wavelength_interpolation_method: Literal["linear", "cubic", "nearest"] = "linear"
    wavelength_resolution: float = 0.1  # nm resolution for standardized grid

    # Feature Selection Configuration - to handle high-dimension/low-sample scenario
    use_feature_selection: bool = True  # Enable/disable feature selection
    feature_selection_method: Literal[
        "selectkbest", "rfe", "lasso", "mutual_info", "tree_importance"
    ] = "selectkbest"
    n_features_to_select: Union[int, float] = (
        0.25  # Number of features to select (int) or fraction (float < 1.0)
    )
    feature_selection_score_func: Literal["f_regression", "mutual_info_regression"] = (
        "f_regression"
    )
    # For SelectKBest
    rfe_estimator: Literal["random_forest", "xgboost", "lightgbm"] = (
        "xgboost"  # For RFE
    )
    lasso_alpha: float = 0.01  # Alpha parameter for LASSO feature selection
    tree_importance_threshold: float = 0.001  # Minimum feature importance threshold

    # SHAP-Based Feature Selection (overrides use_feature_selection if enabled)
    use_shap_feature_selection: bool = False  # Enable SHAP-based feature selection
    shap_importance_file: Optional[str] = (
        "models/K_only_catboost_20251009_141347_shap_importance.csv"  # Path to SHAP importance CSV file (e.g., "models/full_context_lightgbm_*_shap_importance.csv")
    )
    shap_top_n_features: int = (
        40  # Number of top features to select based on SHAP importance
    )
    shap_min_importance: Optional[float] = (
        None  # Minimum SHAP importance threshold (optional)
    )

    # Outlier detection settings - RELAXED for more training data
    outlier_method: str = "SAM"
    outlier_threshold: float = 0.95  # More lenient to keep borderline samples
    max_outlier_percentage: float = 50.0  # Limit removal to preserve training data

    # Alternative options (not used in best run):
    # outlier_threshold: float = 0.85          # 85% similarity threshold
    # max_outlier_percentage: float = 80.0     # Allow 80% outlier removal

    @field_validator("outlier_method")
    @classmethod
    def outlier_method_must_be_valid(cls, v):
        if v.upper() not in ["SAM", "MAD"]:
            raise ValueError("outlier_method must be 'SAM' or 'MAD'")
        return v.upper()

    # Literature-verified Nitrogen LIBS spectral lines for NO3 detection:
    # According to NIST and LIBS literature:
    # Primary N I triplet (742-747 nm):
    # - 742.36 nm: Strong N I line
    # - 744.23 nm: Strong N I line
    # - 746.83 nm: Strong N I line (most sensitive for soil nitrogen)
    # Secondary N I lines:
    # - 821.63 nm, 824.21 nm: N I doublet (near-IR, secondary verification)
    # - 868.03 nm, 871.17 nm: N I lines (additional verification)

    # Primary nitrate detection region - N I triplet around 742-747 nm
    # This is the most sensitive region for nitrogen detection in soil samples
    nitrate_region: PeakRegion = PeakRegion(
        element="N_I_primary",
        lower_wavelength=741.0,
        upper_wavelength=748.0,
        center_wavelengths=[742.36, 744.23, 746.83],
    )

    context_regions: List[PeakRegion] = [
        # Secondary nitrogen atomic lines for NO3 detection
        PeakRegion(
            element="N_I_821",
            lower_wavelength=820.0,
            upper_wavelength=826.0,
            center_wavelengths=[821.63, 824.21],
        ),  # N I doublet (secondary verification)
        PeakRegion(
            element="N_I_868",
            lower_wavelength=867.0,
            upper_wavelength=872.0,
            center_wavelengths=[868.03, 871.17],
        ),  # N I lines (additional verification)
        # Supporting elements for N/C, N/O, N/H ratio calculations (critical for NO3 vs NH4 discrimination)
        PeakRegion(
            element="C_I",
            lower_wavelength=832.5,
            upper_wavelength=834.5,
            center_wavelengths=[833.5],
        ),
        PeakRegion(
            element="O_I",
            lower_wavelength=776.5,
            upper_wavelength=778.5,
            center_wavelengths=[777.4],
        ),
        PeakRegion(
            element="H_alpha",
            lower_wavelength=655.5,
            upper_wavelength=657.5,
            center_wavelengths=[656.3],
        ),
        # NPK context (P and K for nutrient balance analysis)
        PeakRegion(
            element="P_I",
            lower_wavelength=653.56,
            upper_wavelength=655.56,
            center_wavelengths=[654.56],
        ),
        PeakRegion(
            element="K_I",
            lower_wavelength=765.0,
            upper_wavelength=771.0,
            center_wavelengths=[766.49, 769.90],
        ),
        # Calcium for soil chemistry context
        PeakRegion(
            element="CA_II_393",
            lower_wavelength=392.5,
            upper_wavelength=394.5,
            center_wavelengths=[393.37],
        ),
    ]

    # Molecular bands - CRITICAL for nitrate (NO3) detection and nitrogen speciation
    # These bands help discriminate between nitrate (NO3), nitrite (NO2), ammonium (NH4), and organic N
    molecular_bands: List[PeakRegion] = [
        PeakRegion(
            element="CN_violet_1",
            lower_wavelength=385.0,
            upper_wavelength=390.0,
            center_wavelengths=[387.5],
        ),  # Organic nitrogen indicator
        PeakRegion(
            element="CN_violet_2",
            lower_wavelength=415.0,
            upper_wavelength=425.0,
            center_wavelengths=[420.0],
        ),  # Organic nitrogen indicator
        PeakRegion(
            element="CN_red",
            lower_wavelength=606.0,
            upper_wavelength=620.0,
            center_wavelengths=[612.0],
        ),  # Additional CN band
        PeakRegion(
            element="NH_band",
            lower_wavelength=335.0,
            upper_wavelength=338.0,
            center_wavelengths=[336.5],
        ),  # Ammonium (NH4) indicator
        PeakRegion(
            element="NO_band",
            lower_wavelength=226.0,
            upper_wavelength=248.0,
            center_wavelengths=[237.0],
        ),  # Direct nitrate/nitrite indicator
        PeakRegion(
            element="NO_gamma",
            lower_wavelength=235.0,
            upper_wavelength=280.0,
            center_wavelengths=[257.0],
        ),  # NO gamma system (nitrate/nitrite)
    ]

    # Additional macro elements
    macro_elements: List[PeakRegion] = [
        PeakRegion(
            element="S_I",
            lower_wavelength=834.5,
            upper_wavelength=836.5,
            center_wavelengths=[835.5],
        ),
        PeakRegion(
            element="S_I_2",
            lower_wavelength=868.0,
            upper_wavelength=870.0,
            center_wavelengths=[869.0],
        ),
        # Note: Primary P, K and Mg lines are already defined in context_regions for NPK nutrient analysis
        # and soil chemistry context. These elements support nitrate prediction but are not the primary target.
        # P_I is already in context_regions (line 649), so we don't duplicate it here
        # Additional K lines for macro nutrient context if needed:
        # PeakRegion(element="K_I_macro", lower_wavelength=765.0, upper_wavelength=771.0, center_wavelengths=[766.49, 769.90]),
        # PeakRegion(element="K_I_404_macro", lower_wavelength=403.5, upper_wavelength=405.5, center_wavelengths=[404.41, 404.72]),
    ]

    # Micro elements
    micro_elements: List[PeakRegion] = [
        PeakRegion(
            element="Fe_I",
            lower_wavelength=437.5,
            upper_wavelength=439.5,
            center_wavelengths=[438.4],
        ),
        PeakRegion(
            element="Fe_I_2",
            lower_wavelength=439.5,
            upper_wavelength=441.5,
            center_wavelengths=[440.5],
        ),
        PeakRegion(
            element="Mn_I",
            lower_wavelength=401.5,
            upper_wavelength=404.5,
            center_wavelengths=[403.08],
        ),  # WIDENED: 0.8→3.0 nm (was too narrow for preprocessing)
        PeakRegion(
            element="B_I",
            lower_wavelength=248.5,
            upper_wavelength=250.5,
            center_wavelengths=[249.8],
        ),
        PeakRegion(
            element="Zn_I",
            lower_wavelength=480.5,
            upper_wavelength=482.5,
            center_wavelengths=[481.1],
        ),
        PeakRegion(
            element="Cu_I",
            lower_wavelength=323.5,
            upper_wavelength=325.5,
            center_wavelengths=[324.8],
        ),
        PeakRegion(
            element="Mo_I",
            lower_wavelength=378.5,
            upper_wavelength=380.5,
            center_wavelengths=[379.8],
        ),
    ]

    # Oxygen and hydrogen lines
    # Note: O_I and H_alpha are already in context_regions for N/O and N/H ratio calculations
    oxygen_hydrogen: List[PeakRegion] = [
        # PeakRegion(element="O_I", lower_wavelength=776.5, upper_wavelength=778.5, center_wavelengths=[777.4]),  # REMOVED: duplicate from context_regions
        PeakRegion(
            element="O_I_2",
            lower_wavelength=843.5,
            upper_wavelength=845.5,
            center_wavelengths=[844.6],
        ),
        # PeakRegion(element="H_alpha", lower_wavelength=655.5, upper_wavelength=657.5, center_wavelengths=[656.3]),  # REMOVED: duplicate from context_regions
        PeakRegion(
            element="H_beta",
            lower_wavelength=485.0,
            upper_wavelength=487.0,
            center_wavelengths=[486.1],
        ),
    ]

    # Feature configuration flags - OPTIMIZED FOR NITRATE (NO3) DETECTION
    enable_molecular_bands: bool = True  # CRITICAL: CN/NH/NO bands essential for N speciation (NO3 vs NH4 vs organic N)
    enable_macro_elements: bool = (
        True  # S, P, K - NPK context for agricultural interpretation
    )
    enable_micro_elements: bool = True  # Fe, Mn, B, Zn - soil chemistry context
    enable_oxygen_hydrogen: bool = (
        True  # CRITICAL: H/O ratios essential for NO3 vs NH4 discrimination
    )
    enable_advanced_ratios: bool = (
        True  # N/C, N/O, N/H ratios are critical for nitrogen speciation
    )
    enable_spectral_patterns: bool = (
        True  # Peak shapes help identify nitrogen compounds
    )
    enable_interference_correction: bool = (
        True  # Atmospheric N2 can interfere with soil N measurements
    )
    enable_plasma_indicators: bool = False  # Not needed for concentration prediction

    # Nitrate feature generation method
    use_focused_nitrate_features: bool = (
        True  # If True, uses focused features; if False, uses original features
    )

    # Raw spectral data mode - pass filtered intensities directly to models without feature engineering
    use_raw_spectral_data: bool = False  # If True, use raw intensities from PeakRegions instead of engineered features

    # Spectral Preprocessing Configuration
    use_spectral_preprocessing: bool = (
        True  # Enable spectral preprocessing (Savgol, SNV, ALS baseline)
    )
    spectral_preprocessing_method: Literal[
        "none", "savgol", "snv", "baseline", "savgol+snv", "baseline+snv", "full"
    ] = "full"  # Phase 1 recommended

    def get_regions_for_strategy(self, strategy: str) -> List[PeakRegion]:
        """Get spectral regions based on feature strategy."""
        if strategy == "N_only":
            # Only N regions + C, O, H for N/C, N/O, N/H ratios + P, K for NPK context
            n_regions = [self.nitrate_region]
            # Add secondary nitrogen lines
            n_regions.extend(
                [r for r in self.context_regions if r.element.startswith("N_I")]
            )
            # Add C_I for N_C_ratio calculation
            c_region = next(
                (r for r in self.context_regions if r.element == "C_I"), None
            )
            if c_region:
                n_regions.append(c_region)
            # Add O and H for N/O and N/H ratios (critical for NO3 vs NH4)
            o_region = next(
                (r for r in self.context_regions if r.element == "O_I"), None
            )
            if o_region:
                n_regions.append(o_region)
            h_region = next(
                (r for r in self.context_regions if r.element == "H_alpha"), None
            )
            if h_region:
                n_regions.append(h_region)
            # Add P and K for NPK nutrient context
            p_region = next(
                (r for r in self.context_regions if r.element == "P_I"), None
            )
            if p_region:
                n_regions.append(p_region)
            k_region = next(
                (r for r in self.context_regions if r.element == "K_I"), None
            )
            if k_region:
                n_regions.append(k_region)
            return n_regions
        else:
            # Full regions for simple_only and full_context
            return self.all_regions

    @property
    def all_regions(self) -> List[PeakRegion]:
        regions = [self.nitrate_region] + self.context_regions

        # Add enabled enhanced regions
        if self.enable_molecular_bands:
            regions.extend(self.molecular_bands)
        if self.enable_macro_elements:
            regions.extend(self.macro_elements)
        if self.enable_micro_elements:
            regions.extend(self.micro_elements)
        if self.enable_oxygen_hydrogen:
            regions.extend(self.oxygen_hydrogen)

        return regions

    # feature_strategies: List[str] = ["N_only", "simple_only", "full_context"]
    feature_strategies: List[str] = ["N_only"]
    peak_shapes: List[str] = ["lorentzian"]
    fitting_mode: str = "mean_first"
    baseline_correction: bool = True

    # models_to_train: List[str] = [
    #     "ridge", "lasso", "random_forest", "gradient_boost", "xgboost",
    #     "lightgbm", "catboost", "svr", "extratrees", "neural_network", "neural_network_light"
    # ]
    models_to_train: List[str] = [
        "random_forest",
        "gradient_boost",
        "extratrees",
        "xgboost",
        "lightgbm",
        "catboost",
        "svr",
    ]

    # Usage in pipeline_config.py:
    #
    # For spectral denoising:
    # dimension_reduction = DimensionReductionConfig(
    #     method='denoising_ae',
    #     n_components=10,
    #     denoising_ae_params={
    #         'hidden_layers': [128, 64],
    #         'noise_factor': 0.2,
    #         'epochs': 100
    #     }
    # )
    #
    # For probabilistic encoding with uncertainty:
    # dimension_reduction = DimensionReductionConfig(
    #     method='vae',
    #     n_components=8,
    #     vae_params={
    #         'hidden_layers': [128, 64],
    #         'beta': 2.0,  # Slight disentanglement
    #         'epochs': 150
    #     }
    # )
    #
    # For automatic VAE component selection:
    # dimension_reduction = DimensionReductionConfig(
    #     method='vae',
    #     n_components='auto',  # Automatic selection
    #     vae_params={
    #         'auto_components_method': 'elbow',  # or 'reconstruction_threshold'
    #         'auto_components_range': (5, 25),  # Search range
    #         'epochs': 200  # Reduce epochs for faster component search
    #     }
    # )
    # Dimensionality reduction configuration (applies to both standard models and AutoGluon)
    use_dimension_reduction: bool = False
    # dimension_reduction: DimensionReductionConfig = DimensionReductionConfig(
    #     method='pls',
    #     n_components=13,  # Better sample-to-feature ratio: 512/10 = 51:1
    #     pls_params={'scale': True, 'max_iter': 500}
    # )
    dimension_reduction: DimensionReductionConfig = DimensionReductionConfig(
        method="pca",
        n_components=0.95,
        # pls_params={'scale': True, 'max_iter': 3000},
        # n_components=0.97,  # VAE requires integer for latent dimension
        # Uncomment to override default vae_params:
        # vae_params={
        #     'hidden_layers': [128, 64],
        #     'beta': 2.0,  # Slight disentanglement
        #     'epochs': 150
        # }
    )

    # Sample weighting configuration for model training
    use_sample_weights: bool = (
        True  # ENABLED GLOBALLY: Critical for handling extreme concentration ranges
    )
    sample_weight_method: Literal[
        "legacy", "improved", "weighted_r2", "distribution_based", "hybrid"
    ] = "distribution_based"

    # Post-processing calibration for improving prediction accuracy
    use_post_calibration: bool = (
        False  # Disable isotonic regression post-processing (overfitting on test data)
    )
    post_calibration_method: Literal["isotonic", "linear", "piecewise"] = "isotonic"
    post_calibration_target: Literal["within_20.5", "mape"] = (
        "within_20.5"  # Target metric to oeptimize
    )

    # Concentration-aware feature enhancement (Option A alternative to sample weighting)
    use_concentration_features: bool = (
        True  # Enable concentration-range features for AutoGluon
    )

    model_params: ModelParamsConfig = ModelParamsConfig()
    cv_folds: int = 5

    autogluon: AutoGluonConfig = AutoGluonConfig()

    log_file: str = "pipeline.log"
    log_level: str = "INFO"

    # --- Parallel Processing ---
    parallel: ParallelConfig = ParallelConfig()

    # --- Model Tuning ---
    tuner: TunerConfig = TunerConfig()

    # Alternative path properties (using internal field names directly)
    @property
    def data_dir_path(self) -> Path:
        return Path(self.__dict__.get("_data_dir", f"{get_base_path()}/data"))

    @property
    def raw_data_dir_path(self) -> Path:
        return Path(
            self.__dict__.get("_raw_data_dir", f"{get_base_path()}/data/raw/061125")
        )

    @property
    def processed_data_dir_path(self) -> Path:
        return Path(
            self.__dict__.get(
                "_processed_data_dir", f"{get_base_path()}/data/processed"
            )
        )

    @property
    def model_dir_path(self) -> Path:
        return Path(self.__dict__.get("_model_dir", f"{get_base_path()}/models"))

    @property
    def reports_dir_path(self) -> Path:
        return Path(self.__dict__.get("_reports_dir", f"{get_base_path()}/reports"))

    @property
    def log_dir_path(self) -> Path:
        return Path(self.__dict__.get("_log_dir", f"{get_base_path()}/logs"))

    @property
    def bad_files_dir_path(self) -> Path:
        return Path(self.__dict__.get("_bad_files_dir", f"{get_base_path()}/bad_files"))

    @property
    def averaged_files_dir_path(self) -> Path:
        return Path(
            self.__dict__.get(
                "_averaged_files_dir",
                f"{get_base_path()}/data/averaged_files_per_sample",
            )
        )

    @property
    def cleansed_files_dir_path(self) -> Path:
        return Path(
            self.__dict__.get(
                "_cleansed_files_dir",
                f"{get_base_path()}/data/cleansed_files_per_sample",
            )
        )

    @property
    def bad_prediction_files_dir_path(self) -> Path:
        return Path(
            self.__dict__.get(
                "_bad_prediction_files_dir", f"{get_base_path()}/bad_prediction_files"
            )
        )

    @property
    def reference_data_path_obj(self) -> Path:
        return Path(
            self.__dict__.get(
                "_reference_data_path",
                f"{get_base_path()}/data/reference_data/Lab_Values.xlsx",
            )
        )

    # Properties that return Path objects for existing code compatibility
    @property
    def data_dir(self) -> Path:
        return Path(self.__dict__.get("_data_dir", f"{get_base_path()}/data"))

    @property
    def raw_data_dir(self) -> Path:
        return Path(
            self.__dict__.get("_raw_data_dir", f"{get_base_path()}/data/raw/061125")
        )

    @property
    def processed_data_dir(self) -> Path:
        return Path(
            self.__dict__.get(
                "_processed_data_dir", f"{get_base_path()}/data/processed"
            )
        )

    @property
    def model_dir(self) -> Path:
        return Path(self.__dict__.get("_model_dir", f"{get_base_path()}/models"))

    @property
    def reports_dir(self) -> Path:
        return Path(self.__dict__.get("_reports_dir", f"{get_base_path()}/reports"))

    @property
    def log_dir(self) -> Path:
        return Path(self.__dict__.get("_log_dir", f"{get_base_path()}/logs"))

    @property
    def bad_files_dir(self) -> Path:
        return Path(self.__dict__.get("_bad_files_dir", f"{get_base_path()}/bad_files"))

    @property
    def averaged_files_dir(self) -> Path:
        return Path(
            self.__dict__.get(
                "_averaged_files_dir",
                f"{get_base_path()}/data/averaged_files_per_sample",
            )
        )

    @property
    def cleansed_files_dir(self) -> Path:
        return Path(
            self.__dict__.get(
                "_cleansed_files_dir",
                f"{get_base_path()}/data/cleansed_files_per_sample",
            )
        )

    @property
    def bad_prediction_files_dir(self) -> Path:
        return Path(
            self.__dict__.get(
                "_bad_prediction_files_dir", f"{get_base_path()}/bad_prediction_files"
            )
        )

    @property
    def reference_data_path(self) -> Path:
        return Path(
            self.__dict__.get(
                "_reference_data_path",
                f"{get_base_path()}/data/reference_data/Lab_Values.xlsx",
            )
        )

    def ensure_paths_exist(self, create_dirs: bool = True) -> None:
        """Ensure all required directories exist, optionally creating them."""
        import logging

        logger = logging.getLogger(__name__)

        # List of directories that should exist
        directories = [
            self.data_dir_path,
            self.raw_data_dir_path,
            self.processed_data_dir_path,
            self.model_dir_path,
            self.reports_dir_path,
            self.log_dir_path,
            self.bad_files_dir_path,
            self.averaged_files_dir_path,
            self.cleansed_files_dir_path,
            self.bad_prediction_files_dir_path,
        ]

        # Create directories if they don't exist
        for directory in directories:
            if create_dirs:
                directory.mkdir(parents=True, exist_ok=True)
                logger.info(f"Ensured directory exists: {directory}")
            elif not directory.exists():
                logger.warning(f"Directory does not exist: {directory}")

        # Check for reference data file (don't create, just warn if missing)
        reference_path = self.reference_data_path_obj
        if not reference_path.exists():
            # Try to find Excel files in data directory
            data_dir = self.data_dir_path
            excel_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
            if excel_files:
                logger.info(f"Reference file not found at {reference_path}")
                logger.info(
                    f"Found Excel files in data directory: {[f.name for f in excel_files]}"
                )
                # Use the first Excel file found
                self.__dict__["_reference_data_path"] = str(excel_files[0])
                logger.info(f"Using Excel file: {excel_files[0]}")
            else:
                logger.warning(f"Reference data file not found: {reference_path}")

    def is_cloud_deployment(self) -> bool:
        """Check if running in cloud environment."""
        return os.getenv("STORAGE_TYPE") == "gcs"

    class Config:
        validate_assignment = True


# Get the project root directory (two levels up from config directory)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Auto-detect environment: use /app for containers, otherwise use local path
_BASE_PATH_CACHE = None
_BASE_PATH_PRINTED = False


def get_base_path() -> Path:
    """
    Automatically detect if we're running in a container (GCP) or locally.
    Returns appropriate base path. Caches result and only prints message once.
    """
    global _BASE_PATH_CACHE, _BASE_PATH_PRINTED

    if _BASE_PATH_CACHE is not None:
        return _BASE_PATH_CACHE

    # Check if we're in a container environment
    if os.path.exists("/app") and os.access("/app", os.W_OK):
        _BASE_PATH_CACHE = Path("/app")
        if not _BASE_PATH_PRINTED:
            print(f"[CONFIG] Detected container environment, using base path: /app")
            _BASE_PATH_PRINTED = True
    # Check environment variable override
    elif os.getenv("PIPELINE_ROOT"):
        _BASE_PATH_CACHE = Path(os.getenv("PIPELINE_ROOT"))
        if not _BASE_PATH_PRINTED:
            print(f"[CONFIG] Using environment override, base path: {_BASE_PATH_CACHE}")
            _BASE_PATH_PRINTED = True
    # Default to project root for local development
    else:
        _BASE_PATH_CACHE = PROJECT_ROOT
        if not _BASE_PATH_PRINTED:
            print(
                f"[CONFIG] Detected local development, using base path: {PROJECT_ROOT}"
            )
            _BASE_PATH_PRINTED = True

    return _BASE_PATH_CACHE


BASE_PATH = get_base_path()

config = Config(
    run_timestamp="placeholder",
    _data_dir=str(BASE_PATH / "data"),
    _raw_data_dir=str(BASE_PATH / "data" / "raw" / "061125"),
    _processed_data_dir=str(BASE_PATH / "data" / "processed"),
    _model_dir=str(BASE_PATH / "models"),
    _reports_dir=str(BASE_PATH / "reports"),
    _log_dir=str(BASE_PATH / "logs"),
    _reference_data_path=str(BASE_PATH / "data" / "reference_data" / "Lab_Values.xlsx"),
    _bad_files_dir=str(BASE_PATH / "bad_files"),
    _averaged_files_dir=str(BASE_PATH / "data" / "averaged_files_per_sample"),
    _cleansed_files_dir=str(BASE_PATH / "data" / "cleansed_files_per_sample"),
    _bad_prediction_files_dir=str(BASE_PATH / "bad_prediction_files"),
)
