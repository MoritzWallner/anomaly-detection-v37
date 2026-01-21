import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from scipy import stats
from typing import List, Dict, Any
import warnings
import os
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

# Use non-interactive backend for saving plots without display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Enable output by default for local use (set SUPPRESS_OUTPUT=true to disable)
SUPPRESS_OUTPUT = os.environ.get('SUPPRESS_OUTPUT', 'false') == 'true'

def _print(*args, **kwargs):
    """Conditional print that respects SUPPRESS_OUTPUT flag."""
    if not SUPPRESS_OUTPUT:
        print(*args, **kwargs)


def calculate_slope(y: np.ndarray) -> float:
    """Calculate slope of linear regression for y against x=[0,1,2,...,n-1]."""
    n = len(y)
    if n < 2:
        return 0.0
    x = np.arange(n)
    x_mean = (n - 1) / 2
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator if denominator != 0 else 0.0


def calculate_trend(y: np.ndarray) -> np.ndarray:
    """Calculate linear trend line for y against x=[0,1,2,...,n-1]."""
    n = len(y)
    if n < 2:
        return y.copy()
    x = np.arange(n)
    slope = calculate_slope(y)
    intercept = np.mean(y) - slope * (n - 1) / 2
    return slope * x + intercept


def preprocess_enum_values(parameter_anomaly_group_array: List[Dict]) -> None:
    """
    Convert enum features to numeric values in-place.

    Creates global enum-to-int mapping across all groups to ensure consistency.
    Example: {"charging": 0, "driving": 1, "idle": 2}
    """

    # First pass: Identify enum features and collect unique values globally
    enum_features = {}  # {feature_name: set of unique values across all groups}

    for group in parameter_anomaly_group_array:
        for feature in group['featureArray']:
            param_history = feature.get('parameterHistoryArray', [])

            if len(param_history) == 0:
                continue

            # Check if this is an enum feature
            first_type = param_history[0].get('type', 'number')

            if first_type == 'enum':
                feature_name = feature['featureName']

                if feature_name not in enum_features:
                    enum_features[feature_name] = set()

                # Collect all unique enum values for this feature
                for point in param_history:
                    if isinstance(point['value'], str):  # Enum values are strings
                        enum_features[feature_name].add(point['value'])

    if not enum_features:
        return  # No enum features to process

    _print("\n[Preprocessing] Converting enum features to numbers...")

    # Second pass: Create global mappings and convert values
    for feature_name, unique_values in enum_features.items():
        # Sort for consistency (same enum always gets same number across runs)
        sorted_values = sorted(list(unique_values))
        enum_mapping = {val: idx for idx, val in enumerate(sorted_values)}

        _print(f"  Feature '{feature_name}': {enum_mapping}")

        # Convert all values for this feature across all groups
        for group in parameter_anomaly_group_array:
            for feature in group['featureArray']:
                if feature['featureName'] == feature_name:
                    for point in feature['parameterHistoryArray']:
                        if point.get('type') == 'enum' and isinstance(point['value'], str):
                            point['value'] = enum_mapping[point['value']]
                            # Keep type as 'enum' for output, but value is now int


def detect_anomalies(request_data: Dict[str, Any], save_plots_path: str = None) -> Dict[str, Any]:
    """
    Detects outlier groups by analyzing features. Supports both time-series and cross-sectional data.

    Args:
        request_data: Dict with:
            - dataType: "time-series" | "cross-sectional" (default: "time-series")
            - groups: List of groups with features
        save_plots_path: Optional path to save visualization PNG (e.g., "diagrams/vehicles/anomaly_analysis.png")

    For time-series data:
        - Uses monthly aggregation to derive shapValues (slope, max_drop, derivative)
        - Detects degradation trends over time

    For cross-sectional data:
        - Uses raw feature values directly
        - Detects unusual combinations of values

    Returns:
        Dict with groups (isOutlier flags), plots, and metadata
    """

    # Handle both old format (list) and new format (dict with dataType)
    if isinstance(request_data, list):
        # Backward compatibility: treat list as time-series groups
        parameter_anomaly_group_array = request_data
        data_type = 'time-series'
    else:
        parameter_anomaly_group_array = request_data.get('groups', [])
        data_type = request_data.get('dataType', 'time-series')

    _print("=" * 80)
    _print(f"ANOMALY DETECTION PIPELINE (dataType: {data_type})")
    _print("=" * 80)

    if not parameter_anomaly_group_array or len(parameter_anomaly_group_array) == 0:
        _print("ERROR: Empty input array")
        return {
            'groups': parameter_anomaly_group_array,
            'plots': {},
            'metadata': {}
        }

    # Step 0: Convert enum values to numbers (shared for both types)
    preprocess_enum_values(parameter_anomaly_group_array)

    _print(f"\n[1/5] Analyzing {len(parameter_anomaly_group_array)} groups...")

    # Route based on dataType
    if data_type == 'time-series':
        feature_matrix, group_ids, feature_names = build_time_series_features(parameter_anomaly_group_array)
    elif data_type == 'cross-sectional':
        feature_matrix, group_ids, feature_names = build_cross_sectional_features(parameter_anomaly_group_array)
    else:
        raise ValueError(f"Unknown dataType: {data_type}. Use 'time-series' or 'cross-sectional'")

    if len(feature_matrix) < 2:
        _print("WARNING: Need at least 2 groups for comparison")
        return {
            'groups': parameter_anomaly_group_array,
            'plots': {},
            'metadata': {}
        }

    # Step 3: Detect group-level outliers (shared logic)
    _print("\n[4/5] Detecting group-level outliers...")
    group_outlier_flags, feature_importance_dict = detect_group_outliers(
        feature_matrix, group_ids, parameter_anomaly_group_array, feature_names, data_type
    )

    # Step 4: Detect feature-level and point-level outliers
    _print("\n[5/5] Detecting feature-level and point-level outliers...")
    detect_feature_and_point_outliers(parameter_anomaly_group_array, group_outlier_flags, feature_importance_dict, data_type)

    # Print summary
    _print("\n" + "=" * 80)
    _print("RESULTS")
    _print("=" * 80)
    outlier_groups = [g['parameterAnomalyGroupId'] for g in parameter_anomaly_group_array if g.get('isOutlier', False)]
    _print(f"\nOutlier groups detected: {outlier_groups if outlier_groups else 'None'}")

    for group in parameter_anomaly_group_array:
        if group.get('isOutlier', False):
            _print(f"\n  {group['parameterAnomalyGroupId']}: OUTLIER")
            outlier_features = [f['featureName'] for f in group['featureArray'] if f.get('isOutlier', False)]
            if outlier_features:
                _print(f"    Problematic features: {outlier_features}")

    _print("\n" + "=" * 80)

    # Generate plot data for frontend (type-specific)
    _print("\n[6/6] Generating plot data for visualization...")
    plot_data = generate_plot_data(parameter_anomaly_group_array, feature_importance_dict, group_outlier_flags, data_type, feature_matrix, feature_names)

    # Extract metadata
    metadata = extract_metadata(parameter_anomaly_group_array, data_type)

    # Render and save plots if path provided
    if save_plots_path:
        render_plots(plot_data, metadata, parameter_anomaly_group_array, save_plots_path)

    # Return enhanced structure
    return {
        'groups': parameter_anomaly_group_array,
        'plots': plot_data,
        'metadata': metadata
    }


def build_time_series_features(parameter_anomaly_group_array: List[Dict]) -> tuple:
    """
    Build feature matrix for time-series data.
    Uses monthly aggregation to derive [slope, max_drop, derivative] per feature.

    Returns:
        feature_matrix: np.array of shape (n_groups, n_features * 3)
        group_ids: List of group IDs
        feature_names: List of feature names (with _slope, _max_drop, _derivative suffixes)
    """
    _print("\n[2/5] Calculating shapValues (slope, max_drop, derivative)...")
    calculate_shap_values(parameter_anomaly_group_array)

    _print("\n[3/5] Building feature matrix for group comparison...")
    feature_matrix, group_ids = build_feature_matrix(parameter_anomaly_group_array)

    # Build feature names
    feature_names = []
    for feature in parameter_anomaly_group_array[0]['featureArray']:
        feature_name = feature['featureName']
        feature_names.extend([
            f"{feature_name}_slope",
            f"{feature_name}_max_drop",
            f"{feature_name}_derivative"
        ])

    return feature_matrix, group_ids, feature_names


def build_cross_sectional_features(parameter_anomaly_group_array: List[Dict]) -> tuple:
    """
    Build feature matrix for cross-sectional data.
    Uses raw values directly - no temporal feature derivation.

    Returns:
        feature_matrix: np.array of shape (n_groups, n_features)
        group_ids: List of group IDs
        feature_names: List of feature names
    """
    _print("\n[2/5] Extracting raw feature values (cross-sectional mode)...")

    feature_names = []
    feature_vectors = []
    group_ids = []

    # Get feature names from first group
    for feature in parameter_anomaly_group_array[0]['featureArray']:
        feature_names.append(feature['featureName'])

    # Build feature vector for each group
    for group in parameter_anomaly_group_array:
        group_features = []
        group_ids.append(group['parameterAnomalyGroupId'])

        for feature in group['featureArray']:
            values = [p['value'] for p in feature['parameterHistoryArray']]

            # Use mean if multiple values, otherwise single value
            if len(values) == 1:
                group_features.append(float(values[0]))
            elif len(values) > 1:
                group_features.append(float(np.mean(values)))
            else:
                group_features.append(0.0)

            # Set shapValues to None for cross-sectional (not applicable)
            feature['shapValues'] = None

        feature_vectors.append(group_features)

    feature_matrix = np.array(feature_vectors)

    _print(f"\n[3/5] Building feature matrix for group comparison...")
    _print(f"  Feature matrix shape: {feature_matrix.shape}")
    _print(f"  Groups: {group_ids}")
    _print(f"  Features: {feature_names}")

    return feature_matrix, group_ids, feature_names


def calculate_shap_values(parameter_anomaly_group_array: List[Dict]) -> None:
    """
    Calculate shapValues [slope, max_drop, derivative] for each feature in each group.

    Uses monthly aggregation approach (like find_bad_cell_car.py) to detect long-term trends:
    - Slope: Linear regression on monthly std/mean (degradation indicator)
    - Max Drop: Largest consecutive decrease in monthly values
    - Derivative: Volatility of monthly changes
    """

    for group in parameter_anomaly_group_array:
        group_id = group['parameterAnomalyGroupId']

        for feature in group['featureArray']:
            feature_name = feature['featureName']
            param_history = feature['parameterHistoryArray']

            if len(param_history) < 2:
                feature['shapValues'] = [0.0, 0.0, 0.0]
                continue

            # Sort by time
            param_history_sorted = sorted(param_history, key=lambda x: x['createdAt'])

            # Convert to DataFrame for monthly aggregation
            df = pd.DataFrame([
                {
                    'time': pd.to_datetime(p['createdAt']),
                    'value': p['value']
                }
                for p in param_history_sorted
            ])

            # Group by month and calculate statistics
            df['month'] = df['time'].dt.to_period('M')
            monthly_stats = df.groupby('month')['value'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()

            if len(monthly_stats) < 2:
                # Not enough months for trend analysis, use raw data
                values = np.array([p['value'] for p in param_history_sorted])
                slope = calculate_slope(values)

                diffs = np.diff(values) if len(values) > 1 else np.array([0])
                max_drop = np.min(diffs) if len(diffs) > 0 else 0.0
                derivative = np.std(diffs) if len(diffs) > 0 else 0.0

                feature['shapValues'] = [float(slope), float(max_drop), float(derivative)]
                _print(f"  {group_id} - {feature_name}: slope={slope:.4f} (raw), max_drop={max_drop:.4f}, deriv={derivative:.4f}")
                continue

            # Calculate slope on monthly STD (key indicator of degradation - increasing variability)
            monthly_std = monthly_stats['std'].fillna(0).values
            std_slope = calculate_slope(monthly_std)

            # Calculate max drop in monthly mean values
            monthly_mean = monthly_stats['mean'].values
            if len(monthly_mean) > 1:
                diffs = np.diff(monthly_mean)
                max_drop = np.min(diffs)  # Most negative
            else:
                max_drop = 0.0

            # Calculate derivative (volatility) from monthly std
            if len(monthly_std) > 1:
                derivative = np.mean(monthly_std)  # Average variability
            else:
                derivative = 0.0

            feature['shapValues'] = [float(std_slope), float(max_drop), float(derivative)]

            _print(f"  {group_id} - {feature_name}: std_slope={std_slope:.4f} (monthly), max_drop={max_drop:.4f}, avg_std={derivative:.4f}")


def build_feature_matrix(parameter_anomaly_group_array: List[Dict]) -> tuple:
    """
    Build feature matrix for group comparison.

    Returns:
        feature_matrix: Array of shape (n_groups, n_features * 3)
        group_ids: List of group IDs
    """

    feature_matrix = []
    group_ids = []

    for group in parameter_anomaly_group_array:
        row = []
        for feature in group['featureArray']:
            # Flatten shapValues into row
            row.extend(feature.get('shapValues', [0.0, 0.0, 0.0]))

        feature_matrix.append(row)
        group_ids.append(group['parameterAnomalyGroupId'])

    feature_matrix = np.array(feature_matrix)
    _print(f"  Feature matrix shape: {feature_matrix.shape}")
    _print(f"  Groups: {group_ids}")

    return feature_matrix, group_ids


def detect_group_outliers(feature_matrix: np.ndarray, group_ids: List[str],
                         parameter_anomaly_group_array: List[Dict],
                         feature_names: List[str],
                         data_type: str = 'time-series') -> tuple:
    """
    Detect which groups are outliers using multi-method approach.

    Methods:
    1. Degradation score analysis (positive slopes = degradation)
    2. Z-score analysis on features
    3. Isolation Forest on full feature matrix
    4. Combined scoring with degradation priority
    5. Feature importance calculation

    Returns:
        Tuple of (outlier_flags_dict, feature_importance_dict)
    """

    n_groups = len(feature_matrix)

    # Method 1: Degradation Score Analysis (TIME-SERIES SPECIFIC)
    # Positive slopes indicate increasing variability = degradation = BAD
    if data_type == 'time-series':
        _print("\n  Method 1: Degradation Score Analysis (positive slopes = degradation)")
        # Extract slopes (every 3rd value starting from index 0)
        n_features = len(parameter_anomaly_group_array[0]['featureArray'])
        slope_indices = [i * 3 for i in range(n_features)]
        slopes = feature_matrix[:, slope_indices]

        # Degradation score: focus on POSITIVE slopes (increasing variability)
        # Positive slope = degradation, negative slope = stable/improving
        # We want to detect groups with the highest positive slopes
        degradation_scores = np.mean(slopes, axis=1)  # Higher = more degradation

        # Normalize degradation scores to [0, 1] range for comparison
        if np.max(degradation_scores) != np.min(degradation_scores):
            degradation_scores_normalized = (degradation_scores - np.min(degradation_scores)) / (np.max(degradation_scores) - np.min(degradation_scores))
        else:
            degradation_scores_normalized = np.zeros_like(degradation_scores)

        for i, group_id in enumerate(group_ids):
            _print(f"    {group_id}: degradation_score={degradation_scores[i]:.4f} (normalized={degradation_scores_normalized[i]:.3f})")

        analysis_values = slopes
    else:
        _print("\n  Method 1: Z-Score Analysis on Raw Feature Values")
        # Use all features directly for cross-sectional
        analysis_values = feature_matrix
        degradation_scores = np.zeros(n_groups)
        degradation_scores_normalized = np.zeros(n_groups)

    # Method 2: Z-score analysis (statistical deviation)
    _print("\n  Method 2: Z-Score Analysis (statistical deviation)")
    value_zscores = np.zeros_like(analysis_values)
    for i in range(analysis_values.shape[1]):
        if np.std(analysis_values[:, i]) > 0:
            value_zscores[:, i] = stats.zscore(analysis_values[:, i])
        else:
            value_zscores[:, i] = 0.0

    # Anomaly score: mean of absolute Z-scores (higher = more anomalous)
    anomaly_scores = np.mean(np.abs(value_zscores), axis=1)

    for i, group_id in enumerate(group_ids):
        _print(f"    {group_id}: anomaly_score={anomaly_scores[i]:.3f}")

    # Method 3: Isolation Forest
    _print("\n  Method 3: Isolation Forest on All Features")

    feature_importance_dict = {}

    if n_groups >= 3:
        # Normalize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=min(0.3, 1/n_groups),  # Expect at most 1 outlier or 30%
            random_state=42,
            n_estimators=100
        )

        iso_predictions = iso_forest.fit_predict(feature_matrix_scaled)
        iso_scores = iso_forest.score_samples(feature_matrix_scaled)

        if_outliers = [group_ids[i] for i in range(n_groups) if iso_predictions[i] == -1]
        _print(f"    Isolation Forest flagged: {if_outliers}")

        # Calculate feature importance from tree structures
        _print("\n  Method 3b: Calculating Feature Importance...")

        try:
            # Extract feature importance from Isolation Forest trees
            importances = np.zeros(feature_matrix_scaled.shape[1])

            for tree in iso_forest.estimators_:
                tree_structure = tree.tree_
                tree_importances = tree_structure.compute_feature_importances(normalize=False)
                importances += tree_importances

            importances /= len(iso_forest.estimators_)

            if importances.sum() > 0:
                importances = importances / importances.sum()

            # Create importance dictionary using provided feature names
            for i, fname in enumerate(feature_names):
                if i < len(importances):
                    feature_importance_dict[fname] = importances[i]

            # Print top 5 most important features
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            _print(f"    Top features for outlier detection:")
            for fname, importance in sorted_features[:min(5, len(sorted_features))]:
                _print(f"      {fname}: {importance:.4f}")

        except Exception as e:
            _print(f"    Warning: Could not calculate feature importance: {e}")

    else:
        _print(f"    Skipping Isolation Forest (need ≥3 groups, have {n_groups})")
        iso_predictions = np.ones(n_groups)

    # Ranking by Anomaly Score (same as v2)
    _print("\n  Anomaly Score Ranking:")
    ranked_indices = np.argsort(anomaly_scores)[::-1]  # Highest anomaly score first

    _print("  Final Ranking (worst to best):")
    for rank, idx in enumerate(ranked_indices):
        group_id = group_ids[idx]
        score = anomaly_scores[idx]
        marker = "OUTLIER" if rank == 0 else ""
        _print(f"      {rank+1}. {group_id}: {score:.3f} {marker}")

    # Determine outliers: Top anomaly score OR flagged by Isolation Forest
    # For time-series: additionally require POSITIVE degradation (increasing variability)
    outlier_flags = {}

    if data_type == 'time-series':
        # Find the top-ranked group with positive degradation
        top_degrading_group = None
        for idx in ranked_indices:
            if degradation_scores[idx] > 0:
                top_degrading_group = group_ids[idx]
                break

        for i, group_id in enumerate(group_ids):
            is_if_outlier = (iso_predictions[i] == -1)
            has_positive_degradation = degradation_scores[i] > 0

            # Flag if: (top-ranked with positive degradation) OR (IF outlier with positive degradation)
            is_top_degrading = (group_id == top_degrading_group)
            outlier_flags[group_id] = has_positive_degradation and (is_top_degrading or is_if_outlier)
    else:
        # For cross-sectional data, use original v2 logic
        for i, group_id in enumerate(group_ids):
            is_top_anomaly = (i == ranked_indices[0])
            is_if_outlier = (iso_predictions[i] == -1)
            outlier_flags[group_id] = is_top_anomaly or is_if_outlier

    # Store scores in groups for visualization
    for i, group in enumerate(parameter_anomaly_group_array):
        group['zScore'] = float(anomaly_scores[i])
        if data_type == 'time-series':
            group['degradationScore'] = float(degradation_scores[i])

    # Update group-level isOutlier flags
    for group in parameter_anomaly_group_array:
        group['isOutlier'] = outlier_flags.get(group['parameterAnomalyGroupId'], False)

    return outlier_flags, feature_importance_dict


def detect_feature_and_point_outliers(parameter_anomaly_group_array: List[Dict],
                                      group_outlier_flags: Dict[str, bool],
                                      feature_importance_dict: Dict[str, float],
                                      data_type: str = 'time-series') -> None:
    """
    Detect feature-level outliers within groups.

    Feature-level: Use Isolation Forest feature importance to flag features

    Args:
        feature_importance_dict: Dict mapping feature names to importance scores
        data_type: 'time-series' or 'cross-sectional'
    """

    # Calculate importance threshold (flag features above 70th percentile)
    if feature_importance_dict:
        importance_values = list(feature_importance_dict.values())
        if importance_values:
            importance_threshold = np.percentile(importance_values, 70)
        else:
            importance_threshold = 0
    else:
        importance_threshold = 0

    # Detect feature and point outliers
    for group in parameter_anomaly_group_array:
        group_id = group['parameterAnomalyGroupId']
        is_outlier_group = group.get('isOutlier', False)

        for feature in group['featureArray']:
            feature_name = feature['featureName']

            # Feature-level detection using importance scores
            if is_outlier_group and feature_importance_dict:
                if data_type == 'time-series':
                    # Get importance scores for this feature's shapValues
                    slope_importance = feature_importance_dict.get(f"{feature_name}_slope", 0)
                    drop_importance = feature_importance_dict.get(f"{feature_name}_max_drop", 0)
                    deriv_importance = feature_importance_dict.get(f"{feature_name}_derivative", 0)
                    total_importance = slope_importance + drop_importance + deriv_importance
                else:
                    # Cross-sectional: direct feature importance
                    total_importance = feature_importance_dict.get(feature_name, 0)

                # Flag if importance is above threshold
                feature['isOutlier'] = total_importance > importance_threshold
            else:
                feature['isOutlier'] = False

            # No point-level detection - set all points to non-outlier
            for point in feature['parameterHistoryArray']:
                point['isOutlier'] = False


def generate_plot_data(parameter_anomaly_group_array: List[Dict],
                       feature_importance_dict: Dict[str, float],
                       group_outlier_flags: Dict[str, bool],
                       data_type: str = 'time-series',
                       feature_matrix: np.ndarray = None,
                       feature_names: List[str] = None) -> Dict[str, Any]:
    """
    Generate plot-ready data for visualization panels.
    Unavailable plots for a given dataType are set to None.

    Returns dict matching TypeScript PlotData interface.
    """

    # Color palette
    OUTLIER_COLOR = '#FF4444'
    NORMAL_COLOR = '#4682B4'
    PALETTE = ['#4682B4', '#FF8C00', '#32CD32', '#FF69B4', '#FFD700', '#9370DB', '#00CED1', '#FF6347']

    # Initialize all plots as None (unavailable by default)
    plots = {
        'volume': None,      # Time-series only
        'violin': None,      # Both
        'std': None,         # Time-series only
        'heatmap': None,     # Both (different implementation)
        'boxPlot': None,     # Both
        'shapValues': None,  # Time-series only
        'scatter': None,     # Both (different implementation)
        'featureImportance': None,  # Both
        'distribution': None,       # Cross-sectional only
    }

    # ========== SHARED PLOTS (both data types) ==========

    # Violin plot (feature contribution)
    all_feature_names = set()
    for group in parameter_anomaly_group_array:
        for feature in group['featureArray']:
            all_feature_names.add(feature['featureName'])

    plots['violin'] = {'features': []}
    for idx, fname in enumerate(sorted(all_feature_names)):
        values = []
        outlier_flags = []

        for group in parameter_anomaly_group_array:
            is_group_outlier = group.get('isOutlier', False)
            for feature in group['featureArray']:
                if feature['featureName'] == fname:
                    if data_type == 'time-series':
                        # Use slope as contribution metric
                        shapvals = feature.get('shapValues', [0, 0, 0])
                        if shapvals:
                            values.append(abs(shapvals[0]))
                        else:
                            values.append(0)
                    else:
                        # Cross-sectional: use raw value
                        raw_values = [p['value'] for p in feature.get('parameterHistoryArray', [])]
                        values.append(float(np.mean(raw_values)) if raw_values else 0)
                    outlier_flags.append(is_group_outlier)

        plots['violin']['features'].append({
            'featureName': fname,
            'color': PALETTE[idx % len(PALETTE)],
            'values': values,
            'outlierFlags': outlier_flags
        })

    # Box plot (distribution per group)
    plots['boxPlot'] = {'boxes': []}
    for idx, group in enumerate(parameter_anomaly_group_array):
        group_id = group['parameterAnomalyGroupId']
        is_outlier = group.get('isOutlier', False)

        if group['featureArray']:
            feature = group['featureArray'][0]
            values = [p['value'] for p in feature.get('parameterHistoryArray', [])]

            if len(values) > 0:
                q1 = float(np.percentile(values, 25))
                median = float(np.percentile(values, 50))
                q3 = float(np.percentile(values, 75))
                iqr = q3 - q1
                lower_whisker = float(max(min(values), q1 - 1.5 * iqr))
                upper_whisker = float(min(max(values), q3 + 1.5 * iqr))
                box_outliers = [float(v) for v in values if v < lower_whisker or v > upper_whisker]

                plots['boxPlot']['boxes'].append({
                    'groupId': group_id,
                    'groupName': group_id,
                    'color': '#FFA07A' if is_outlier else '#87CEEB',
                    'min': lower_whisker,
                    'q1': q1,
                    'median': median,
                    'q3': q3,
                    'max': upper_whisker,
                    'mean': float(np.mean(values)),
                    'outliers': box_outliers
                })

    # Feature importance (shared, but aggregation differs by type)
    plots['featureImportance'] = {'features': []}
    if feature_importance_dict:
        if data_type == 'time-series':
            # Aggregate by feature (sum importance across slope/max_drop/derivative)
            feature_importance_agg = {}
            for key, importance in feature_importance_dict.items():
                parts = key.rsplit('_', 1)
                if len(parts) == 2:
                    fname = parts[0]
                    feature_importance_agg[fname] = feature_importance_agg.get(fname, 0) + importance
        else:
            # Cross-sectional: direct feature importance
            feature_importance_agg = dict(feature_importance_dict)

        # Count affected groups per feature
        affected_groups_count = {}
        for group in parameter_anomaly_group_array:
            if group.get('isOutlier', False):
                for feature in group['featureArray']:
                    if feature.get('isOutlier', False):
                        fname = feature['featureName']
                        affected_groups_count[fname] = affected_groups_count.get(fname, 0) + 1

        sorted_features = sorted(feature_importance_agg.items(), key=lambda x: x[1], reverse=True)

        for idx, (fname, importance) in enumerate(sorted_features):
            plots['featureImportance']['features'].append({
                'featureName': fname,
                'importance': float(importance),
                'color': PALETTE[idx % len(PALETTE)],
                'affectedGroups': affected_groups_count.get(fname, 0)
            })

    # ========== TIME-SERIES ONLY PLOTS ==========
    if data_type == 'time-series':
        # Volume (time series) - include all features
        plots['volume'] = {'series': []}
        for idx, group in enumerate(parameter_anomaly_group_array):
            group_id = group['parameterAnomalyGroupId']
            is_outlier = group.get('isOutlier', False)

            if group['featureArray']:
                for feature in group['featureArray']:
                    feature_name = feature.get('featureName', 'unknown')
                    points = []

                    for point in feature.get('parameterHistoryArray', []):
                        timestamp_iso = pd.to_datetime(point['createdAt']).isoformat()
                        value = point['value']
                        point_is_outlier = point.get('isOutlier', False)

                        points.append({
                            'timestamp': timestamp_iso,
                            'value': float(value),
                            'isOutlier': point_is_outlier
                        })

                    plots['volume']['series'].append({
                        'groupId': group_id,
                        'isOutlier': is_outlier,
                        'featureName': feature_name,
                        'points': points
                    })

        # Monthly STD trends - include all features
        plots['std'] = {'series': []}
        for idx, group in enumerate(parameter_anomaly_group_array):
            group_id = group['parameterAnomalyGroupId']
            is_outlier = group.get('isOutlier', False)

            if group['featureArray']:
                for feature in group['featureArray']:
                    feature_name = feature.get('featureName', 'unknown')
                    param_history = feature.get('parameterHistoryArray', [])

                    if len(param_history) > 0:
                        df = pd.DataFrame([
                            {'time': pd.to_datetime(p['createdAt']), 'value': p['value']}
                            for p in param_history
                        ])

                        df['month'] = df['time'].dt.to_period('M')
                        monthly_stats = df.groupby('month')['value'].std().reset_index()
                        monthly_stats['timestamp'] = monthly_stats['month'].dt.to_timestamp()

                        data_points = [
                            [int(row['timestamp'].timestamp() * 1000), row['value']]
                            for _, row in monthly_stats.iterrows()
                        ]

                        plots['std']['series'].append({
                            'groupId': group_id,
                            'groupName': group_id,
                            'featureName': feature_name,
                            'color': OUTLIER_COLOR if is_outlier else PALETTE[idx % len(PALETTE)],
                            'lineWidth': 2 if is_outlier else 1,
                            'dataPoints': data_points
                        })

        # Heatmap (hourly patterns)
        hours = list(range(24))
        heatmap_data = []
        column_labels = []

        for group in parameter_anomaly_group_array:
            column_labels.append(group['parameterAnomalyGroupId'])

            if group['featureArray']:
                feature = group['featureArray'][0]
                param_history = feature.get('parameterHistoryArray', [])

                df = pd.DataFrame([
                    {'time': pd.to_datetime(p['createdAt']), 'value': p['value']}
                    for p in param_history
                ])

                if len(df) > 0:
                    df['hour'] = df['time'].dt.hour
                    hourly_avg = df.groupby('hour')['value'].mean()
                    hour_values = [hourly_avg.get(h, 0) for h in hours]
                else:
                    hour_values = [0] * 24
            else:
                hour_values = [0] * 24

            heatmap_data.append(hour_values)

        heatmap_data_transposed = list(map(list, zip(*heatmap_data)))
        all_values = [v for row in heatmap_data_transposed for v in row]

        plots['heatmap'] = {
            'data': heatmap_data_transposed,
            'rowLabels': [str(h) for h in hours],
            'columnLabels': column_labels,
            'colorScale': {
                'min': float(min(all_values)) if all_values else 0,
                'max': float(max(all_values)) if all_values else 1,
                'colors': ['#FFFF00', '#FF0000']
            }
        }

        # ShapValues bar chart
        plots['shapValues'] = {'groups': []}
        for idx, group in enumerate(parameter_anomaly_group_array):
            group_id = group['parameterAnomalyGroupId']
            is_outlier = group.get('isOutlier', False)

            features = []
            for feature in group['featureArray']:
                shapvals = feature.get('shapValues', [0, 0, 0])
                if shapvals:
                    features.append({
                        'featureName': feature.get('featureName', 'unknown'),
                        'slope': float(shapvals[0]),
                        'maxDrop': float(shapvals[1]),
                        'derivative': float(shapvals[2])
                    })

            plots['shapValues']['groups'].append({
                'groupId': group_id,
                'isOutlier': is_outlier,
                'features': features
            })

        # Scatter plot (degradation vs z-score)
        plots['scatter'] = {'points': []}
        for idx, group in enumerate(parameter_anomaly_group_array):
            group_id = group['parameterAnomalyGroupId']
            is_outlier = group.get('isOutlier', False)

            slopes = [f.get('shapValues', [0, 0, 0])[0] for f in group['featureArray'] if f.get('shapValues')]
            degradation_score = float(np.mean([abs(s) for s in slopes])) if slopes else 0.0
            z_score = group.get('zScore', 0.0)

            plots['scatter']['points'].append({
                'groupId': group_id,
                'groupName': group_id,
                'x': degradation_score,
                'y': z_score,
                'color': OUTLIER_COLOR if is_outlier else NORMAL_COLOR,
                'size': 8 if is_outlier else 5,
                'isOutlier': is_outlier
            })

    # ========== CROSS-SECTIONAL ONLY PLOTS ==========
    if data_type == 'cross-sectional':
        # Heatmap (features × groups)
        heatmap_data = []
        row_labels = list(sorted(all_feature_names))
        column_labels = [g['parameterAnomalyGroupId'] for g in parameter_anomaly_group_array]

        for fname in row_labels:
            row_values = []
            for group in parameter_anomaly_group_array:
                for feature in group['featureArray']:
                    if feature['featureName'] == fname:
                        values = [p['value'] for p in feature.get('parameterHistoryArray', [])]
                        row_values.append(float(np.mean(values)) if values else 0)
                        break
                else:
                    row_values.append(0)
            heatmap_data.append(row_values)

        all_values = [v for row in heatmap_data for v in row]
        plots['heatmap'] = {
            'data': heatmap_data,
            'rowLabels': row_labels,
            'columnLabels': column_labels,
            'colorScale': {
                'min': float(min(all_values)) if all_values else 0,
                'max': float(max(all_values)) if all_values else 1,
                'colors': ['#FFFF00', '#FF0000']
            }
        }

        # Scatter plot (PCA 2D projection)
        if feature_matrix is not None and len(feature_matrix) >= 2:
            # Scale and apply PCA
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            pca = PCA(n_components=min(2, feature_matrix_scaled.shape[1]))
            pca_result = pca.fit_transform(feature_matrix_scaled)

            plots['scatter'] = {
                'points': [],
                'axisLabels': {
                    'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                    'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)' if len(pca.explained_variance_ratio_) > 1 else 'PC2'
                }
            }

            for idx, group in enumerate(parameter_anomaly_group_array):
                group_id = group['parameterAnomalyGroupId']
                is_outlier = group.get('isOutlier', False)

                plots['scatter']['points'].append({
                    'groupId': group_id,
                    'groupName': group_id,
                    'x': float(pca_result[idx, 0]),
                    'y': float(pca_result[idx, 1]) if pca_result.shape[1] > 1 else 0,
                    'color': OUTLIER_COLOR if is_outlier else NORMAL_COLOR,
                    'size': 8 if is_outlier else 5,
                    'isOutlier': is_outlier
                })

        # Anomaly score distribution
        if feature_matrix is not None:
            scaler = StandardScaler()
            feature_matrix_scaled = scaler.fit_transform(feature_matrix)

            iso_forest = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
            iso_forest.fit(feature_matrix_scaled)
            anomaly_scores = iso_forest.decision_function(feature_matrix_scaled)

            plots['distribution'] = {
                'scores': [float(s) for s in anomaly_scores],
                'groupIds': [g['parameterAnomalyGroupId'] for g in parameter_anomaly_group_array],
                'isOutlier': [g.get('isOutlier', False) for g in parameter_anomaly_group_array],
                'threshold': float(np.percentile(anomaly_scores, 10))  # Approximate threshold
            }

    return plots


def extract_metadata(parameter_anomaly_group_array: List[Dict], data_type: str = 'time-series') -> Dict[str, Any]:
    """
    Extract metadata about the analysis for frontend display.
    """

    outlier_groups = [g for g in parameter_anomaly_group_array if g.get('isOutlier', False)]

    # Extract date range from first group's first feature (only for time-series)
    timestamps = []
    if data_type == 'time-series':
        for group in parameter_anomaly_group_array:
            for feature in group.get('featureArray', []):
                for point in feature.get('parameterHistoryArray', []):
                    if 'createdAt' in point:
                        try:
                            timestamps.append(pd.to_datetime(point['createdAt']))
                        except:
                            pass

    # Get unique feature names
    feature_names = set()
    for group in parameter_anomaly_group_array:
        for feature in group['featureArray']:
            feature_names.add(feature['featureName'])

    metadata = {
        'dataType': data_type,
        'totalGroups': len(parameter_anomaly_group_array),
        'anomalousGroups': len(outlier_groups),
        'featuresAnalyzed': sorted(list(feature_names))
    }

    # Add date range only for time-series
    if data_type == 'time-series' and timestamps:
        metadata['dateRange'] = {
            'start': min(timestamps).isoformat(),
            'end': max(timestamps).isoformat()
        }
    else:
        metadata['dateRange'] = None

    return metadata


def render_plots(plots: Dict[str, Any], metadata: Dict[str, Any], groups: List[Dict], save_path: str):
    """
    Render matplotlib visualizations and save each as a separate image file.

    Args:
        plots: Plot data from generate_plot_data()
        metadata: Metadata from extract_metadata()
        groups: The analyzed groups with outlier flags
        save_path: Base path for saving PNG files (directory will be used)
    """
    # Color constants
    OUTLIER_COLOR = '#FF4444'
    NORMAL_COLOR = '#4682B4'
    PALETTE = ['#4682B4', '#FF8C00', '#32CD32', '#FF69B4', '#FFD700', '#9370DB', '#00CED1', '#FF6347']

    # Create output directory
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # 1. Time Series Plot - one subplot per feature
    if plots.get('volume') and plots['volume'].get('series'):
        # Group series by feature name
        feature_names = sorted(set(s.get('featureName', 'unknown') for s in plots['volume']['series']))
        n_features = len(feature_names)

        fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features), squeeze=False)
        axes = axes[:, 0]  # Flatten to 1D array

        for feat_idx, feature_name in enumerate(feature_names):
            ax = axes[feat_idx]
            group_idx = 0  # Track group index for coloring

            for series in plots['volume']['series']:
                if series.get('featureName', 'unknown') != feature_name:
                    continue

                group_id = series['groupId']
                is_outlier = series['isOutlier']
                points = series['points']

                if points:
                    times = [pd.to_datetime(p['timestamp']) for p in points]
                    values = [p['value'] for p in points]

                    color = OUTLIER_COLOR if is_outlier else PALETTE[group_idx % len(PALETTE)]
                    linewidth = 2 if is_outlier else 1
                    alpha = 1.0 if is_outlier else 0.6
                    label = f"{group_id} {'(OUTLIER)' if is_outlier else ''}"

                    ax.plot(times, values, color=color, linewidth=linewidth, alpha=alpha, label=label)
                    group_idx += 1

            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'{feature_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        filepath = save_dir / '01_time_series.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(filepath)

    # 2. Box Plot - one subplot per feature
    if plots.get('boxPlot') and plots['boxPlot'].get('boxes'):
        # Get all feature names from groups
        feature_names = []
        for group in groups:
            if group.get('featureArray'):
                for feature in group['featureArray']:
                    fname = feature.get('featureName', 'unknown')
                    if fname not in feature_names:
                        feature_names.append(fname)
        feature_names = sorted(feature_names)
        n_features = len(feature_names) if feature_names else 1

        fig, axes = plt.subplots(n_features, 1, figsize=(10, 5 * n_features), squeeze=False)
        axes = axes[:, 0]

        boxes = plots['boxPlot']['boxes']

        for feat_idx, feature_name in enumerate(feature_names):
            ax = axes[feat_idx]
            positions = list(range(len(boxes)))
            labels = []

            for idx, box in enumerate(boxes):
                group_id = box['groupId']
                labels.append(group_id)

                for group in groups:
                    if group['parameterAnomalyGroupId'] == group_id:
                        if group['featureArray']:
                            for feature in group['featureArray']:
                                if feature.get('featureName', 'unknown') == feature_name:
                                    values = [p['value'] for p in feature.get('parameterHistoryArray', [])]
                                    if values:
                                        bp = ax.boxplot([values], positions=[idx], widths=0.6, patch_artist=True)
                                        color = OUTLIER_COLOR if group.get('isOutlier') else NORMAL_COLOR
                                        bp['boxes'][0].set_facecolor(color)
                                        bp['boxes'][0].set_alpha(0.6)
                                    break
                        break

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'{feature_name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filepath = save_dir / '02_box_plot.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(filepath)

    # 3. Monthly STD Trends - one subplot per feature
    if plots.get('std') and plots['std'].get('series'):
        # Group series by feature name
        feature_names = sorted(set(s.get('featureName', 'unknown') for s in plots['std']['series']))
        n_features = len(feature_names)

        fig, axes = plt.subplots(n_features, 1, figsize=(12, 5 * n_features), squeeze=False)
        axes = axes[:, 0]

        for feat_idx, feature_name in enumerate(feature_names):
            ax = axes[feat_idx]
            group_idx = 0

            for series in plots['std']['series']:
                if series.get('featureName', 'unknown') != feature_name:
                    continue

                group_id = series['groupId']
                data_points = series['dataPoints']
                color = series['color']
                linewidth = series['lineWidth']

                if data_points:
                    times = [pd.to_datetime(p[0], unit='ms') for p in data_points]
                    values = [p[1] for p in data_points]

                    ax.plot(times, values, label=group_id, color=color, linewidth=linewidth, marker='o', markersize=4)
                    group_idx += 1

            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Standard Deviation', fontsize=12)
            ax.set_title(f'{feature_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        filepath = save_dir / '03_monthly_std.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(filepath)

    # 4. Heatmap
    if plots.get('heatmap') and plots['heatmap'].get('data'):
        fig, ax = plt.subplots(figsize=(10, 8))
        heatmap_data = np.array(plots['heatmap']['data'])
        row_labels = plots['heatmap']['rowLabels']
        col_labels = plots['heatmap']['columnLabels']

        if len(row_labels) > 10:
            hour_indices = [0, 6, 12, 18, 23] if len(row_labels) == 24 else list(range(0, len(row_labels), max(1, len(row_labels)//5)))
            heatmap_subset = heatmap_data[hour_indices, :]
            row_labels_subset = [row_labels[i] for i in hour_indices]
        else:
            heatmap_subset = heatmap_data
            row_labels_subset = row_labels

        im = ax.imshow(heatmap_subset, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45)
        ax.set_yticks(range(len(row_labels_subset)))
        ax.set_yticklabels([f'{h}:00' if h.isdigit() else h for h in row_labels_subset])
        ax.set_xlabel('Group', fontsize=12)
        ax.set_ylabel('Hour', fontsize=12)
        ax.set_title('Hourly Patterns Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        filepath = save_dir / '04_heatmap.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(filepath)

    # 5. Feature Importance
    if plots.get('featureImportance') and plots['featureImportance'].get('features'):
        fig, ax = plt.subplots(figsize=(10, 6))
        features = plots['featureImportance']['features'][:10]  # Top 10
        names = [f['featureName'] for f in features]
        importances = [f['importance'] for f in features]
        colors = [f['color'] for f in features]

        ax.barh(range(len(names)), importances, color=colors, alpha=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        filepath = save_dir / '05_feature_importance.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(filepath)

    # 7. Slope Values per Group (Degradation Indicator)
    if plots.get('shapValues') and plots['shapValues'].get('groups'):
        fig, ax = plt.subplots(figsize=(10, 6))

        groups_data = plots['shapValues']['groups']
        group_ids = [g['groupId'] for g in groups_data]

        # Calculate average slope per group (across all features)
        avg_slopes = []
        for g in groups_data:
            slopes = [f['slope'] for f in g['features']]
            avg_slopes.append(np.mean(slopes) if slopes else 0)

        # Color bars: red for positive (degradation), blue for negative
        colors = [OUTLIER_COLOR if s > 0 else NORMAL_COLOR for s in avg_slopes]

        y_pos = range(len(group_ids))
        ax.barh(y_pos, avg_slopes, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(group_ids)
        ax.set_xlabel('Slope (positive = increasing variability)', fontsize=12)
        ax.set_title('Degradation Score by Group', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        filepath = save_dir / '07_slope_by_group.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        saved_files.append(filepath)

    # 8. ShapValues per Feature per Group
    if plots.get('shapValues') and plots['shapValues'].get('groups'):
        groups_data = plots['shapValues']['groups']
        n_groups = len(groups_data)

        if n_groups > 0 and groups_data[0]['features']:
            feature_names = [f['featureName'] for f in groups_data[0]['features']]
            n_features = len(feature_names)

            fig, axes = plt.subplots(1, n_features, figsize=(5 * n_features, 6))
            if n_features == 1:
                axes = [axes]

            for feat_idx, feature_name in enumerate(feature_names):
                ax = axes[feat_idx]

                group_ids = []
                slopes = []
                max_drops = []
                derivatives = []
                is_outliers = []

                for g in groups_data:
                    group_ids.append(g['groupId'])
                    is_outliers.append(g['isOutlier'])
                    for f in g['features']:
                        if f['featureName'] == feature_name:
                            slopes.append(f['slope'])
                            max_drops.append(f['maxDrop'])
                            derivatives.append(f['derivative'])
                            break

                x = np.arange(len(group_ids))
                width = 0.25

                ax.bar(x - width, slopes, width, label='Slope', color='#4682B4')
                ax.bar(x, max_drops, width, label='Max Drop', color='#FF8C00')
                ax.bar(x + width, derivatives, width, label='Derivative', color='#32CD32')

                # Highlight outlier groups
                for i, is_outlier in enumerate(is_outliers):
                    if is_outlier:
                        ax.axvspan(i - 0.4, i + 0.4, alpha=0.2, color='red')

                ax.set_xlabel('Group')
                ax.set_ylabel('Value')
                ax.set_title(f'{feature_name}')
                ax.set_xticks(x)
                ax.set_xticklabels(group_ids, rotation=45)
                ax.legend(loc='best', fontsize=8)
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.grid(True, alpha=0.3, axis='y')

            plt.suptitle('ShapValues by Feature', fontsize=14, fontweight='bold')
            plt.tight_layout()
            filepath = save_dir / '08_shapvalues_by_feature.png'
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            saved_files.append(filepath)

    # 6. Summary
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    outlier_groups = [g for g in groups if g.get('isOutlier', False)]
    normal_groups = [g for g in groups if not g.get('isOutlier', False)]

    summary_text = f"""
ANOMALY DETECTION SUMMARY

Data Type: {metadata.get('dataType', 'N/A')}
Total Groups: {metadata.get('totalGroups', len(groups))}
Anomalous: {len(outlier_groups)} ({len(outlier_groups)/len(groups)*100:.1f}%)
Normal: {len(normal_groups)}

Features Analyzed:
  {', '.join(metadata.get('featuresAnalyzed', [])[:5])}

OUTLIERS DETECTED:
"""

    if outlier_groups:
        for og in outlier_groups:
            gid = og['parameterAnomalyGroupId']
            outlier_features = [f['featureName'] for f in og.get('featureArray', []) if f.get('isOutlier', False)]
            summary_text += f"\n  {gid}"
            if outlier_features:
                summary_text += f": {', '.join(outlier_features[:3])}"
    else:
        summary_text += "\n  None detected"

    if metadata.get('dateRange'):
        dr = metadata['dateRange']
        summary_text += f"\n\nDate Range:\n  {dr['start'][:10]} to {dr['end'][:10]}"

    ax.text(0.05, 0.95, summary_text, fontsize=12, family='monospace',
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             transform=ax.transAxes)
    plt.tight_layout()
    filepath = save_dir / '06_summary.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    saved_files.append(filepath)

    _print(f"\n  Diagrams saved to: {save_dir}/")
    for f in saved_files:
        _print(f"    - {f.name}")
