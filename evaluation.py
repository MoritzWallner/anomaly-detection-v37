#!/usr/bin/env python3
"""
Evaluation / Ablation Pipeline for Anomaly Detection

Runs the anomaly detector across all config combinations (aggregation window,
detection method, contamination, group subsets) and produces a results CSV.

Optimized: feature extraction (the slow part) runs once per unique
(dataset, subset, window) combination. Only the fast detection logic
is varied across method/contamination configs.

Usage:
    python evaluation.py
"""

import os
os.environ['SUPPRESS_OUTPUT'] = 'true'  # Must be set before importing anomaly_detector

import sys
import copy
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from anomaly_detector import (
    detect_anomalies, preprocess_enum_values,
    build_time_series_features, build_cross_sectional_features,
    detect_group_outliers, _resolve_config,
)
from traffic_transform import transform as transform_traffic
from vehicles_transform import transform as transform_vehicles
from customers_transform import transform as transform_customers


# --- Dataset definitions ---

DATASETS = {
    'vehicles': {
        'transform': transform_vehicles,
        'ground_truth': 'ID1',
        'data_type': 'time-series',
    },
    'traffic': {
        'transform': transform_traffic,
        'ground_truth': 'Junction1',
        'data_type': 'time-series',
    },
    'customers': {
        'transform': transform_customers,
        'ground_truth': 'CUST_010',
        'data_type': 'cross-sectional',
    },
}

# --- Config axes ---

AGGREGATION_WINDOWS = ['W', 'M', 'Q']
DETECTION_METHODS = ['zscore', 'iforest', 'ensemble']
CONTAMINATIONS = [0.1, 0.5, 'adaptive']

# Vehicles group subsets: always include ID1, pick N others
VEHICLES_OTHER_GROUPS = ['CUP1', 'CUP2', 'CUP3', 'CUP4', 'CUP5', 'ID2']
VEHICLES_SUBSETS = [
    ['ID1'] + VEHICLES_OTHER_GROUPS[:2],   # ID1 + 2 others (3 total)
    ['ID1'] + VEHICLES_OTHER_GROUPS[:4],   # ID1 + 4 others (5 total)
    ['ID1'] + VEHICLES_OTHER_GROUPS,       # ID1 + all 6 others (7 total)
]


def subset_groups(groups, group_ids_to_keep):
    """Filter groups list to the specified subset."""
    keep_set = set(group_ids_to_keep)
    return [g for g in groups if g['parameterAnomalyGroupId'] in keep_set]


def extract_results(groups, ground_truth_id):
    """Extract evaluation metrics from processed groups."""
    flagged_outliers = [g['parameterAnomalyGroupId'] for g in groups if g.get('isOutlier', False)]

    sorted_by_zscore = sorted(groups, key=lambda g: g.get('zScore', 0), reverse=True)
    gt_rank = next(
        (i + 1 for i, g in enumerate(sorted_by_zscore)
         if g['parameterAnomalyGroupId'] == ground_truth_id),
        None
    )

    return {
        'flagged_outliers': flagged_outliers,
        'gt_zscore_rank': gt_rank,
        'gt_detected': ground_truth_id in flagged_outliers,
        'n_groups': len(groups),
    }


def run_evaluation():
    """Run all evaluation configs and collect results."""
    total_start = time.time()

    # Phase 1: Load all datasets once
    print("Loading datasets...")
    raw_data = {}
    for name, cfg in DATASETS.items():
        raw_data[name] = cfg['transform']()
        n_groups = len(raw_data[name]['groups'])
        print(f"  {name}: {n_groups} groups")

    # Phase 2: Pre-parse timestamps once for all time-series data
    print("Pre-parsing timestamps...")
    for name, data in raw_data.items():
        if data.get('dataType') == 'time-series':
            for group in data['groups']:
                for feature in group['featureArray']:
                    for point in feature['parameterHistoryArray']:
                        point['_parsed_dt'] = pd.to_datetime(point['createdAt'])

    # Phase 3: Build unique (dataset, subset) combinations
    print("Building group subsets...")
    subset_configs = []
    for dataset_name, dataset_cfg in DATASETS.items():
        if dataset_name == 'vehicles':
            for subset_ids in VEHICLES_SUBSETS:
                subset_configs.append((dataset_name, subset_ids))
        else:
            all_ids = [g['parameterAnomalyGroupId'] for g in raw_data[dataset_name]['groups']]
            subset_configs.append((dataset_name, all_ids))

    # Phase 4: For each (dataset, subset, window), extract features ONCE,
    # then run detection with each (method, contamination) — the fast part.
    results_rows = []
    run_count = 0

    # Count total runs for progress display
    total_runs = 0
    for dataset_name, subset_ids in subset_configs:
        is_ts = DATASETS[dataset_name]['data_type'] == 'time-series'
        windows = AGGREGATION_WINDOWS if is_ts else [None]
        for _ in windows:
            for method in DETECTION_METHODS:
                contams = ['adaptive'] if method == 'zscore' else CONTAMINATIONS
                total_runs += len(contams)

    print(f"\nTotal evaluation runs: {total_runs}")
    print(f"{'='*80}\n")

    for dataset_name, subset_ids in subset_configs:
        data_type = DATASETS[dataset_name]['data_type']
        ground_truth = DATASETS[dataset_name]['ground_truth']
        is_ts = data_type == 'time-series'
        windows = AGGREGATION_WINDOWS if is_ts else [None]

        # Get the subset of groups (shared reference — we'll deep-copy per window)
        full_groups = raw_data[dataset_name]['groups']
        subset_groups_list = subset_groups(full_groups, subset_ids)

        for window in windows:
            # Deep-copy groups for this (subset, window) — feature extraction modifies in-place
            groups_copy = copy.deepcopy(subset_groups_list)

            # Preprocess enums
            preprocess_enum_values(groups_copy)

            # Extract features (THE SLOW PART — runs once per subset+window)
            config = _resolve_config({'aggregation_window': window} if window else {})
            t0 = time.time()

            if is_ts:
                feature_matrix, group_ids, feature_names = build_time_series_features(groups_copy, config=config)
            else:
                feature_matrix, group_ids, feature_names = build_cross_sectional_features(groups_copy)

            feat_time = time.time() - t0

            subset_label = '+'.join(sorted(subset_ids))
            print(f"  Features extracted: {dataset_name} | {subset_label} | "
                  f"window={window or 'N/A'} ({feat_time:.1f}s)")

            # Now run detection with each (method, contamination) — FAST
            for method in DETECTION_METHODS:
                contams = ['adaptive'] if method == 'zscore' else CONTAMINATIONS

                for contamination in contams:
                    run_count += 1
                    det_config = _resolve_config({
                        'aggregation_window': window or 'M',
                        'detection_method': method,
                        'contamination': contamination,
                    })

                    # Deep-copy groups for detection (it modifies groups in-place with isOutlier, zScore)
                    det_groups = copy.deepcopy(groups_copy)

                    t1 = time.time()
                    outlier_flags, feat_importance = detect_group_outliers(
                        feature_matrix, group_ids, det_groups, feature_names, data_type, config=det_config
                    )
                    det_time = time.time() - t1

                    result = extract_results(det_groups, ground_truth)

                    row = {
                        'dataset': dataset_name,
                        'n_groups': result['n_groups'],
                        'group_subset': subset_label,
                        'aggregation_window': window or 'N/A',
                        'detection_method': method,
                        'contamination': str(contamination),
                        'flagged_outliers': ', '.join(result['flagged_outliers']) or 'None',
                        'gt_zscore_rank': result['gt_zscore_rank'],
                        'gt_detected': result['gt_detected'],
                    }
                    results_rows.append(row)

                    status = "OK" if result['gt_detected'] else "MISS"
                    print(f"    [{run_count}/{total_runs}] {method:8s} | "
                          f"contam={str(contamination):8s} | {status} ({det_time:.2f}s)")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.1f}s")

    return results_rows


def format_results(results_rows):
    """Format and save results."""
    df = pd.DataFrame(results_rows)

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {len(df)}")
    print(f"Ground truth detected: {df['gt_detected'].sum()}/{len(df)} "
          f"({100*df['gt_detected'].mean():.1f}%)")

    # Per-dataset
    print(f"\nBy dataset:")
    for dataset in df['dataset'].unique():
        sub = df[df['dataset'] == dataset]
        print(f"  {dataset}: {sub['gt_detected'].sum()}/{len(sub)} detected")

    # Per-method
    print(f"\nBy detection method:")
    for method in df['detection_method'].unique():
        sub = df[df['detection_method'] == method]
        print(f"  {method}: {sub['gt_detected'].sum()}/{len(sub)} detected")

    # Per-contamination
    print(f"\nBy contamination:")
    for contam in df['contamination'].unique():
        sub = df[df['contamination'] == contam]
        print(f"  {contam}: {sub['gt_detected'].sum()}/{len(sub)} detected")

    # Per-window (time-series only)
    print(f"\nBy aggregation window:")
    for window in df['aggregation_window'].unique():
        sub = df[df['aggregation_window'] == window]
        print(f"  {window}: {sub['gt_detected'].sum()}/{len(sub)} detected")

    # Save CSV
    output_path = Path(__file__).parent / "output" / "evaluation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print full table
    print(f"\n{'='*60}")
    print("FULL RESULTS TABLE")
    print(f"{'='*60}")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    results = run_evaluation()
    format_results(results)
