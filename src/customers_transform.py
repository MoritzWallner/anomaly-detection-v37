"""
Transform customers.csv into the format expected by detect_anomalies().

This is CROSS-SECTIONAL data (no time dimension).

Input format (customers.csv):
    customer_id,avg_transaction,monthly_logins,support_tickets,account_age_days
    CUST_001,48.97,11.58,2.65,521.84

Output format:
    {
        "dataType": "cross-sectional",
        "groups": [
            {
                "parameterAnomalyGroupId": "CUST_001",
                "featureArray": [
                    {"featureName": "avg_transaction", "parameterHistoryArray": [{"value": 48.97}]},
                    {"featureName": "monthly_logins", "parameterHistoryArray": [{"value": 11.58}]}
                ]
            }
        ]
    }
"""

import pandas as pd
import uuid
import json
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration file with min/max constraints.

    Args:
        config_path: Path to config JSON. If None, uses default location.

    Returns:
        Config dict with 'constraints' key mapping feature names to {min, max}
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "customers.json"

    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"constraints": {}}


def transform(csv_path: str = None, config_path: str = None) -> Dict[str, Any]:
    """
    Transform customers.csv to anomaly detector input format.

    Args:
        csv_path: Path to customers.csv. If None, uses default location.
        config_path: Path to config JSON with min/max constraints. If None, uses default.

    Returns:
        Dict in the format expected by detect_anomalies()
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "datasets" / "customers.csv"

    # Load config with min/max constraints and units
    config = load_config(config_path)
    constraints = config.get("constraints", {})
    units = config.get("units", {})

    df = pd.read_csv(csv_path)

    # Feature columns (exclude ID column)
    feature_cols = [c for c in df.columns if c != 'customer_id']

    groups = []
    filtered_counts = {}

    for _, row in df.iterrows():
        customer_id = row['customer_id']
        feature_array = []

        for col in feature_cols:
            value = row[col]

            # Skip NaN values
            if pd.notna(value):
                # Apply min/max filtering if constraints exist
                if col in constraints:
                    min_val = constraints[col].get('min', float('-inf'))
                    max_val = constraints[col].get('max', float('inf'))
                    if not (min_val <= value <= max_val):
                        filtered_counts[col] = filtered_counts.get(col, 0) + 1
                        continue

                # Cross-sectional: single value per feature (no time dimension)
                feature_array.append({
                    "featureName": col,
                    "parameterHistoryArray": [{
                        "parameterHistoryId": str(uuid.uuid4()),
                        "type": "number",
                        "value": float(value)
                    }]
                })

        groups.append({
            "parameterAnomalyGroupId": customer_id,
            "featureArray": feature_array
        })

    # Log filtered values
    if filtered_counts:
        print("      Min/max filtering applied:")
        for feature, count in filtered_counts.items():
            constraint = constraints[feature]
            print(f"        {feature}: {count} values filtered (valid range: {constraint['min']}-{constraint['max']})")

    return {
        "dataType": "cross-sectional",
        "groups": groups,
        "units": units
    }


if __name__ == "__main__":
    # Test the transform
    result = transform()
    print(f"Transformed {len(result['groups'])} groups (cross-sectional)")
    for group in result['groups']:
        print(f"  {group['parameterAnomalyGroupId']}:")
        for feature in group['featureArray']:
            value = feature['parameterHistoryArray'][0]['value']
            print(f"    {feature['featureName']}: {value:.2f}")
