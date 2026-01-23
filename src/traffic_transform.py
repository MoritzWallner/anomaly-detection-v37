"""
Transform traffic.csv into the format expected by detect_anomalies().

Input format (traffic.csv):
    DateTime,Junction,Vehicles,ID
    2015-11-01 00:00:00,1,15,20151101001

Output format:
    {
        "dataType": "time-series",
        "groups": [
            {
                "parameterAnomalyGroupId": "Junction1",
                "featureArray": [
                    {
                        "featureName": "vehicle_count",
                        "parameterHistoryArray": [
                            {"parameterHistoryId": "uuid", "createdAt": "ISO8601", "type": "number", "value": 15}
                        ]
                    }
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
        config_path = Path(__file__).parent.parent / "config" / "traffic.json"

    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"constraints": {}}


def transform(csv_path: str = None, config_path: str = None) -> Dict[str, Any]:
    """
    Transform traffic.csv to anomaly detector input format.

    Args:
        csv_path: Path to traffic.csv. If None, uses default location.
        config_path: Path to config JSON with min/max constraints. If None, uses default.

    Returns:
        Dict in the format expected by detect_anomalies()
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "datasets" / "traffic.csv"

    # Load config with min/max constraints and units
    config = load_config(config_path)
    constraints = config.get("constraints", {})
    units = config.get("units", {})

    df = pd.read_csv(csv_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    groups = []
    filtered_counts = {}  # Track filtered values per feature

    for junction_id in sorted(df['Junction'].unique()):
        junction_data = df[df['Junction'] == junction_id].sort_values('DateTime')

        param_history = []
        feature_name = "vehicle_count"

        for _, row in junction_data.iterrows():
            value = float(row['Vehicles'])

            # Apply min/max filtering if constraints exist for this feature
            if feature_name in constraints:
                min_val = constraints[feature_name].get('min', float('-inf'))
                max_val = constraints[feature_name].get('max', float('inf'))
                if not (min_val <= value <= max_val):
                    filtered_counts[feature_name] = filtered_counts.get(feature_name, 0) + 1
                    continue  # Skip this value

            param_history.append({
                "parameterHistoryId": str(uuid.uuid4()),
                "createdAt": row['DateTime'].isoformat(),
                "type": "number",
                "value": value
            })

        groups.append({
            "parameterAnomalyGroupId": f"Junction{junction_id}",
            "featureArray": [
                {
                    "featureName": feature_name,
                    "parameterHistoryArray": param_history
                }
            ]
        })

    # Log filtered values
    if filtered_counts:
        print("      Min/max filtering applied:")
        for feature, count in filtered_counts.items():
            constraint = constraints[feature]
            print(f"        {feature}: {count} values filtered (valid range: {constraint['min']}-{constraint['max']})")

    return {
        "dataType": "time-series",
        "groups": groups,
        "units": units
    }


if __name__ == "__main__":
    # Test the transform
    result = transform()
    print(f"Transformed {len(result['groups'])} groups")
    for group in result['groups']:
        print(f"  {group['parameterAnomalyGroupId']}: {len(group['featureArray'][0]['parameterHistoryArray'])} data points")
