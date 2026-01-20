"""
Transform vehicles.csv into the format expected by detect_anomalies().

Input format (vehicles.csv):
    vehicle,time,voltage,soc,usage
    ID1,2022-10-01T00:00:00,380.5,85.2,driving

Output format:
    {
        "dataType": "time-series",
        "groups": [
            {
                "parameterAnomalyGroupId": "ID1",
                "featureArray": [
                    {"featureName": "voltage", "parameterHistoryArray": [...]},
                    {"featureName": "soc", "parameterHistoryArray": [...]}
                ]
            }
        ]
    }

Note: The 'usage' column is present in the CSV but excluded from analysis.
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
        config_path = Path(__file__).parent.parent / "config" / "vehicles.json"

    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"constraints": {}}


def transform(csv_path: str = None, config_path: str = None) -> Dict[str, Any]:
    """
    Transform vehicles.csv to anomaly detector input format.

    Args:
        csv_path: Path to vehicles.csv. If None, uses default location.
        config_path: Path to config JSON with min/max constraints. If None, uses default.

    Returns:
        Dict in the format expected by detect_anomalies()
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "datasets" / "vehicles.csv"

    # Load config with min/max constraints
    config = load_config(config_path)
    constraints = config.get("constraints", {})

    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])

    # Feature columns (exclude metadata columns and usage)
    feature_cols = [c for c in df.columns if c not in ['vehicle', 'time', 'usage']]

    groups = []
    filtered_counts = {}  # Track filtered values per feature

    for vehicle in sorted(df['vehicle'].unique()):
        vehicle_data = df[df['vehicle'] == vehicle].sort_values('time')

        feature_array = []

        for col in feature_cols:
            param_history = []
            for _, row in vehicle_data.iterrows():
                value = row[col]
                # Skip NaN values
                if pd.notna(value):
                    value_out = float(value)

                    # Apply min/max filtering if constraints exist for this feature
                    if col in constraints:
                        min_val = constraints[col].get('min', float('-inf'))
                        max_val = constraints[col].get('max', float('inf'))
                        if not (min_val <= value_out <= max_val):
                            filtered_counts[col] = filtered_counts.get(col, 0) + 1
                            continue  # Skip this value

                    param_history.append({
                        "parameterHistoryId": str(uuid.uuid4()),
                        "createdAt": row['time'].isoformat(),
                        "type": "number",
                        "value": value_out
                    })

            feature_array.append({
                "featureName": col,
                "parameterHistoryArray": param_history
            })

        groups.append({
            "parameterAnomalyGroupId": vehicle,
            "featureArray": feature_array
        })

    # Log filtered values
    if filtered_counts:
        print("      Min/max filtering applied:")
        for feature, count in filtered_counts.items():
            constraint = constraints[feature]
            print(f"        {feature}: {count} values filtered (valid range: {constraint['min']}-{constraint['max']})")

    return {
        "dataType": "time-series",
        "groups": groups
    }


if __name__ == "__main__":
    # Test the transform
    result = transform()
    print(f"Transformed {len(result['groups'])} groups")
    for group in result['groups']:
        print(f"  {group['parameterAnomalyGroupId']}:")
        for feature in group['featureArray']:
            print(f"    {feature['featureName']}: {len(feature['parameterHistoryArray'])} data points")
