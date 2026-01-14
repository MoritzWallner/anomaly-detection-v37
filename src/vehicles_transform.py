"""
Transform vehicles.csv into the format expected by detect_anomalies().

Input format (vehicles.csv):
    vehicle,time,voltage,soc
    ID1,2022-10-01T00:00:00,380.5,85.2

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
"""

import pandas as pd
import uuid
from typing import Dict, Any
from pathlib import Path


def transform(csv_path: str = None) -> Dict[str, Any]:
    """
    Transform vehicles.csv to anomaly detector input format.

    Args:
        csv_path: Path to vehicles.csv. If None, uses default location.

    Returns:
        Dict in the format expected by detect_anomalies()
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "datasets" / "vehicles.csv"

    df = pd.read_csv(csv_path)
    df['time'] = pd.to_datetime(df['time'])

    # Feature columns (exclude metadata columns)
    feature_cols = [c for c in df.columns if c not in ['vehicle', 'time']]

    groups = []

    for vehicle in sorted(df['vehicle'].unique()):
        vehicle_data = df[df['vehicle'] == vehicle].sort_values('time')

        feature_array = []

        for col in feature_cols:
            param_history = []
            for _, row in vehicle_data.iterrows():
                value = row[col]
                # Skip NaN values
                if pd.notna(value):
                    param_history.append({
                        "parameterHistoryId": str(uuid.uuid4()),
                        "createdAt": row['time'].isoformat(),
                        "type": "number",
                        "value": float(value)
                    })

            feature_array.append({
                "featureName": col,
                "parameterHistoryArray": param_history
            })

        groups.append({
            "parameterAnomalyGroupId": vehicle,
            "featureArray": feature_array
        })

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
