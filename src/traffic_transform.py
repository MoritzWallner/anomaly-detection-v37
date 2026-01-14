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
from typing import Dict, Any
from pathlib import Path


def transform(csv_path: str = None) -> Dict[str, Any]:
    """
    Transform traffic.csv to anomaly detector input format.

    Args:
        csv_path: Path to traffic.csv. If None, uses default location.

    Returns:
        Dict in the format expected by detect_anomalies()
    """
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "datasets" / "traffic.csv"

    df = pd.read_csv(csv_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    groups = []

    for junction_id in sorted(df['Junction'].unique()):
        junction_data = df[df['Junction'] == junction_id].sort_values('DateTime')

        param_history = []
        for _, row in junction_data.iterrows():
            param_history.append({
                "parameterHistoryId": str(uuid.uuid4()),
                "createdAt": row['DateTime'].isoformat(),
                "type": "number",
                "value": float(row['Vehicles'])
            })

        groups.append({
            "parameterAnomalyGroupId": f"Junction{junction_id}",
            "featureArray": [
                {
                    "featureName": "vehicle_count",
                    "parameterHistoryArray": param_history
                }
            ]
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
        print(f"  {group['parameterAnomalyGroupId']}: {len(group['featureArray'][0]['parameterHistoryArray'])} data points")
