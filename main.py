#!/usr/bin/env python3
"""
Anomaly Detection Analysis - Main Entry Point

Interactive menu to choose which dataset to analyze:
1. Traffic (Junction data)
2. Vehicles (EV battery data)
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from anomaly_detector import detect_anomalies
from traffic_transform import transform as transform_traffic
from vehicles_transform import transform as transform_vehicles


def print_header():
    """Print application header."""
    print("=" * 60)
    print("         ANOMALY DETECTION ANALYSIS")
    print("=" * 60)
    print()


def print_menu():
    """Print dataset selection menu."""
    print("Select dataset to analyze:")
    print()
    print("  1. Traffic (Junction vehicle counts)")
    print("  2. Vehicles (EV battery data)")
    print()
    print("  q. Quit")
    print()


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    output_dir = Path(__file__).parent / "output" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return output_path


def print_summary(results: dict, dataset_name: str):
    """Print analysis summary."""
    groups = results.get('groups', [])
    outliers = [g['parameterAnomalyGroupId'] for g in groups if g.get('isOutlier', False)]

    print()
    print("=" * 60)
    print(f"ANALYSIS RESULTS - {dataset_name}")
    print("=" * 60)
    print()
    print(f"Total groups analyzed: {len(groups)}")
    print(f"Outliers detected: {len(outliers)}")

    if outliers:
        print()
        print("Outlier groups:")
        for outlier in outliers:
            print(f"  - {outlier}")
    else:
        print("  No outliers detected")

    print()


def analyze_traffic():
    """Run anomaly detection on traffic data."""
    print()
    print("[1/4] Loading and transforming traffic data...")
    request_data = transform_traffic()
    print(f"      Loaded {len(request_data['groups'])} junctions")

    # Set up diagram save path
    diagram_path = Path(__file__).parent / "diagrams" / "traffic" / "anomaly_analysis.png"

    print()
    print("[2/4] Running anomaly detection...")
    results = detect_anomalies(request_data, save_plots_path=str(diagram_path))

    print()
    print("[3/4] Saving results...")
    output_path = save_results(results, "traffic_results.json")
    print(f"      Saved to: {output_path}")

    print()
    print(f"[4/4] Diagram saved to: {diagram_path}")

    print_summary(results, "Traffic")

    return results


def analyze_vehicles():
    """Run anomaly detection on vehicle data."""
    print()
    print("[1/4] Loading and transforming vehicle data...")
    request_data = transform_vehicles()
    print(f"      Loaded {len(request_data['groups'])} vehicles")

    # Set up diagram save path
    diagram_path = Path(__file__).parent / "diagrams" / "vehicles" / "anomaly_analysis.png"

    print()
    print("[2/4] Running anomaly detection...")
    results = detect_anomalies(request_data, save_plots_path=str(diagram_path))

    print()
    print("[3/4] Saving results...")
    output_path = save_results(results, "vehicles_results.json")
    print(f"      Saved to: {output_path}")

    print()
    print(f"[4/4] Diagram saved to: {diagram_path}")

    print_summary(results, "Vehicles")

    return results


def main():
    """Main entry point."""
    print_header()

    while True:
        print_menu()

        choice = input("Enter choice (1, 2, or q): ").strip().lower()

        if choice == '1':
            analyze_traffic()
            print()
        elif choice == '2':
            analyze_vehicles()
            print()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or q.")
            print()


if __name__ == "__main__":
    main()
