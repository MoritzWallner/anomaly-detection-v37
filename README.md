# Anomaly Detection Pipeline

Companion code for the semester thesis *"Detecting Anomalous Behavior in Fleet
Data: A Generalized Approach to Multi-Feature Outlier Detection"* (Moritz
Wallner, TUM FTM). The pipeline compares multiple instances of similar objects
(vehicles, traffic junctions, customers) and identifies which instance behaves
anomalously, and why.

## Requirements

- Python 3.9 or newer
- The dependencies listed in `requirements.txt` (NumPy, pandas, scikit-learn,
  SciPy, matplotlib, PyArrow)

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the pipeline

There are two entry points.

### 1. `main.py` — interactive analysis of one dataset

```bash
python main.py
```

A menu lets you pick one of the three datasets:

1. **Traffic** (`datasets/traffic.csv`) — hourly vehicle counts at four
   junctions. Time-series.
2. **Vehicles** (`datasets/vehicles.csv`) — battery voltage and state of charge
   for seven electric vehicles. Time-series.
3. **Customers** (`datasets/customers.csv`) — synthetic customer profiles with
   four attributes each. Cross-sectional.

For the chosen dataset the script:

- writes a JSON result file to `output/results/<dataset>_results.json`
- renders the diagrams used in Chapter 4 of the thesis to
  `diagrams/<dataset>/`
- prints a short summary listing the detected outlier groups

The expected ground-truth outliers are `Junction1` (traffic), `ID1` (vehicles)
and `CUST_010` (customers).

### 2. `evaluation.py` — ablation runner

```bash
python evaluation.py
```

Reproduces the 91-configuration ablation study reported in Section 4.5 of the
thesis (three aggregation windows × three detection methods × three
contamination settings × three vehicle subset sizes, plus the traffic and
customer datasets). The full result table is written to
`output/evaluation_results.csv`.

## Project layout

```
.
├── main.py                    # Interactive CLI
├── evaluation.py              # Ablation / sensitivity-analysis runner
├── requirements.txt           # Python dependencies
├── config/                    # Per-dataset settings (units, min/max constraints, default eval config)
│   ├── traffic.json
│   ├── vehicles.json
│   └── customers.json
├── datasets/                  # Raw input CSV files
│   ├── traffic.csv
│   ├── vehicles.csv
│   └── customers.csv
├── src/
│   ├── anomaly_detector.py    # The pipeline itself (preprocessing, feature extraction, detection, plot data)
│   ├── traffic_transform.py   # Loads traffic.csv into the pipeline's input format
│   ├── vehicles_transform.py  # Loads vehicles.csv into the pipeline's input format
│   └── customers_transform.py # Loads customers.csv into the pipeline's input format
├── diagrams/                  # Rendered PNG plots, one subfolder per dataset
└── output/                    # Generated result JSON and the ablation CSV
```

## How the code maps to the thesis

All algorithmic logic lives in `src/anomaly_detector.py`. The table below maps
each section of Chapter 3 ("Method") to the function that implements it.

| Thesis section                                            | File                                 | Function                                                            |
| --------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------- |
| §3.3 Minimum / Maximum filtering                          | `src/<dataset>_transform.py`         | `transform()` (reads `constraints` from `config/<dataset>.json`)    |
| §3.4 Enum to number conversion                            | `src/anomaly_detector.py`            | `preprocess_enum_values()`                                          |
| §3.5 Feature Extraction (time-series)                     | `src/anomaly_detector.py`            | `build_time_series_features()`, `calculate_temporal_features()`     |
| §3.5 Feature Extraction (cross-sectional)                 | `src/anomaly_detector.py`            | `build_cross_sectional_features()`                                  |
| §3.5 Slope formula                                        | `src/anomaly_detector.py`            | `calculate_slope()`                                                 |
| §3.6 Normalization (StandardScaler)                       | `src/anomaly_detector.py`            | inside `detect_group_outliers()` (calls `StandardScaler`)           |
| §3.7 Outlier Detection on Parameter Group Level (Z-Score, Isolation Forest, ensemble, positive-slope gate, adaptive contamination, score ranking) | `src/anomaly_detector.py` | `detect_group_outliers()`                                           |
| §3.8 Outlier Detection on Parameter Level (feature importance, 70th percentile threshold) | `src/anomaly_detector.py` | `detect_feature_and_point_outliers()`                               |
| §3.9 Plot Data generation                                 | `src/anomaly_detector.py`            | `generate_plot_data()`, `render_plots()`                            |
| §4.5 Ablation and sensitivity analysis                    | `evaluation.py`                      | `run_evaluation()`                                                  |

The top-level function is `detect_anomalies()` in `src/anomaly_detector.py`. It
orchestrates all of the steps above in the order they appear in the thesis.

## Datasets

| Dataset   | Type             | Source                                                                                          | Ground-truth outlier |
| --------- | ---------------- | ----------------------------------------------------------------------------------------------- | -------------------- |
| Vehicles  | Time-series      | TUM FTM electric-vehicle UDS dataset (https://github.com/TUMFTM/electric-vehicle-uds-dataset)   | `ID1`                |
| Traffic   | Time-series      | Kaggle traffic prediction dataset (https://www.kaggle.com/code/karnikakapoor/traffic-prediction-gru) | `Junction1`     |
| Customers | Cross-sectional  | Synthetic, generated as described in Chapter 4 of the thesis                                    | `CUST_010`           |

## Default configuration

The default pipeline configuration (used for all results in Chapter 4 of the
thesis) is:

- Aggregation window: monthly (`M`)
- Detection method: ensemble (Z-Score + Isolation Forest with the
  positive-slope degradation filter)
- Isolation Forest: 100 trees, `random_state=42`
- Contamination: adaptive, `min(0.3, 1/n)` where `n` is the number of groups

This configuration is stored in `config/<dataset>.json` under the `eval_config`
key. The `evaluation.py` script systematically varies these settings to
reproduce the ablation study.
