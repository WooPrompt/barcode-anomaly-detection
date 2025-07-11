# -*- coding: utf-8 -*-
"""
@Title: Anomaly Detection Benchmark
@Description:
This script benchmarks the performance of the modular anomaly detection functions.
"""

import pandas as pd
import os
from anomaly_detection_decomposed import detect_epc_fake, detect_epc_dup, detect_loc_err, detect_evt_order_err
from transition_time_analyzer_v2 import load_and_merge_raw_data

def run_benchmarks():
    # --- Configuration ---
    BASE_PROJECT_DIR = os.getcwd()
    RAW_PATH = os.path.join(BASE_PROJECT_DIR, "data", "raw")
    CSV_PATH = os.path.join(BASE_PROJECT_DIR, "data", "processed")

    FACTORY_FILES = ["hws", "icn", "kum", "ygs"]
    GEOSPATIAL_FILE = os.path.join(
        CSV_PATH, "location_id_withGeospatial - location_id.csv"
    )

    # --- Load Data ---
    print("Loading data...")
    raw_df = load_and_merge_raw_data(RAW_PATH, FACTORY_FILES)
    geo_df = pd.read_csv(GEOSPATIAL_FILE)[
        ["scan_location", "Latitude", "Longitude"]
    ].drop_duplicates()
    df = pd.merge(raw_df, geo_df, on="scan_location", how="left")
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df.dropna(subset=["event_time"], inplace=True)
    print(f"Data loaded. Total records: {len(df)}")

    # --- Run Benchmarks ---
    print("\n--- Running Benchmarks ---")
    functions_to_benchmark = [
        detect_epc_fake,
        detect_epc_dup,
        detect_loc_err
    ]

    results = []
    for func in functions_to_benchmark:
        anomalies, duration = func(df)
        result = {
            "function": func.__name__,
            "anomalies_found": len(anomalies),
            "runtime_sec": duration
        }
        results.append(result)
        print(f"- {result['function']}: Found {result['anomalies_found']} anomalies in {result['runtime_sec']:.4f} seconds.")
    
    print("\n--- Benchmark Summary ---")
    benchmark_df = pd.DataFrame(results)
    print(benchmark_df.to_string(index=False))

if __name__ == "__main__":
    run_benchmarks()
