# -*- coding: utf-8 -*- *거점에 재고 체류시간은 평균 이동시간에 제외하도록 변경한 코드*
"""
Created on Tue Jul  8 16:00:00 2025

@author: user

@Title: Transition Time Analyzer v2.1 (Corrected)
@Description:
This script calculates the average time for products to move between scanning locations.
It now reads the original, raw factory data to avoid the circular reference pointed out by the user.
This version ensures the resulting transition averages are unbiased and can be used as a reliable rulebook for anomaly detection.

@Changelog:
- v2.1 (Corrected):
  - Removed the dependency on the pre-cleaned 'all_factories_clean_v1.csv' to fix a circular logic error.
  - Added a function to load and merge the raw data from hws.csv, icn.csv, kum.csv, and ygs.csv.
  - The analysis now runs on the complete, unfiltered dataset.
- v2.0 (Previous):
  - Calculated transition times but incorrectly used a pre-cleaned input file.
"""

import pandas as pd
import numpy as np
import os


def load_and_merge_raw_data(base_path, factory_names):
    """Loads and merges data from a list of raw factory CSVs."""
    all_dfs = []
    print(f"Starting to load raw data from: {base_path}")
    for name in factory_names:
        file_path = os.path.join(base_path, f"{name}.csv")
        try:
            df = pd.read_csv(file_path, sep="\t", engine="python")
            df["factory"] = name  # Add a column to identify the source factory
            all_dfs.append(df)
            print(f"- Successfully loaded and added {name}.csv")
        except FileNotFoundError:
            print(f"- Warning: {file_path} not found. Skipping.")
        except Exception as e:
            print(f"- Error loading {file_path}: {e}")

    if not all_dfs:
        print("Error: No data was loaded. Exiting.")
        return None

    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(
        f"Successfully merged {len(all_dfs)} files with a total of {len(merged_df)} records."
    )
    return merged_df


def analyze_transition_times(df, output_path):
    """
    Analyzes the transition times between outbound and inbound scan events.
    """
    print("Preparing data for analysis...")
    # Convert event_time to datetime objects, coercing errors to NaT (Not a Time)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    # Drop rows where the conversion failed, as they cannot be analyzed
    df.dropna(subset=["event_time"], inplace=True)

    print("Sorting data by epc_code and event_time...")
    df.sort_values(["epc_code", "event_time"], inplace=True)

    print("Identifying previous event details for each step...")
    df["from_scan_location"] = df.groupby("epc_code")["scan_location"].shift(1)
    df["from_event_time"] = df.groupby("epc_code")["event_time"].shift(1)
    df["from_event_type"] = df.groupby("epc_code")["event_type"].shift(1)

    df.dropna(subset=["from_scan_location"], inplace=True)

    print("Filtering for valid 'Outbound' -> 'Inbound' transitions...")
    transitions_df = df[
        (df["from_event_type"].str.contains("Outbound", na=False))
        & (df["event_type"].str.contains("Inbound", na=False))
        & (df["from_scan_location"] != df["scan_location"])
    ].copy()

    print(f"Found {len(transitions_df)} valid transitions.")

    if len(transitions_df) == 0:
        print("No valid transitions found. Cannot generate summary.")
        return

    print("Calculating transition time in hours...")
    transitions_df["time_taken_hours"] = (
        transitions_df["event_time"] - transitions_df["from_event_time"]
    ) / np.timedelta64(1, "h")

    print("Aggregating results for each unique route...")
    agg_funcs = {"time_taken_hours": ["mean", "std", "min", "max", "count"]}
    transition_summary = (
        transitions_df.groupby(["from_scan_location", "scan_location"])
        .agg(agg_funcs)
        .reset_index()
    )

    transition_summary.columns = [
        "_".join(col).strip() for col in transition_summary.columns.values
    ]
    transition_summary.rename(
        columns={
            "from_scan_location_": "from_scan_location",
            "scan_location_": "to_scan_location",
        },
        inplace=True,
    )

    print(f"Saving analysis to {output_path}...")
    transition_summary.to_csv(output_path, index=False, sep=",")

    print("Analysis complete.")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    # Define base paths
    BASE_PROJECT_DIR = os.getcwd()
    RAW_PATH = os.path.join(BASE_PROJECT_DIR, "raw")
    CSV_PATH = os.path.join(BASE_PROJECT_DIR, "csv")
    output_file = os.path.join(CSV_PATH, "business_step_transition_avg_v2.csv")

    # List of raw factory data files
    factory_files = ["hws", "icn", "kum", "ygs"]

    # Load and merge the raw data first
    raw_df = load_and_merge_raw_data(RAW_PATH, factory_files)

    # Proceed with analysis only if data was loaded successfully
    if raw_df is not None:
        analyze_transition_times(raw_df, output_file)
