
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 15:00:00 2025

@author: user

@Title: Decomposed Anomaly Detection for Performance Analysis
@Description:
This script breaks down the monolithic rule-based detection logic into five separate,
independent functions. Each function detects a single type of anomaly (epcFake,
epcDup, locErr, evtOrderErr, jump) from the entire raw dataset. The primary purpose
is to measure the performance (execution time) and effectiveness (number of anomalies
found) of each rule individually. This helps in understanding the cost and benefit of
each check before building a sequential pipeline.

Each function returns a DataFrame containing:
- The problematic `epc_code`.
- The specific event details (`event_time`, `scan_location`, etc.).
- The `anomaly_type`.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta

# --- Constants & Helpers from anomaly_detection_v5 ---
VALID_COMPANY_CODES = {"8804823", "8805843", "8809437"}
Z_SCORE_THRESHOLD = -3.0

def check_epc_format(epc):
    if not isinstance(epc, str): return False
    parts = epc.split(".")
    if len(parts) != 6: return False
    header, company, product, lot, manufacture, serial = parts
    if header != "001": return False
    if company not in VALID_COMPANY_CODES: return False
    try:
        manufacture_date = datetime.strptime(manufacture, "%Y%m%d")
        if not (datetime.now() - timedelta(days=5*365) < manufacture_date < datetime.now()):
            return False
    except (ValueError, IndexError):
        return False
    return True

def get_location_level(scan_location):
    if not isinstance(scan_location, str): return 99
    if "공장" in scan_location: return 0
    if "물류센터" in scan_location: return 1
    if "도매상" in scan_location: return 2
    if "소매상" in scan_location: return 3
    return 99

# --- Decomposed Anomaly Detection Functions ---

def detect_epc_fake(df: pd.DataFrame) -> pd.DataFrame:
    """Detects records with invalid EPC format."""
    fake_mask = ~df["epc_code"].apply(check_epc_format)
    anomalies = df[fake_mask].copy()
    anomalies["anomaly_type"] = "epcFake"
    return anomalies

def detect_epc_dup(df: pd.DataFrame) -> pd.DataFrame:
    """Detects EPCs scanned in multiple locations at the exact same time."""
    dup_mask = df.duplicated(subset=["epc_code", "event_time"], keep=False)
    duplicates = df[dup_mask]
    
    # Find groups where the same EPC/time has more than one unique location
    anomalous_groups = duplicates.groupby(["epc_code", "event_time"]).filter(lambda x: x["scan_location"].nunique() > 1)
    anomalies = anomalous_groups.copy()
    anomalies["anomaly_type"] = "epcDup"
    return anomalies

def detect_loc_err(df: pd.DataFrame) -> pd.DataFrame:
    """Detects backward movements in the supply chain (e.g., wholesaler to factory)."""
    journeys = df.sort_values(["epc_code", "event_time"]).copy()
    journeys["from_scan_location"] = journeys.groupby("epc_code")["scan_location"].shift(1)
    journeys.dropna(subset=["from_scan_location"], inplace=True)

    journeys["from_level"] = journeys["from_scan_location"].apply(get_location_level)
    journeys["to_level"] = journeys["scan_location"].apply(get_location_level)
    
    loc_err_mask = journeys["to_level"] < journeys["from_level"]
    anomalies = journeys[loc_err_mask].copy()
    anomalies["anomaly_type"] = "locErr"
    return anomalies

def detect_evt_order_err(df: pd.DataFrame) -> pd.DataFrame:
    """Detects 'Inbound' events happening before 'Outbound' at the same location."""
    journeys = df.sort_values(["epc_code", "event_time"]).copy()
    journeys["from_scan_location"] = journeys.groupby("epc_code")["scan_location"].shift(1)
    journeys["from_event_type"] = journeys.groupby("epc_code")["event_type"].shift(1)
    journeys.dropna(subset=["from_scan_location", "from_event_type"], inplace=True)

    evt_err_mask = (
        (journeys["scan_location"] == journeys["from_scan_location"])
        & (journeys["from_event_type"].str.contains("Outbound", na=False))
        & (journeys["event_type"].str.contains("Inbound", na=False))
    )
    anomalies = journeys[evt_err_mask].copy()
    anomalies["anomaly_type"] = "evtOrderErr"
    return anomalies

def detect_jump(df: pd.DataFrame, transition_stats: pd.DataFrame) -> pd.DataFrame:
    """Detects impossibly fast travel between two locations."""
    journeys = df.sort_values(["epc_code", "event_time"]).copy()
    journeys["event_time"] = pd.to_datetime(journeys["event_time"])
    
    # Shift columns to create transitions
    journeys["from_scan_location"] = journeys.groupby("epc_code")["scan_location"].shift(1)
    journeys["from_event_time"] = journeys.groupby("epc_code")["event_time"].shift(1)
    journeys.dropna(subset=["from_scan_location", "from_event_time"], inplace=True)

    # Merge with transition statistics
    journeys = pd.merge(
        journeys,
        transition_stats,
        how="left",
        left_on=["from_scan_location", "scan_location"],
        right_on=["from_scan_location", "to_scan_location"],
    )

    # Calculate Z-score for time taken
    journeys["time_taken_hours"] = (journeys["event_time"] - journeys["from_event_time"]) / np.timedelta64(1, "h")
    journeys["z_score"] = np.nan
    valid_std = journeys["time_taken_hours_std"] > 0
    journeys.loc[valid_std, "z_score"] = (
        journeys["time_taken_hours"] - journeys["time_taken_hours_mean"]
    ) / journeys["time_taken_hours_std"]

    jump_mask = journeys["z_score"] < Z_SCORE_THRESHOLD
    anomalies = journeys[jump_mask].copy()
    anomalies["anomaly_type"] = "jump"
    return anomalies


if __name__ == "__main__":
    # --- 1. Load Data ---
    print("Loading data for analysis...")
    BASE_PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..\..'))
    RAW_PATH = os.path.join(BASE_PROJECT_DIR, "data", "raw")
    PROCESSED_PATH = os.path.join(BASE_PROJECT_DIR, "data", "processed")

    # Load and combine all raw factory data
    factory_files = [f for f in os.listdir(RAW_PATH) if f.endswith('.csv')]
    # Define column names as the files don't have headers
    # The raw files have a header row. We read it, using tab as a separator.
    # low_memory=False is used to prevent DtypeWarning from mixed types in columns.
    df_list = [pd.read_csv(os.path.join(RAW_PATH, f), sep='	', low_memory=False) for f in factory_files]
    raw_df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded a total of {len(raw_df)} records from {len(factory_files)} files.")

    # Load transition statistics needed for the 'jump' detector
    transition_stats_file = os.path.join(PROCESSED_PATH, "business_step_transition_avg_v2.csv")
    transition_stats = pd.read_csv(transition_stats_file)
    print("Transition statistics loaded.")

    # --- 2. Define and Run Detectors ---
    detectors = {
        "EPC Fake": detect_epc_fake,
        "EPC Duplication": detect_epc_dup,
        "Location Error": detect_loc_err,
        "Event Order Error": detect_evt_order_err,
        "Time/Space Jump": detect_jump,
    }

    print("\n--- Running Decomposed Anomaly Detectors ---")
    
    results = {}
    for name, func in detectors.items():
        print(f"\n--- Analyzing: {name} ---")
        start_time = time.time()
        
        # The 'jump' detector has a different signature
        if name == "Time/Space Jump":
            anomalies_df = func(raw_df, transition_stats)
        else:
            anomalies_df = func(raw_df)
            
        end_time = time.time()
        duration = end_time - start_time
        num_anomalies = len(anomalies_df)
        results[name] = {"count": num_anomalies, "time": duration}

        print(f">> Found {num_anomalies} anomalies.")
        print(f">> Execution time: {duration:.4f} seconds.")
        if not anomalies_df.empty:
            print("Sample of detected anomalies:")
            # Display relevant columns
            display_cols = ['epc_code', 'event_time', 'scan_location', 'event_type', 'anomaly_type']
            print(anomalies_df[display_cols].head().to_string())

    # --- 3. Final Summary ---
    print("\n--- Performance Summary ---")
    summary_df = pd.DataFrame.from_dict(results, orient='index')
    summary_df.index.name = "Anomaly Type"
    summary_df.columns = ["Anomaly Count", "Execution Time (s)"]
    print(summary_df.sort_values(by="Anomaly Count", ascending=False).to_string())
