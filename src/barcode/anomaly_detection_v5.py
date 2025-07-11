# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 21:00:00 2025

@author: user

@Title: Anomaly Detection v5 (Refactored for API)
@Description:
This script refactors the comprehensive rule-based anomaly detection logic into a callable function.
It takes a DataFrame as input, performs all five anomaly checks (epcFake, epcDup, locErr, evtOrderErr,
and statistical jump), and returns both the detected anomalies and the cleaned DataFrame.
This version is designed to be integrated into a larger model serving API.

@Changelog:
- v5.0 (New):
  - Refactored `run_anomaly_detection` into `detect_rule_based_anomalies(input_df, transition_stats, geo_df)`.
  - The function now takes the raw data, transition statistics, and geospatial data as DataFrames.
  - It returns two DataFrames: `anomalies_df` (rule-based anomalies) and `clean_df` (data after rule-based cleaning).
  - Removed all direct file I/O (loading/saving) from the core logic.
  - Removed factory-by-factory processing loop, assuming a single combined input DataFrame.
- v4.0 (Previous):
  - Contained all anomaly detection logic but handled file I/O internally.
"""

import pandas as pd
import numpy as np
import os
import math
from datetime import datetime, timedelta
from transition_time_analyzer_v2 import load_and_merge_raw_data

VALID_COMPANY_CODES = {"8804823", "8805843", "8809437"}
Z_SCORE_THRESHOLD = -3.0  # For jump anomalies (faster than average)


def check_epc_format(epc):
    if not isinstance(epc, str):
        return False
    parts = epc.split(".")
    if len(parts) != 6:
        return False
    header, company, product, lot, manufacture, serial = parts
    if header != "001":
        return False
    if company not in VALID_COMPANY_CODES:
        return False
    if not (product.isdigit() and len(product) == 7):
        return False
    if not (lot.isdigit() and len(lot) == 6):
        return False
    if not (serial.isdigit() and len(serial) == 9):
        return False
    try:
        manufacture_date = datetime.strptime(manufacture, "%Y%m%d")
        if (
            manufacture_date > datetime.now()
            or manufacture_date < datetime.now() - timedelta(days=5 * 365)
        ):
            return False
    except ValueError:
        return False
    return True


def get_location_level(scan_location):
    if not isinstance(scan_location, str):
        return 99
    if "공장" in scan_location:
        return 0
    if "물류센터" in scan_location:
        return 1
    if "도매상" in scan_location:
        return 2
    if "소매상" in scan_location:
        return 3
    return 99


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    if any(v is None or pd.isna(v) for v in [lat1, lon1, lat2, lon2]):
        return 0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def detect_rule_based_anomalies(
    input_df: pd.DataFrame, transition_stats: pd.DataFrame, geo_df: pd.DataFrame
):
    """
    Detects rule-based anomalies (epcFake, epcDup, locErr, evtOrderErr, statistical jump).
    Takes raw data, transition statistics, and geospatial data as DataFrames.
    Returns a DataFrame of detected anomalies and a DataFrame of rule-based cleaned data.
    """
    df = (
        input_df.copy()
    )  # Work on a copy to avoid modifying the original input DataFrame

    # 1. Prepare Data (similar to v4, but without file loading)
    df = pd.merge(df, geo_df, on="scan_location", how="left")
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df.dropna(subset=["event_time"], inplace=True)
    df["original_index"] = (
        df.index.astype(str) + df["factory"]
    )  # Unique identifier for each row

    # Sort for journey analysis
    df.sort_values(["epc_code", "event_time"], inplace=True)

    anomalies_list = []  # Use a list to collect anomaly records

    # --- Initial Raw Checks (epcFake, epcDup) ---
    # epcFake
    fake_mask = ~df["epc_code"].apply(check_epc_format)
    if fake_mask.any():
        anomalies_df = df[fake_mask].copy()
        anomalies_df["anomaly_type"] = "epcFake"
        anomalies_list.extend(anomalies_df.to_dict(orient="records"))

    # epcDup
    dup_mask = df.duplicated(subset=["epc_code", "event_time"], keep=False)
    if dup_mask.any():
        duplicates = df[dup_mask]
        for _, group in duplicates.groupby(["epc_code", "event_time"]):
            if group["scan_location"].nunique() > 1:
                anomalies_df = group.copy()
                anomalies_df["anomaly_type"] = "epcDup"
                anomalies_list.extend(anomalies_df.to_dict(orient="records"))

    # Filter out initial anomalies before transition-based checks
    initial_anomalous_indices = (
        pd.DataFrame(anomalies_list)["original_index"].unique()
        if anomalies_list
        else []
    )
    df_after_initial_clean = df[
        ~df["original_index"].isin(initial_anomalous_indices)
    ].copy()

    # --- Transition-Based Checks (locErr, evtOrderErr, jump) ---
    cols_to_shift = [
        "scan_location",
        "event_type",
        "event_time",
        "Latitude",
        "Longitude",
    ]
    journeys = df_after_initial_clean.copy()
    for col in cols_to_shift:
        journeys[f"from_{col}"] = journeys.groupby("epc_code")[col].shift(1)

    journeys.dropna(subset=[f"from_{col}" for col in cols_to_shift], inplace=True)

    if not journeys.empty:
        # Merge transition stats for jump detection
        journeys = pd.merge(
            journeys,
            transition_stats,
            how="left",
            left_on=["from_scan_location", "scan_location"],
            right_on=["from_scan_location", "to_scan_location"],
        )

        # locErr
        journeys["from_level"] = journeys["from_scan_location"].apply(
            get_location_level
        )
        journeys["to_level"] = journeys["scan_location"].apply(get_location_level)
        loc_err_mask = journeys["to_level"] < journeys["from_level"]
        if loc_err_mask.any():
            anomalies_df = journeys[loc_err_mask].copy()
            anomalies_df["anomaly_type"] = "locErr"
            anomalies_list.extend(anomalies_df.to_dict(orient="records"))

        # evtOrderErr
        evt_err_mask = (
            (journeys["scan_location"] == journeys["from_scan_location"])
            & (journeys["from_event_type"].str.contains("Outbound", na=False))
            & (journeys["event_type"].str.contains("Inbound", na=False))
        )
        if evt_err_mask.any():
            anomalies_df = journeys[evt_err_mask].copy()
            anomalies_df["anomaly_type"] = "evtOrderErr"
            anomalies_list.extend(anomalies_df.to_dict(orient="records"))

        # jump (Statistical Method)
        journeys["time_taken_hours"] = (
            journeys["event_time"] - journeys["from_event_time"]
        ) / np.timedelta64(1, "h")
        journeys["z_score"] = np.nan
        valid_std = journeys["time_taken_hours_std"] > 0
        journeys.loc[valid_std, "z_score"] = (
            journeys["time_taken_hours"] - journeys["time_taken_hours_mean"]
        ) / journeys["time_taken_hours_std"]

        jump_conditions = journeys["z_score"] < Z_SCORE_THRESHOLD
        if jump_conditions.any():
            anomalies_df = journeys[jump_conditions].copy()
            anomalies_df["anomaly_type"] = "jump"
            anomalies_list.extend(anomalies_df.to_dict(orient="records"))

    # Consolidate all anomalies found
    if anomalies_list:
        final_anomalies_df = pd.DataFrame(anomalies_list).drop_duplicates(
            subset=["original_index"]
        )
    else:
        final_anomalies_df = pd.DataFrame()

    # Create the final clean dataset by excluding all anomalous EPCs
    # This ensures that if any event for an EPC is anomalous, the entire EPC's journey is removed.
    if not final_anomalies_df.empty:
        anomalous_epcs = set(final_anomalies_df["epc_code"])
        clean_df = df[~df["epc_code"].isin(anomalous_epcs)].copy()
    else:
        clean_df = df.copy()

    # Drop the helper columns before returning the clean data
    columns_to_drop = [
        "from_scan_location",
        "from_event_time",
        "from_event_type",
        "from_Latitude",
        "from_Longitude",
        "to_scan_location",
        "time_taken_hours_mean",
        "time_taken_hours_std",
        "time_taken_hours_min",
        "time_taken_hours_max",
        "time_taken_hours_count",
        "time_taken_hours",
        "z_score",
        "distance_km",
        "time_diff_hours",
        "speed_kmh",
        "from_level",
        "to_level",
        "Latitude",
        "Longitude",
        "original_index",
    ]
    clean_df_cols = [col for col in clean_df.columns if col not in columns_to_drop]
    clean_df = clean_df[clean_df_cols]

        # 🔍 Debug: What anomalies were detected?
    print("▶ Final anomaly types found:")
    if anomalies_list:
        df_temp = pd.DataFrame(anomalies_list)
        print(df_temp["anomaly_type"].value_counts())
    else:
        print("❌ No rule-based anomalies were detected.")

    return final_anomalies_df, clean_df



if __name__ == "__main__":
    # --- Configuration (for local testing) ---
    BASE_PROJECT_DIR = os.getcwd()
    RAW_PATH = os.path.join(BASE_PROJECT_DIR, "data", "raw")
    CSV_PATH = os.path.join(BASE_PROJECT_DIR, "data", "processed")

    FACTORY_FILES = ["hws", "icn", "kum", "ygs"]  # Use real data for testing
    TRANSITION_STATS_FILE = os.path.join(
        CSV_PATH, "business_step_transition_avg_v2.csv"
    )
    GEOSPATIAL_FILE = os.path.join(
        CSV_PATH, "location_id_withGeospatial - location_id.csv"
    )

    raw_df = load_and_merge_raw_data(RAW_PATH, FACTORY_FILES)
    transition_stats = pd.read_csv(TRANSITION_STATS_FILE)
    geo_df = pd.read_csv(GEOSPATIAL_FILE)[
        ["scan_location", "Latitude", "Longitude"]
    ].drop_duplicates()

    if raw_df is not None:
        rule_anomalies, rule_clean_df = detect_rule_based_anomalies(
            raw_df, transition_stats, geo_df
        )
        print("--- Rule-Based Anomaly Detection Results (Local Test) ---")
        if not rule_anomalies.empty:
            print("Total Rule-Based Anomaly Counts:")
            print(rule_anomalies["anomaly_type"].value_counts())
            rule_anomalies.to_csv(
                os.path.join(CSV_PATH, "local_rule_anomalies_test.csv"), index=False
            )
        else:
            print("No rule-based anomalies detected.")
        rule_clean_df.to_csv(
            os.path.join(CSV_PATH, "local_rule_clean_test.csv"), index=False, sep="\t"
        )
        print(
            f"Rule-based clean data saved to: {os.path.join(CSV_PATH, 'local_rule_clean_test.csv')}"
        )
