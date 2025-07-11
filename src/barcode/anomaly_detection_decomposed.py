# -*- coding: utf-8 -*-
"""
@Title: Anomaly Detection Decomposed
@Description:
This script contains modular, standalone functions for detecting specific types of anomalies
in barcode scan data. Each function is designed to be benchmarked and eventually replaced
by a machine learning model.
"""

import pandas as pd
from datetime import datetime, timedelta
import time

# A set of valid company codes for EPC format validation.
VALID_COMPANY_CODES = {"8804823", "8805843", "8809437"}


def check_epc_format(epc: str) -> bool:
    """
    Validates the structure and content of a given EPC code.

    Args:
        epc: The EPC code string to validate.

    Returns:
        True if the format is valid, False otherwise.
    """
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
        # Check if the manufacture date is not in the future or more than 5 years in the past.
        if not (datetime.now() - timedelta(days=5*365) < manufacture_date < datetime.now()):
            return False
    except ValueError:
        return False
    return True


def get_location_level(scan_location: str) -> int:
    """
    Assigns a hierarchical level to a scan location.

    Args:
        scan_location: The name of the scan location.

    Returns:
        An integer representing the supply chain level (0 is highest).
    """
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
    return 99 # Default for unknown locations


def detect_epc_fake(df: pd.DataFrame) -> tuple[list[dict], float]:
    """
    Detects records with malformed EPC codes (epcFake).

    Args:
        df: DataFrame with barcode data, must include 'epc_code' and 'event_time'.

    Returns:
        A tuple containing:
        - A list of dictionaries, where each dict represents an epcFake anomaly.
        - The execution time of the function in seconds.
    """
    start_time = time.time()
    
    # Apply the format check to the 'epc_code' column.
    fake_mask = ~df["epc_code"].apply(check_epc_format)
    
    anomalies = []
    if fake_mask.any():
        # Select the anomalous rows.
        anomalous_df = df.loc[fake_mask, ["epc_code", "event_time"]].copy()
        anomalous_df["anomaly_type"] = "epcFake"
        
        # Format the output as a list of dictionaries.
        anomalies = anomalous_df.rename(columns={"anomaly_type": "anomaly"}).to_dict(orient="records")

    end_time = time.time()
    duration = end_time - start_time
    
    return anomalies, duration

def detect_epc_dup(df: pd.DataFrame) -> tuple[list[dict], float]:
    """
    Detects EPCs scanned at different locations at the same time (epcDup).

    Args:
        df: DataFrame with barcode data, must include 'epc_code', 'event_time', and 'scan_location'.

    Returns:
        A tuple containing:
        - A list of dictionaries, where each dict represents an epcDup anomaly.
        - The execution time of the function in seconds.
    """
    start_time = time.time()
    
    # We use groupby and filter to find groups with the same EPC and time but different locations.
    anomalous_groups = df.groupby(['epc_code', 'event_time']).filter(lambda g: g['scan_location'].nunique() > 1)
    
    anomalies = []
    if not anomalous_groups.empty:
        # Format the output to match the required structure.
        result_df = anomalous_groups[['epc_code', 'event_time']].copy()
        result_df['anomaly'] = 'epcDup'
        anomalies = result_df.to_dict(orient='records')

    end_time = time.time()
    duration = end_time - start_time
    
    return anomalies, duration

def detect_loc_err(df: pd.DataFrame) -> tuple[list[dict], float]:
    """
    Detects illogical backward movements in the supply chain (locErr).

    Args:
        df: DataFrame with barcode data, must include 'epc_code', 'event_time', and 'scan_location'.

    Returns:
        A tuple containing:
        - A list of dictionaries, where each dict represents a locErr anomaly.
        - The execution time of the function in seconds.
    """
    start_time = time.time()
    
    # Sort data to represent journeys correctly.
    journeys = df.sort_values(["epc_code", "event_time"]).copy()
    
    # Get the previous scan location for each event.
    journeys['from_scan_location'] = journeys.groupby('epc_code')['scan_location'].shift(1)
    journeys.dropna(subset=['from_scan_location'], inplace=True)

    anomalies = []
    if not journeys.empty:
        # Get the hierarchical level for both current and previous locations.
        journeys["from_level"] = journeys["from_scan_location"].apply(get_location_level)
        journeys["to_level"] = journeys["scan_location"].apply(get_location_level)
        
        # Anomaly is when the current level is less than the previous level.
        loc_err_mask = journeys["to_level"] < journeys["from_level"]
        
        if loc_err_mask.any():
            anomalous_df = journeys.loc[loc_err_mask, ["epc_code", "event_time"]].copy()
            anomalous_df["anomaly"] = "locErr"
            anomalies = anomalous_df.to_dict(orient="records")

    end_time = time.time()
    duration = end_time - start_time
    
    return anomalies, duration

def detect_evt_order_err(df: pd.DataFrame) -> tuple[list[dict], float]:
    """
    Detects illogical event orders within the same location (evtOrderErr).

    Args:
        df: DataFrame with barcode data, must include 'epc_code', 'event_time', 'scan_location', and 'event_type'.

    Returns:
        A tuple containing:
        - A list of dictionaries, where each dict represents an evtOrderErr anomaly.
        - The execution time of the function in seconds.
    """
    start_time = time.time()
    
    # Sort data to represent journeys correctly.
    journeys = df.sort_values(["epc_code", "event_time"]).copy()
    
    # Get the previous event details for each event.
    journeys['from_scan_location'] = journeys.groupby('epc_code')['scan_location'].shift(1)
    journeys['from_event_type'] = journeys.groupby('epc_code')['event_type'].shift(1)
    journeys.dropna(subset=['from_scan_location', 'from_event_type'], inplace=True)

    anomalies = []
    if not journeys.empty:
        # Anomaly is when an Outbound event is followed by an Inbound event at the same location.
        evt_err_mask = (
            (journeys["scan_location"] == journeys["from_scan_location"])
            & (journeys["from_event_type"].str.contains("Outbound", na=False))
            & (journeys["event_type"].str.contains("Inbound", na=False))
        )
        
        if evt_err_mask.any():
            anomalous_df = journeys.loc[evt_err_mask, ["epc_code", "event_time"]].copy()
            anomalous_df["anomaly"] = "evtOrderErr"
            anomalies = anomalous_df.to_dict(orient="records")

    end_time = time.time()
    duration = end_time - start_time
    
    return anomalies, duration
