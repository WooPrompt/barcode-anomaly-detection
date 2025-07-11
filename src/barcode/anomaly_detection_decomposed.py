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