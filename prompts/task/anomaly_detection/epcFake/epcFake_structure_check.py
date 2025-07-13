# -*- coding: utf-8 -*-
"""
EPC Structure Check - Standalone Function
Detects malformed EPC codes that don't match expected structure format.
Generated from: prompts/task/anomaly_detection/epcFake/
"""

import pandas as pd
from datetime import datetime, timedelta
import time

# Valid company codes for EPC format validation
VALID_COMPANY_CODES = {"8804823", "8805843", "8809437"}


def check_epc_format(epc: str) -> bool:
    """Validates EPC code structure and content."""
    if not isinstance(epc, str):
        return False
    parts = epc.split(".")
    if len(parts) != 6:
        return False
    header, company, product, lot, manufacture, serial = parts
    if header != "001" or company not in VALID_COMPANY_CODES:
        return False
    if not (product.isdigit() and len(product) == 7):
        return False
    if not (lot.isdigit() and len(lot) == 6):
        return False
    if not (serial.isdigit() and len(serial) == 9):
        return False
    try:
        manufacture_date = datetime.strptime(manufacture, "%Y%m%d")
        if not (datetime.now() - timedelta(days=5*365) < manufacture_date < datetime.now()):
            return False
    except ValueError:
        return False
    return True


def detect_epc_fake(df: pd.DataFrame) -> tuple[list[dict], float]:
    """
    Detects malformed EPC codes.
    
    Args:
        df: DataFrame with 'epc_code' and 'event_time' columns
        
    Returns:
        tuple: (anomalies_list, execution_time_seconds)
    """
    start_time = time.time()
    
    fake_mask = ~df["epc_code"].apply(check_epc_format)
    anomalies = []
    
    if fake_mask.any():
        anomalous_df = df.loc[fake_mask, ["epc_code", "event_time"]].copy()
        anomalies = [
            {
                "epc_code": row["epc_code"],
                "anomaly": "epcFake", 
                "event_time": str(row["event_time"])
            }
            for _, row in anomalous_df.iterrows()
        ]
    
    execution_time = time.time() - start_time
    return anomalies, execution_time


if __name__ == "__main__":
    # Benchmark test
    import pandas as pd
    test_data = pd.DataFrame({
        'epc_code': [
            '001.8804823.1234567.123456.20250101.123456789',  # Valid
            '002.8804823.1234567.123456.20250101.123456789',  # Invalid header
            '001.9999999.1234567.123456.20250101.123456789',  # Invalid company
            'invalid.format'  # Invalid structure
        ],
        'event_time': ['2025-01-01 10:00:00'] * 4
    })
    
    anomalies, runtime = detect_epc_fake(test_data)
    print(f"Runtime: {runtime:.4f}s")
    print(f"Anomalies found: {len(anomalies)}")
    for anomaly in anomalies:
        print(f"  {anomaly}")