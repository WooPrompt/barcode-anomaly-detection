#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to validate the backend format function fixes
"""

import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.barcode.multi_anomaly_detector import detect_anomalies_backend_format

def test_backend_format():
    """Test the backend format function with the problematic input from prompts/d.txt"""
    
    # Test data from prompts/d.txt
    test_input = {
        "data": [
            {
                "eventId": 101,
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "location_id": 1,
                "business_step": "Factory",
                "event_type": "Outbound",
                "event_time": "2024-07-02 09:00:00",
                "file_id": 1
            },
            {
                "eventId": 102,
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "location_id": 2,
                "business_step": "WMS",
                "event_type": "Inbound",
                "event_time": "2024-07-02 11:00:00",
                "file_id": 1
            },
            {
                "eventId": 103,
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "location_id": 3,
                "business_step": "Wholesaler",
                "event_type": "Inbound",
                "event_time": "2024-07-03 09:30:00",
                "file_id": 1
            },
            {
                "eventId": 104,
                "epc_code": "INVALID.FORMAT.EPC",
                "location_id": 4,
                "business_step": "Retailer",
                "event_type": "Inbound",
                "event_time": "2024-07-03 13:00:00",
                "file_id": 1
            },
            {
                "eventId": 105,
                "epc_code": "001.8804823.0000001.000001.20240701.000000002",
                "location_id": 1,
                "business_step": "Factory",
                "event_type": "Outbound",
                "event_time": "2024-07-02 09:00:00",
                "file_id": 1
            },
            {
                "eventId": 106,
                "epc_code": "001.8804823.0000001.000001.20240701.000000002",
                "location_id": 5,
                "business_step": "WMS",
                "event_type": "Inbound",
                "event_time": "2024-07-02 09:00:00",
                "file_id": 1
            }
        ]
    }
    
    print("=== Testing Backend Format Function ===")
    print("Input data:")
    print(json.dumps(test_input, indent=2, ensure_ascii=False))
    print("\n" + "="*50 + "\n")
    
    try:
        # Convert to JSON string as the function expects
        input_json = json.dumps(test_input)
        
        # Call the function
        result_json = detect_anomalies_backend_format(input_json)
        
        # Parse and display result
        result = json.loads(result_json)
        
        print("=== RESULT ===")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n=== ANALYSIS ===")
        
        # Check EventHistory for null values
        event_history = result.get("EventHistory", [])
        print(f"EventHistory count: {len(event_history)}")
        
        has_null_values = False
        for event in event_history:
            for key, value in event.items():
                if value is None and key != "eventId":
                    has_null_values = True
                    break
        
        print(f"Contains null values: {has_null_values}")
        
        # Check epcAnomalyStats
        epc_stats = result.get("epcAnomalyStats", [])
        print(f"epcAnomalyStats count: {len(epc_stats)}")
        
        epc_codes_in_stats = [stat["epcCode"] for stat in epc_stats]
        print(f"EPC codes in stats: {epc_codes_in_stats}")
        
        # Expected EPC codes (should include all that have anomalies)
        expected_epcs = [
            "INVALID.FORMAT.EPC",
            "001.8804823.0000001.000001.20240701.000000001", 
            "001.8804823.0000001.000001.20240701.000000002"
        ]
        
        missing_epcs = [epc for epc in expected_epcs if epc not in epc_codes_in_stats]
        print(f"Missing EPC codes: {missing_epcs}")
        
        # Check totalEvents calculation
        for stat in epc_stats:
            total_calculated = (
                stat["jumpCount"] + 
                stat["evtOrderErrCount"] + 
                stat["epcFakeCount"] + 
                stat["epcDupCount"] + 
                stat["locErrCount"]
            )
            print(f"EPC {stat['epcCode']}: totalEvents={stat['totalEvents']}, calculated={total_calculated}")
        
        # Check file stats
        file_stats = result.get("fileAnomalyStats", {})
        file_total_calculated = (
            file_stats.get("jumpCount", 0) +
            file_stats.get("evtOrderErrCount", 0) +
            file_stats.get("epcFakeCount", 0) +
            file_stats.get("epcDupCount", 0) +
            file_stats.get("locErrCount", 0)
        )
        print(f"File totalEvents: {file_stats.get('totalEvents', 0)}, calculated: {file_total_calculated}")
        
        print("\n=== VALIDATION ===")
        print(f"[OK] Null values removed: {not has_null_values}")
        print(f"[OK] All EPC codes included: {len(missing_epcs) == 0}")
        print(f"[OK] Correct totalEvents calculation: {all(stat['totalEvents'] == (stat['jumpCount'] + stat['evtOrderErrCount'] + stat['epcFakeCount'] + stat['epcDupCount'] + stat['locErrCount']) for stat in epc_stats)}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backend_format()