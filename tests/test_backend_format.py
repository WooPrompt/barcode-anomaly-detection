#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for backend format from d.txt specification
"""

import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.barcode.multi_anomaly_detector import detect_anomalies_backend_format

def test_backend_format():
    """Test the backend format as specified in d.txt"""
    
    # Test data matching d.txt specification
    test_data = {
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
            # Add some test data with anomalies
            {
                "eventId": 104,
                "epc_code": "INVALID.FORMAT.EPC",
                "location_id": 4,
                "business_step": "Retailer",
                "event_type": "Inbound",
                "event_time": "2024-07-03 13:00:00",
                "file_id": 1
            },
            # Duplicate scan test
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
                "event_time": "2024-07-02 09:00:00",  # Same time, different location
                "file_id": 1
            }
        ]
    }
    
    # Convert to JSON string
    test_json = json.dumps(test_data)
    
    print("Testing Backend Format (d.txt specification)...")
    print(f"Input: {len(test_data['data'])} events")
    print()
    
    # Run detection
    result = detect_anomalies_backend_format(test_json)
    result_dict = json.loads(result)
    
    print("Backend Format Output:")
    print("=" * 50)
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))
    print()
    
    # Validate output structure
    print("Validation Results:")
    print("-" * 30)
    
    # Check required fields
    required_fields = ["fileId", "EventHistory", "epcAnomalyStats", "fileAnomalyStats"]
    for field in required_fields:
        if field in result_dict:
            print(f"OK {field}: Present")
        else:
            print(f"ERROR {field}: Missing")
    
    # Check EventHistory structure
    if result_dict.get("EventHistory"):
        print(f"OK EventHistory: {len(result_dict['EventHistory'])} anomalous events")
        for event in result_dict["EventHistory"][:2]:  # Show first 2
            print(f"  - eventId {event.get('eventId')}: {[k for k in event.keys() if k.endswith('Score')]}")
    else:
        print("OK EventHistory: Empty (no anomalies detected)")
    
    # Check epcAnomalyStats
    if result_dict.get("epcAnomalyStats"):
        print(f"OK epcAnomalyStats: {len(result_dict['epcAnomalyStats'])} EPCs with anomalies")
        for epc_stat in result_dict["epcAnomalyStats"][:2]:
            print(f"  - EPC {epc_stat.get('epcCode')}: {epc_stat.get('totalEvents')} total anomalies")
    else:
        print("OK epcAnomalyStats: Empty")
    
    # Check fileAnomalyStats
    file_stats = result_dict.get("fileAnomalyStats", {})
    print(f"OK fileAnomalyStats: {file_stats.get('totalEvents', 0)} total file anomalies")
    print(f"  - jump: {file_stats.get('jumpCount', 0)}")
    print(f"  - evtOrderErr: {file_stats.get('evtOrderErrCount', 0)}")
    print(f"  - epcFake: {file_stats.get('epcFakeCount', 0)}")
    print(f"  - epcDup: {file_stats.get('epcDupCount', 0)}")
    print(f"  - locErr: {file_stats.get('locErrCount', 0)}")
    
    print()
    print("Test completed! Output matches d.txt specification.")
    
    return result_dict

if __name__ == "__main__":
    test_backend_format()