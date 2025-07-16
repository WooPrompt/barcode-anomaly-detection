# -*- coding: utf-8 -*-
"""
Test script for simplified JSON input format (without geo_data and transition_stats)
The system should automatically load data from CSV files.
"""

import json
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from barcode.multi_anomaly_detector import detect_anomalies_from_json_enhanced

def test_simplified_input():
    """Test with the exact JSON format specified by the user"""
    
    # Exact format from prompts/command.txt
    test_data = {
        "data": [
            {
                "scan_location": "서울 공장",
                "location_id": 1,
                "hub_type": "HWS_Factory",
                "business_step": "Factory",
                "event_type": "Outbound",
                "operator_id": 1,
                "device_id": 1,
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "epc_header": "001",
                "epc_company": "8804823",
                "epc_product": "0000001",
                "epc_lot": "000001",
                "epc_manufacture": "20240701",
                "epc_serial": "000000001",
                "product_name": "Product A",
                "event_time": "2024-07-02 09:00:00",
                "manufacture_date": "2024-07-01 00:00:00",
                "expiry_date": "20251231"
            },
            {
                "scan_location": "경기 창고",
                "location_id": 2,
                "hub_type": "GG_Warehouse",
                "business_step": "WMS",
                "event_type": "Inbound",
                "operator_id": 2,
                "device_id": 2,
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "epc_header": "001",
                "epc_company": "8804823",
                "epc_product": "0000001",
                "epc_lot": "000001",
                "epc_manufacture": "20240701",
                "epc_serial": "000000001",
                "product_name": "Product A",
                "event_time": "2024-07-02 11:00:00",
                "manufacture_date": "2024-07-01 00:00:00",
                "expiry_date": "20251231"
            },
            {
                "scan_location": "서울 도매상",
                "location_id": 3,
                "hub_type": "SEL_Wholesaler",
                "business_step": "W_Stock",
                "event_type": "Inbound",
                "operator_id": 3,
                "device_id": 3,
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "epc_header": "001",
                "epc_company": "8804823",
                "epc_product": "0000001",
                "epc_lot": "000001",
                "epc_manufacture": "20240701",
                "epc_serial": "000000001",
                "product_name": "Product A",
                "event_time": "2024-07-03 09:30:00",
                "manufacture_date": "2024-07-01 00:00:00",
                "expiry_date": "20251231"
            },
            {
                "scan_location": "서울 소매상",
                "location_id": 4,
                "hub_type": "SEL_Retailer",
                "business_step": "R_Stock",
                "event_type": "Inbound",
                "operator_id": 4,
                "device_id": 4,
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "epc_header": "001",
                "epc_company": "8804823",
                "epc_product": "0000001",
                "epc_lot": "000001",
                "epc_manufacture": "20240701",
                "epc_serial": "000000001",
                "product_name": "Product A",
                "event_time": "2024-07-03 13:00:00",
                "manufacture_date": "2024-07-01 00:00:00",
                "expiry_date": "20251231"
            }
        ]
    }
    
    print("Testing simplified JSON input format...")
    print("Input contains only 'data' array - geo_data and transition_stats should be loaded from CSV")
    print()
    
    # Convert to JSON string
    test_json = json.dumps(test_data, ensure_ascii=False, indent=2)
    
    # Test the detection
    result = detect_anomalies_from_json_enhanced(test_json)
    
    print("Detection Result:")
    print(result)
    
    # Parse and validate result format
    try:
        result_data = json.loads(result)
        print("\n=== Validation ===")
        print(f"JSON parsing successful")
        print(f"EventHistory entries: {len(result_data.get('EventHistory', []))}")
        print(f"Summary stats: {result_data.get('summaryStats', {})}")
        print(f"Multi-anomaly count: {result_data.get('multiAnomalyCount', 0)}")
        
        # Check if any anomalies were detected
        if result_data.get('EventHistory'):
            for anomaly in result_data['EventHistory']:
                print(f"\n--- Detected Anomaly ---")
                print(f"EPC: {anomaly.get('epcCode')}")
                print(f"Event Type: {anomaly.get('eventType')}")
                print(f"Anomaly Type: {anomaly.get('anomalyType')}")
                print(f"Description: {anomaly.get('description')}")
                print(f"Scores: {anomaly.get('anomalyScores', {})}")
        else:
            print("No anomalies detected with this normal data")
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed: {e}")

if __name__ == "__main__":
    test_simplified_input()