# -*- coding: utf-8 -*-
"""
Test current multi-anomaly detector output to show complete JSON format
"""

import json
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from barcode.multi_anomaly_detector import detect_anomalies_from_json_enhanced

def test_current_output():
    """Show complete current output format"""
    
    # Test data with multiple anomaly types
    test_data = {
        "data": [
            # EPC with multiple issues
            {
                "scan_location": "서울 공장",
                "location_id": 1,
                "business_step": "Factory",
                "event_type": "Outbound",
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "product_name": "Product A",
                "event_time": "2024-07-02 09:00:00"
            },
            {
                "scan_location": "경기 창고",
                "location_id": 2,
                "business_step": "WMS",
                "event_type": "Inbound",
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "product_name": "Product A",
                "event_time": "2024-07-02 11:00:00"
            },
            {
                "scan_location": "서울 도매상",
                "location_id": 3,
                "business_step": "W_Stock",
                "event_type": "Inbound",
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "product_name": "Product A",
                "event_time": "2024-07-03 09:30:00"
            },
            {
                "scan_location": "서울 소매상",
                "location_id": 4,
                "business_step": "R_Stock",
                "event_type": "Inbound",
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "product_name": "Product A",
                "event_time": "2024-07-03 13:00:00"
            },
            # Fake EPC
            {
                "scan_location": "부산 공장",
                "location_id": 5,
                "business_step": "Factory",
                "event_type": "Outbound",
                "epc_code": "INVALID.FORMAT.WRONG",
                "product_name": "Product B",
                "event_time": "2024-07-02 10:00:00"
            },
            # Duplicate scan (same time, different location)
            {
                "scan_location": "인천공장창고",
                "location_id": 6,
                "business_step": "Factory",
                "event_type": "Outbound",
                "epc_code": "001.8804823.0000002.000002.20240701.000000002",
                "product_name": "Product C",
                "event_time": "2024-07-02 15:00:00"
            },
            {
                "scan_location": "수도권물류센터",
                "location_id": 7,
                "business_step": "Logistics",
                "event_type": "Inbound",
                "epc_code": "001.8804823.0000002.000002.20240701.000000002",
                "product_name": "Product C",
                "event_time": "2024-07-02 15:00:00"  # Same time = duplicate
            }
        ]
    }
    
    print("=== 현재 멀티 이상 탐지 시스템 출력 ===")
    
    # Convert to JSON string
    test_json = json.dumps(test_data, ensure_ascii=False)
    
    # Get result
    result = detect_anomalies_from_json_enhanced(test_json)
    
    print("완전한 JSON 결과:")
    print(result)

if __name__ == "__main__":
    test_current_output()