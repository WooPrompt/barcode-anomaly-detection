#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test API integration with backend format
"""

import requests
import json

def test_api_integration():
    """Test the FastAPI server with backend format"""
    
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
                "epc_code": "INVALID.FORMAT.EPC",
                "location_id": 3,
                "business_step": "Wholesaler",
                "event_type": "Inbound",
                "event_time": "2024-07-03 09:30:00",
                "file_id": 1
            }
        ]
    }
    
    # Test API endpoint
    url = "http://localhost:8000/api/v1/barcode-anomaly-detect"
    
    try:
        print("Testing API integration...")
        print(f"POST {url}")
        
        response = requests.post(url, json=test_data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS: API returned expected format")
            print(f"FileId: {result.get('fileId')}")
            print(f"EventHistory entries: {len(result.get('EventHistory', []))}")
            print(f"EPC stats: {len(result.get('epcAnomalyStats', []))}")
            print(f"File stats total events: {result.get('fileAnomalyStats', {}).get('totalEvents', 0)}")
            
            # Test the export endpoint
            print("\nTesting export endpoint...")
            export_url = "http://localhost:8000/api/manager/export"
            export_response = requests.get(export_url, timeout=30)
            print(f"Export endpoint status: {export_response.status_code}")
            
            if export_response.status_code == 200:
                export_result = export_response.json()
                print(f"Export fileId: {export_result.get('fileId')}")
                print(f"Export EventHistory entries: {len(export_result.get('EventHistory', []))}")
            else:
                print(f"Export error: {export_response.text}")
            
        else:
            print(f"ERROR: API returned status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API server")
        print("Please run: python fastapi_server.py")
        print("Then run this test again")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    test_api_integration()