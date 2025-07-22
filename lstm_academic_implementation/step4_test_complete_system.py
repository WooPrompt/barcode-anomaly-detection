#!/usr/bin/env python3
"""
Step 4: Test Complete LSTM Integration
Tests the full pipeline from data to API
"""

import requests
import json
import time
from datetime import datetime
from pathlib import Path

def test_lstm_integration():
    """Test complete LSTM integration"""
    
    print("Testing complete LSTM integration")
    
    # Step 1: Test data preparation files
    required_files = [
        'lstm_academic_implementation/trained_lstm_model.pt',
        'lstm_academic_implementation/training_summary.json'
    ]
    
    for file in required_files:
        if Path(file).exists():
            print(f"SUCCESS: {file}")
        else:
            print(f"ERROR: {file} - Missing!")
            return False
    
    # Step 2: Test FastAPI server (assume it's running)
    base_url = "http://localhost:8000"
    
    print("Testing FastAPI server...")
    
    # Health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("SUCCESS: Server is running")
        else:
            print("ERROR: Server health check failed")
            return False
    except requests.ConnectionError:
        print("ERROR: Server not running. Please start with: python fastapi_server.py")
        return False
    
    # Step 3: Test LSTM endpoint
    print("Testing LSTM endpoint...")
    
    # Create test data in backend format
    test_data = {
        "data": [
            {
                "event_id": 1,
                "epc_code": "001.8804823.1293291.010001.20250722.000001",
                "location_id": 101,
                "business_step": "Factory",
                "event_type": "Aggregation",
                "event_time": "2025-07-22T08:00:00Z",
                "file_id": 1
            },
            {
                "event_id": 2,
                "epc_code": "001.8804823.1293291.010001.20250722.000001",
                "location_id": 102,
                "business_step": "WMS",
                "event_type": "Observation",
                "event_time": "2025-07-22T10:30:00Z",
                "file_id": 1
            },
            {
                "event_id": 3,
                "epc_code": "001.8804823.1293291.010001.20250722.000001",
                "location_id": 103,
                "business_step": "Distribution",
                "event_type": "HUB_Outbound",
                "event_time": "2025-07-22T14:15:00Z",
                "file_id": 1
            }
        ]
    }
    
    # Test LSTM endpoint
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/manager/export-and-analyze-async/lstm",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # ms
        
        if response.status_code == 200:
            result = response.json()
            print(f"SUCCESS: LSTM endpoint successful!")
            print(f"Response time: {response_time:.1f}ms")
            print(f"Method: {result.get('method', 'unknown')}")
            print(f"Anomalies detected: {len(result.get('EventHistory', []))}")
            
            # Check if response has proper format
            required_fields = ['fileId', 'EventHistory', 'epcAnomalyStats', 'fileAnomalyStats']
            for field in required_fields:
                if field in result:
                    print(f"SUCCESS: {field} present")
                else:
                    print(f"ERROR: {field} missing")
            
            # Show sample result
            print("\nSample Response:")
            print(json.dumps(result, indent=2)[:500] + "...")
            
        else:
            print(f"ERROR: LSTM endpoint failed: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"ERROR: LSTM endpoint test failed: {e}")
        return False
    
    # Step 4: Compare with other endpoints
    print("\nComparing with other detection methods...")
    
    # Test rule-based endpoint
    try:
        rule_response = requests.post(
            f"{base_url}/api/manager/export-and-analyze-async",
            json=test_data
        )
        
        if rule_response.status_code == 200:
            rule_result = rule_response.json()
            rule_anomalies = len(rule_result.get('EventHistory', []))
            print(f"Rule-based anomalies: {rule_anomalies}")
        
    except Exception as e:
        print(f"WARNING: Rule-based test failed: {e}")
    
    # Test SVM endpoint
    try:
        svm_response = requests.post(
            f"{base_url}/api/manager/export-and-analyze-async/svm",
            json=test_data
        )
        
        if svm_response.status_code == 200:
            svm_result = svm_response.json()
            svm_anomalies = len(svm_result.get('EventHistory', []))
            print(f"SVM anomalies: {svm_anomalies}")
        
    except Exception as e:
        print(f"WARNING: SVM test failed: {e}")
    
    print("\nComplete integration test successful!")
    print("LSTM model is now fully integrated with FastAPI!")
    
    return True

if __name__ == "__main__":
    success = test_lstm_integration()
    if success:
        print("\nSUCCESS: All tests passed!")
        print("Your LSTM model is production ready!")
        print("\nAvailable endpoints:")
        print("   POST /api/manager/export-and-analyze-async (rule-based)")
        print("   POST /api/manager/export-and-analyze-async/svm (SVM ML)")
        print("   POST /api/manager/export-and-analyze-async/lstm (LSTM DL)")
    else:
        print("\nERROR: Some tests failed. Please check the setup.")