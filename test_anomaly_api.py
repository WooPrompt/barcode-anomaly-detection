#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Enhanced Multi-Anomaly Detection API
Tests the new JSON output format with sample data
"""

import json
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'barcode'))

from multi_anomaly_detector import detect_anomalies_from_json_enhanced

def test_anomaly_detection():
    """
    Test the enhanced anomaly detection with sample data.
    Expected anomalies in test_data_sample.json:
    1. epcDup: EPC 001.xxx.001 appears simultaneously at 구미공장 and 인천공장
    2. epcFake: "invalid.fake.epc.format.wrong.123456789" has wrong format
    3. jump: 구미공장 → 서울물류센터 in 0.5 hours (expected 4±0.5 hours)
    4. locErr: EPC goes from 강릉소매상 (level 3) back to 구미공장 (level 0)
    """
    
    print("Testing Enhanced Multi-Anomaly Detection System")
    print("=" * 60)
    
    # Load test data
    try:
        with open('test_data_sample.json', 'r', encoding='utf-8') as f:
            test_data = f.read()
        print("Test data loaded successfully")
    except FileNotFoundError:
        print("test_data_sample.json not found!")
        return
    
    # Run detection
    try:
        result = detect_anomalies_from_json_enhanced(test_data)
        result_dict = json.loads(result)
        print("Anomaly detection completed successfully")
    except Exception as e:
        print(f"Error during detection: {e}")
        return
    
    # Display results
    print("\nDETECTION RESULTS:")
    print("-" * 40)
    
    event_history = result_dict.get('EventHistory', [])
    summary_stats = result_dict.get('summaryStats', {})
    multi_anomaly_count = result_dict.get('multiAnomalyCount', 0)
    
    print(f"Total anomalies found: {len(event_history)}")
    print(f"Multi-anomaly EPCs: {multi_anomaly_count}")
    print(f"Summary: {summary_stats}")
    
    print("\nDETAILED ANOMALY ANALYSIS:")
    print("-" * 40)
    
    for i, anomaly in enumerate(event_history, 1):
        print(f"\n{i}. EPC: {anomaly['epcCode']}")
        print(f"   Location: {anomaly['scanLocation']}")
        print(f"   Anomaly Types: {anomaly['anomalyTypes']}")
        print(f"   Scores: {anomaly['anomalyScores']}")
        print(f"   Sequence Position: {anomaly['sequencePosition']}/{anomaly['totalSequenceLength']}")
        print(f"   Primary Issue: {anomaly['primaryAnomaly']}")
        print(f"   Description: {anomaly['description']}")
    
    # Test specific expected anomalies
    print("\nVALIDATION CHECKS:")
    print("-" * 40)
    
    # Check for expected anomaly types
    all_anomaly_types = set()
    for anomaly in event_history:
        all_anomaly_types.update(anomaly['anomalyTypes'])
    
    expected_types = {'epcDup', 'epcFake', 'jump', 'locErr'}
    found_types = all_anomaly_types & expected_types
    
    print(f"Expected anomaly types: {expected_types}")
    print(f"Found anomaly types: {found_types}")
    
    if found_types == expected_types:
        print("All expected anomaly types detected!")
    else:
        missing = expected_types - found_types
        print(f"Missing anomaly types: {missing}")
    
    # Check multi-anomaly detection
    multi_anomaly_epcs = [a for a in event_history if len(a['anomalyTypes']) > 1]
    print(f"\nMulti-anomaly EPCs found: {len(multi_anomaly_epcs)}")
    
    for epc in multi_anomaly_epcs:
        print(f"   {epc['epcCode']}: {epc['anomalyTypes']}")
    
    print("\n" + "=" * 60)
    print("Test completed! Check results above.")
    
    # Save results for inspection
    with open('test_results.json', 'w', encoding='utf-8') as f:
        f.write(result)
    print("Results saved to test_results.json")

if __name__ == '__main__':
    test_anomaly_detection()