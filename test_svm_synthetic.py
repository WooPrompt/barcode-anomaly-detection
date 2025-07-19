#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta

# Add project path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

def create_synthetic_data_with_anomalies(n_samples=5000):
    """Create synthetic supply chain data with known anomalies like professor's code"""
    
    print(f"Creating {n_samples} synthetic supply chain events...")
    
    # Base parameters
    n_epcs = n_samples // 10  # Each EPC has ~10 events
    base_time = datetime(2025, 7, 1, 10, 0, 0)
    
    events = []
    
    for epc_idx in range(n_epcs):
        epc_code = f"001.8805843.2932031.{epc_idx:06d}.20250701.000000001"
        
        # Normal supply chain sequence
        locations = [
            (1, "인천공장", "Factory", "Aggregation"),
            (5, "인천공장창고", "WMS", "WMS_Inbound"), 
            (13, "수도권물류센터", "Logistics_HUB", "HUB_Inbound"),
            (23, "수도권_도매상1", "W_Stock_Inbound", "W_Stock_Inbound"),
            (53, "수도권_도매상1_권역_소매상1", "R_Stock_Inbound", "R_Stock_Inbound")
        ]
        
        current_time = base_time + timedelta(hours=epc_idx * 0.1)  # Stagger start times
        
        for i, (location_id, scan_location, business_step, event_type) in enumerate(locations):
            # Normal progression - add some random time between events
            if i > 0:
                current_time += timedelta(hours=np.random.normal(5, 1))  # 5±1 hours between steps
            
            event = {
                "eventId": len(events) + 1,
                "epc_code": epc_code,
                "location_id": location_id,
                "scan_location": scan_location,
                "business_step": business_step,
                "event_type": event_type,
                "event_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_id": 1,
                "is_anomaly": 0  # Normal event
            }
            events.append(event)
    
    # Now inject anomalies (like professor's 10% rate)
    n_anomalies = int(len(events) * 0.1)  # 10% anomalies
    anomaly_indices = np.random.choice(len(events), n_anomalies, replace=False)
    
    print(f"Injecting {n_anomalies} anomalies into {len(events)} events...")
    
    for idx in anomaly_indices:
        event = events[idx]
        anomaly_type = np.random.choice(['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump'])
        
        if anomaly_type == 'epcFake':
            # Make EPC format invalid
            event['epc_code'] = "INVALID_EPC_FORMAT_" + str(idx)
            
        elif anomaly_type == 'epcDup':
            # Create duplicate event at same time
            if idx < len(events) - 1:
                events[idx + 1]['event_time'] = event['event_time']  # Same timestamp
                events[idx + 1]['is_anomaly'] = 1
                
        elif anomaly_type == 'locErr':
            # Create hierarchy violation (go backwards)
            if event['business_step'] == 'WMS':
                event['business_step'] = 'Factory'  # Go backwards
                event['location_id'] = 1
                event['scan_location'] = "인천공장"
                
        elif anomaly_type == 'evtOrderErr':
            # Create time ordering error
            if idx > 0:
                prev_time = datetime.strptime(events[idx-1]['event_time'], "%Y-%m-%d %H:%M:%S")
                event['event_time'] = (prev_time - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
                
        elif anomaly_type == 'jump':
            # Create impossible time jump (too fast travel)
            if idx > 0:
                prev_time = datetime.strptime(events[idx-1]['event_time'], "%Y-%m-%d %H:%M:%S")
                event['event_time'] = (prev_time + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
                # Keep different locations to make it impossible travel
        
        event['is_anomaly'] = 1
        event['anomaly_type'] = anomaly_type
    
    df = pd.DataFrame(events)
    
    print(f"Created synthetic dataset:")
    print(f"  Total events: {len(df)}")
    print(f"  Normal events: {len(df[df['is_anomaly'] == 0])}")
    print(f"  Anomaly events: {len(df[df['is_anomaly'] == 1])}")
    print(f"  Anomaly rate: {len(df[df['is_anomaly'] == 1]) / len(df) * 100:.1f}%")
    
    return df

def test_svm_on_synthetic_data():
    """Test SVM anomaly detection on synthetic data"""
    
    print("=== TESTING SVM ON SYNTHETIC DATA ===")
    
    # Create synthetic data
    synthetic_df = create_synthetic_data_with_anomalies(2000)
    
    # Save synthetic data for inspection
    output_dir = "data/analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    synthetic_df.to_csv(os.path.join(output_dir, "synthetic_data.csv"), index=False)
    print(f"Synthetic data saved to {output_dir}/synthetic_data.csv")
    
    # Test 1: Rule-based detection on synthetic data
    print("\n--- Testing Rule-based Detection ---")
    try:
        from src.barcode.multi_anomaly_detector import detect_anomalies_backend_format
        
        # Prepare sample for rule-based detection
        test_sample = synthetic_df.head(500).copy()
        test_sample['eventId'] = range(1, len(test_sample) + 1)
        
        # Remove scan_location to avoid encoding issues
        test_sample_clean = test_sample.drop('scan_location', axis=1)
        
        detection_input = {"data": test_sample_clean.to_dict('records')}
        detection_json = json.dumps(detection_input)
        
        rule_results = detect_anomalies_backend_format(detection_json)
        if isinstance(rule_results, str):
            rule_results = json.loads(rule_results)
        
        if 'error' in rule_results:
            print(f"Rule-based detection error: {rule_results['error']}")
        else:
            event_history = rule_results.get('EventHistory', [])
            file_stats = rule_results.get('fileAnomalyStats', {})
            
            print(f"Rule-based results:")
            print(f"  Events with anomalies: {len(event_history)}")
            print(f"  File statistics: {file_stats}")
            
            # Compare with ground truth
            actual_anomalies = len(test_sample[test_sample['is_anomaly'] == 1])
            detected_anomalies = len(event_history)
            print(f"  Ground truth anomalies: {actual_anomalies}")
            print(f"  Detected anomalies: {detected_anomalies}")
            
    except Exception as e:
        print(f"Rule-based detection failed: {e}")
    
    # Test 2: SVM detection on synthetic data
    print("\n--- Testing SVM Detection ---")
    try:
        from src.barcode.svm_anomaly_detector import SVMAnomalyDetector
        
        # Create SVM detector
        svm_detector = SVMAnomalyDetector()
        
        # Check if models are trained
        models_loaded = svm_detector.load_models()
        print(f"SVM models loaded: {models_loaded}")
        
        if not models_loaded:
            print("No trained SVM models found. Need to train first.")
            
            # Quick training on synthetic data
            print("Training SVM on synthetic data...")
            
            # Convert synthetic data to training format
            training_json_list = []
            for chunk_start in range(0, len(synthetic_df), 100):
                chunk_end = min(chunk_start + 100, len(synthetic_df))
                chunk_df = synthetic_df.iloc[chunk_start:chunk_end].copy()
                chunk_df['eventId'] = range(1, len(chunk_df) + 1)
                
                chunk_data = {"data": chunk_df.to_dict('records')}
                training_json_list.append(json.dumps(chunk_data))
            
            # Train models
            training_results = svm_detector.train_models(json_data_list=training_json_list[:10])  # Use first 10 chunks
            print(f"Training results: {training_results}")
        
        # Test SVM prediction
        test_sample = synthetic_df.head(200).copy()
        test_sample['eventId'] = range(1, len(test_sample) + 1)
        
        test_input = {"data": test_sample.to_dict('records')}
        test_json = json.dumps(test_input)
        
        svm_results = svm_detector.predict_anomalies(test_json)
        if isinstance(svm_results, str):
            svm_results = json.loads(svm_results)
        
        if 'error' in svm_results:
            print(f"SVM detection error: {svm_results['error']}")
        else:
            event_history = svm_results.get('EventHistory', [])
            file_stats = svm_results.get('fileAnomalyStats', {})
            
            print(f"SVM results:")
            print(f"  Events with anomalies: {len(event_history)}")
            print(f"  File statistics: {file_stats}")
            
            # Compare with ground truth
            actual_anomalies = len(test_sample[test_sample['is_anomaly'] == 1])
            detected_anomalies = len(event_history)
            print(f"  Ground truth anomalies: {actual_anomalies}")
            print(f"  Detected anomalies: {detected_anomalies}")
            
            if len(event_history) > 0:
                print(f"  Sample detections:")
                for event in event_history[:3]:
                    print(f"    Event {event.get('eventId')}: {event}")
        
    except Exception as e:
        print(f"SVM detection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_svm_on_synthetic_data()