#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import random

# Add project path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(current_dir)
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

def load_real_supply_chain_data():
    """Load real geolocation and transition data for realistic synthetic generation"""
    
    # Load geolocation data
    try:
        geo_df = pd.read_csv('data/processed/location_id_withGeospatial.csv', encoding='utf-8')
        print(f"Loaded {len(geo_df)} locations with coordinates")
    except:
        # Fallback locations if file has encoding issues
        geo_df = pd.DataFrame({
            'location_id': [1, 5, 13, 23, 53],
            'scan_location': ['인천공장', '인천공장창고', '수도권물류센터', '수도권_도매상1', '수도권_도매상1_권역_소매상1'],
            'Latitude': [37.45, 37.46, 37.35, 37.55, 37.555],
            'Longitude': [126.65, 126.66, 127.2, 127.05, 127.055]
        })
        print(f"Using fallback geolocation data: {len(geo_df)} locations")
    
    # Load transition statistics
    try:
        trans_df = pd.read_csv('data/processed/business_step_transition_avg_v2.csv', encoding='utf-8')
        print(f"Loaded {len(trans_df)} transition patterns")
    except:
        # Fallback transition data
        trans_df = pd.DataFrame({
            'from_scan_location': ['인천공장', '인천공장창고', '수도권물류센터', '수도권_도매상1'],
            'to_scan_location': ['인천공장창고', '수도권물류센터', '수도권_도매상1', '수도권_도매상1_권역_소매상1'],
            'time_taken_hours_mean': [5.5, 5.8, 5.3, 5.2],
            'time_taken_hours_std': [2.3, 2.4, 2.7, 2.6]
        })
        print(f"Using fallback transition data: {len(trans_df)} transitions")
    
    return geo_df, trans_df

def calculate_distance_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def create_realistic_synthetic_data(n_samples=5000):
    """Create realistic synthetic supply chain data that matches detection logic"""
    
    print(f"Creating {n_samples} realistic synthetic supply chain events...")
    
    # Load real data for realistic patterns
    geo_df, trans_df = load_real_supply_chain_data()
    
    # Define realistic supply chain sequence
    supply_chain_sequence = [
        {'location_id': 1, 'scan_location': '인천공장', 'business_step': 'Factory', 'event_type': 'Aggregation'},
        {'location_id': 5, 'scan_location': '인천공장창고', 'business_step': 'WMS', 'event_type': 'WMS_Inbound'},
        {'location_id': 13, 'scan_location': '수도권물류센터', 'business_step': 'Logistics_HUB', 'event_type': 'HUB_Inbound'},
        {'location_id': 23, 'scan_location': '수도권_도매상1', 'business_step': 'W_Stock_Inbound', 'event_type': 'W_Stock_Inbound'},
        {'location_id': 53, 'scan_location': '수도권_도매상1_권역_소매상1', 'business_step': 'R_Stock_Inbound', 'event_type': 'R_Stock_Inbound'}
    ]
    
    # Valid EPC companies and headers
    valid_companies = ["8804823", "8805843", "8809437"]
    valid_header = "001"
    
    n_epcs = n_samples // len(supply_chain_sequence)
    base_time = datetime(2025, 7, 1, 10, 0, 0)
    
    events = []
    
    print(f"Creating {n_epcs} EPC sequences...")
    
    for epc_idx in range(n_epcs):
        # Create valid EPC format matching real data structure
        company = random.choice(valid_companies)
        product_code = f"{random.randint(1000000, 9999999):07d}"
        lot_code = f"{random.randint(100000, 999999):06d}"  # 6 digits as expected
        manufacture_date = "20250701"
        serial = f"{epc_idx+1:09d}"
        
        epc_code = f"{valid_header}.{company}.{product_code}.{lot_code}.{manufacture_date}.{serial}"
        
        # Start time for this EPC (stagger starts)
        current_time = base_time + timedelta(hours=epc_idx * 0.05)
        
        # Create normal sequence
        for step_idx, step in enumerate(supply_chain_sequence):
            # Add realistic time progression based on transition statistics
            if step_idx > 0:
                # Get expected transition time
                prev_location = supply_chain_sequence[step_idx-1]['scan_location']
                curr_location = step['scan_location']
                
                # Find transition time from real data
                transition = trans_df[
                    (trans_df['from_scan_location'] == prev_location) & 
                    (trans_df['to_scan_location'] == curr_location)
                ]
                
                if not transition.empty:
                    mean_hours = transition.iloc[0]['time_taken_hours_mean']
                    std_hours = transition.iloc[0]['time_taken_hours_std']
                    # Add realistic variation
                    actual_hours = max(1.0, np.random.normal(mean_hours, std_hours))
                else:
                    # Fallback to reasonable default
                    actual_hours = np.random.normal(5.0, 2.0)
                    actual_hours = max(1.0, actual_hours)
                
                current_time += timedelta(hours=actual_hours)
            
            event = {
                "eventId": len(events) + 1,
                "epc_code": epc_code,
                "location_id": step['location_id'],
                "scan_location": step['scan_location'],
                "business_step": step['business_step'],
                "event_type": step['event_type'],
                "event_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "file_id": 1,
                "is_anomaly": 0,
                "anomaly_type": "normal"
            }
            events.append(event)
    
    print(f"Created {len(events)} normal events")
    
    # Now inject realistic anomalies (10% rate)
    n_anomalies = int(len(events) * 0.1)
    print(f"Injecting {n_anomalies} realistic anomalies...")
    
    anomaly_count = {
        'epcFake': 0,
        'epcDup': 0, 
        'locErr': 0,
        'evtOrderErr': 0,
        'jump': 0
    }
    
    # Create each type of anomaly with specific logic
    anomalies_per_type = n_anomalies // 5
    
    # 1. EpcFake anomalies - Invalid EPC formats
    print("Creating epcFake anomalies...")
    for _ in range(anomalies_per_type):
        event_idx = random.randint(0, len(events) - 1)
        event = events[event_idx]
        
        # Create various types of invalid EPCs
        anomaly_type = random.choice([
            'invalid_header',
            'invalid_company', 
            'malformed_structure',
            'missing_parts'
        ])
        
        if anomaly_type == 'invalid_header':
            event['epc_code'] = event['epc_code'].replace('001.', '999.')
        elif anomaly_type == 'invalid_company':
            event['epc_code'] = event['epc_code'].replace('.8805843.', '.1234567.')
        elif anomaly_type == 'malformed_structure':
            event['epc_code'] = 'INVALID_FORMAT_' + str(event_idx)
        elif anomaly_type == 'missing_parts':
            parts = event['epc_code'].split('.')
            event['epc_code'] = '.'.join(parts[:3])  # Remove last parts
        
        event['is_anomaly'] = 1
        event['anomaly_type'] = 'epcFake'
        anomaly_count['epcFake'] += 1
    
    # 2. EpcDup anomalies - Same timestamp duplicates (not factory-warehouse)
    print("Creating epcDup anomalies...")
    for _ in range(anomalies_per_type):
        # Find events with same EPC but different locations
        epc_groups = {}
        for i, event in enumerate(events):
            epc = event['epc_code']
            if epc not in epc_groups:
                epc_groups[epc] = []
            epc_groups[epc].append((i, event))
        
        # Pick an EPC with multiple events
        epc_with_multiple = [epc for epc, evts in epc_groups.items() if len(evts) > 1]
        if not epc_with_multiple:
            continue
            
        selected_epc = random.choice(epc_with_multiple)
        epc_events = epc_groups[selected_epc]
        
        if len(epc_events) >= 2:
            # Make two events have same timestamp at DIFFERENT locations (not factory-warehouse)
            idx1, event1 = random.choice(epc_events)
            idx2, event2 = random.choice(epc_events)
            
            # Ensure they're not factory-warehouse pair
            if not ((event1['business_step'] == 'Factory' and event2['business_step'] == 'WMS') or
                    (event1['business_step'] == 'WMS' and event2['business_step'] == 'Factory')):
                events[idx2]['event_time'] = event1['event_time']
                events[idx1]['is_anomaly'] = 1
                events[idx2]['is_anomaly'] = 1
                events[idx1]['anomaly_type'] = 'epcDup'
                events[idx2]['anomaly_type'] = 'epcDup'
                anomaly_count['epcDup'] += 2
    
    # 3. LocErr anomalies - Hierarchy violations
    print("Creating locErr anomalies...")
    for _ in range(anomalies_per_type):
        event_idx = random.randint(1, len(events) - 1)
        event = events[event_idx]
        
        # Create hierarchy violation
        violation_type = random.choice([
            'reverse_hierarchy',
            'skip_level',
            'impossible_transition'
        ])
        
        if violation_type == 'reverse_hierarchy':
            # WMS -> Factory (backwards)
            if event['business_step'] == 'WMS':
                event['business_step'] = 'Factory'
                event['location_id'] = 1
                event['scan_location'] = '인천공장'
        elif violation_type == 'skip_level':
            # Factory -> Retailer (skip WMS and Wholesaler)
            if event['business_step'] == 'Factory':
                event['business_step'] = 'R_Stock_Inbound'
                event['location_id'] = 53
                event['scan_location'] = '수도권_도매상1_권역_소매상1'
        elif violation_type == 'impossible_transition':
            # WMS -> Retailer (skip Wholesaler)
            if event['business_step'] == 'WMS':
                event['business_step'] = 'R_Stock_Inbound'
                event['location_id'] = 53
                event['scan_location'] = '수도권_도매상1_권역_소매상1'
        
        event['is_anomaly'] = 1
        event['anomaly_type'] = 'locErr'
        anomaly_count['locErr'] += 1
    
    # 4. EvtOrderErr anomalies - Time ordering violations
    print("Creating evtOrderErr anomalies...")
    for _ in range(anomalies_per_type):
        event_idx = random.randint(1, len(events) - 1)
        event = events[event_idx]
        prev_event = events[event_idx - 1]
        
        # Only create time reversal for same EPC
        if event['epc_code'] == prev_event['epc_code']:
            # Make current event happen BEFORE previous event
            prev_time = datetime.strptime(prev_event['event_time'], "%Y-%m-%d %H:%M:%S")
            # Go back 1-12 hours
            hours_back = random.randint(1, 12)
            new_time = prev_time - timedelta(hours=hours_back)
            event['event_time'] = new_time.strftime("%Y-%m-%d %H:%M:%S")
            
            event['is_anomaly'] = 1
            event['anomaly_type'] = 'evtOrderErr'
            anomaly_count['evtOrderErr'] += 1
    
    # 5. Jump anomalies - Impossible space-time travel
    print("Creating jump anomalies...")
    for _ in range(anomalies_per_type):
        event_idx = random.randint(1, len(events) - 1)
        event = events[event_idx]
        prev_event = events[event_idx - 1]
        
        # Only create jumps for same EPC with different locations
        if (event['epc_code'] == prev_event['epc_code'] and 
            event['location_id'] != prev_event['location_id']):
            
            # Get geographic coordinates
            prev_geo = geo_df[geo_df['location_id'] == prev_event['location_id']]
            curr_geo = geo_df[geo_df['location_id'] == event['location_id']]
            
            if not prev_geo.empty and not curr_geo.empty:
                # Calculate actual distance
                prev_lat, prev_lon = prev_geo.iloc[0]['Latitude'], prev_geo.iloc[0]['Longitude']
                curr_lat, curr_lon = curr_geo.iloc[0]['Latitude'], curr_geo.iloc[0]['Longitude']
                
                distance_km = calculate_distance_km(prev_lat, prev_lon, curr_lat, curr_lon)
                
                # Create impossible travel time (need at least distance/100 hours for 100km/h)
                min_required_hours = distance_km / 100.0
                impossible_hours = min_required_hours * random.uniform(0.1, 0.8)  # 10-80% of required time
                
                prev_time = datetime.strptime(prev_event['event_time'], "%Y-%m-%d %H:%M:%S")
                new_time = prev_time + timedelta(hours=impossible_hours)
                event['event_time'] = new_time.strftime("%Y-%m-%d %H:%M:%S")
                
                event['is_anomaly'] = 1
                event['anomaly_type'] = 'jump'
                anomaly_count['jump'] += 1
    
    df = pd.DataFrame(events)
    
    print(f"\n=== REALISTIC SYNTHETIC DATA CREATED ===")
    print(f"Total events: {len(df)}")
    print(f"Normal events: {len(df[df['is_anomaly'] == 0])}")
    print(f"Anomaly events: {len(df[df['is_anomaly'] == 1])}")
    print(f"Anomaly rate: {len(df[df['is_anomaly'] == 1]) / len(df) * 100:.1f}%")
    
    print(f"\nAnomaly breakdown:")
    for anomaly_type, count in anomaly_count.items():
        print(f"  {anomaly_type}: {count}")
    
    return df

def test_realistic_synthetic_data():
    """Test both detection systems on realistic synthetic data"""
    
    print("=== TESTING REALISTIC SYNTHETIC DATA ===")
    
    # Create realistic synthetic data
    synthetic_df = create_realistic_synthetic_data(2000)
    
    # Save synthetic data
    output_dir = "data/analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    synthetic_df.to_csv(os.path.join(output_dir, "realistic_synthetic_data.csv"), index=False)
    print(f"Realistic synthetic data saved to {output_dir}/realistic_synthetic_data.csv")
    
    # Test rule-based detection
    print("\n--- Testing Rule-based Detection on Realistic Data ---")
    try:
        from src.barcode.multi_anomaly_detector import detect_anomalies_backend_format
        
        # Test on sample
        test_sample = synthetic_df.head(1000).copy()
        test_sample['eventId'] = range(1, len(test_sample) + 1)
        
        # Remove problematic columns
        test_sample_clean = test_sample.drop(['scan_location', 'is_anomaly', 'anomaly_type'], axis=1, errors='ignore')
        
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
            detection_rate = (detected_anomalies / actual_anomalies * 100) if actual_anomalies > 0 else 0
            
            print(f"  Ground truth anomalies: {actual_anomalies}")
            print(f"  Detected anomalies: {detected_anomalies}")
            print(f"  Detection rate: {detection_rate:.1f}%")
            
            if len(event_history) > 0:
                print(f"  Sample detections:")
                for event in event_history[:5]:
                    anomaly_types = [k for k, v in event.items() if k != 'eventId' and v is True]
                    print(f"    Event {event.get('eventId')}: {', '.join(anomaly_types)}")
        
    except Exception as e:
        print(f"Rule-based detection failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_realistic_synthetic_data()