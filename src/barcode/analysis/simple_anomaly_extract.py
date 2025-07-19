import pandas as pd
import os
import sys
import json

# Add project path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

from src.barcode.multi_anomaly_detector import detect_anomalies_backend_format

def simple_anomaly_extract():
    """Simple extraction focusing on one file first"""
    
    raw_data_dir = os.path.join(project_root, "data", "raw")
    output_dir = os.path.join(project_root, "data", "analysis_output")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("SIMPLE ANOMALY EXTRACTION - ONE FILE")
    print("=" * 40)
    
    # Load just one file completely to start
    file_path = os.path.join(raw_data_dir, "hws.csv")
    print(f"Loading complete {file_path}...")
    
    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")
    
    # Clean data - remove rows with NaN in critical columns
    critical_cols = ['epc_code', 'location_id', 'business_step', 'event_type', 'event_time']
    before_clean = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"After cleaning: {len(df):,} rows (removed {before_clean - len(df):,} rows with NaN)")
    
    # Add required columns
    df['eventId'] = range(1, len(df) + 1)
    df['file_id'] = 1
    df['source_file'] = 'hws.csv'
    
    print(f"\nData overview:")
    print(f"Business steps: {df['business_step'].unique()}")
    print(f"Event types: {df['event_type'].unique()}")
    print(f"Location IDs: {sorted(df['location_id'].unique())}")
    print(f"Time range: {df['event_time'].min()} to {df['event_time'].max()}")
    
    # Test on smaller chunk first
    test_size = 10000
    test_df = df.head(test_size).copy()
    test_df['eventId'] = range(1, len(test_df) + 1)
    
    print(f"\nTesting on first {test_size:,} events...")
    
    detection_input = {"data": test_df.to_dict('records')}
    detection_json = json.dumps(detection_input)
    
    try:
        results_raw = detect_anomalies_backend_format(detection_json)
        
        if isinstance(results_raw, str):
            results = json.loads(results_raw)
        else:
            results = results_raw
        
        event_history = results.get('EventHistory', [])
        file_stats = results.get('fileAnomalyStats', {})
        
        print(f"\nTEST RESULTS:")
        print(f"Events with anomalies: {len(event_history)}")
        print(f"File statistics: {file_stats}")
        
        total_anomalies = sum(file_stats.get(f'{t}Count', 0) for t in ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump'])
        
        if total_anomalies > 0:
            print(f"SUCCESS! Found {total_anomalies} anomalies in test sample!")
            
            # Show sample anomalies
            print(f"\nSample anomalies:")
            for event in event_history[:3]:
                anomaly_types = [k for k, v in event.items() if k != 'eventId' and v is True]
                event_id = event['eventId']
                event_data = test_df[test_df['eventId'] == event_id].iloc[0]
                print(f"  Event {event_id}: {', '.join(anomaly_types)}")
                print(f"    EPC: {event_data['epc_code']}")
                print(f"    Location: {event_data['location_id']}")
                print(f"    Time: {event_data['event_time']}")
            
            # Now process full file
            print(f"\nProcessing full file with {len(df):,} events...")
            
            # Process in chunks to avoid memory issues
            chunk_size = 5000
            all_anomaly_events = []
            total_anomaly_count = 0
            
            for start in range(0, len(df), chunk_size):
                end = min(start + chunk_size, len(df))
                chunk_df = df.iloc[start:end].copy()
                chunk_df['eventId'] = range(1, len(chunk_df) + 1)
                
                chunk_input = {"data": chunk_df.to_dict('records')}
                chunk_json = json.dumps(chunk_input)
                
                try:
                    chunk_results_raw = detect_anomalies_backend_format(chunk_json)
                    if isinstance(chunk_results_raw, str):
                        chunk_results = json.loads(chunk_results_raw)
                    else:
                        chunk_results = chunk_results_raw
                    
                    chunk_history = chunk_results.get('EventHistory', [])
                    
                    # Map back to original indices
                    for event in chunk_history:
                        original_idx = start + event['eventId'] - 1
                        event['original_index'] = original_idx
                        all_anomaly_events.append(event)
                    
                    chunk_stats = chunk_results.get('fileAnomalyStats', {})
                    chunk_total = sum(chunk_stats.get(f'{t}Count', 0) for t in ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump'])
                    total_anomaly_count += chunk_total
                    
                    if chunk_total > 0:
                        print(f"  Chunk {start//chunk_size + 1}: {chunk_total} anomalies")
                    
                except Exception as e:
                    print(f"  Error in chunk {start//chunk_size + 1}: {e}")
                    continue
            
            print(f"\nFINAL RESULTS:")
            print(f"Total anomaly events: {len(all_anomaly_events)}")
            print(f"Total anomaly count: {total_anomaly_count}")
            
            if len(all_anomaly_events) > 0:
                # Extract anomaly and normal data
                anomaly_indices = [event['original_index'] for event in all_anomaly_events]
                anomaly_df = df.iloc[anomaly_indices].copy()
                normal_df = df.drop(df.index[anomaly_indices]).copy()
                
                # Save to CSV
                anomaly_output = os.path.join(output_dir, "hws_anomaly_events.csv")
                normal_output = os.path.join(output_dir, "hws_normal_events.csv")
                
                anomaly_df.to_csv(anomaly_output, index=False)
                normal_df.to_csv(normal_output, index=False)
                
                print(f"\nSaved {len(anomaly_df):,} anomaly events to: {anomaly_output}")
                print(f"Saved {len(normal_df):,} normal events to: {normal_output}")
                
                # Calculate rates
                anomaly_rate = len(anomaly_df) / len(df) * 100
                print(f"\nANOMALY RATE: {anomaly_rate:.4f}%")
                
                # Calculate nu parameters
                print(f"\nNU PARAMETER RECOMMENDATIONS:")
                for anomaly_type in ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']:
                    # Count this type in all anomaly events
                    type_count = sum(1 for event in all_anomaly_events 
                                   if event.get(anomaly_type) is True)
                    ratio = type_count / len(df)
                    nu = min(ratio * 0.8, 0.1) if ratio > 0 else 0.01
                    print(f"  {anomaly_type}: count={type_count}, ratio={ratio:.6f}, nu={nu:.6f}")
            
        else:
            print(f"No anomalies found in test sample")
            print(f"Saving all data as normal...")
            
            normal_output = os.path.join(output_dir, "hws_normal_events.csv")
            df.to_csv(normal_output, index=False)
            print(f"Saved {len(df):,} normal events to: {normal_output}")
    
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_anomaly_extract()