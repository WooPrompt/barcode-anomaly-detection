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

def quick_extract_sample():
    """Extract anomalies from a large sample quickly"""
    
    raw_data_dir = os.path.join(project_root, "data", "raw")
    output_dir = os.path.join(project_root, "data", "analysis_output")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("QUICK ANOMALY EXTRACTION")
    print("=" * 30)
    
    # Load a substantial sample from each file
    sample_size = 50000  # 50K rows per file = 200K total
    
    df_list = []
    for filename in ['hws.csv', 'icn.csv', 'kum.csv', 'ygs.csv']:
        file_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(file_path):
            # Sample from different parts of each file
            total_rows = sum(1 for line in open(file_path, 'r', encoding='utf-8')) - 1  # -1 for header
            
            # Take samples from beginning, middle, end
            samples = []
            samples.append(pd.read_csv(file_path, sep='\t', nrows=sample_size//3))  # Beginning
            
            middle_start = max(0, total_rows//2 - sample_size//6)
            samples.append(pd.read_csv(file_path, sep='\t', skiprows=middle_start, nrows=sample_size//3))  # Middle
            
            end_start = max(0, total_rows - sample_size//3)
            samples.append(pd.read_csv(file_path, sep='\t', skiprows=end_start))  # End
            
            df = pd.concat(samples, ignore_index=True)
            df['source_file'] = filename
            df_list.append(df)
            print(f"  Sampled {filename}: {len(df):,} rows (from {total_rows:,} total)")
    
    # Merge all samples
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"Total sample data: {len(merged_df):,} rows")
    
    # Clean data - remove rows with NaN in critical columns
    critical_cols = ['epc_code', 'location_id', 'business_step', 'event_type', 'event_time']
    before_clean = len(merged_df)
    merged_df = merged_df.dropna(subset=critical_cols)
    print(f"After cleaning: {len(merged_df):,} rows (removed {before_clean - len(merged_df):,} rows with NaN)")
    
    # Add required columns
    merged_df['eventId'] = range(1, len(merged_df) + 1)
    merged_df['file_id'] = 1
    
    # Show sample diversity
    print(f"\nSample diversity:")
    print(f"Business steps: {list(merged_df['business_step'].unique())}")
    print(f"Location IDs: {len(merged_df['location_id'].unique())} unique locations")
    if len(merged_df) > 0:
        print(f"Time range: {merged_df['event_time'].min()} to {merged_df['event_time'].max()}")
    else:
        print("No valid data after cleaning")
    
    # Run detection on entire sample
    print(f"\nRunning anomaly detection on {len(merged_df):,} events...")
    
    detection_input = {"data": merged_df.to_dict('records')}
    detection_json = json.dumps(detection_input)
    
    try:
        results_raw = detect_anomalies_backend_format(detection_json)
        
        if isinstance(results_raw, str):
            results = json.loads(results_raw)
        else:
            results = results_raw
        
        event_history = results.get('EventHistory', [])
        file_stats = results.get('fileAnomalyStats', {})
        
        print(f"\nRESULTS:")
        print(f"Events with anomalies: {len(event_history)}")
        print(f"File statistics: {file_stats}")
        
        total_anomalies = sum([
            file_stats.get('epcFakeCount', 0),
            file_stats.get('epcDupCount', 0),
            file_stats.get('locErrCount', 0),
            file_stats.get('evtOrderErrCount', 0),
            file_stats.get('jumpCount', 0)
        ])
        
        if total_anomalies > 0:
            print(f"SUCCESS! Found {total_anomalies} anomalies!")
            
            # Extract anomaly events
            anomaly_event_ids = [event['eventId'] for event in event_history]
            anomaly_df = merged_df[merged_df['eventId'].isin(anomaly_event_ids)].copy()
            normal_df = merged_df[~merged_df['eventId'].isin(anomaly_event_ids)].copy()
            
            # Add anomaly type information
            anomaly_info = {}
            for event in event_history:
                event_id = event['eventId']
                anomaly_types = []
                for key, value in event.items():
                    if key != 'eventId' and value is True:
                        anomaly_types.append(key)
                anomaly_info[event_id] = anomaly_types
            
            # Add anomaly flags
            for anomaly_type in ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']:
                anomaly_df[f'{anomaly_type}_detected'] = anomaly_df['eventId'].apply(
                    lambda x: anomaly_type in anomaly_info.get(x, [])
                )
            
            # Save files
            anomaly_output = os.path.join(output_dir, "sample_anomaly_events.csv")
            normal_output = os.path.join(output_dir, "sample_normal_events.csv")
            
            anomaly_df.to_csv(anomaly_output, index=False)
            normal_df.to_csv(normal_output, index=False)
            
            print(f"\nSaved {len(anomaly_df):,} anomaly events to: {anomaly_output}")
            print(f"Saved {len(normal_df):,} normal events to: {normal_output}")
            
            # Calculate anomaly rates and nu parameters
            anomaly_rate = len(anomaly_df) / len(merged_df) * 100
            print(f"\nANOMALY ANALYSIS:")
            print(f"Total events: {len(merged_df):,}")
            print(f"Anomaly events: {len(anomaly_df):,} ({anomaly_rate:.3f}%)")
            print(f"Normal events: {len(normal_df):,}")
            
            print(f"\nNU PARAMETER RECOMMENDATIONS:")
            for anomaly_type in ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']:
                count = file_stats.get(f'{anomaly_type}Count', 0)
                ratio = count / len(merged_df)
                nu = min(ratio * 0.8, 0.1) if ratio > 0 else 0.01
                print(f"  {anomaly_type}: count={count}, ratio={ratio:.6f}, nu={nu:.6f}")
            
            # Show sample anomalies
            print(f"\nSAMPLE ANOMALIES:")
            for i, (_, row) in enumerate(anomaly_df.head(5).iterrows()):
                event_id = row['eventId']
                anomaly_types = anomaly_info.get(event_id, [])
                print(f"  Event {event_id}: {', '.join(anomaly_types)}")
                print(f"    EPC: {row['epc_code']}")
                print(f"    Location: {row['scan_location']} (ID: {row['location_id']})")
                print(f"    Business Step: {row['business_step']}")
                print(f"    Time: {row['event_time']}")
                print()
            
        else:
            print(f"No anomalies found in sample of {len(merged_df):,} events")
            print(f"Saving all data as normal...")
            
            normal_output = os.path.join(output_dir, "sample_normal_events.csv")
            merged_df.to_csv(normal_output, index=False)
            print(f"Saved {len(merged_df):,} normal events to: {normal_output}")
            
            # Create empty anomaly file
            empty_anomaly = os.path.join(output_dir, "sample_anomaly_events.csv")
            empty_df = merged_df.iloc[0:0]
            empty_df.to_csv(empty_anomaly, index=False)
            print(f"Created empty anomaly file: {empty_anomaly}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    quick_extract_sample()