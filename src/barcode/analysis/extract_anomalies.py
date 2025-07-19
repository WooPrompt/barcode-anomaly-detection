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

def extract_anomalies_to_csv():
    """Run rule-based detection and save anomalies/normal data to separate CSV files"""
    
    raw_data_dir = os.path.join(project_root, "data", "raw")
    output_dir = os.path.join(project_root, "data", "analysis_output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("EXTRACTING ANOMALIES FROM COMPLETE DATASET")
    print("=" * 50)
    
    # Load all data
    print("Loading data...")
    df_list = []
    for filename in ['hws.csv', 'icn.csv', 'kum.csv', 'ygs.csv']:
        file_path = os.path.join(raw_data_dir, filename)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep='\t')
            df['source_file'] = filename
            df_list.append(df)
            print(f"  Loaded {filename}: {len(df):,} rows")
    
    # Merge all dataframes
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"Total merged data: {len(merged_df):,} rows")
    
    # Add required columns
    merged_df['eventId'] = range(1, len(merged_df) + 1)
    merged_df['file_id'] = 1
    
    # Process in chunks to avoid memory issues
    chunk_size = 10000
    total_chunks = (len(merged_df) + chunk_size - 1) // chunk_size
    
    all_anomaly_events = []
    anomaly_summary = {
        'epcFakeCount': 0,
        'epcDupCount': 0,
        'locErrCount': 0,
        'evtOrderErrCount': 0,
        'jumpCount': 0
    }
    
    print(f"\nProcessing {total_chunks} chunks of {chunk_size:,} events each...")
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(merged_df))
        
        chunk_df = merged_df.iloc[start_idx:end_idx].copy()
        chunk_df['eventId'] = range(1, len(chunk_df) + 1)  # Reset eventId for each chunk
        
        print(f"  Processing chunk {chunk_idx + 1}/{total_chunks} (rows {start_idx:,}-{end_idx:,})...")
        
        # Run detection on chunk
        detection_input = {"data": chunk_df.to_dict('records')}
        detection_json = json.dumps(detection_input)
        
        try:
            results_raw = detect_anomalies_backend_format(detection_json)
            
            if isinstance(results_raw, str):
                results = json.loads(results_raw)
            else:
                results = results_raw
            
            event_history = results.get('EventHistory', [])
            file_stats = results.get('fileAnomalyStats', {})
            
            # Update summary counts
            for anomaly_type in anomaly_summary.keys():
                anomaly_summary[anomaly_type] += file_stats.get(anomaly_type, 0)
            
            # Collect anomaly events with original eventId
            if event_history:
                for event in event_history:
                    chunk_event_id = event.get('eventId')
                    # Map back to original eventId
                    original_event_id = start_idx + chunk_event_id
                    event['original_eventId'] = original_event_id
                    all_anomaly_events.append(event)
                
                print(f"    Found {len(event_history)} anomaly events in this chunk")
            
        except Exception as e:
            print(f"    ERROR in chunk {chunk_idx + 1}: {e}")
            continue
    
    print(f"\nANOMALY EXTRACTION COMPLETE")
    print(f"Total anomaly events found: {len(all_anomaly_events):,}")
    print(f"Anomaly breakdown:")
    for anomaly_type, count in anomaly_summary.items():
        print(f"  {anomaly_type}: {count:,}")
    
    if len(all_anomaly_events) > 0:
        # Create anomaly dataframe
        anomaly_event_ids = [event['original_eventId'] for event in all_anomaly_events]
        anomaly_df = merged_df[merged_df['eventId'].isin(anomaly_event_ids)].copy()
        
        # Add anomaly type information
        anomaly_info = {}
        for event in all_anomaly_events:
            event_id = event['original_eventId']
            anomaly_types = []
            for key, value in event.items():
                if key not in ['eventId', 'original_eventId'] and value is True:
                    anomaly_types.append(key)
            anomaly_info[event_id] = anomaly_types
        
        # Add anomaly type columns
        for anomaly_type in ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']:
            anomaly_df[f'{anomaly_type}_detected'] = anomaly_df['eventId'].apply(
                lambda x: anomaly_type in anomaly_info.get(x, [])
            )
        
        # Save anomaly events to CSV
        anomaly_output_path = os.path.join(output_dir, "anomaly_events.csv")
        anomaly_df.to_csv(anomaly_output_path, index=False)
        print(f"\nSaved {len(anomaly_df):,} anomaly events to: {anomaly_output_path}")
        
        # Create normal events dataframe (events NOT in anomaly list)
        normal_df = merged_df[~merged_df['eventId'].isin(anomaly_event_ids)].copy()
        
        # Save normal events to CSV
        normal_output_path = os.path.join(output_dir, "normal_events.csv")
        normal_df.to_csv(normal_output_path, index=False)
        print(f"Saved {len(normal_df):,} normal events to: {normal_output_path}")
        
        # Create summary statistics
        total_events = len(merged_df)
        anomaly_rate = len(anomaly_df) / total_events * 100
        
        summary = {
            'total_events': total_events,
            'anomaly_events': len(anomaly_df),
            'normal_events': len(normal_df),
            'anomaly_rate_percent': anomaly_rate,
            'anomaly_breakdown': anomaly_summary
        }
        
        # Save summary
        summary_output_path = os.path.join(output_dir, "anomaly_summary.json")
        with open(summary_output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSUMMARY:")
        print(f"Total events: {total_events:,}")
        print(f"Anomaly events: {len(anomaly_df):,} ({anomaly_rate:.3f}%)")
        print(f"Normal events: {len(normal_df):,} ({100-anomaly_rate:.3f}%)")
        print(f"Summary saved to: {summary_output_path}")
        
        # Calculate nu parameters
        print(f"\nRECOMMENDED NU PARAMETERS:")
        for anomaly_type, count in anomaly_summary.items():
            if count > 0:
                ratio = count / total_events
                nu = min(ratio * 0.8, 0.1)
                print(f"  {anomaly_type}: count={count:,}, ratio={ratio:.6f}, nu={nu:.6f}")
            else:
                print(f"  {anomaly_type}: count=0, ratio=0.000000, nu=0.010000")
        
        return anomaly_df, normal_df, summary
        
    else:
        print(f"\nNo anomalies detected in the entire dataset!")
        print(f"This confirms extremely high data quality.")
        
        # Save all data as normal
        normal_output_path = os.path.join(output_dir, "normal_events.csv")
        merged_df.to_csv(normal_output_path, index=False)
        print(f"Saved all {len(merged_df):,} events as normal to: {normal_output_path}")
        
        # Create empty anomaly file
        anomaly_output_path = os.path.join(output_dir, "anomaly_events.csv")
        empty_df = merged_df.iloc[0:0].copy()  # Empty dataframe with same structure
        empty_df.to_csv(anomaly_output_path, index=False)
        print(f"Created empty anomaly file: {anomaly_output_path}")
        
        return None, merged_df, {'total_events': len(merged_df), 'anomaly_events': 0}

if __name__ == "__main__":
    extract_anomalies_to_csv()