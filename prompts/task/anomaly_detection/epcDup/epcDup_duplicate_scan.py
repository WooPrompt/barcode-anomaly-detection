import pandas as pd
import time
from typing import List, Dict, Tuple

def detect_epc_duplicate(df: pd.DataFrame) -> Tuple[List[Dict], float]:
    """
    Detect impossible simultaneous EPC scans at different locations.
    
    Based on question.txt specifications:
    - Same EPC at different locations with same timestamp = duplicate
    - Second precision sufficient (not microsecond)
    - Handle missing timestamps as errors
    - Single timezone assumption
    
    Args:
        df: DataFrame with columns ['epc_code', 'scan_location', 'event_time']
    
    Returns:
        tuple: (anomaly_list, execution_time_seconds)
    """
    start_time = time.time()
    anomalies = []
    
    try:
        # Handle missing timestamps - flag as error
        null_timestamp_mask = df['event_time'].isnull()
        if null_timestamp_mask.any():
            for idx in df[null_timestamp_mask].index:
                anomalies.append({
                    'epc_code': df.loc[idx, 'epc_code'],
                    'scan_location': df.loc[idx, 'scan_location'],
                    'event_time': None,
                    'anomaly_type': 'epcDup',
                    'reason': 'missing_timestamp'
                })
        
        # Work with valid timestamps only
        valid_df = df[~null_timestamp_mask].copy()
        
        if valid_df.empty:
            return anomalies, time.time() - start_time
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(valid_df['event_time']):
            valid_df['event_time'] = pd.to_datetime(valid_df['event_time'])
        
        # Round to second precision as specified
        valid_df['event_time_rounded'] = valid_df['event_time'].dt.floor('S')
        
        # Group by EPC and timestamp to find duplicates at different locations
        grouped = valid_df.groupby(['epc_code', 'event_time_rounded'])
        
        for (epc, timestamp), group in grouped:
            # Check if same EPC appears at different locations at same time
            unique_locations = group['scan_location'].nunique()
            
            if unique_locations > 1:
                # This is an impossible duplicate - same EPC can't be at multiple locations simultaneously
                for _, row in group.iterrows():
                    anomalies.append({
                        'epc_code': epc,
                        'scan_location': row['scan_location'],
                        'event_time': row['event_time'],
                        'anomaly_type': 'epcDup',
                        'reason': f'simultaneous_scan_at_{unique_locations}_locations',
                        'duplicate_count': len(group)
                    })
    
    except Exception as e:
        # Log error but continue processing
        anomalies.append({
            'epc_code': 'SYSTEM_ERROR',
            'scan_location': 'UNKNOWN',
            'event_time': None,
            'anomaly_type': 'epcDup',
            'reason': f'processing_error: {str(e)}'
        })
    
    execution_time = time.time() - start_time
    return anomalies, execution_time

# Benchmark test function
def benchmark_epc_duplicate_detection(df: pd.DataFrame, iterations: int = 100) -> Dict:
    """Benchmark the duplicate detection function performance."""
    times = []
    
    for _ in range(iterations):
        _, exec_time = detect_epc_duplicate(df)
        times.append(exec_time)
    
    return {
        'function': 'detect_epc_duplicate',
        'iterations': iterations,
        'avg_time_seconds': sum(times) / len(times),
        'min_time_seconds': min(times),
        'max_time_seconds': max(times),
        'total_records': len(df)
    }

if __name__ == "__main__":
    # Test with sample data
    test_data = {
        'epc_code': ['EPC001', 'EPC001', 'EPC002', 'EPC003', 'EPC003'],
        'scan_location': ['Location_A', 'Location_B', 'Location_A', 'Location_A', 'Location_A'],
        'event_time': ['2025-07-13 10:00:00', '2025-07-13 10:00:00', '2025-07-13 10:01:00', 
                      '2025-07-13 10:02:00', '2025-07-13 10:03:00']
    }
    
    test_df = pd.DataFrame(test_data)
    anomalies, exec_time = detect_epc_duplicate(test_df)
    
    print(f"Found {len(anomalies)} anomalies in {exec_time:.4f} seconds")
    for anomaly in anomalies:
        print(f"  - {anomaly}")
    
    # Run benchmark
    benchmark_results = benchmark_epc_duplicate_detection(test_df, 10)
    print(f"\nBenchmark results: {benchmark_results}")