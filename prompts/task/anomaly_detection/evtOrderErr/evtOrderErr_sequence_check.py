import pandas as pd
import time
from typing import List, Dict, Tuple
import re

def detect_event_order_error(df: pd.DataFrame) -> Tuple[List[Dict], float]:
    """
    Detect event sequence violations within same location.
    
    Based on question.txt specifications:
    - Valid sequence: Inbound → Outbound at same location
    - Multiple consecutive Inbound or Outbound events are errors
    - Partial keyword matching (e.g., "Partial_Outbound" → "Outbound")
    - Case insensitive event type matching
    - Single-event EPCs excluded from analysis
    - Events with null/empty event_type flagged
    
    Args:
        df: DataFrame with columns ['epc_code', 'scan_location', 'event_time', 'event_type']
    
    Returns:
        tuple: (anomaly_list, execution_time_seconds)
    """
    start_time = time.time()
    anomalies = []
    
    def classify_event_type(event_type):
        """Classify event type as Inbound, Outbound, or Other using partial matching."""
        if pd.isna(event_type) or str(event_type).strip() == '':
            return 'NULL'
        
        event_str = str(event_type).lower()
        
        # Partial keyword matching for Inbound events
        inbound_keywords = ['inbound', 'stock_inbound', 'hub_inbound', 'wms_inbound', 'aggregation']
        for keyword in inbound_keywords:
            if keyword.lower() in event_str:
                return 'Inbound'
        
        # Partial keyword matching for Outbound events  
        outbound_keywords = ['outbound', 'stock_outbound', 'hub_outbound', 'wms_outbound', 'pos_sell']
        for keyword in outbound_keywords:
            if keyword.lower() in event_str:
                return 'Outbound'
        
        # Other events (intermediate states) are valid per specifications
        return 'Other'
    
    try:
        # Add event classification
        df_copy = df.copy()
        df_copy['event_category'] = df_copy['event_type'].apply(classify_event_type)
        
        # Flag events with null/empty event_type
        null_events = df_copy[df_copy['event_category'] == 'NULL']
        for _, row in null_events.iterrows():
            anomalies.append({
                'epc_code': row['epc_code'],
                'scan_location': row['scan_location'],
                'event_time': row['event_time'],
                'event_type': row['event_type'],
                'anomaly_type': 'evtOrderErr',
                'reason': 'null_empty_event_type'
            })
        
        # Sort by EPC, location, and time for sequence analysis
        df_copy = df_copy.sort_values(['epc_code', 'scan_location', 'event_time'])
        
        # Group by EPC and location to analyze sequences
        for (epc, location), group in df_copy.groupby(['epc_code', 'scan_location']):
            # Skip single-event EPCs as specified
            if len(group) <= 1:
                continue
            
            # Filter to Inbound/Outbound events only for sequence checking
            io_events = group[group['event_category'].isin(['Inbound', 'Outbound'])]
            
            if len(io_events) <= 1:
                continue
            
            # Check for consecutive same-type events (violation)
            for i in range(1, len(io_events)):
                current_category = io_events.iloc[i]['event_category']
                previous_category = io_events.iloc[i-1]['event_category']
                
                # Consecutive Inbound or Outbound events are errors
                if current_category == previous_category:
                    anomalies.append({
                        'epc_code': epc,
                        'scan_location': location,
                        'event_time': io_events.iloc[i]['event_time'],
                        'event_type': io_events.iloc[i]['event_type'],
                        'anomaly_type': 'evtOrderErr',
                        'reason': f'consecutive_{current_category.lower()}_events',
                        'previous_event_type': io_events.iloc[i-1]['event_type'],
                        'sequence_violation': f'{previous_category} → {current_category}'
                    })
    
    except Exception as e:
        # Log error but continue processing
        anomalies.append({
            'epc_code': 'SYSTEM_ERROR',
            'scan_location': 'UNKNOWN',
            'event_time': None,
            'event_type': None,
            'anomaly_type': 'evtOrderErr',
            'reason': f'processing_error: {str(e)}'
        })
    
    execution_time = time.time() - start_time
    return anomalies, execution_time

# Benchmark test function
def benchmark_event_order_detection(df: pd.DataFrame, iterations: int = 100) -> Dict:
    """Benchmark the event order error detection function performance."""
    times = []
    
    for _ in range(iterations):
        _, exec_time = detect_event_order_error(df)
        times.append(exec_time)
    
    return {
        'function': 'detect_event_order_error',
        'iterations': iterations,
        'avg_time_seconds': sum(times) / len(times),
        'min_time_seconds': min(times),
        'max_time_seconds': max(times),
        'total_records': len(df)
    }

if __name__ == "__main__":
    # Test with sample data showing sequence violations
    test_data = {
        'epc_code': ['EPC001', 'EPC001', 'EPC001', 'EPC002', 'EPC002', 'EPC003', 'EPC004'],
        'scan_location': ['Location_A', 'Location_A', 'Location_A', 'Location_B', 'Location_B', 'Location_C', 'Location_D'],
        'event_time': ['2025-07-13 10:00:00', '2025-07-13 11:00:00', '2025-07-13 12:00:00',
                      '2025-07-13 10:00:00', '2025-07-13 11:00:00', '2025-07-13 10:00:00', '2025-07-13 10:00:00'],
        'event_type': ['WMS_Inbound', 'WMS_Inbound', 'WMS_Outbound', 'HUB_Inbound', 'HUB_Outbound', None, 'POS_Sell']
    }
    
    test_df = pd.DataFrame(test_data)
    anomalies, exec_time = detect_event_order_error(test_df)
    
    print(f"Found {len(anomalies)} anomalies in {exec_time:.4f} seconds")
    for anomaly in anomalies:
        print(f"  - {anomaly}")
    
    # Run benchmark
    benchmark_results = benchmark_event_order_detection(test_df, 10)
    print(f"\nBenchmark results: {benchmark_results}")