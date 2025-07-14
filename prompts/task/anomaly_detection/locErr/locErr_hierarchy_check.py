import pandas as pd
import time
from typing import List, Dict, Tuple

def detect_location_error(df: pd.DataFrame) -> Tuple[List[Dict], float]:
    """
    Detect location hierarchy violations in EPC movement.
    
    Based on question.txt specifications:
    - Hierarchy: Factory(0) → Logistics(1) → Wholesale(2) → Retail(3)
    - Normal flow is downstream (lower to higher numbers)
    - Reverse flow should be flagged as error
    - Unknown locations default to error level 99
    - Single-scan EPCs excluded from analysis
    
    Args:
        df: DataFrame with columns ['epc_code', 'scan_location', 'event_time']
    
    Returns:
        tuple: (anomaly_list, execution_time_seconds)
    """
    start_time = time.time()
    anomalies = []
    
    # Location hierarchy mapping (Korean names as per specifications)
    location_hierarchy = {
        # Factory level (0)
        '공장': 0, '인천공장': 0, '화성공장': 0, '금산공장': 0, '영광공장': 0,
        'factory': 0, 'plant': 0,
        
        # Logistics level (1) 
        '물류센터': 1, '물류': 1, '창고': 1, '수도권물류센터': 1, '충청물류센터': 1,
        'logistics': 1, 'warehouse': 1, 'hub': 1,
        
        # Wholesale level (2)
        '도매상': 2, '도매': 2, '대리점': 2, '총판': 2,
        'wholesale': 2, 'distributor': 2,
        
        # Retail level (3)
        '소매상': 3, '소매': 3, '매장': 3, '점포': 3,
        'retail': 3, 'store': 3, 'shop': 3
    }
    
    def get_location_level(location_name):
        """Get hierarchy level for a location, case insensitive."""
        if pd.isna(location_name):
            return 99  # Error level for null/NaN
        
        location_str = str(location_name).lower()
        
        # Check for partial matches in location name
        for loc_key, level in location_hierarchy.items():
            if loc_key.lower() in location_str:
                return level
        
        # Unknown location defaults to error level
        return 99
    
    try:
        # Add location levels
        df_copy = df.copy()
        df_copy['location_level'] = df_copy['scan_location'].apply(get_location_level)
        
        # Sort by EPC and time for sequence analysis
        df_copy = df_copy.sort_values(['epc_code', 'event_time'])
        
        # Group by EPC to analyze movement patterns
        for epc, group in df_copy.groupby('epc_code'):
            # Skip single-scan EPCs as specified
            if len(group) <= 1:
                continue
            
            # Check for location hierarchy violations
            for i in range(1, len(group)):
                current_level = group.iloc[i]['location_level']
                previous_level = group.iloc[i-1]['location_level']
                
                # Flag error level locations
                if current_level == 99 or previous_level == 99:
                    anomalies.append({
                        'epc_code': epc,
                        'scan_location': group.iloc[i]['scan_location'],
                        'event_time': group.iloc[i]['event_time'],
                        'anomaly_type': 'locErr',
                        'reason': 'unknown_location',
                        'location_level': current_level
                    })
                    continue
                
                # Check for reverse flow (higher to lower level)
                if current_level < previous_level:
                    anomalies.append({
                        'epc_code': epc,
                        'scan_location': group.iloc[i]['scan_location'],
                        'event_time': group.iloc[i]['event_time'],
                        'anomaly_type': 'locErr',
                        'reason': 'reverse_flow',
                        'current_level': current_level,
                        'previous_level': previous_level,
                        'previous_location': group.iloc[i-1]['scan_location']
                    })
                
                # Note: Same-level movement is allowed per specifications
    
    except Exception as e:
        # Log error but continue processing
        anomalies.append({
            'epc_code': 'SYSTEM_ERROR',
            'scan_location': 'UNKNOWN',
            'event_time': None,
            'anomaly_type': 'locErr',
            'reason': f'processing_error: {str(e)}'
        })
    
    execution_time = time.time() - start_time
    return anomalies, execution_time

# Benchmark test function
def benchmark_location_error_detection(df: pd.DataFrame, iterations: int = 100) -> Dict:
    """Benchmark the location error detection function performance."""
    times = []
    
    for _ in range(iterations):
        _, exec_time = detect_location_error(df)
        times.append(exec_time)
    
    return {
        'function': 'detect_location_error',
        'iterations': iterations,
        'avg_time_seconds': sum(times) / len(times),
        'min_time_seconds': min(times),
        'max_time_seconds': max(times),
        'total_records': len(df)
    }

if __name__ == "__main__":
    # Test with sample data showing hierarchy violations
    test_data = {
        'epc_code': ['EPC001', 'EPC001', 'EPC001', 'EPC002', 'EPC002', 'EPC003'],
        'scan_location': ['인천공장', '물류센터', '도매상', '도매상', '공장', '알수없는장소'],
        'event_time': ['2025-07-13 10:00:00', '2025-07-13 11:00:00', '2025-07-13 12:00:00',
                      '2025-07-13 10:00:00', '2025-07-13 11:00:00', '2025-07-13 10:00:00']
    }
    
    test_df = pd.DataFrame(test_data)
    anomalies, exec_time = detect_location_error(test_df)
    
    print(f"Found {len(anomalies)} anomalies in {exec_time:.4f} seconds")
    for anomaly in anomalies:
        print(f"  - {anomaly}")
    
    # Run benchmark
    benchmark_results = benchmark_location_error_detection(test_df, 10)
    print(f"\nBenchmark results: {benchmark_results}")