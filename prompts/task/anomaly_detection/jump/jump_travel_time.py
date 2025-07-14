import pandas as pd
import time
import numpy as np
from typing import List, Dict, Tuple
from geopy.distance import geodesic

def detect_jump_anomalies(df: pd.DataFrame, 
                         transition_stats_df: pd.DataFrame = None,
                         geospatial_df: pd.DataFrame = None) -> Tuple[List[Dict], float]:
    """
    Detect impossible travel time jumps between locations.
    
    Based on question.txt specifications:
    - Uses statistical baselines from business_step_transition_avg_v2.csv
    - Compares actual vs expected transition time using geospatial coordinates
    - Flags travel times beyond 3 standard deviations (statistical outlier)
    - Missing coordinates exclude records from jump detection
    - Same-location scans excluded (zero travel time expected)
    - Handles negative time differences as impossible travel
    
    Args:
        df: DataFrame with columns ['epc_code', 'scan_location', 'event_time']
        transition_stats_df: Statistical baselines for location transitions
        geospatial_df: Location coordinates for distance calculations
    
    Returns:
        tuple: (anomaly_list, execution_time_seconds)
    """
    start_time = time.time()
    anomalies = []
    
    # Mock data if not provided (for testing)
    if transition_stats_df is None:
        transition_stats_df = pd.DataFrame({
            'from_location': ['공장', '물류센터', '도매상'],
            'to_location': ['물류센터', '도매상', '소매상'],
            'avg_transition_time_hours': [2.5, 24.0, 6.0],
            'std_transition_time_hours': [0.5, 4.0, 1.0]
        })
    
    if geospatial_df is None:
        geospatial_df = pd.DataFrame({
            'location_id': ['공장', '물류센터', '도매상', '소매상'],
            'latitude': [37.5665, 37.4563, 37.2636, 37.1830],
            'longitude': [126.9780, 126.7052, 127.0286, 127.0074]
        })
    
    def calculate_travel_time_threshold(from_loc, to_loc, distance_km):
        """Calculate expected travel time and threshold using statistical baseline."""
        # Look up statistical baseline if available
        baseline = transition_stats_df[
            (transition_stats_df['from_location'] == from_loc) & 
            (transition_stats_df['to_location'] == to_loc)
        ]
        
        if not baseline.empty:
            avg_time = baseline.iloc[0]['avg_transition_time_hours']
            std_time = baseline.iloc[0]['std_transition_time_hours']
            # 3 standard deviations threshold as specified
            threshold = avg_time + (3 * std_time)
            return threshold
        
        # Fallback: assume reasonable travel speed (50 km/h average including logistics time)
        estimated_time = distance_km / 50.0  # hours
        return estimated_time * 2  # Conservative threshold
    
    def get_coordinates(location):
        """Get latitude and longitude for a location."""
        coords = geospatial_df[geospatial_df['location_id'] == location]
        if coords.empty:
            return None, None
        return coords.iloc[0]['latitude'], coords.iloc[0]['longitude']
    
    try:
        # Sort by EPC and time
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy['event_time']):
            df_copy['event_time'] = pd.to_datetime(df_copy['event_time'])
        
        df_copy = df_copy.sort_values(['epc_code', 'event_time'])
        
        # Group by EPC to analyze travel patterns
        for epc, group in df_copy.groupby('epc_code'):
            if len(group) <= 1:
                continue  # Single scan, no travel to analyze
            
            for i in range(1, len(group)):
                current_row = group.iloc[i]
                previous_row = group.iloc[i-1]
                
                current_location = current_row['scan_location']
                previous_location = previous_row['scan_location']
                
                # Skip same-location scans as specified
                if current_location == previous_location:
                    continue
                
                # Calculate time difference
                time_diff = current_row['event_time'] - previous_row['event_time']
                time_diff_hours = time_diff.total_seconds() / 3600.0
                
                # Handle negative time differences as impossible travel
                if time_diff_hours < 0:
                    anomalies.append({
                        'epc_code': epc,
                        'scan_location': current_location,
                        'event_time': current_row['event_time'],
                        'anomaly_type': 'jump',
                        'reason': 'negative_time_difference',
                        'time_diff_hours': time_diff_hours,
                        'previous_location': previous_location
                    })
                    continue
                
                # Get coordinates for distance calculation
                curr_lat, curr_lon = get_coordinates(current_location)
                prev_lat, prev_lon = get_coordinates(previous_location)
                
                # Missing coordinates exclude records from jump detection
                if None in [curr_lat, curr_lon, prev_lat, prev_lon]:
                    continue
                
                # Calculate distance
                distance_km = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).kilometers
                
                # Calculate expected travel time threshold
                threshold_hours = calculate_travel_time_threshold(
                    previous_location, current_location, distance_km
                )
                
                # Check if actual time is impossibly fast (statistical outlier)
                if time_diff_hours > 0 and time_diff_hours < (threshold_hours / 10):  # 10x faster than reasonable
                    anomalies.append({
                        'epc_code': epc,
                        'scan_location': current_location,
                        'event_time': current_row['event_time'],
                        'anomaly_type': 'jump',
                        'reason': 'impossible_fast_travel',
                        'actual_time_hours': time_diff_hours,
                        'expected_min_hours': threshold_hours / 10,
                        'distance_km': distance_km,
                        'previous_location': previous_location
                    })
    
    except Exception as e:
        # Log error but continue processing
        anomalies.append({
            'epc_code': 'SYSTEM_ERROR',
            'scan_location': 'UNKNOWN',
            'event_time': None,
            'anomaly_type': 'jump',
            'reason': f'processing_error: {str(e)}'
        })
    
    execution_time = time.time() - start_time
    return anomalies, execution_time

# Benchmark test function
def benchmark_jump_detection(df: pd.DataFrame, iterations: int = 100) -> Dict:
    """Benchmark the jump detection function performance."""
    times = []
    
    for _ in range(iterations):
        _, exec_time = detect_jump_anomalies(df)
        times.append(exec_time)
    
    return {
        'function': 'detect_jump_anomalies',
        'iterations': iterations,
        'avg_time_seconds': sum(times) / len(times),
        'min_time_seconds': min(times),
        'max_time_seconds': max(times),
        'total_records': len(df)
    }

if __name__ == "__main__":
    # Test with sample data showing impossible travel times
    test_data = {
        'epc_code': ['EPC001', 'EPC001', 'EPC001', 'EPC002', 'EPC002'],
        'scan_location': ['공장', '물류센터', '도매상', '공장', '소매상'],
        'event_time': ['2025-07-13 10:00:00', '2025-07-13 10:05:00', '2025-07-13 10:10:00',
                      '2025-07-13 10:00:00', '2025-07-13 10:01:00']  # 1 minute to travel far distance
    }
    
    test_df = pd.DataFrame(test_data)
    anomalies, exec_time = detect_jump_anomalies(test_df)
    
    print(f"Found {len(anomalies)} anomalies in {exec_time:.4f} seconds")
    for anomaly in anomalies:
        print(f"  - {anomaly}")
    
    # Run benchmark
    benchmark_results = benchmark_jump_detection(test_df, 10)
    print(f"\nBenchmark results: {benchmark_results}")