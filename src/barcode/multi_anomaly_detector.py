# -*- coding: utf-8 -*-
"""
Enhanced Multi-Anomaly Detection System v4.0
Author: Data Analysis Team
Date: 2025-07-13

Key Features:
- Multi-anomaly detection per EPC (one EPC can have multiple anomaly types)
- Probability scoring (0-100%) for each anomaly type
- Sequence position identification for problematic steps
- EventHistory format output for frontend integration
- CatBoost consideration for categorical data

Based on question.txt specifications from prompts/task/anomaly_detection/
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any

# Constants for magic numbers
SCORE_THRESHOLDS = {
    'HIGH': 80,
    'MEDIUM': 50, 
    'LOW': 20,
    'NORMAL': 0
}

EPC_VALIDATION_SCORES = {
    'STRUCTURE_ERROR': 40,
    'HEADER_ERROR': 20,
    'COMPANY_ERROR': 25,
    'DATE_ERROR': 20,
    'DATE_OLD_ERROR': 15,
    'SERIAL_ERROR': 10,
    'MISSING_EVENT': 30,
    'CONSECUTIVE_EVENT': 25,
    'LOCATION_UNKNOWN': 20,
    'LOCATION_REVERSE': 30
}

VALID_COMPANIES = {"8804823", "8805843", "8809437"}
VALID_HEADER = "001"
MAX_PRODUCT_AGE_YEARS = 5

def validate_epc_parts(parts: List[str]) -> Dict[str, bool]:
    """
    Validate individual EPC parts for better readability and reusability.
    
    Args:
        parts: List of EPC code parts split by '.'
    
    Returns:
        Dictionary with validation results for each part
    """
    if len(parts) != 6:
        return {'structure': False, 'header': False, 'company': False, 
                'product': False, 'lot': False, 'date': False, 'serial': False}
    
    validations = {
        'structure': len(parts) == 6,
        'header': parts[0] == VALID_HEADER,
        'company': parts[1] in VALID_COMPANIES,
        'product': parts[2].isdigit() and len(parts[2]) == 7,
        'lot': parts[3].isdigit() and len(parts[3]) == 6,
        'serial': parts[5].isdigit() and len(parts[5]) == 9,
        'date': True  # Will be validated separately due to complexity
    }
    
    return validations

def validate_manufacture_date(date_string: str) -> Tuple[bool, str]:
    """
    Validate manufacture date with detailed error reporting.
    
    Args:
        date_string: Date in YYYYMMDD format
    
    Returns:
        Tuple of (is_valid, error_type)
    """
    try:
        manufacture_date = datetime.strptime(date_string, '%Y%m%d')
        today = datetime.now()
        
        if manufacture_date > today:
            return False, 'future_date'
        elif (today - manufacture_date).days > (MAX_PRODUCT_AGE_YEARS * 365):
            return False, 'too_old'
        else:
            return True, 'valid'
    except ValueError:
        return False, 'invalid_format'

def calculate_epc_fake_score(epc_code: str) -> int:
    """
    Calculate probability score for EPC format violations.
    Based on epcFake/question.txt specifications.
    
    Args:
        epc_code: EPC code string to validate
    
    Returns:
        int: 0-100 probability score (0=valid, 100=definitely fake)
    """
    # Pre-validation checks
    if pd.isna(epc_code) or not epc_code or not isinstance(epc_code, str):
        return 100
    
    total_score = 0
    parts = epc_code.strip().split('.')
    
    # Structure validation - CRITICAL: Early return to prevent IndexError
    if len(parts) != 6:
        return 100  # Structural error = definitely fake
    
    # Validate individual parts (safe now that we know parts has 6 elements)
    validations = validate_epc_parts(parts)
    
    if not validations['header']:
        total_score += EPC_VALIDATION_SCORES['HEADER_ERROR']
    
    if not validations['company']:
        total_score += EPC_VALIDATION_SCORES['COMPANY_ERROR']
    
    if not validations['product']:
        total_score += 15  # Product code error
    
    if not validations['lot']:
        total_score += 15  # Lot code error
    
    if not validations['serial']:
        total_score += EPC_VALIDATION_SCORES['SERIAL_ERROR']
    
    # Date validation with detailed error handling
    date_valid, date_error = validate_manufacture_date(parts[4])
    date_error_occurred = False
    
    if not date_valid:
        date_error_occurred = True
        if date_error == 'future_date':
            total_score += EPC_VALIDATION_SCORES['DATE_ERROR']
        elif date_error == 'too_old':
            total_score += EPC_VALIDATION_SCORES['DATE_OLD_ERROR']
        else:  # invalid_format
            total_score += EPC_VALIDATION_SCORES['DATE_ERROR']
    
    # Critical combination check: if 3+ core validations fail
    critical_failures = sum([
        not validations['header'], 
        not validations['company'],
        date_error_occurred  # Now explicitly tracking date errors
    ])
    
    if critical_failures >= 3:
        return 100  # Definitely fake
    
    return min(100, total_score)

def calculate_duplicate_score(epc_code: str, group_data: pd.DataFrame) -> int:
    """
    Calculate probability score for EPC duplicate violations.
    Based on epcDup/question.txt specifications.
    
    Args:
        epc_code: EPC code being analyzed
        group_data: DataFrame grouped by EPC and timestamp (same second)
                   Contains all scans of this EPC at the same timestamp
    
    Returns:
        int: 0-100 probability score
    """
    # Efficiency check: early return for single scan
    if len(group_data) <= 1:
        return 0
    
    # Count unique locations for this timestamp
    unique_locations = group_data['scan_location'].nunique()
    
    if unique_locations <= 1:
        return 0  # Same location = not impossible (valid duplicate scan)
    
    # Multiple locations at same time = physically impossible
    base_score = 80
    location_penalty = (unique_locations - 1) * 10
    
    # Cap at 100 to prevent overflow
    return min(100, base_score + location_penalty)

def calculate_time_jump_score(time_diff_hours: float, expected_hours: float, std_hours: float) -> int:
    """
    Calculate probability score for impossible travel times.
    Based on jump/question.txt specifications.
    """
    if expected_hours == 0 or std_hours == 0:
        return 0
    
    if time_diff_hours < 0:
        return 95  # Negative time = impossible
    
    # Statistical Z-score approach
    z_score = abs(time_diff_hours - expected_hours) / std_hours
    
    if z_score <= 2:
        return 0  # Within normal range
    elif z_score <= 3:
        return 60  # Suspicious
    elif z_score <= 4:
        return 80  # Highly suspicious
    else:
        return 95  # Almost certainly impossible

def classify_event_type(event: str) -> str:
    """
    Classify event into inbound/outbound/other categories.
    
    Args:
        event: Event type string
    
    Returns:
        str: 'inbound', 'outbound', or 'other'
    """
    if pd.isna(event) or not event:
        return 'missing'
    
    event_lower = event.lower()
    
    # Inbound event patterns
    inbound_keywords = ['inbound', 'aggregation', 'receiving', 'arrival']
    if any(keyword in event_lower for keyword in inbound_keywords):
        return 'inbound'
    
    # Outbound event patterns  
    outbound_keywords = ['outbound', 'shipping', 'dispatch', 'departure']
    if any(keyword in event_lower for keyword in outbound_keywords):
        return 'outbound'
    
    return 'other'  # inspection, return, etc.

def calculate_event_order_score(event_sequence: List[str]) -> int:
    """
    Calculate probability score for event order violations.
    Based on evtOrderErr/question.txt specifications.
    
    Args:
        event_sequence: List of event types in chronological order
    
    Returns:
        int: 0-100 probability score
    """
    if len(event_sequence) <= 1:
        return 0  # Single event = no sequence to analyze
    
    total_score = 0
    consecutive_inbound = 0
    consecutive_outbound = 0
    
    for event in event_sequence:
        event_type = classify_event_type(event)
        
        if event_type == 'missing':
            total_score += EPC_VALIDATION_SCORES['MISSING_EVENT']
            continue
        elif event_type == 'inbound':
            consecutive_inbound += 1
            consecutive_outbound = 0
            
            if consecutive_inbound > 1:
                total_score += EPC_VALIDATION_SCORES['CONSECUTIVE_EVENT']
                
                # Progressive penalty for 3+ consecutive events
                if consecutive_inbound >= 3:
                    total_score += (consecutive_inbound - 2) * 15
                    
        elif event_type == 'outbound':
            consecutive_outbound += 1
            consecutive_inbound = 0
            
            if consecutive_outbound > 1:
                total_score += EPC_VALIDATION_SCORES['CONSECUTIVE_EVENT']
                
                # Progressive penalty for 3+ consecutive events
                if consecutive_outbound >= 3:
                    total_score += (consecutive_outbound - 2) * 15
        else:
            # Other events reset the consecutive counters
            consecutive_inbound = 0
            consecutive_outbound = 0
    
    return min(100, total_score)

# Location hierarchy constants
LOCATION_HIERARCHY = {
    # Factory level (0)
    '공장': 0, 'factory': 0, '제조': 0, 'manufacturing': 0,
    # Logistics center level (1) 
    '물류센터': 1, '물류': 1, 'logistics': 1, 'hub': 1, '센터': 1, 'center': 1,
    # Wholesale level (2)
    '도매상': 2, '도매': 2, 'wholesale': 2, 'w_stock': 2, '창고': 2, 'warehouse': 2,
    # Retail level (3)
    '소매상': 3, '소매': 3, 'retail': 3, 'r_stock': 3, 'pos': 3, '매장': 3, 'store': 3
}

def get_location_hierarchy_level(location: str) -> int:
    """
    Get hierarchy level for a location with comprehensive keyword matching.
    
    Args:
        location: Location name string
    
    Returns:
        int: Hierarchy level (0-3) or 99 for unknown locations
    """
    if pd.isna(location) or not location:
        return 99
    
    location_lower = location.lower().strip()
    
    for keyword, level in LOCATION_HIERARCHY.items():
        if keyword in location_lower:
            return level
    
    return 99

def calculate_location_error_score(location_sequence: List[str]) -> int:
    """
    Calculate probability score for location hierarchy violations.
    Based on locErr/question.txt specifications.
    
    Args:
        location_sequence: List of locations in chronological order
    
    Returns:
        int: 0-100 probability score
    """
    if len(location_sequence) <= 1:
        return 0
    
    total_score = 0
    
    for i in range(1, len(location_sequence)):
        current_level = get_location_hierarchy_level(location_sequence[i])
        previous_level = get_location_hierarchy_level(location_sequence[i-1])
        
        if current_level == 99:
            total_score += EPC_VALIDATION_SCORES['LOCATION_UNKNOWN']
        if previous_level == 99:
            total_score += EPC_VALIDATION_SCORES['LOCATION_UNKNOWN']
        
        if current_level != 99 and previous_level != 99:
            if current_level < previous_level:
                total_score += EPC_VALIDATION_SCORES['LOCATION_REVERSE']
    
    return min(100, total_score)

def classify_anomaly_severity(score: int) -> str:
    """
    Classify anomaly severity based on score thresholds.
    
    Args:
        score: Anomaly score (0-100)
    
    Returns:
        str: Severity level ('HIGH', 'MEDIUM', 'LOW', 'NORMAL')
    """
    if score >= SCORE_THRESHOLDS['HIGH']:
        return 'HIGH'
    elif score >= SCORE_THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    elif score >= SCORE_THRESHOLDS['LOW']:
        return 'LOW'
    else:
        return 'NORMAL'

def preprocess_scan_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Common preprocessing for all anomaly detection algorithms.
    
    Args:
        df: Raw scan data DataFrame
    
    Returns:
        pd.DataFrame: Preprocessed data (copy, no side effects)
    """
    df_processed = df.copy()
    
    # 1. Time data normalization (critical shared function)
    df_processed['event_time'] = pd.to_datetime(df_processed['event_time'])
    df_processed['event_time_rounded'] = df_processed['event_time'].dt.floor('s')
    
    # 2. Missing value handling - avoid inplace to prevent side effects
    df_processed['event_type'] = df_processed['event_type'].fillna('UNKNOWN')
    df_processed['scan_location'] = df_processed['scan_location'].fillna('UNKNOWN_LOCATION')
    
    # 3. EPC code normalization - handle None values safely
    df_processed['epc_code'] = df_processed['epc_code'].astype(str).str.strip().str.upper()
    
    # 4. Sort by EPC and time for sequence analysis
    df_processed = df_processed.sort_values(['epc_code', 'event_time']).reset_index(drop=True)
    
    return df_processed

def detect_multi_anomalies_enhanced(df: pd.DataFrame, transition_stats: pd.DataFrame, geo_df: pd.DataFrame) -> List[Dict]:
    """
    Enhanced multi-anomaly detection with probability scoring.
    Each EPC is checked against all 5 anomaly types.
    
    Args:
        df: Raw scan data
        transition_stats: Expected transition times between locations
        geo_df: Geospatial data for locations
    
    Returns:
        List[Dict]: List of detected anomalies with scores and metadata
    """
    # Preprocess data using common function
    df_processed = preprocess_scan_data(df)
    
    detection_results = []
    
    # Group by EPC to analyze each product journey
    for epc_code, epc_group in df_processed.groupby('epc_code'):
        # Fix: Use drop=True to prevent index column creation
        epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
        
        detected_anomaly_types = []
        anomaly_score_map = {}
        problematic_sequence_steps = []
        
        # 1. EPC Format Validation (epcFake)
        fake_detection_score = calculate_epc_fake_score(epc_code)
        if fake_detection_score > 0:
            detected_anomaly_types.append('epcFake')
            anomaly_score_map['epcFake'] = fake_detection_score
        
        # 2. Duplicate Scan Detection (epcDup) - with deduplication logic
        max_duplicate_score = 0
        for timestamp, time_group in epc_group.groupby('event_time_rounded'):
            duplicate_score = calculate_duplicate_score(epc_code, time_group)
            if duplicate_score > 0:
                max_duplicate_score = max(max_duplicate_score, duplicate_score)
        
        # Only add if duplicates were found
        if max_duplicate_score > 0:
            detected_anomaly_types.append('epcDup')
            anomaly_score_map['epcDup'] = max_duplicate_score
        
        # 3. Time Jump Detection (jump) - only if transition_stats has data
        if not transition_stats.empty and 'from_scan_location' in transition_stats.columns:
            for i in range(1, len(epc_group)):
                current_row = epc_group.iloc[i]
                previous_row = epc_group.iloc[i-1]
                
                # Calculate actual time difference
                time_diff = (pd.to_datetime(current_row['event_time']) - 
                            pd.to_datetime(previous_row['event_time'])).total_seconds() / 3600
                
                # Look up expected transition time
                transition_match = transition_stats[
                    (transition_stats['from_scan_location'] == previous_row['scan_location']) &
                    (transition_stats['to_scan_location'] == current_row['scan_location'])
                ]
                
                if not transition_match.empty:
                    expected_time = transition_match.iloc[0]['time_taken_hours_mean']
                    std_time = transition_match.iloc[0]['time_taken_hours_std']
                    
                    jump_score = calculate_time_jump_score(time_diff, expected_time, std_time)
                    if jump_score > 0:
                        # Fix: Use correct variable names and deduplication
                        if 'jump' not in detected_anomaly_types:
                            detected_anomaly_types.append('jump')
                            anomaly_score_map['jump'] = jump_score
                            problematic_sequence_steps.append(f"Step_{i}_to_{i+1}")
                        else:
                            # Keep highest jump score
                            anomaly_score_map['jump'] = max(anomaly_score_map['jump'], jump_score)
        
        # 4. Event Order Check (evtOrderErr) - with deduplication
        event_sequence = epc_group['event_type'].tolist()
        order_score = calculate_event_order_score(event_sequence)
        if order_score > 0:
            detected_anomaly_types.append('evtOrderErr')
            anomaly_score_map['evtOrderErr'] = order_score
        
        # 5. Location Hierarchy Check (locErr) - with deduplication
        location_sequence = epc_group['scan_location'].tolist()
        location_score = calculate_location_error_score(location_sequence)
        if location_score > 0:
            detected_anomaly_types.append('locErr')
            anomaly_score_map['locErr'] = location_score
        
        # Create result entry if any anomalies found
        if detected_anomaly_types:
            # Calculate problem step for requirements format
            if problematic_sequence_steps:
                # Extract first problematic step and create problemStep
                try:
                    first_problem_step = int(problematic_sequence_steps[0].split('_')[1])
                    calculated_position = first_problem_step
                    # Create problem step description
                    if calculated_position < len(epc_group) - 1:
                        current_loc = epc_group.iloc[calculated_position]['scan_location']
                        next_loc = epc_group.iloc[calculated_position + 1]['scan_location']
                        problem_step = f"{current_loc}_to_{next_loc}"
                    else:
                        problem_step = f"Step_{calculated_position}_issue"
                except (ValueError, IndexError):
                    calculated_position = len(epc_group) // 2
                    problem_step = "Sequence_analysis"
            else:
                # For non-sequence anomalies (epcFake, epcDup)
                if 'epcFake' in detected_anomaly_types:
                    calculated_position = 0
                    problem_step = "EPC_Format_Error"
                elif 'epcDup' in detected_anomaly_types:
                    calculated_position = 1
                    problem_step = "Duplicate_Scan_Error"
                else:
                    calculated_position = len(epc_group) // 2
                    problem_step = "General_Analysis"
            
            # Ensure position is within bounds
            safe_position = min(calculated_position, len(epc_group) - 1)
            
            primary_anomaly = max(anomaly_score_map.items(), key=lambda x: x[1])[0]
            
            # Get representative row data
            rep_row = epc_group.iloc[safe_position]
            
            # Get primary anomaly for Korean description
            primary_anomaly_kr = {
                'epcFake': 'EPC 형식 위반',
                'epcDup': '불가능한 중복 스캔',
                'jump': '비논리적인 시공간 이동',
                'evtOrderErr': '이벤트 순서 오류',
                'locErr': '위치 계층 위반'
            }
            
            # Get primary anomaly Korean name
            primary_anomaly_korean = primary_anomaly_kr.get(primary_anomaly, primary_anomaly)
            
            # Create description based on primary anomaly
            if primary_anomaly == 'jump':
                description = f"{rep_row['scan_location']}에서 비논리적 시간점프 이동 발생"
            elif primary_anomaly == 'evtOrderErr':
                description = f"{rep_row.get('business_step', '단계')} 이벤트 순서 오류"
            elif primary_anomaly == 'epcFake':
                description = f"EPC 코드 형식 위반 감지"
            elif primary_anomaly == 'epcDup':
                description = f"동일 시간 다른 위치 중복 스캔 감지"
            elif primary_anomaly == 'locErr':
                description = f"위치 계층 순서 위반"
            else:
                description = f"{primary_anomaly_korean} 감지"
            
            # Backend required format
            anomaly_result = {
                'epcCode': epc_code,
                'productName': rep_row.get('product_name', 'Unknown'),
                'eventType': primary_anomaly,  # Backend required: primary anomaly code
                'businessStep': rep_row.get('business_step', rep_row.get('event_type', 'Unknown')),
                'scanLocation': rep_row['scan_location'],
                'eventTime': str(rep_row['event_time']),
                'anomaly': True,
                'anomalyType': primary_anomaly_korean,  # Backend required: Korean description
                'anomalyCode': primary_anomaly,  # Backend required: code
                'description': description,  # Backend required: specific description
                
                # Additional fields for enhanced functionality (keep multi-anomaly support)
                'anomalyTypes': detected_anomaly_types,  # All detected anomalies
                'anomalyScores': anomaly_score_map,      # Scores for each
                'sequencePosition': safe_position + 1,
                'totalSequenceLength': len(epc_group),
                'primaryAnomaly': primary_anomaly,       # For backward compatibility
                'problemStep': problem_step
            }
            
            detection_results.append(anomaly_result)
    
    return detection_results

def load_csv_data():
    """
    Load geo_data, transition_stats, and location_mapping from CSV files
    """
    import os
    
    try:
        # Load geo data
        geo_path = "data/processed/location_id_withGeospatial.csv"
        if os.path.exists(geo_path):
            geo_df = pd.read_csv(geo_path)
        else:
            geo_df = pd.DataFrame()
        
        # Load transition stats
        transition_path = "data/processed/business_step_transition_avg_v2.csv"
        if os.path.exists(transition_path):
            transition_df = pd.read_csv(transition_path)
        else:
            transition_df = pd.DataFrame()
            
        # Load location mapping
        location_mapping_path = "data/processed/location_id_scan_location_matching.csv"
        if os.path.exists(location_mapping_path):
            location_mapping_df = pd.read_csv(location_mapping_path)
        else:
            location_mapping_df = pd.DataFrame()
            
        return geo_df, transition_df, location_mapping_df
        
    except Exception as e:
        print(f"Warning: Could not load CSV files: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def save_detection_result(input_data: dict, output_data: dict) -> str:
    """
    탐지 결과를 JSON 파일로 저장 (ML 학습 데이터 축적용)
    
    Args:
        input_data: 입력 데이터 (백엔드에서 받은 데이터)
        output_data: 출력 데이터 (탐지 결과)
    
    Returns:
        str: 저장된 파일의 로그 ID
    """
    try:
        # 저장 디렉토리 생성
        log_dir = "data/detection_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 타임스탬프 기반 로그 ID 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초 포함
        log_id = f"detection_{timestamp}"
        
        # 메타데이터 계산
        file_id = input_data.get("data", [{}])[0].get("file_id", 1) if input_data.get("data") else 1
        total_events = len(input_data.get("data", []))
        anomaly_count = output_data.get("fileAnomalyStats", {}).get("totalEvents", 0)
        detection_rate = anomaly_count / total_events if total_events > 0 else 0
        
        # 로그 엔트리 생성
        log_entry = {
            "logId": log_id,
            "timestamp": datetime.now().isoformat(),
            "fileId": file_id,
            "input": input_data,
            "output": output_data,
            "metadata": {
                "total_input_events": total_events,
                "total_anomaly_events": anomaly_count,
                "detection_rate": round(detection_rate, 4),
                "anomaly_breakdown": {
                    "jumpCount": output_data.get("fileAnomalyStats", {}).get("jumpCount", 0),
                    "evtOrderErrCount": output_data.get("fileAnomalyStats", {}).get("evtOrderErrCount", 0),
                    "epcFakeCount": output_data.get("fileAnomalyStats", {}).get("epcFakeCount", 0),
                    "epcDupCount": output_data.get("fileAnomalyStats", {}).get("epcDupCount", 0),
                    "locErrCount": output_data.get("fileAnomalyStats", {}).get("locErrCount", 0)
                },
                "unique_epc_count": len(output_data.get("epcAnomalyStats", [])),
                "processing_timestamp": timestamp
            }
        }
        
        # JSON 파일로 저장
        log_file_path = os.path.join(log_dir, f"{log_id}.json")
        with open(log_file_path, "w", encoding="utf-8") as f:
            json.dump(log_entry, f, ensure_ascii=False, indent=2)
        
        print(f"Detection result saved: {log_file_path}")
        return log_id
        
    except Exception as e:
        print(f"Warning: Could not save detection result: {e}")
        return f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def detect_anomalies_backend_format(json_input_str: str) -> str:
    """
    Backend-compatible anomaly detection function.
    Input: JSON with eventId, location_id based format
    Output: fileId, EventHistory, epcAnomalyStats, fileAnomalyStats format
    """
    try:
        input_data = json.loads(json_input_str)
        raw_df = pd.DataFrame(input_data['data'])
        
        # Get fileId from first record
        file_id = int(raw_df['file_id'].iloc[0]) if not raw_df.empty else 1
        
        # Load CSV data for processing
        geo_df, transition_stats, location_mapping_df = load_csv_data()
        
        # Map location_id to scan_location if location mapping is available
        if not location_mapping_df.empty and 'location_id' in raw_df.columns:
            raw_df = raw_df.merge(
                location_mapping_df[['location_id', 'scan_location']], 
                on='location_id', 
                how='left'
            )
            # Fill missing scan_location with location_id as string
            raw_df['scan_location'] = raw_df['scan_location'].fillna(raw_df['location_id'].astype(str))
        
    except (json.JSONDecodeError, KeyError) as e:
        return json.dumps({"error": f"Invalid JSON input: {e}"}, indent=2, ensure_ascii=False)

    if raw_df.empty:
        return json.dumps({
            "fileId": 1,
            "EventHistory": [],
            "epcAnomalyStats": [],
            "fileAnomalyStats": {
                "totalEvents": 0,
                "jumpCount": 0,
                "evtOrderErrCount": 0,
                "epcFakeCount": 0,
                "epcDupCount": 0,
                "locErrCount": 0
            }
        }, indent=2, ensure_ascii=False)

    # Detect anomalies with enhanced multi-anomaly support
    anomaly_results = detect_multi_anomalies_enhanced(raw_df, transition_stats, geo_df)

    # Build EventHistory in backend format with proper multi-anomaly detection
    event_history = []
    epc_anomaly_stats = {}
    file_anomaly_stats = {
        "totalEvents": 0,
        "jumpCount": 0,
        "evtOrderErrCount": 0, 
        "epcFakeCount": 0,
        "epcDupCount": 0,
        "locErrCount": 0
    }

    # Process each event individually for event-specific multi-anomaly detection
    for _, event_row in raw_df.iterrows():
        event_id = int(event_row.get('eventId', 0))
        epc_code = event_row.get('epc_code', '')
        location_id = int(event_row.get('location_id', 0))
        event_time = event_row.get('event_time', '')
        business_step = event_row.get('business_step', '')
        event_type = event_row.get('event_type', '')
        
        event_anomalies = {}
        
        # 1. EPC Format Check (epcFake) - applies to this specific event
        fake_score = calculate_epc_fake_score(epc_code)
        if fake_score > 0:
            event_anomalies['epcFake'] = True
            event_anomalies['epcFakeScore'] = float(fake_score)
        
        # Get EPC group for this event's EPC
        epc_events = raw_df[raw_df['epc_code'] == epc_code].sort_values('event_time').reset_index(drop=True)
        current_event_index = epc_events[epc_events['eventId'] == event_id].index
        
        if len(current_event_index) > 0:
            current_idx = current_event_index[0]
            
            # 2. Event Order Error Check (evtOrderErr) - look for temporal sequence violations  
            if current_idx > 0:
                prev_event = epc_events.iloc[current_idx - 1]
                prev_event_type = prev_event.get('event_type', '')
                prev_business_step = prev_event.get('business_step', '')
                prev_time = prev_event.get('event_time', '')
                current_event_type = event_type
                
                try:
                    from datetime import datetime
                    prev_datetime = datetime.strptime(prev_time, '%Y-%m-%d %H:%M:%S')
                    current_datetime = datetime.strptime(event_time, '%Y-%m-%d %H:%M:%S')
                    
                    # Detect temporal disorder: current event happens BEFORE previous event
                    if current_datetime < prev_datetime:
                        event_anomalies['evtOrderErr'] = True
                        event_anomalies['evtOrderErrScore'] = 25.0
                    
                    # Also detect consecutive Inbound at same location (impossible)
                    elif (current_event_type == 'Inbound' and prev_event_type == 'Inbound' and 
                          business_step == prev_business_step):
                        event_anomalies['evtOrderErr'] = True
                        event_anomalies['evtOrderErrScore'] = 25.0
                        
                except (ValueError, TypeError):
                    # If datetime parsing fails, fall back to simpler logic
                    if (current_event_type == 'Inbound' and prev_event_type == 'Inbound' and 
                        business_step == prev_business_step):
                        event_anomalies['evtOrderErr'] = True
                        event_anomalies['evtOrderErrScore'] = 25.0
            
            # 3. Duplicate Scan Check (epcDup) - check if this specific event has time conflicts
            same_time_events = epc_events[epc_events['event_time'] == event_time]
            if len(same_time_events) > 1:
                # Multiple events at same time for same EPC is always suspicious
                event_anomalies['epcDup'] = True
                event_anomalies['epcDupScore'] = 90.0
            
            # 4. Location Error Check (locErr) - check if this specific transition violates hierarchy
            if current_idx > 0:
                prev_event = epc_events.iloc[current_idx - 1]
                prev_business_step = prev_event.get('business_step', '')
                
                # Check for hierarchy violations
                hierarchy = {'Factory': 1, 'WMS': 2, 'Wholesaler': 3, 'Retailer': 4}
                current_level = hierarchy.get(business_step, 99)
                prev_level = hierarchy.get(prev_business_step, 99)
                
                # Check for reverse hierarchy (higher to lower level)
                if current_level < prev_level and current_level != 99 and prev_level != 99:
                    event_anomalies['locErr'] = True
                    event_anomalies['locErrScore'] = 30.0
                
                # Check for level skipping and suspicious direct jumps
                if ((prev_level == 2 and current_level == 4) or   # WMS -> Retailer (skip Wholesaler)
                    (prev_level == 1 and current_level == 3) or   # Factory -> Wholesaler (skip WMS)
                    (prev_level == 1 and current_level == 4)):    # Factory -> Retailer (skip WMS+Wholesaler)
                    event_anomalies['locErr'] = True
                    event_anomalies['locErrScore'] = 30.0
                
                # Special case: Factory->WMS with same timestamp (suspicious direct jump)
                elif (prev_level == 1 and current_level == 2):  # Factory -> WMS
                    prev_time = prev_event.get('event_time', '')
                    if event_time == prev_time:  # Same timestamp = suspicious
                        event_anomalies['locErr'] = True
                        event_anomalies['locErrScore'] = 30.0
        
        # If this event has anomalies, add to EventHistory
        if event_anomalies:
            event_record = {"eventId": event_id}
            event_record.update(event_anomalies)
            event_history.append(event_record)
    
    # Build EPC anomaly statistics from actual event history
    for event in event_history:
        event_id = event['eventId']
        # Find the EPC code for this event
        event_row = raw_df[raw_df['eventId'] == event_id].iloc[0]
        epc_code = event_row['epc_code']
        
        # Initialize EPC stats if not exists
        if epc_code not in epc_anomaly_stats:
            epc_anomaly_stats[epc_code] = {
                "epcCode": epc_code,
                "totalEvents": 0,
                "jumpCount": 0,
                "evtOrderErrCount": 0,
                "epcFakeCount": 0, 
                "epcDupCount": 0,
                "locErrCount": 0
            }
        
        # Count each anomaly type for this EPC based on actual detections
        for anomaly_type in ['jump', 'evtOrderErr', 'epcFake', 'epcDup', 'locErr']:
            if event.get(anomaly_type, False):
                epc_anomaly_stats[epc_code][f"{anomaly_type}Count"] += 1
    
    # Calculate totalEvents for each EPC (sum of all anomaly counts)
    for epc_code, stats in epc_anomaly_stats.items():
        stats["totalEvents"] = (
            stats["jumpCount"] + 
            stats["evtOrderErrCount"] + 
            stats["epcFakeCount"] + 
            stats["epcDupCount"] + 
            stats["locErrCount"]
        )
    
    # Calculate file anomaly statistics
    for epc_stats in epc_anomaly_stats.values():
        file_anomaly_stats["jumpCount"] += epc_stats["jumpCount"]
        file_anomaly_stats["evtOrderErrCount"] += epc_stats["evtOrderErrCount"]
        file_anomaly_stats["epcFakeCount"] += epc_stats["epcFakeCount"]
        file_anomaly_stats["epcDupCount"] += epc_stats["epcDupCount"]
        file_anomaly_stats["locErrCount"] += epc_stats["locErrCount"]
    
    # Calculate file totalEvents (sum of all anomaly counts)
    file_anomaly_stats["totalEvents"] = (
        file_anomaly_stats["jumpCount"] +
        file_anomaly_stats["evtOrderErrCount"] +
        file_anomaly_stats["epcFakeCount"] +
        file_anomaly_stats["epcDupCount"] +
        file_anomaly_stats["locErrCount"]
    )

    # Convert epc_anomaly_stats to list (only include EPCs with anomalies)
    epc_stats_list = [stats for stats in epc_anomaly_stats.values() if stats["totalEvents"] > 0]

    output = {
        "fileId": file_id,
        "EventHistory": event_history,
        "epcAnomalyStats": epc_stats_list,
        "fileAnomalyStats": file_anomaly_stats
    }

    return json.dumps(output, indent=2, ensure_ascii=False)

def detect_anomalies_from_json_enhanced(json_input_str: str) -> str:
    """
    Main enhanced function with EventHistory format output.
    Auto-loads geo_data and transition_stats from CSV files.
    DEPRECATED: Use detect_anomalies_backend_format for new backend integration
    """
    try:
        input_data = json.loads(json_input_str)
        raw_df = pd.DataFrame(input_data['data'])
        
        # Auto-load from CSV files if not provided
        transition_stats = pd.DataFrame(input_data.get('transition_stats', []))
        geo_df = pd.DataFrame(input_data.get('geo_data', []))
        
        # Load from CSV if empty
        if transition_stats.empty or geo_df.empty:
            csv_geo_df, csv_transition_df, _ = load_csv_data()
            
            if geo_df.empty and not csv_geo_df.empty:
                geo_df = csv_geo_df
                
            if transition_stats.empty and not csv_transition_df.empty:
                transition_stats = csv_transition_df
                
    except (json.JSONDecodeError, KeyError) as e:
        return json.dumps({"error": f"Invalid JSON input: {e}"}, indent=2, ensure_ascii=False)

    if raw_df.empty:
        return json.dumps({
            "EventHistory": [],
            "summaryStats": {"epcFake": 0, "epcDup": 0, "locErr": 0, "evtOrderErr": 0, "jump": 0},
            "multiAnomalyCount": 0
        }, indent=2, ensure_ascii=False)

    # Use all data without filtering
    filtered_df = raw_df

    # Detect anomalies
    anomaly_results = detect_multi_anomalies_enhanced(filtered_df, transition_stats, geo_df)

    # Calculate summary statistics
    summary_stats = {"epcFake": 0, "epcDup": 0, "locErr": 0, "evtOrderErr": 0, "jump": 0}
    multi_anomaly_count = 0
    
    for result in anomaly_results:
        if len(result['anomalyTypes']) > 1:
            multi_anomaly_count += 1
        
        for anomaly_type in result['anomalyTypes']:
            if anomaly_type in summary_stats:
                summary_stats[anomaly_type] += 1

    output = {
        "EventHistory": anomaly_results,
        "summaryStats": summary_stats,
        "multiAnomalyCount": multi_anomaly_count,
        "totalAnomalyCount": len(anomaly_results)
    }

    return json.dumps(output, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    # Test with enhanced multi-anomaly detection
    test_data = {
        "product_id": "0000001",
        "lot_id": "000001",
        "data": [
            {
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "event_time": "2024-07-02 09:00:00",
                "scan_location": "서울 공장",
                "event_type": "Inbound",
                "product_name": "Product 1"
            },
            {
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "event_time": "2024-07-02 09:00:00",
                "scan_location": "부산 공장",
                "event_type": "Inbound",
                "product_name": "Product 1"
            },
            {
                "epc_code": "invalid.format.epc",
                "event_time": "2024-07-02 10:00:00",
                "scan_location": "인천 물류센터",
                "event_type": "Outbound",
                "product_name": "Product 2"
            }
        ],
        "transition_stats": [
            {
                "from_scan_location": "서울 공장",
                "to_scan_location": "부산 공장",
                "time_taken_hours_mean": 5.0,
                "time_taken_hours_std": 1.0
            }
        ],
        "geo_data": []
    }
    
    test_json = json.dumps(test_data)
    result = detect_anomalies_from_json_enhanced(test_json)
    print("Enhanced Multi-Anomaly Detection Result:")
    print(result)