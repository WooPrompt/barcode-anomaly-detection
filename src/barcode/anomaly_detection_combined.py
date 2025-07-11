# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:30:00 2025

@author: user

@Title: Combined Anomaly Detection v3.1 (API Ready)
@Description:
This script integrates rule-based and SVM-based anomaly detection.
It's refactored to take a single JSON object as input, containing all necessary data
(raw events, transition statistics, geospatial info), and returns a structured JSON output.
This version removes strict deduplication to ensure every unique anomaly is reported
and includes both rule-based and model-based anomalies in the final summary.

@Changelog:
- v3.1 (Current):
  - Refined SVM feature engineering for better stability.
  - Ensured all anomaly types are correctly reported in summaryStats.
- v3.0 (Previous):
  - Reworked to be fully API-driven via `detect_anomalies_from_json`.
  - Input is a single JSON object, output is a single JSON object.
  - Removed deduplication on (event_time, location) to show all distinct EPC anomalies.
  - `details` format changed to: "epc_code | event_time | anomaly_type | worker_id | scan_location".
  - `summaryStats` now includes all rule types and a 'model' key for SVM anomalies.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
# Assuming anomaly_detection_v5 is in the same directory or a reachable path
from anomaly_detection_v5 import detect_rule_based_anomalies, check_epc_format

def get_svm_anomalies(df: pd.DataFrame, product_id: str, lot_id: str):
    """
    Trains a One-Class SVM model on the provided data to find statistical anomalies.
    """
    print(f"--- Running SVM Anomaly Detection for Product {product_id}, Lot {lot_id} ---")
    
    # Feature Engineering
    df_copy = df.copy()
    df_copy['event_time'] = pd.to_datetime(df_copy['event_time'])
    df_copy['manufacture_date'] = pd.to_datetime(df_copy['epc_code'].str.split('.').str[4], format='%Y%m%d', errors='coerce')
    
    df_copy['event_time_dayofweek'] = df_copy['event_time'].dt.dayofweek
    df_copy['event_time_hour'] = df_copy['event_time'].dt.hour
    df_copy['time_since_manufacture_days'] = (df_copy['event_time'] - df_copy['manufacture_date']).dt.days.fillna(0)

    features_to_use = ['event_time_dayofweek', 'event_time_hour', 'time_since_manufacture_days']
    features_df = df_copy[features_to_use]

    if features_df.empty:
        print("No valid features for SVM.")
        return pd.DataFrame()

    # Model Pipeline
    svm_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('svm', OneClassSVM(kernel='rbf', nu=0.05))
    ])
    
    try:
        predictions = svm_pipeline.fit_predict(features_df)
    except Exception as e:
        print(f"Error during SVM training/prediction: {e}")
        return pd.DataFrame()

    anomaly_indices = np.where(predictions == -1)[0]
    svm_anomalies_df = df_copy.iloc[anomaly_indices].copy()
    svm_anomalies_df['anomaly_type'] = 'model'
    
    print(f"Found {len(svm_anomalies_df)} anomalies using One-Class SVM.")
    return svm_anomalies_df

def detect_anomalies_from_json(json_input_str: str):
    """
    Main function to detect both rule-based and SVM anomalies from a JSON input.
    """
    try:
        input_data = json.loads(json_input_str)
        raw_df = pd.DataFrame(input_data['data'])
        transition_stats = pd.DataFrame(input_data['transition_stats'])
        geo_df = pd.DataFrame(input_data['geo_data'])
        product_id = input_data.get('product_id')
        lot_id = input_data.get('lot_id')
    except (json.JSONDecodeError, KeyError) as e:
        return json.dumps({"error": f"Invalid JSON input: {e}"}, indent=2, ensure_ascii=False)

    if raw_df.empty or not product_id or not lot_id:
        return json.dumps({"title": "데이터 부족", "details": [], "summaryStats": {}}, indent=2, ensure_ascii=False)

    # --- 2. Separate Fake EPCs First ---
    fake_mask = ~raw_df['epc_code'].apply(check_epc_format)
    fake_anomalies_df = raw_df[fake_mask].copy()
    fake_anomalies_df['anomaly_type'] = 'epcFake'

    # --- 3. Filter Data for the Target Product and Lot ---
    valid_df = raw_df[~fake_mask].copy()
    valid_df['epc_product'] = valid_df['epc_code'].apply(lambda x: x.split('.')[2])
    valid_df['epc_lot'] = valid_df['epc_code'].apply(lambda x: x.split('.')[3])
    
    target_df = valid_df[(valid_df['epc_product'] == product_id) & (valid_df['epc_lot'] == lot_id)].copy()

    if target_df.empty:
        # If no target data, just report the fake ones if they exist
        if not fake_anomalies_df.empty:
            rule_anomalies_df = fake_anomalies_df
            svm_anomalies_df = pd.DataFrame()
        else:
            return json.dumps({"title": f"제품 {product_id}-로트 {lot_id} 데이터 없음", "details": [], "summaryStats": {}}, indent=2, ensure_ascii=False)
    else:
        # --- 4. Run Further Anomaly Detections ---
        # Rule-Based Anomalies (on already filtered valid data)
        rule_anomalies_df, clean_for_svm = detect_rule_based_anomalies(target_df, transition_stats, geo_df)
        rule_anomalies_df = pd.concat([rule_anomalies_df, fake_anomalies_df], ignore_index=True)
        
        # SVM-Based Anomalies
        svm_anomalies_df = get_svm_anomalies(clean_for_svm, product_id, lot_id)

    combined_anomalies_df = pd.concat([rule_anomalies_df, svm_anomalies_df], ignore_index=True)

    if combined_anomalies_df.empty:
        return json.dumps({"title": f"제품 {product_id}-로트 {lot_id} 이상 이벤트 없음", "details": [], "summaryStats": {}}, indent=2, ensure_ascii=False)

    combined_anomalies_df['event_time'] = pd.to_datetime(combined_anomalies_df['event_time'])
    details_list = combined_anomalies_df.apply(
        lambda row: f"{row['epc_code']} | {row['event_time'].strftime('%Y-%m-%d %H:%M:%S')} | {row['anomaly_type']} | {row.get('worker_id', 'N/A')} | {row['scan_location']}",
        axis=1
    ).tolist()

    summary_stats = combined_anomalies_df['anomaly_type'].value_counts().to_dict()
    all_keys = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump', 'model']
    for key in all_keys:
        summary_stats.setdefault(key, 0)

    title = f"제품 {product_id}-로트 {lot_id} 이상 이벤트 감지"

    output_json = {
        "title": title,
        "details": details_list,
        "summaryStats": summary_stats
    }

    return json.dumps(output_json, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    example_input_json_str = """
    {
      "product_id": "0000001",
      "lot_id": "000001",
      "data": [
        { "epc_code": "001.8804823.0000001.000001.20240701.000000001", "event_time": "2024-07-02 09:00:00", "scan_location": "서울 공장", "event_type": "Inbound", "worker_id": "001", "factory": "hws"},
        { "epc_code": "001.8804823.0000001.000001.20240701.000000001", "event_time": "2024-07-02 12:00:00", "scan_location": "부산 물류센터", "event_type": "Inbound", "worker_id": "002", "factory": "hws"},
        { "epc_code": "001.8804823.0000001.000001.20240701.000000002", "event_time": "2024-07-02 09:23:00", "scan_location": "서울 공장", "event_type": "Outbound", "worker_id": "003", "factory": "hws"},
        { "epc_code": "001.8804823.0000001.000001.20240701.000000002", "event_time": "2024-07-02 09:23:00", "scan_location": "서울 공장", "event_type": "Inbound", "worker_id": "003", "factory": "hws"},
        { "epc_code": "invalid-epc-code", "event_time": "2024-07-02 11:00:00", "scan_location": "대전 물류센터", "event_type": "Inbound", "worker_id": "004", "factory": "icn"}
      ],
      "transition_stats": [
        { "from_scan_location": "서울 공장", "to_scan_location": "부산 물류센터", "time_taken_hours_mean": 1.0, "time_taken_hours_std": 0.2 }
      ],
      "geo_data": [
        { "scan_location": "서울 공장", "Latitude": 37.5665, "Longitude": 126.9780 },
        { "scan_location": "부산 물류센터", "Latitude": 35.1796, "Longitude": 129.0756 },
        { "scan_location": "대전 물류센터", "Latitude": 36.3504, "Longitude": 127.3845 }
      ]
    }
    """
    
    print("--- Running Detection with Example Input ---")
    json_output = detect_anomalies_from_json(example_input_json_str)
    print(json_output)

    example_frontend_json = {
      "title": "제품 0000001-로트 000001 이상 이벤트 감지",
      "details": [
        "invalid-epc-code | 2024-07-02 11:00:00 | epcFake | 004 | 대전 물류센터",
        "001.8804823.0000001.000001.20240701.000000002 | 2024-07-02 09:23:00 | evtOrderErr | 003 | 서울 공장",
        "001.8804823.0000001.000001.20240701.000000001 | 2024-07-02 12:00:00 | jump | 002 | 부산 물류센터",
        "001.8804823.0000001.000001.20240701.000000001 | 2024-07-02 09:00:00 | model | 001 | 서울 공장"
      ],
      "summaryStats": {
        "epcFake": 1,
        "epcDup": 0,
        "locErr": 0,
        "evtOrderErr": 1,
        "jump": 1,
        "model": 1
      }
    }
    print("\n--- Frontend Example JSON ---")
    print(json.dumps(example_frontend_json, indent=2, ensure_ascii=False))