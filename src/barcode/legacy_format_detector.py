# -*- coding: utf-8 -*-
"""
Legacy Format Multi-Anomaly Detector
Provides the traditional title/details/summaryStats format with multi-anomaly support
For compatibility with existing frontend implementations
"""

import pandas as pd
import json
from multi_anomaly_detector import detect_multi_anomalies_enhanced

def detect_anomalies_legacy_format(json_input_str: str) -> str:
    """
    Enhanced anomaly detection with legacy format output + multi-anomaly support.
    
    Returns the traditional format:
    {
      "title": "제품 XXX-로트 XXX 이상 이벤트 감지", 
      "details": [
        "epc_code | timestamp | anomaly_types | worker_id | location"
      ],
      "summaryStats": {"epcFake": 0, "epcDup": 0, ...}
    }
    
    Multi-anomaly support: anomaly_types field shows comma-separated list
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

    if raw_df.empty:
        return json.dumps({
            "title": "데이터 없음",
            "details": [],
            "summaryStats": {"epcFake": 0, "epcDup": 0, "locErr": 0, "evtOrderErr": 0, "jump": 0}
        }, indent=2, ensure_ascii=False)

    # Filter by product and lot if specified
    if product_id and lot_id:
        raw_df['epc_product'] = raw_df['epc_code'].apply(lambda x: x.split('.')[2] if len(x.split('.')) >= 3 else None)
        raw_df['epc_lot'] = raw_df['epc_code'].apply(lambda x: x.split('.')[3] if len(x.split('.')) >= 4 else None)
        filtered_df = raw_df[(raw_df['epc_product'] == product_id) & (raw_df['epc_lot'] == lot_id)]
    else:
        filtered_df = raw_df

    # Detect anomalies using the enhanced multi-anomaly detector
    anomaly_results = detect_multi_anomalies_enhanced(filtered_df, transition_stats, geo_df)

    # Convert to legacy format
    details_list = []
    for result in anomaly_results:
        # Join multiple anomaly types with comma
        anomaly_types_str = ",".join(result['anomalyTypes'])
        
        # Get worker_id from first occurrence of this EPC
        epc_data = filtered_df[filtered_df['epc_code'] == result['epcCode']]
        worker_id = epc_data.iloc[0].get('worker_id', 'N/A') if not epc_data.empty else 'N/A'
        
        # Format: "epc_code | timestamp | anomaly_types | worker_id | location"
        detail_str = f"{result['epcCode']} | {result['eventTime']} | {anomaly_types_str} | {worker_id} | {result['scanLocation']}"
        details_list.append(detail_str)

    # Calculate summary statistics
    summary_stats = {"epcFake": 0, "epcDup": 0, "locErr": 0, "evtOrderErr": 0, "jump": 0}
    
    for result in anomaly_results:
        for anomaly_type in result['anomalyTypes']:
            if anomaly_type in summary_stats:
                summary_stats[anomaly_type] += 1

    # Generate title
    if product_id and lot_id:
        title = f"제품 {product_id}-로트 {lot_id} 이상 이벤트 감지"
    else:
        title = "전체 데이터 이상 이벤트 감지"

    # Additional statistics for multi-anomaly tracking
    multi_anomaly_count = sum(1 for result in anomaly_results if len(result['anomalyTypes']) > 1)
    
    output = {
        "title": title,
        "details": details_list,
        "summaryStats": summary_stats,
        "multiAnomalyCount": multi_anomaly_count,  # Extra field for multi-anomaly tracking
        "totalAnomalyCount": len(anomaly_results)
    }

    return json.dumps(output, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    # Test with sample data
    test_data = {
        "product_id": "0000001",
        "lot_id": "000001", 
        "data": [
            {
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "event_time": "2024-07-02 09:00:00",
                "scan_location": "서울 공장",
                "event_type": "Inbound",
                "worker_id": "001"
            },
            {
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "event_time": "2024-07-02 09:00:00", 
                "scan_location": "부산 공장",
                "event_type": "Inbound",
                "worker_id": "002"
            },
            {
                "epc_code": "invalid.format.epc",
                "event_time": "2024-07-02 10:00:00",
                "scan_location": "인천 물류센터", 
                "event_type": "Outbound",
                "worker_id": "003"
            }
        ],
        "transition_stats": [],
        "geo_data": []
    }
    
    test_json = json.dumps(test_data)
    result = detect_anomalies_legacy_format(test_json)
    print("Legacy Format Multi-Anomaly Detection Result:")
    print(result)