import sys
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List

# For demo, use mock SVMInferenceEngine if not implemented
class MockSVMInferenceEngine:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.anomaly_types = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
    def event_predict(self, event_row: pd.Series) -> Dict:
        # 랜덤 예측 (실제 모델 대체)
        preds = {atype: float(np.random.rand()) for atype in self.anomaly_types}
        detected = {atype: preds[atype] for atype in self.anomaly_types if preds[atype] > 0.5}
        return detected

def preprocess_input(input_path: str) -> pd.DataFrame:
    # Accept CSV or JSON
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            df = pd.DataFrame(data)
    else:
        raise ValueError('Input file must be .csv or .json')
    return df

def build_api_json(df: pd.DataFrame, engine: MockSVMInferenceEngine) -> Dict:
    # EventHistory: eventId별로 감지된 이상치만 dict로
    event_history = []
    epc_stats = {}
    file_stats = {atype+'Count': 0 for atype in engine.anomaly_types}
    file_stats['totalEvents'] = 0
    file_id = int(df['file_id'].iloc[0]) if 'file_id' in df.columns and not df.empty else 1

    for idx, row in df.iterrows():
        event_id = int(row.get('eventId', idx+1))
        epc_code = row.get('epc_code', 'UNKNOWN')
        detected = engine.event_predict(row)
        if not detected:
            continue  # 이상치 없으면 EventHistory에 포함하지 않음
        event_record = {'eventId': event_id}
        for atype, score in detected.items():
            event_record[atype] = True
            event_record[atype+'Score'] = round(float(score)*100, 1)
        event_history.append(event_record)

        # epcAnomalyStats 집계
        if epc_code not in epc_stats:
            epc_stats[epc_code] = {
                'epcCode': epc_code,
                'jumpCount': 0,
                'evtOrderErrCount': 0,
                'epcFakeCount': 0,
                'epcDupCount': 0,
                'locErrCount': 0,
                'totalEvents': 0
            }
        for atype in detected:
            key = atype+'Count'
            if key in epc_stats[epc_code]:
                epc_stats[epc_code][key] += 1
                file_stats[key] += 1
                epc_stats[epc_code]['totalEvents'] += 1
                file_stats['totalEvents'] += 1

    # epcAnomalyStats: 모든 EPC 포함
    epc_anomaly_stats = list(epc_stats.values())

    output = {
        'fileId': file_id,
        'EventHistory': event_history,
        'epcAnomalyStats': epc_anomaly_stats,
        'fileAnomalyStats': file_stats
    }
    return output

def main():
    if len(sys.argv) < 2:
        print('Usage: python svm_predict_demo.py <input_file.csv|json>')
        sys.exit(1)
    input_path = sys.argv[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else 'models/svm'
    df = preprocess_input(input_path)
    engine = MockSVMInferenceEngine(model_dir)
    api_json = build_api_json(df, engine)
    print(json.dumps(api_json, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main() 