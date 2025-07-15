# -*- coding: utf-8 -*-
"""
JSON 출력 기본 테스트
사용자가 요청한 대로 JSON 출력부터 확인
"""

import json
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from barcode.multi_anomaly_detector import detect_anomalies_from_json_enhanced

def test_basic_json_output():
    """기본적인 JSON 출력이 되는지 테스트"""
    
    print("=== JSON 출력 테스트 ===")
    
    # 간단한 테스트 데이터
    test_data = {
        "data": [
            {
                "scan_location": "서울 공장",
                "location_id": 1,
                "business_step": "Factory",
                "event_type": "Outbound",
                "epc_code": "001.8804823.0000001.000001.20240701.000000001",
                "product_name": "Product A",
                "event_time": "2024-07-02 09:00:00"
            }
        ]
    }
    
    print("입력 데이터:")
    print(json.dumps(test_data, ensure_ascii=False, indent=2))
    print()
    
    # JSON 변환
    test_json = json.dumps(test_data, ensure_ascii=False)
    
    try:
        # 함수 실행
        result = detect_anomalies_from_json_enhanced(test_json)
        
        print("출력 결과:")
        print(result)
        print()
        
        # JSON 파싱 테스트
        result_data = json.loads(result)
        
        print("=== 파싱 검증 ===")
        print(f"EventHistory 존재: {'EventHistory' in result_data}")
        print(f"summaryStats 존재: {'summaryStats' in result_data}")
        print(f"multiAnomalyCount 존재: {'multiAnomalyCount' in result_data}")
        print(f"totalAnomalyCount 존재: {'totalAnomalyCount' in result_data}")
        
        print()
        print("=== 내용 확인 ===")
        print(f"탐지된 이상 개수: {result_data.get('totalAnomalyCount', 0)}")
        print(f"요약 통계: {result_data.get('summaryStats', {})}")
        
        print("\nJSON 출력 테스트 성공")
        return True
        
    except Exception as e:
        print(f"테스트 실패: {e}")
        return False

def test_empty_data():
    """빈 데이터 처리 테스트"""
    
    print("\n=== 빈 데이터 테스트 ===")
    
    empty_data = {"data": []}
    test_json = json.dumps(empty_data, ensure_ascii=False)
    
    try:
        result = detect_anomalies_from_json_enhanced(test_json)
        result_data = json.loads(result)
        
        print(f"빈 데이터 결과: {result_data.get('totalAnomalyCount', 0)}개 이상")
        print("빈 데이터 처리 성공")
        return True
        
    except Exception as e:
        print(f"빈 데이터 테스트 실패: {e}")
        return False

def test_invalid_json():
    """잘못된 JSON 처리 테스트"""
    
    print("\n=== 잘못된 JSON 테스트 ===")
    
    invalid_json = "{'invalid': 'json'}"  # 작은따옴표 사용
    
    try:
        result = detect_anomalies_from_json_enhanced(invalid_json)
        result_data = json.loads(result)
        
        if "error" in result_data:
            print("잘못된 JSON 오류 처리 성공")
            return True
        else:
            print("오류 처리되지 않음")
            return False
            
    except Exception as e:
        print(f"잘못된 JSON 테스트 실패: {e}")
        return False

if __name__ == "__main__":
    print("JSON 출력 기본 테스트 시작...\n")
    
    # 테스트 실행
    test1 = test_basic_json_output()
    test2 = test_empty_data()
    test3 = test_invalid_json()
    
    print(f"\n=== 테스트 결과 ===")
    print(f"기본 JSON 출력: {'PASS' if test1 else 'FAIL'}")
    print(f"빈 데이터 처리: {'PASS' if test2 else 'FAIL'}")
    print(f"잘못된 JSON 처리: {'PASS' if test3 else 'FAIL'}")
    
    if all([test1, test2, test3]):
        print("\n모든 JSON 테스트 통과!")
    else:
        print("\n일부 테스트 실패")