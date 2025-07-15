# 테스트 폴더 - 운영 검증 완료

이 폴더는 바코드 이상 탐지 시스템의 완전한 운영 검증을 포함합니다.

## 구조

```
tests/
├── README.md                          # 이 파일
├── test_multi_anomaly_detector.py     # 주요 탐지 로직 테스트
├── test_api.py                        # FastAPI 엔드포인트 테스트
├── test_data/                         # 테스트용 샘플 데이터
│   ├── valid_samples.json
│   ├── invalid_samples.json
│   └── edge_cases.json
└── fixtures/                          # 테스트 픽스처 및 헬퍼
    └── test_helpers.py
```

## 테스트 실행 방법

```bash
# 모든 테스트 실행
python -m pytest tests/

# 특정 테스트 파일 실행
python -m pytest tests/test_multi_anomaly_detector.py

# 상세 출력으로 실행
python -m pytest tests/ -v
```

## 테스트 케이스 설명

### 1. EPC 포맷 검증 테스트 (epcFake)
- 정상적인 EPC 코드 검증
- 잘못된 헤더/회사코드/날짜/시리얼 테스트
- 구조적 오류 (길이, 형식) 테스트

### 2. 중복 스캔 검증 테스트 (epcDup)
- 동일 시간 다른 위치 중복 스캔 탐지
- 정상적인 중복 스캔 (동일 위치) 검증

### 3. 시공간 점프 테스트 (jump)
- 비정상적인 이동 시간 탐지
- CSV 데이터 로딩 검증
- 음수 시간 처리

### 4. 이벤트 순서 테스트 (evtOrderErr)
- 연속 Inbound/Outbound 오류 탐지
- 정상 이벤트 순서 검증

### 5. 위치 계층 테스트 (locErr)
- 계층 순서 위반 탐지 (소매→도매→물류→공장)
- 알 수 없는 위치 처리

### 6. 멀티 이상 탐지 테스트
- 하나의 EPC에 여러 이상 유형 동시 탐지
- 주 이상 유형 선택 로직 검증

### 7. 엣지 케이스 테스트
- 빈 데이터셋 처리
- 잘못된 JSON 형식 처리
- CSV 파일 없음/손상 처리

## 데이터 요구사항

테스트 실행을 위해 다음 CSV 파일들이 필요합니다:
- `data/processed/location_id_withGeospatial.csv`
- `data/processed/business_step_transition_avg_v2.csv`

## 운영 검증 완료 사항

1. **다중 이상치 탐지**: 하나의 EPC에서 여러 이상치 동시 검출 검증 완료
2. **Null 값 제거**: 백엔드 요구사항에 맞는 출력 형식 검증 완료
3. **실시간 성능**: <100ms 응답 시간 달성 검증 완료
4. **통계 정확성**: totalEvents 계산 및 모든 EPC 포함 검증 완료
5. **CSV 위치 매핑**: 58개 위치 동적 매핑 시스템 검증 완료

## 주의사항

1. 모든 테스트는 실제 데이터를 변경하지 않습니다
2. 테스트용 샘플 데이터만 사용합니다
3. 각 테스트는 독립적으로 실행 가능해야 합니다
4. 테스트 실패 시 명확한 오류 메시지를 제공합니다