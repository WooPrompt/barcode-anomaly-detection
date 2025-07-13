# 🎯 완전한 팀 프로젝트 요구사항 분석 (업데이트)
## AI프로젝트-3조: 2차원 바코드 기반 물류 공급망 불법 유통 분석 웹서비스

---

## 👥 **팀 구성 및 역할**
- **프론트엔드 팀**: UI/UX 구현 담당 개발자들
- **백엔드 개발자**: API 개발 및 데이터 관리 담당  
- **데이터 분석가**: 룰베이스 이상치 탐지 시스템 개발 담당

## 🔄 **새로운 핵심 요구사항 (필수 업데이트)**

### **1. EPC별 다중 이상치 동시 탐지**
- **요구사항**: 하나의 EPC 코드가 여러 종류의 이상치에 동시에 걸릴 수 있음
- **구현 방식**: 모든 탐지 함수가 각 EPC를 5가지 이상치 유형 전체에 대해 검증
- **예시**: 시간점프로 걸린 EPC도 epcFake, epcDup, locErr, evtOrderErr 추가 검증 필요

### **2. 시퀀스 단계별 이상 지점 식별**
- **요구사항**: EPC가 공장→창고→물류센터→소매상 경로 중 어느 단계에서 문제인지 정확히 파악
- **시각화**: 문제 지점을 빨간색으로 하이라이트 (프론트엔드 요청)
- **데이터 구조**: 각 이상치마다 발생 시퀀스 위치와 단계 상세 정보 포함

### **3. 이상 확률 점수 시스템**
- **프론트엔드 요청**: 각 EPC의 이상 확률을 퍼센트로 표시
- **구현 방식**: 룰베이스 신뢰도 점수 (0-100%)
- **향후 확장**: 머신러닝 모델 통합을 위한 기반 구조

---

## 🚀 **데이터 분석가 최우선 작업 계획**

### **🔥 즉시 필요한 작업 (오늘~내일)**

#### **1. 다중 이상치 탐지 API 엔드포인트 완성**
```python
# POST /api/v1/barcode-anomaly-detect
# 5개 함수 통합 + EPC별 다중 이상치 동시 탐지 기능
```

#### **2. 업데이트된 응답 형식 (시퀀스 정보 포함)**
```json
{
  "EventHistory": [
    {
      "epcCode": "001.8809437.1203199.150002.20250701.000000002",
      "productName": "Product 2", 
      "businessStep": "Factory",
      "scanLocation": "구미공장",
      "eventTime": "2025-07-01 10:23:39",
      "anomaly": true,
      "anomalyTypes": ["jump", "epcFake"],
      "anomalyScores": {"jump": 85, "epcFake": 72},
      "sequencePosition": 3,
      "totalSequenceLength": 7,
      "primaryAnomaly": "jump",
      "problemStep": "Factory_to_Logistics",
      "description": "다중 이상치 탐지: 시간점프 + EPC 형식 위반"
    }
  ]
}
```

#### **3. 다중 이상치 지원 리포트 API 개발**
팀에서 이미 프론트엔드 구현 완료, **다중 이상치 지원 백엔드 API만 기다리는 상황**:

```python
# GET /api/reports?product=제품A&lot=로트1-001&menu=이상탐지리포트&type=all
# 다중 이상치 유형 동시 조회 지원
# GET /api/report/detail?reportId=report_002&includeSequence=true
```

---

## 📋 **상세 기술 요구사항**

### **백엔드 API 명세 (업데이트된 요구사항)**

#### **1. 핵심 데이터 처리 API (다중 이상치 지원)**
```python
POST /api/v1/barcode-anomaly-detect
Input: JSON with data array (CSV 파일 전체 처리)
Output: EventHistory format with multiple anomaly detection per EPC
기능: 
- 시퀀스 단계별 이상 지점 식별
- 각 이상치 유형별 확률 점수
- EPC별 다중 이상치 동시 탐지
```

#### **2. 향상된 리포트 생성 API**
```python
# 다중 이상치 지원 리포트 목록 조회
GET /api/reports
Response: [{"id": "report_001", "label": "제품A-로트1-001 다중 이상치 탐지 1"}]

# 시퀀스 정보 포함 상세 리포트
GET /api/report/detail?reportId=report_002  
Response: {
  "title": "제품 0000001-로트 000001 다중 이상치 탐지",
  "details": [
    {
      "epcCode": "001.8804823.0000001.000001.20240701.000000002",
      "timestamp": "2024-07-02 09:23:00",
      "anomalyTypes": ["evtOrderErr", "jump"],
      "sequencePosition": 3,
      "location": "서울 공장",
      "scores": {"evtOrderErr": 78, "jump": 65},
      "problemStep": "Inbound_to_Outbound"
    }
  ],
  "summaryStats": {
    "epcFake": 0, "epcDup": 0, "locErr": 0, "evtOrderErr": 1, "jump": 1
  },
  "multiAnomalyCount": 1,
  "sequenceProblems": ["Factory", "Logistics_HUB"]
}
```

#### **3. 지도 시각화 API**
```python
# 노드 정보 조회 (58개 위치)
GET /api/manager/nodes
Response: [{"hubType": "ICN_Factory", "scanLocation": "인천공장", "businessStep": "Factory", "coord": [126.65, 37.45]}]

# 이상 데이터 조회
GET /api/manager/anomalies
Response: Trip 객체 배열 with anomaly types
```

#### **4. 대시보드 KPI API**
```python
GET /api/manager/kpi
Response: {
  "totalTripCount": 854320000,
  "uniqueProductCount": 128, 
  "anomalyCount": 125,
  "anomalyRate": 0.0146,
  "salesRate": 92.5
}
```

### **Frontend 요구사항 **

#### **1. 완료된 기능들 (API 연동 대기 중)**
- ✅ **파일 업로드 시스템**: CSV 업로드, 미리보기, 필터링, 검색
- ✅ **리포트 생성 시스템**: Chart.js 도넛차트, PDF 저장, Excel 내보내기  
- ✅ **사용자 관리**: 로그인, 회원가입, 승인 시스템
- ✅ **대시보드**: 공장별 통계, KPI 카드, 지도 시각화
- ✅ **시각화 지도**: DeckGL, 58개 위치 좌표, 이상치 하이라이트

#### **2. API 연동 필요한 부분**
- **실시간 이상치 데이터** 표시 
- **비즈니스 스텝별 필터링** (Factory, WMS, Logistics_HUB, W_Stock, R_Stock, POS_Sell)
- **5가지 이상치 유형별 색상 코딩** (jump: 보라색, evtOrderErr: 주황색, etc.)

---

## 🛠 **데이터 분석가 구체적 작업 계획**

### **Phase 1: 즉시 (오늘)**
1. **5개 함수 통합**: 단일 API 엔드포인트로 통합
2. **응답 형식 맞추기**: EventHistory JSON 형식으로 변경
3. **비즈니스 스텝 필터링**: 6개 카테고리별 필터링 로직 추가

### **Phase 2: 내일**  
1. **리포트 API 구현**: product/lot별 이상치 목록 및 상세 정보
2. **통계 데이터 API**: summaryStats 형식으로 각 이상치 유형별 카운트
3. **성능 최적화**: 920,000 레코드 처리 최적화

### **Phase 3: 이번 주 내**
1. **지도 API 연동**: 58개 위치 + 이상치 트립 데이터
2. **KPI 대시보드 API**: 총 처리 건수, 이상 발생률, 판매율 등
3. **실시간 모니터링**: WebSocket 또는 폴링 기반 실시간 업데이트

---

## 📊 **실제 데이터 현황**

### **규모**
- **총 레코드**: 920,000개 (4개 공장 통합)
- **지역**: 58개 위치 (공장→창고→물류센터→도매상→소매상)
- **현재 탐지된 이상치**: 600개 (evtOrderErr 기준)

### **5가지 이상치 정의 (최종 확정)**
| 이상치 유형 | 변수명 | 설명 | 우선순위 |
|------------|--------|------|----------|
| 시공간 점프 | `jump` | 비논리적인 시공간 이동 | HIGH |
| 이벤트 순서 오류 | `evtOrderErr` | 인바운드/아웃바운드 순서 오류 | HIGH |
| EPC 위조 | `epcFake` | EPC 생성 규칙 위반 | MEDIUM |
| EPC 복제 | `epcDup` | 동시 다발적 스캔 | MEDIUM |
| 경로 위조 | `locErr` | 승인되지 않은 경로 이탈 | MEDIUM |

---

## 🎨 **시각화 요구사항**

### **지도 시각화 (강나현 완성)**
- **DeckGL 기반** 인터랙티브 맵
- **58개 위치** 실제 좌표 반영
- **이상치별 색상 코딩**:
  - jump: 보라색 (#8800ff)
  - evtOrderErr: 주황색 (#ff8800) 
  - epcFake: 빨간색 (#cc0000)
  - epcDup: 노란색 (#ffff00)
  - locErr: 파란색 (#0088ff)

### **대시보드 차트 (이유리 완성)**
- **Chart.js 도넛차트**: 정상/이상 비율
- **막대 차트**: 이상치 유형별 발생 건수
- **PDF/Excel 내보내기**: html2pdf 호환성 해결 완료

---

## 🔄 **현재 프로젝트 상태**

### **✅ 완료됨**
- **데이터 분석**: 5개 탐지 함수 개발 완료 ✅
- **Frontend UI**: 모든 화면 구현 완료 ✅  
- **Backend 인증**: 로그인/회원가입 완료 ✅
- **데이터베이스**: MySQL 스키마 완료 ✅

### **🔄 진행 중 (API 연동 대기)**
- **이상치 탐지 API**: 통합 필요 ⏳
- **리포트 생성 API**: 데이터 연동 필요 ⏳
- **대시보드 KPI**: 실시간 데이터 필요 ⏳

### **⏰ 대기 중**
- **성능 테스트**: API 완성 후 진행
- **배포**: 전체 통합 후 진행

---

## 🎯 **팀 기대사항 요약**

### **Backend 팀이 당신에게 기대하는 것:**
1. **즉시**: `/api/v1/barcode-anomaly-detect` 완성된 엔드포인트 
2. **내일**: 리포트 API (`/api/reports`, `/api/report/detail`) 
3. **이번 주**: KPI 대시보드 API (`/api/manager/kpi`)

### **Frontend 팀이 당신에게 기대하는 것:**
1. **정확한 JSON 형식**: EventHistory 스키마 준수
2. **실시간 성능**: <100ms 응답 시간 목표  
3. **비즈니스 로직**: 6개 스텝 필터링 지원

### **전체 팀이 당신에게 기대하는 것:**
1. **주도적 역할**: 데이터 분석 전문가로서 기술적 가이드 제공
2. **빠른 통합**: 이미 완성된 프론트엔드와 백엔드 연결
3. **안정적 성능**: 920,000 레코드 실시간 처리



*문서 생성일: 2025-07-13*  
*기반 자료: 7개 팀 프로젝트 문서 통합 분석*