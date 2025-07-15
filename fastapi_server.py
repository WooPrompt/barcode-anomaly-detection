#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI Server for Multi-Anomaly Detection
Author: Data Analysis Team
Date: 2025-07-14

FastAPI implementation with automatic documentation and validation.
Supports both detection API (for BE) and report API (for FE).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.barcode.multi_anomaly_detector import detect_anomalies_from_json_enhanced, detect_anomalies_backend_format, save_detection_result

app = FastAPI(
    title="바코드 이상치 탐지 API",
    description="""
    ## 공급망 바코드 이상치 탐지 시스템
    
    ### 기능
    - **실시간 이상치 탐지**: 5가지 이상치 유형 동시 탐지
    - **다중 이상치 지원**: 하나의 EPC에서 여러 이상치 동시 발견 가능
    - **확률 점수**: 각 이상치 유형별 0-100% 확률 점수 제공
    - **시퀀스 분석**: 문제 발생 단계 정확히 식별
    
    ### 탐지 가능한 이상치 유형
    1. **epcFake**: EPC 코드 형식 위반 (구조, 회사코드, 날짜 오류)
    2. **epcDup**: 불가능한 중복 스캔 (동일시간, 다른장소)
    3. **jump**: 불가능한 이동시간 (물리적으로 불가능한 시공간 점프)
    4. **evtOrderErr**: 이벤트 순서 오류 (연속 인바운드/아웃바운드)
    5. **locErr**: 위치 계층 위반 (소매→도매 역순 이동)
    
    ### 개발팀
    - **데이터 분석**: 이상치 탐지 알고리즘 개발
    - **백엔드**: API 서버 및 데이터베이스 관리  
    - **프론트엔드**: 웹 UI 및 시각화 구현
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class ScanRecord(BaseModel):
    epc_code: str
    event_time: str
    scan_location: str
    event_type: str
    product_name: Optional[str] = "Unknown"
    business_step: Optional[str] = None
    worker_id: Optional[str] = None

# Backend format models (d.txt specification)
class BackendScanRecord(BaseModel):
    eventId: int
    epc_code: str
    location_id: int
    business_step: str
    event_type: str
    event_time: str
    file_id: int

class BackendAnomalyDetectionRequest(BaseModel):
    data: List[BackendScanRecord]

class EventHistoryRecord(BaseModel):
    eventId: int
    jump: Optional[bool] = None
    jumpScore: Optional[float] = None
    evtOrderErr: Optional[bool] = None
    evtOrderErrScore: Optional[float] = None
    epcDup: Optional[bool] = None
    epcDupScore: Optional[float] = None
    epcFake: Optional[bool] = None
    epcFakeScore: Optional[float] = None
    locErr: Optional[bool] = None
    locErrScore: Optional[float] = None

class EpcAnomalyStats(BaseModel):
    epcCode: str
    totalEvents: int
    jumpCount: int
    evtOrderErrCount: int
    epcFakeCount: int
    epcDupCount: int
    locErrCount: int

class FileAnomalyStats(BaseModel):
    totalEvents: int
    jumpCount: int
    evtOrderErrCount: int
    epcFakeCount: int
    epcDupCount: int
    locErrCount: int

class BackendAnomalyDetectionResponse(BaseModel):
    fileId: int
    EventHistory: List[EventHistoryRecord]
    epcAnomalyStats: List[EpcAnomalyStats]
    fileAnomalyStats: FileAnomalyStats

class TransitionStat(BaseModel):
    from_scan_location: str
    to_scan_location: str
    time_taken_hours_mean: float
    time_taken_hours_std: float

class GeoData(BaseModel):
    scan_location: str
    Latitude: float
    Longitude: float

class AnomalyDetectionRequest(BaseModel):
    data: List[ScanRecord]
    transition_stats: Optional[List[TransitionStat]] = []
    geo_data: Optional[List[GeoData]] = []

class EventHistoryItem(BaseModel):
    epcCode: str
    productName: str
    businessStep: str
    scanLocation: str
    eventTime: str
    anomaly: bool
    anomalyTypes: List[str]
    anomalyScores: Dict[str, int]
    sequencePosition: int
    totalSequenceLength: int
    primaryAnomaly: str
    problemStep: str
    description: str

class AnomalyDetectionResponse(BaseModel):
    EventHistory: List[EventHistoryItem]
    summaryStats: Dict[str, int]
    multiAnomalyCount: int
    totalAnomalyCount: int

class ReportDetail(BaseModel):
    epcCode: str
    timestamp: str
    anomalyTypes: List[str]
    sequencePosition: int
    location: str
    scores: Dict[str, int]
    problemStep: str

class ReportResponse(BaseModel):
    title: str
    details: List[ReportDetail]
    summaryStats: Dict[str, int]
    multiAnomalyCount: int
    sequenceProblems: List[str]

# Global storage for reports (in production, use database)
reports_storage = {}

@app.get("/")
async def root():
    """루트 엔드포인트 - API 정보 제공"""
    return {
        "message": "바코드 이상치 탐지 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "이상치_탐지": "POST /api/v1/barcode-anomaly-detect",
            "리포트_목록": "GET /api/reports",
            "리포트_상세": "GET /api/report/detail?reportId=xxx",
            "헬스체크": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트 - 서버 상태 확인"""
    return {"status": "정상", "service": "이상치-탐지-서비스"}

@app.post(
    "/api/v1/barcode-anomaly-detect",
    summary="다중 이상치 탐지 (백엔드용 - 즉시 응답)",
    description="백엔드에서 데이터를 전송하고 즉시 결과를 받는 엔드포인트: epcFake, epcDup, jump, evtOrderErr, locErr"
)
async def detect_anomalies_backend(request: BackendAnomalyDetectionRequest):
    """
    백엔드 통합용 다중 이상치 탐지 엔드포인트 (즉시 응답)
    
    **입력**: eventId, location_id 기반 스캔 데이터
    **출력**: fileId, EventHistory, epcAnomalyStats, fileAnomalyStats 형식 (즉시 응답)
    
    **탐지 가능한 이상치 유형:**
    - epcFake: 잘못된 EPC 형식
    - epcDup: 불가능한 중복 스캔
    - jump: 불가능한 이동 시간
    - evtOrderErr: 잘못된 이벤트 순서
    - locErr: 위치 계층 위반
    """
    try:
        # Convert Pydantic model to JSON string for backend function
        request_json = request.json()
        
        # Call backend-compatible detection function
        result_json = detect_anomalies_backend_format(request_json)
        
        # Parse result and return immediately
        result_dict = json.loads(result_json)
        
        # Optional: Save result for ML training data accumulation
        save_detection_result(result_dict, request_json)
        
        return result_dict
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

@app.get(
    "/api/manager/export",
    response_model=BackendAnomalyDetectionResponse,
    summary="최근 탐지 결과 조회 (GET 방식)",
    description="가장 최근에 저장된 이상치 탐지 결과를 조회 (ML 훈련 데이터용)"
)
async def export_anomaly_data():
    """
    최근 이상치 탐지 결과를 GET 방식으로 조회
    
    **사용 목적**: ML 모델 훈련을 위한 축적된 데이터 조회
    **주 사용자**: POST API 호출 후 결과 재확인이 필요한 경우
    """
    try:
        # Read the most recent saved detection result from JSON files
        import glob
        
        # Find the most recent JSON file in ml_training_data directory
        json_files = glob.glob("ml_training_data/*.json")
        if not json_files:
            # Return empty result if no saved data available
            return {
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
            }
        
        # Get the most recent file by modification time
        most_recent_file = max(json_files, key=os.path.getmtime)
        
        # Load and return the result
        with open(most_recent_file, 'r', encoding='utf-8') as f:
            result_data = json.load(f)
            return result_data["detection_result"]
            
    except Exception as e:
        # Fallback to empty result on any error
        return {
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
        }

@app.post(
    "/api/v1/barcode-anomaly-detect-legacy",
    response_model=AnomalyDetectionResponse,
    summary="다중 이상치 탐지 (레거시)",
    description="이전 버전 호환성을 위한 엔드포인트"
)
async def detect_anomalies_legacy(request: AnomalyDetectionRequest):
    """
    레거시 호환성을 위한 다중 이상치 탐지 엔드포인트
    """
    try:
        # Convert Pydantic model to JSON string for existing function
        request_json = request.json()
        
        # Call existing detection function
        result_json = detect_anomalies_from_json_enhanced(request_json)
        
        # Parse result and return
        result_dict = json.loads(result_json)
        
        # Store result for report generation
        report_id = f"legacy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        reports_storage[report_id] = {
            "result": result_dict,
            "created_at": datetime.now().isoformat(),
            "type": "legacy_format"
        }
        
        return result_dict
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

@app.get("/api/reports")
async def get_reports():
    """사용 가능한 리포트 목록 조회"""
    report_list = []
    for report_id, data in reports_storage.items():
        label = f"이상치 탐지 리포트 {report_id.split('_')[-1]}"
        report_list.append({
            "id": report_id,
            "label": label,
            "created_at": data["created_at"]
        })
    return report_list

@app.get(
    "/api/report/detail",
    response_model=ReportResponse,
    summary="리포트 상세 조회 (프론트엔드용)",
    description="시퀀스 정보가 포함된 UI 표시용 포맷 리포트 조회"
)
async def get_report_detail(reportId: str):
    """
    프론트엔드 통합용 상세 리포트 조회
    
    **응답**: UI 표시에 최적화된 리포트 형식
    """
    if reportId not in reports_storage:
        raise HTTPException(status_code=404, detail="Report not found")
    
    stored_data = reports_storage[reportId]
    result_data = stored_data["result"]
    
    # Convert EventHistory format to Report format
    details = []
    sequence_problems = set()
    
    for event in result_data.get("EventHistory", []):
        detail = ReportDetail(
            epcCode=event["epcCode"],
            timestamp=event["eventTime"],
            anomalyTypes=event["anomalyTypes"],
            sequencePosition=event["sequencePosition"],
            location=event["scanLocation"],
            scores=event["anomalyScores"],
            problemStep=event["problemStep"]
        )
        details.append(detail)
        
        # Extract sequence problems
        if "Factory" in event["problemStep"]:
            sequence_problems.add("Factory")
        if "Logistics" in event["problemStep"]:
            sequence_problems.add("Logistics_HUB")
        if "Warehouse" in event["problemStep"]:
            sequence_problems.add("W_Stock")
        if "Retail" in event["problemStep"]:
            sequence_problems.add("R_Stock")
    
    report_response = ReportResponse(
        title=f"다중 이상치 탐지 리포트",
        details=details,
        summaryStats=result_data.get("summaryStats", {}),
        multiAnomalyCount=result_data.get("multiAnomalyCount", 0),
        sequenceProblems=list(sequence_problems)
    )
    
    return report_response

@app.post("/api/v1/test-with-sample")
async def test_with_sample_data():
    """샘플 데이터를 사용한 테스트 엔드포인트"""
    try:
        # Load sample data
        with open('test_data_sample.json', 'r', encoding='utf-8') as f:
            sample_data = f.read()
        
        # Run detection
        result = detect_anomalies_from_json_enhanced(sample_data)
        result_dict = json.loads(result)
        
        # Store for report testing
        report_id = "sample_test_report"
        reports_storage[report_id] = {
            "result": result_dict,
            "product_id": "1203199",
            "lot_id": "150002",
            "created_at": datetime.now().isoformat()
        }
        
        return {
            "detection_result": result_dict,
            "report_id": report_id,
            "test_report_url": f"/api/report/detail?reportId={report_id}"
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="test_data_sample.json not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test error: {e}")

if __name__ == "__main__":
    import uvicorn
    
    print("바코드 이상치 탐지 FastAPI 서버 시작")
    print("API 문서: http://localhost:8000/docs")
    print("대체 문서: http://localhost:8000/redoc")
    print("테스트 엔드포인트: http://localhost:8000/api/v1/test-with-sample")
    print("리포트 API: http://localhost:8000/api/reports")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True  # Auto-reload on code changes
    )