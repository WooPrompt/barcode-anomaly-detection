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

from src.barcode.multi_anomaly_detector import detect_anomalies_from_json_enhanced

app = FastAPI(
    title="Barcode Anomaly Detection API",
    description="Multi-anomaly detection system for supply chain barcode analysis",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc"  # ReDoc at http://localhost:8000/redoc
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
    product_id: Optional[str] = None
    lot_id: Optional[str] = None
    data: List[ScanRecord]
    transition_stats: List[TransitionStat]
    geo_data: List[GeoData]

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
    """Root endpoint with API information."""
    return {
        "message": "Barcode Anomaly Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "detect": "POST /api/v1/barcode-anomaly-detect",
            "reports": "GET /api/reports",
            "report_detail": "GET /api/report/detail?reportId=xxx",
            "health": "GET /health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "anomaly-detection"}

@app.post(
    "/api/v1/barcode-anomaly-detect",
    response_model=AnomalyDetectionResponse,
    summary="Detect Multiple Anomaly Types (for Backend)",
    description="Analyze scan data for 5 types of anomalies: epcFake, epcDup, jump, evtOrderErr, locErr"
)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Multi-anomaly detection endpoint for Backend integration.
    
    **Input**: Scan data with transition statistics and geo data
    **Output**: EventHistory format with multi-anomaly detection per EPC
    
    **Anomaly Types Detected:**
    - epcFake: Invalid EPC format
    - epcDup: Impossible duplicate scans 
    - jump: Impossible travel times
    - evtOrderErr: Invalid event sequences
    - locErr: Location hierarchy violations
    """
    try:
        # Convert Pydantic model to JSON string for existing function
        request_json = request.json()
        
        # Call existing detection function
        result_json = detect_anomalies_from_json_enhanced(request_json)
        
        # Parse result and return
        result_dict = json.loads(result_json)
        
        # Store result for report generation
        report_id = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        reports_storage[report_id] = {
            "result": result_dict,
            "product_id": request.product_id,
            "lot_id": request.lot_id,
            "created_at": datetime.now().isoformat()
        }
        
        return result_dict
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

@app.get("/api/reports")
async def get_reports():
    """Get list of available reports."""
    report_list = []
    for report_id, data in reports_storage.items():
        product_id = data.get("product_id", "Unknown")
        lot_id = data.get("lot_id", "Unknown")
        label = f"제품{product_id}-로트{lot_id} 다중 이상치 탐지"
        report_list.append({
            "id": report_id,
            "label": label,
            "created_at": data["created_at"]
        })
    return report_list

@app.get(
    "/api/report/detail",
    response_model=ReportResponse,
    summary="Get Report Details (for Frontend)",
    description="Get formatted report for UI display with sequence information"
)
async def get_report_detail(reportId: str):
    """
    Get detailed report for Frontend integration.
    
    **Response**: Report format optimized for UI display
    """
    if reportId not in reports_storage:
        raise HTTPException(status_code=404, detail="Report not found")
    
    stored_data = reports_storage[reportId]
    result_data = stored_data["result"]
    product_id = stored_data.get("product_id", "Unknown")
    lot_id = stored_data.get("lot_id", "Unknown")
    
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
        title=f"제품 {product_id}-로트 {lot_id} 다중 이상치 탐지",
        details=details,
        summaryStats=result_data.get("summaryStats", {}),
        multiAnomalyCount=result_data.get("multiAnomalyCount", 0),
        sequenceProblems=list(sequence_problems)
    )
    
    return report_response

@app.post("/api/v1/test-with-sample")
async def test_with_sample_data():
    """Test endpoint using sample data from test_data_sample.json"""
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
    
    print("🚀 Starting FastAPI Anomaly Detection Server")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("🔍 Alternative Docs: http://localhost:8000/redoc")
    print("🧪 Test Endpoint: http://localhost:8000/api/v1/test-with-sample")
    print("📊 Reports API: http://localhost:8000/api/reports")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True  # Auto-reload on code changes
    )