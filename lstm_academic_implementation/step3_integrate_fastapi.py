#!/usr/bin/env python3
"""
Step 3: Integrate LSTM with FastAPI Server
Adds LSTM endpoint to existing fastapi_server.py
"""

import sys
import os
from pathlib import Path

def integrate_lstm_with_fastapi():
    """Add LSTM endpoint to existing FastAPI server"""
    
    print("Integrating LSTM with FastAPI server...")
    
    # Step 1: Check if trained model exists
    model_path = Path('lstm_academic_implementation/trained_lstm_model.pt')
    if not model_path.exists():
        print("ERROR: No trained LSTM model found!")
        print("Please run step2_train_lstm_model.py first!")
        return False
    
    print("Found trained LSTM model")
    
    # Step 2: Create LSTM integration code
    lstm_integration_code = '''

# ================================
# LSTM Integration - Added by step3_integrate_fastapi.py
# ================================

# Add LSTM imports
try:
    sys.path.append('lstm_academic_implementation/src')
    from lstm_inferencer import LSTMInferencer, InferenceRequest
    
    # Initialize LSTM inferencer globally
    LSTM_INFERENCER = None
    
    def get_lstm_inferencer():
        """Get or create LSTM inferencer instance"""
        global LSTM_INFERENCER
        if LSTM_INFERENCER is None:
            try:
                LSTM_INFERENCER = LSTMInferencer(
                    model_path='lstm_academic_implementation/trained_lstm_model.pt',
                    enable_explanations=True
                )
                print("LSTM model loaded successfully")
            except Exception as e:
                print(f"Failed to load LSTM model: {e}")
                return None
        return LSTM_INFERENCER
    
except ImportError as e:
    print(f"LSTM integration not available: {e}")
    LSTM_INFERENCER = None
    
    def get_lstm_inferencer():
        return None

# LSTM Endpoint
@app.post(
    "/api/manager/export-and-analyze-async/lstm",
    summary="LSTM 기반 다중 이상치 탐지 (딥러닝)",
    description="LSTM 딥러닝 모델을 사용한 시계열 이상치 탐지: epcFake, epcDup, jump, evtOrderErr, locErr"
)
async def detect_anomalies_lstm_endpoint(request: BackendAnomalyDetectionRequest):
    """
    LSTM 딥러닝 기반 다중 이상치 탐지 엔드포인트
    
    **특징**: 
    - 양방향 LSTM + 어텐션 메커니즘 사용
    - 시계열 패턴 학습으로 정확도 향상
    - 실시간 추론 (<10ms 목표)
    - Integrated Gradients 기반 설명 가능성
    
    **입력**: event_id, location_id 기반 스캔 데이터
    **출력**: fileId, EventHistory(eventId 필드), epcAnomalyStats, fileAnomalyStats 형식
    
    **LSTM 모델 특징:**
    - 양방향 LSTM: 과거/미래 정보 모두 활용
    - 어텐션 메커니즘: 중요한 시점 자동 식별
    - 멀티라벨 분류: 5가지 이상치 동시 검출
    - 포칼 로스: 클래스 불균형 해결
    """
    try:
        # Get LSTM inferencer
        inferencer = get_lstm_inferencer()
        
        if inferencer is None:
            # Fallback to rule-based detection with warning
            rule_result_json = detect_anomalies_backend_format(request.json())
            rule_result_dict = json.loads(rule_result_json)
            
            rule_result_dict["warning"] = "LSTM model not available. Using rule-based detection. Please check LSTM model training."
            rule_result_dict["method"] = "rule-based-fallback"
            
            return rule_result_dict
        
        # Convert request to LSTM format
        events_data = []
        for record in request.data:
            events_data.append({
                'event_time': record.event_time,
                'location_id': str(record.location_id),
                'business_step': record.business_step,
                'scan_location': f'LOC_{record.location_id}',
                'event_type': record.event_type,
                'operator_id': 'UNKNOWN'
            })
        
        # Group by EPC for LSTM processing
        epc_groups = {}
        for i, record in enumerate(request.data):
            epc_code = record.epc_code
            if epc_code not in epc_groups:
                epc_groups[epc_code] = []
            epc_groups[epc_code].append((record.event_id, events_data[i]))
        
        # Process each EPC with LSTM
        all_event_history = []
        epc_anomaly_stats = []
        total_anomaly_counts = {
            'jumpCount': 0,
            'evtOrderErrCount': 0,
            'epcFakeCount': 0,
            'epcDupCount': 0,
            'locErrCount': 0
        }
        
        for epc_code, epc_data in epc_groups.items():
            event_ids = [item[0] for item in epc_data]
            events = [item[1] for item in epc_data]
            
            # Create LSTM inference request
            lstm_request = InferenceRequest(
                epc_code=epc_code,
                events=events,
                request_id=f"lstm_{epc_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            # Get LSTM prediction
            try:
                lstm_response = inferencer.predict(lstm_request)
                
                # Convert LSTM response to backend format
                if lstm_response.predictions:
                    # Map predictions to event IDs
                    for i, event_id in enumerate(event_ids):
                        event_anomalies = {}
                        event_scores = {}
                        
                        for pred in lstm_response.predictions:
                            anomaly_type = pred.anomaly_type
                            confidence = pred.confidence
                            
                            # Map LSTM confidence to backend score (0-100)
                            score = min(100, max(0, confidence * 100))
                            
                            event_anomalies[anomaly_type] = True
                            event_scores[f"{anomaly_type}Score"] = score
                        
                        if event_anomalies:
                            event_record = {"eventId": event_id}
                            event_record.update(event_anomalies)
                            event_record.update(event_scores)
                            all_event_history.append(event_record)
                    
                    # Count anomalies for this EPC
                    epc_counts = {
                        'jumpCount': 0,
                        'evtOrderErrCount': 0,
                        'epcFakeCount': 0,
                        'epcDupCount': 0,
                        'locErrCount': 0
                    }
                    
                    for pred in lstm_response.predictions:
                        anomaly_type = pred.anomaly_type
                        if anomaly_type == 'jump':
                            epc_counts['jumpCount'] += 1
                        elif anomaly_type == 'evtOrderErr':
                            epc_counts['evtOrderErrCount'] += 1
                        elif anomaly_type == 'epcFake':
                            epc_counts['epcFakeCount'] += 1
                        elif anomaly_type == 'epcDup':
                            epc_counts['epcDupCount'] += 1
                        elif anomaly_type == 'locErr':
                            epc_counts['locErrCount'] += 1
                    
                    # Add to EPC stats if any anomalies found
                    if sum(epc_counts.values()) > 0:
                        epc_stats = {
                            "epcCode": epc_code,
                            "totalEvents": sum(epc_counts.values()),
                            **epc_counts
                        }
                        epc_anomaly_stats.append(epc_stats)
                        
                        # Add to total counts
                        for key in total_anomaly_counts:
                            total_anomaly_counts[key] += epc_counts[key]
                
            except Exception as lstm_error:
                print(f"LSTM prediction failed for EPC {epc_code}: {lstm_error}")
                continue
        
        # Determine file_id (use first one found)
        file_id = request.data[0].file_id if request.data else 1
        
        # Create response
        response = {
            "fileId": file_id,
            "method": "lstm-deep-learning",
            "EventHistory": all_event_history,
            "epcAnomalyStats": epc_anomaly_stats,
            "fileAnomalyStats": {
                "totalEvents": sum(total_anomaly_counts.values()),
                **total_anomaly_counts
            }
        }
        
        # Save result for ML improvement
        save_detection_result(response, request.json())
        
        return response
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON input: {e}")
    except Exception as e:
        # Fallback to rule-based on any error
        try:
            rule_result_json = detect_anomalies_backend_format(request.json())
            rule_result_dict = json.loads(rule_result_json)
            
            rule_result_dict["warning"] = f"LSTM detection failed: {e}. Using rule-based fallback."
            rule_result_dict["method"] = "rule-based-fallback"
            
            return rule_result_dict
        except Exception as fallback_error:
            raise HTTPException(status_code=500, detail=f"LSTM and fallback both failed: {e}, {fallback_error}")

# Update root endpoint to include LSTM
@app.get("/")
async def root():
    """루트 엔드포인트 - API 정보 제공"""
    return {
        "message": "바코드 이상치 탐지 API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "이상치_탐지": "POST /api/manager/export-and-analyze-async",
            "SVM_이상치_탐지": "POST /api/manager/export-and-analyze-async/svm",
            "LSTM_이상치_탐지": "POST /api/manager/export-and-analyze-async/lstm",  # NEW!
            "SVM_모델_훈련": "POST /api/v1/svm/train",
            "리포트_목록": "GET /api/reports",
            "리포트_상세": "GET /api/report/detail?reportId=xxx",
            "헬스체크": "GET /health"
        }
    }

# ================================
# End LSTM Integration
# ================================
'''
    
    # Step 3: Read existing fastapi_server.py
    fastapi_path = Path('fastapi_server.py')
    if not fastapi_path.exists():
        print("ERROR: fastapi_server.py not found!")
        return False
    
    with open(fastapi_path, 'r', encoding='utf-8') as f:
        fastapi_content = f.read()
    
    # Step 4: Check if LSTM integration already exists
    if 'LSTM Integration' in fastapi_content:
        print("WARNING: LSTM integration already exists in fastapi_server.py")
        print("No changes needed - LSTM endpoint already available")
        return True
    
    # Step 5: Add LSTM integration before the final if __name__ == "__main__" block
    insertion_point = fastapi_content.find('if __name__ == "__main__":')
    if insertion_point == -1:
        print("ERROR: Could not find insertion point in fastapi_server.py")
        return False
    
    # Insert LSTM integration code
    new_content = (
        fastapi_content[:insertion_point] + 
        lstm_integration_code + 
        "\n\n" + 
        fastapi_content[insertion_point:]
    )
    
    # Step 6: Write updated fastapi_server.py
    with open(fastapi_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Successfully integrated LSTM with FastAPI!")
    print("New endpoint added: POST /api/manager/export-and-analyze-async/lstm")
    
    return True

if __name__ == "__main__":
    success = integrate_lstm_with_fastapi()
    if success:
        print("\nIntegration complete!")
        print("You can now start the server:")
        print("   conda activate ds")
        print("   python fastapi_server.py")
        print("API documentation: http://localhost:8000/docs")
        print("Test LSTM endpoint: POST /api/manager/export-and-analyze-async/lstm")
    else:
        print("\nERROR: Integration failed. Please check the errors above.")