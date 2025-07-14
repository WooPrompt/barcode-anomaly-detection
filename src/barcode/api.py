import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import json
from typing import List, Dict, Any

# Import the enhanced multi-anomaly detection functions
from multi_anomaly_detector import detect_anomalies_from_json_enhanced
from legacy_format_detector import detect_anomalies_legacy_format

# Initialize the FastAPI app
app = FastAPI(
    title="Barcode Anomaly Detection API",
    description="An API to detect anomalies in barcode scan data using rule-based and model-based approaches.",
    version="1.0.0"
)

# --- Pydantic Models for Request Body Validation ---

class ScanData(BaseModel):
    scan_location: str = Field(..., example="양산공장")
    location_id: int = Field(..., example=3)
    hub_type: str = Field(..., example="YGS_Factory")
    business_step: str = Field(..., example="Factory")
    event_type: str = Field(..., example="Aggregation")
    operator_id: int = Field(..., example=3)
    device_id: int = Field(..., example=3)
    epc_code: str = Field(..., example="001.8805843.3842332.100006.20250701.000008002")
    epc_header: int = Field(..., example=1)
    epc_company: int = Field(..., example=8805843)
    epc_product: int = Field(..., example=3842332)
    epc_lot: int = Field(..., example=100006)
    epc_manufacture: int = Field(..., example=20250701)
    epc_serial: int = Field(..., example=8002)
    product_name: str = Field(..., example="Product 6")
    event_time: str = Field(..., example="2025-07-01 10:45:09")
    manufacture_date: str = Field(..., example="2025-07-01 10:45:09")
    expiry_date: str = Field(..., example="20251231")

class TransitionStat(BaseModel):
    from_scan_location: str = Field(..., example="서울 공장")
    to_scan_location: str = Field(..., example="부산 물류센터")
    time_taken_hours_mean: float = Field(..., example=5.0)
    time_taken_hours_std: float = Field(..., example=1.0)

class GeoData(BaseModel):
    scan_location: str = Field(..., example="서울 공장")
    Latitude: float = Field(..., example=37.5665)
    Longitude: float = Field(..., example=126.9780)

class AnomalyRequest(BaseModel):
    product_id: str = Field(..., example="0000001")
    lot_id: str = Field(..., example="000001")
    data: List[ScanData]
    transition_stats: List[TransitionStat]
    geo_data: List[GeoData]

# --- API Endpoint ---

@app.post("/api/v1/barcode-anomaly-detect", response_model=Dict[str, Any])
async def detect_anomalies_eventhistory(request: AnomalyRequest):
    """
    Enhanced anomaly detection with multi-anomaly support and probability scoring.

    New Features:
    - **Multi-anomaly detection**: Each EPC can have multiple anomaly types
    - **Probability scores**: 0-100% confidence for each anomaly type  
    - **Sequence position**: Identifies which step in the journey has problems
    - **EventHistory format**: Frontend-ready JSON structure

    Parameters:
    - **product_id**: The specific product ID to filter by.
    - **lot_id**: The specific lot ID to filter by.
    - **data**: A list of all scan events.
    - **transition_stats**: Statistics about travel time between locations.
    - **geo_data**: Geospatial information for each location.

    Returns EventHistory format:
    ```json
    {
      "EventHistory": [
        {
          "epcCode": "001.8809437.1203199.150002.20250701.000000002",
          "anomalyTypes": ["jump", "epcFake"],
          "anomalyScores": {"jump": 85, "epcFake": 72},
          "sequencePosition": 3,
          "primaryAnomaly": "jump"
        }
      ],
      "summaryStats": {"jump": 1, "epcFake": 1},
      "multiAnomalyCount": 1
    }
    ```
    """
    try:
        # Convert the Pydantic model to a dictionary, then to a JSON string
        input_json_str = request.model_dump_json()

        # Call the enhanced multi-anomaly detection function
        result_json_str = detect_anomalies_from_json_enhanced(input_json_str)

        # Parse result back to dictionary for FastAPI response
        response_data = json.loads(result_json_str)

        return response_data

    except Exception as e:
        # Catch any unexpected errors during the process
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/api/reports", response_model=Dict[str, Any])
async def detect_anomalies_legacy(request: AnomalyRequest):
    """
    Legacy format anomaly detection with multi-anomaly support.
    
    Provides traditional title/details/summaryStats format for existing frontend.
    Now enhanced with multi-anomaly detection capabilities.

    Returns legacy format:
    ```json
    {
      "title": "제품 0000001-로트 000001 이상 이벤트 감지",
      "details": [
        "001.xxx | 2024-07-02 09:23:00 | evtOrderErr,jump | 003 | 서울 공장"
      ],
      "summaryStats": {"epcFake": 0, "epcDup": 0, "locErr": 0, "evtOrderErr": 1, "jump": 1},
      "multiAnomalyCount": 1
    }
    ```
    
    Multi-anomaly feature: anomaly types are comma-separated in details field.
    """
    try:
        # Convert the Pydantic model to a dictionary, then to a JSON string
        input_json_str = request.model_dump_json()

        # Call the legacy format detection function
        result_json_str = detect_anomalies_legacy_format(input_json_str)

        # Parse result back to dictionary for FastAPI response
        response_data = json.loads(result_json_str)

        return response_data

    except Exception as e:
        # Catch any unexpected errors during the process
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- How to Run ---
# To run this API server, execute the following command in your terminal:
# uvicorn src.barcode.api:app --reload

if __name__ == "__main__":
    # This allows running the app directly for development/testing
    print("Starting FastAPI server...")
    print("Access the API documentation at http://127.0.0.1:8000/docs")
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)