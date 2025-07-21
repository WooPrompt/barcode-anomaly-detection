# Barcode Anomaly Detection API - Complete Documentation

**Version:** 1.0.0  
**Author:** Data Analysis & Backend Team  
**Date:** 2025-07-21  
**Context:** Production-Ready Multi-Anomaly Detection System

---

## 📋 Overview

The Barcode Anomaly Detection API provides real-time detection of 5 types of supply chain anomalies using both rule-based and machine learning approaches. The system processes EPC barcode scan events and identifies fraudulent, duplicate, or logistically impossible patterns.

### **Detection Methods Available:**
1. **Rule-Based Detection** (Production Ready) - Fast, interpretable logic
2. **SVM Machine Learning** (Training Required) - Advanced pattern recognition
3. **LSTM Deep Learning** (Future Enhancement) - Temporal sequence analysis

---

## 🎯 Core API Endpoints

### **1. Primary Anomaly Detection (Rule-Based)**

**Endpoint:** `POST /api/manager/export-and-analyze-async`

**Description:** Production-ready rule-based anomaly detection with immediate response. Detects 5 anomaly types using business logic and statistical validation.

#### **Input Format:**
```json
{
    "data": [
        {
            "event_id": 12345,
            "epc_code": "001-1234567-8901234-567890-20240115-123456789",
            "location_id": 101,
            "business_step": "Factory",
            "event_type": "Aggregation", 
            "event_time": "2024-01-15T08:30:00Z",
            "file_id": 1
        },
        {
            "event_id": 12346,
            "epc_code": "001-1234567-8901234-567890-20240115-123456789",
            "location_id": 102,
            "business_step": "WMS",
            "event_type": "HUB_Outbound",
            "event_time": "2024-01-15T09:45:00Z", 
            "file_id": 1
        },
        {
            "event_id": 12347,
            "epc_code": "001-1234567-8901234-567890-20240115-123456789",
            "location_id": 103,
            "business_step": "Logistics_HUB",
            "event_type": "HUB_Inbound",
            "event_time": "2024-01-15T11:15:00Z", 
            "file_id": 1
        },
        {
            "event_id": 12348,
            "epc_code": "001-7654321-1098765-098765-20240116-987654321",
            "location_id": 201,
            "business_step": "Factory",
            "event_type": "Aggregation",
            "event_time": "2024-01-16T08:00:00Z",
            "file_id": 1
        },
        {
            "event_id": 12349,
            "epc_code": "001-7654321-1098765-098765-20240116-987654321",
            "location_id": 202,
            "business_step": "WMS",
            "event_type": "Stock_In",
            "event_time": "2024-01-16T09:30:00Z",
            "file_id": 1
        },
        {
            "event_id": 12350,
            "epc_code": "001-9876543-2109876-109876-20240117-456789012",
            "location_id": 301,
            "business_step": "Factory",
            "event_type": "Aggregation",
            "event_time": "2024-01-17T07:45:00Z",
            "file_id": 1
        }
    ]
}
```

#### **Input Field Explanations:**
- **event_id** (int): Unique event identifier from backend database
- **epc_code** (str): Full EPC barcode (format: header-company-product-lot-manufacture-serial)
- **location_id** (int): Numerical location identifier (maps to scan_location)
- **business_step** (str): Supply chain stage (Factory/WMS/Logistics_HUB/Distribution/Retail)
- **event_type** (str): Operation type (Aggregation/HUB_Outbound/Stock_In/etc.)
- **event_time** (str): ISO 8601 timestamp when scan occurred
- **file_id** (int): Data batch identifier for multi-file processing

#### **Output Format:**
```json
{
    "fileId": 1,
    "EventHistory": [
        {
            "eventId": 12346,
            "jump": true,
            "jumpScore": 95.8
        },
        {
            "eventId": 12347,
            "epcFake": true,
            "epcFakeScore": 85.2,
            "locErr": true,
            "locErrScore": 67.4
        }
    ],
    "epcAnomalyStats": [
        {
            "epcCode": "001-1234567-8901234-567890-20240115-123456789",
            "totalEvents": 3,
            "jumpCount": 1,
            "evtOrderErrCount": 0,
            "epcFakeCount": 1,
            "epcDupCount": 0,
            "locErrCount": 1
        },
        {
            "epcCode": "001-7654321-1098765-098765-20240116-987654321",
            "totalEvents": 1,
            "jumpCount": 0,
            "evtOrderErrCount": 0,
            "epcFakeCount": 0,
            "epcDupCount": 1,
            "locErrCount": 0
        }
    ],
    "fileAnomalyStats": {
        "totalEvents": 4,
        "jumpCount": 1,
        "evtOrderErrCount": 0,
        "epcFakeCount": 1,
        "epcDupCount": 1,
        "locErrCount": 1
    }
}
```

#### **Output Field Explanations:**
- **EventHistory**: Only events with detected anomalies are included. Each event shows only true anomaly flags and their confidence scores (0-100)
- **epcAnomalyStats**: Only EPC codes with detected anomalies are included. Shows total anomaly counts per EPC code
  - **totalEvents**: Total number of anomalies detected for this specific EPC code
- **fileAnomalyStats**: Overall file-level summary aggregating all anomalies across all EPC codes
  - **totalEvents**: Total number of anomalies detected in the entire file

**Note:** 
- Events without anomalies (like eventId 12345) are omitted from EventHistory
- EPC codes without anomalies (like "001-9876543-2109876-109876-20240117-456789012") are omitted from epcAnomalyStats
- This keeps the response clean and focused on actual issues only

#### **How It Works:**
1. **Data Validation**: Validates EPC format and temporal consistency
2. **Geographic Processing**: Loads geospatial data for distance calculations
3. **Transition Analysis**: Uses statistical baselines for travel time validation
4. **Multi-Detection**: Applies 5 independent anomaly detection algorithms
5. **Scoring**: Generates confidence scores based on deviation magnitude
6. **Response Assembly**: Formats results in backend-compatible JSON structure

---

### **2. Machine Learning Anomaly Detection (SVM)**

**Endpoint:** `POST /api/manager/export-and-analyze-async/svm`

**Description:** Advanced anomaly detection using 5 trained One-Class SVM models. Requires prior model training. Falls back to rule-based detection if models unavailable.

#### **Input Format:**
```json
{
    "data": [
        {
            "event_id": 12345,
            "epc_code": "001-1234567-8901234-567890-20240115-123456789",
            "location_id": 101,
            "business_step": "Factory",
            "event_type": "Aggregation",
            "event_time": "2024-01-15T08:30:00Z",
            "file_id": 1
        }
    ]
}
```

#### **Output Format (SVM Available):**
```json
{
    "fileId": 1,
    "EventHistory": [
        {
            "eventId": 12346,
            "jump": true,
            "jumpScore": 87.2
        },
        {
            "eventId": 12347,
            "epcFake": true,
            "epcFakeScore": 92.1,
            "locErr": true,
            "locErrScore": 78.6
        },
        {
            "eventId": 12349,
            "epcDup": true,
            "epcDupScore": 85.3
        }
    ],
    "epcAnomalyStats": [
        {
            "epcCode": "001-1234567-8901234-567890-20240115-123456789",
            "totalEvents": 3,
            "jumpCount": 1,
            "evtOrderErrCount": 0,
            "epcFakeCount": 1,
            "epcDupCount": 0,
            "locErrCount": 1
        },
        {
            "epcCode": "001-7654321-1098765-098765-20240116-987654321",
            "totalEvents": 1,
            "jumpCount": 0,
            "evtOrderErrCount": 0,
            "epcFakeCount": 0,
            "epcDupCount": 1,
            "locErrCount": 0
        }
    ],
    "fileAnomalyStats": {
        "totalEvents": 4,
        "jumpCount": 1,
        "evtOrderErrCount": 0,
        "epcFakeCount": 1,
        "epcDupCount": 1,
        "locErrCount": 1
    }
}
```

#### **Output Format (SVM Unavailable - Fallback):**
```json
{
    "fileId": 1,
    "warning": "SVM models not trained yet. Using rule-based detection. Please train models first: python train_svm_models.py",
    "EventHistory": [
        {
            "eventId": 12346,
            "jump": true,
            "jumpScore": 95.8
        },
        {
            "eventId": 12347,
            "epcFake": true,
            "epcFakeScore": 85.2,
            "locErr": true,
            "locErrScore": 67.4
        },
        {
            "eventId": 12349,
            "epcDup": true,
            "epcDupScore": 78.9
        }
    ],
    "epcAnomalyStats": [
        {
            "epcCode": "001-1234567-8901234-567890-20240115-123456789",
            "totalEvents": 3,
            "jumpCount": 1,
            "evtOrderErrCount": 0,
            "epcFakeCount": 1,
            "epcDupCount": 0,
            "locErrCount": 1
        },
        {
            "epcCode": "001-7654321-1098765-098765-20240116-987654321",
            "totalEvents": 1,
            "jumpCount": 0,
            "evtOrderErrCount": 0,
            "epcFakeCount": 0,
            "epcDupCount": 1,
            "locErrCount": 0
        }
    ],
    "fileAnomalyStats": {
        "totalEvents": 4,
        "jumpCount": 1,
        "evtOrderErrCount": 0,
        "epcFakeCount": 1,
        "epcDupCount": 1,
        "locErrCount": 1
    }
}
```

#### **How It Works:**
1. **Model Loading**: Loads 5 pre-trained SVM models (one per anomaly type)
2. **Feature Extraction**: Extracts 10-15 dimensional features per anomaly type
3. **SVM Prediction**: Uses One-Class SVM to detect outliers from normal patterns
4. **Probability Scoring**: Converts SVM decision scores to 0-100% confidence
5. **Fallback Logic**: Automatically falls back to rule-based if models unavailable

---

## 🔍 Anomaly Types Detected

### **1. epcFake - EPC Format Violations**
**Detection Logic:** Validates EPC structure, company codes, and date consistency
- Invalid EPC structure (wrong segment lengths)
- Unregistered company codes
- Future manufacture dates
- Invalid date formats

**Score Calculation:** Based on number and severity of format violations

### **2. epcDup - Impossible Duplicate Scans**
**Detection Logic:** Identifies simultaneous scans at different locations
- Same EPC, same timestamp, different locations
- Excludes normal factory-warehouse simultaneous operations
- Considers geographic feasibility

**Score Calculation:** Based on geographic distance and time impossibility

### **3. jump - Impossible Travel Times**
**Detection Logic:** Validates space-time constraints using geographic data
- Calculates Haversine distance between locations
- Compares with statistical travel time baselines
- Considers transportation mode and route feasibility

**Score Calculation:** Based on speed ratio (actual vs. maximum feasible speed)

### **4. evtOrderErr - Event Sequence Violations**
**Detection Logic:** Validates business process order within locations
- Detects consecutive inbound/outbound events
- Validates event type sequences
- Checks business step progression

**Score Calculation:** Based on logical impossibility and frequency

### **5. locErr - Location Hierarchy Violations**
**Detection Logic:** Validates supply chain flow direction
- Prevents backward movement (Retail → Factory)
- Enforces business step hierarchy
- Validates geographic flow patterns

**Score Calculation:** Based on hierarchy level violations and business impact

---

## 🚀 Multi-File Processing

### **Input Format (Multiple Files):**
```json
{
    "data": [
        {
            "event_id": 1,
            "epc_code": "001-1234567-8901234-567890-20240115-123456789",
            "location_id": 101,
            "business_step": "Factory",
            "event_type": "Aggregation",
            "event_time": "2024-01-15T08:30:00Z",
            "file_id": 1
        },
        {
            "event_id": 2,
            "epc_code": "001-7654321-1098765-098765-20240116-987654321",
            "location_id": 201,
            "business_step": "WMS",
            "event_type": "Stock_In",
            "event_time": "2024-01-16T10:15:00Z",
            "file_id": 2
        }
    ]
}
```

### **Output Format (Multiple Files):**
```json
[
    {
        "fileId": 1,
        "EventHistory": [...],
        "epcAnomalyStats": [...],
        "fileAnomalyStats": {...}
    },
    {
        "fileId": 2,
        "EventHistory": [...],
        "epcAnomalyStats": [...],
        "fileAnomalyStats": {...}
    }
]
```

**Note:** API automatically detects multiple file_ids and returns array format. Single file requests return object format for backward compatibility.

---

## 📊 Performance Characteristics

### **Rule-Based Detection:**
- **Latency:** <100ms for 1000 events
- **Throughput:** >10,000 events/second
- **Accuracy:** 56.6% on synthetic anomalies
- **Dependencies:** Geographic data files required

### **SVM Detection:**
- **Latency:** <1 second for 1000 events  
- **Throughput:** >1,000 events/second
- **Accuracy:** Varies by model training quality
- **Dependencies:** Pre-trained models required

### **Training Requirements:**
- **Data Volume:** Minimum 10,000 events for stable training
- **Training Time:** 5-15 minutes for 920K events
- **Model Size:** ~50MB total for all 5 models
- **Retraining:** Recommended monthly for concept drift

---

## 🔧 Integration Requirements

### **Required Data Files:**
```
data/processed/location_id_withGeospatial.csv
data/processed/business_step_transition_avg_v2.csv
```

### **Model Storage:**
```
models/svm_models/epcFake_svm.joblib
models/svm_models/epcDup_svm.joblib
models/svm_models/locErr_svm.joblib
models/svm_models/evtOrderErr_svm.joblib
models/svm_models/jump_svm.joblib
```

### **Training Data Sources:**
```
data/raw/icn.csv
data/raw/kum.csv
data/raw/ygs.csv
data/raw/hws.csv
```

---

## 🏁 Quick Start Guide

### **1. Start API Server:**
```bash
conda activate ds
python fastapi_server.py
```

### **2. Access Documentation:**
- Interactive API Docs: http://localhost:8000/docs
- Alternative Docs: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

### **3. Test Rule-Based Detection:**
```bash
curl -X POST "http://localhost:8000/api/manager/export-and-analyze-async" \
     -H "Content-Type: application/json" \
     -d @test_data_sample.json
```

### **4. Train SVM Models (Optional):**
```bash
python train_svm_models.py
```

### **5. Test SVM Detection:**
```bash
curl -X POST "http://localhost:8000/api/manager/export-and-analyze-async/svm" \
     -H "Content-Type: application/json" \
     -d @test_data_sample.json
```

---

## 🚨 Error Handling

### **Common Error Responses:**

#### **400 Bad Request - Invalid JSON:**
```json
{
    "detail": "Invalid JSON input: Expecting ',' delimiter: line 5 column 12 (char 89)"
}
```

#### **500 Internal Server Error - Detection Failure:**
```json
{
    "detail": "Detection error: Required data file not found: location_id_withGeospatial.csv"
}
```

#### **SVM Model Missing (Automatic Fallback):**
```json
{
    "fileId": 1,
    "method": "rule-based-fallback",
    "warning": "SVM models not trained yet. Using rule-based detection.",
    "EventHistory": [...],
    "epcAnomalyStats": [...],
    "fileAnomalyStats": {...}
}
```

---

## 📈 Monitoring & Maintenance

### **Health Check Endpoint:**
```
GET /health
Response: {"status": "정상", "service": "이상치-탐지-서비스"}
```

### **Performance Monitoring:**
- Monitor response times for detection endpoints
- Track model prediction accuracy over time
- Monitor false positive/negative rates
- Set up alerts for API failures or slow responses

### **Model Maintenance:**
- Retrain SVM models monthly or when accuracy degrades
- Update geographic data when new locations added
- Refresh transition statistics with recent operational data
- Archive old training data to manage disk space

### **Log Analysis:**
- Check logs for feature extraction failures
- Monitor memory usage during large batch processing
- Track concept drift in model performance
- Analyze error patterns for system improvements

---

This documentation provides complete integration guidance for backend teams implementing the barcode anomaly detection API. All examples use actual data formats from the production system.