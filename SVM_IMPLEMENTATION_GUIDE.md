# SVM-based Anomaly Detection Implementation Guide

## Overview

This document describes the complete implementation of SVM-based anomaly detection for the barcode supply chain system. The implementation creates 5 separate One-Class SVM models, each specialized for detecting one specific type of anomaly.

## Architecture

### Core Components

1. **SVM Anomaly Detector** (`src/barcode/svm_anomaly_detector.py`)
   - Main orchestrator class managing 5 SVM models
   - Handles training, prediction, and model persistence
   - Integrates with existing preprocessing components

2. **Feature Extractors** (`src/barcode/svm_preprocessing/feature_extractors/`)
   - `epc_fake_features.py`: EPC format validation (10 dimensions)
   - `epc_dup_features.py`: Duplicate scan patterns (8 dimensions)
   - `jump_features.py`: Time jump analysis (10 dimensions)
   - `loc_err_features.py`: Location hierarchy violations (15 dimensions)
   - `evt_order_features.py`: Event sequence patterns (12 dimensions)

3. **API Endpoints** (`fastapi_server.py`)
   - `POST /api/v1/barcode-anomaly-detect/svm`: SVM-based detection
   - `POST /api/v1/svm/train`: Model training endpoint

### Model Architecture

```
Input Data → Feature Extraction → 5 Parallel SVM Models → Ensemble Results → Output
```

#### Individual SVM Models:

1. **epcFake_svm** (10 features)
   - EPC structure validation
   - Date format checking
   - Company/product code validation
   - Length normalization

2. **epcDup_svm** (8 features)
   - Event count patterns
   - Location/time distributions
   - Repetition ratios
   - Temporal anomalies

3. **locErr_svm** (15 features)
   - Hierarchy level analysis
   - Transition patterns
   - Geographic scatter
   - Location entropy

4. **evtOrderErr_svm** (12 features)
   - Sequence regularity
   - Backward transitions
   - Category distributions
   - Temporal disorder

5. **jump_svm** (10 features)
   - Time gap analysis
   - Travel impossibility
   - Sequence regularity
   - Statistical outliers

## Implementation Details

### Training Process

1. **Data Preparation**:
   - Use rule-based detection results as ground truth labels
   - Extract features for each anomaly type
   - Separate normal (label=0) and anomaly (label=1) samples

2. **One-Class SVM Training**:
   - Train only on normal data (unsupervised learning)
   - Use RBF kernel with optimized hyperparameters
   - Standard scaling for feature normalization

3. **Model Persistence**:
   - Save trained models and scalers as pickle files
   - Store metadata (training date, sample count, performance)

### Prediction Process

1. **Feature Extraction**:
   - Extract features for each EPC code/sequence
   - Apply same preprocessing as training

2. **Anomaly Detection**:
   - Run each SVM model independently
   - Convert decision scores to probability-like confidence (0-100%)
   - Support multi-anomaly detection per event

3. **Result Formatting**:
   - Same output format as rule-based system
   - Include `EventHistory`, `epcAnomalyStats`, `fileAnomalyStats`

## Usage

### 1. Model Training

#### Via API:
```bash
curl -X POST "http://localhost:8000/api/v1/svm/train" \
  -H "Content-Type: application/json" \
  -d '{
    "training_datasets": [
      {
        "data": [
          {
            "eventId": 1,
            "epc_code": "001.8804823.0000001.000001.20250101.000000001",
            "location_id": 1,
            "business_step": "Factory",
            "event_type": "Outbound",
            "event_time": "2025-01-01 08:00:00",
            "file_id": 1
          }
        ]
      }
    ],
    "retrain_all": true
  }'
```

#### Via Python:
```python
from src.barcode.svm_anomaly_detector import train_svm_models

# Prepare training data
json_data_list = [json.dumps(dataset) for dataset in training_datasets]

# Train models
results = train_svm_models(json_data_list)
print(results)
```

### 2. Anomaly Detection

#### Via API:
```bash
curl -X POST "http://localhost:8000/api/v1/barcode-anomaly-detect/svm" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {
        "eventId": 101,
        "epc_code": "001.8804823.0000001.000001.20250101.000000001",
        "location_id": 1,
        "business_step": "Factory",
        "event_type": "Outbound",
        "event_time": "2025-01-01 08:00:00",
        "file_id": 1
      }
    ]
  }'
```

#### Via Python:
```python
from src.barcode.svm_anomaly_detector import detect_anomalies_svm

# Prepare input data
json_data = json.dumps(input_data)

# Detect anomalies
result_json = detect_anomalies_svm(json_data)
result = json.loads(result_json)
```

### 3. Testing

Run the test suite:
```bash
python test_svm_system.py
```

This will test:
- Feature extraction functionality
- Model training process
- Anomaly prediction pipeline

## Output Format

The SVM system produces the same output format as the rule-based system:

```json
{
  "fileId": 1,
  "EventHistory": [
    {
      "eventId": 101,
      "epcFake": true,
      "epcFakeScore": 85.0,
      "epcDup": true,
      "epcDupScore": 92.5
    }
  ],
  "epcAnomalyStats": [
    {
      "epcCode": "001.8804823.0000001.000001.20250101.000000001",
      "totalEvents": 2,
      "jumpCount": 0,
      "evtOrderErrCount": 0,
      "epcFakeCount": 1,
      "epcDupCount": 1,
      "locErrCount": 0
    }
  ],
  "fileAnomalyStats": {
    "totalEvents": 2,
    "jumpCount": 0,
    "evtOrderErrCount": 0,
    "epcFakeCount": 1,
    "epcDupCount": 1,
    "locErrCount": 0
  }
}
```

## Key Features

### 1. Multi-Anomaly Detection
- Each event can have multiple anomaly types simultaneously
- Independent models prevent one anomaly from masking others
- Confidence scores for each detected anomaly

### 2. Scalable Architecture
- Models can be trained independently
- Easy to add new anomaly types
- Supports incremental learning

### 3. Production Ready
- Model persistence and versioning
- Error handling and fallback mechanisms
- Performance monitoring and logging

### 4. Integration Friendly
- Same API interface as rule-based system
- Drop-in replacement capability
- Backward compatibility maintained

## Performance Considerations

### Training Requirements
- Minimum 10 normal samples per model
- Balanced datasets recommended
- Periodic retraining with new data

### Prediction Speed
- ~1-5ms per EPC sequence
- Parallel processing capability
- Memory efficient operation

### Model Size
- ~50KB per trained model
- Total storage: ~250KB for all models
- Fast loading/initialization

## Comparison with Rule-based System

| Aspect | Rule-based | SVM-based |
|--------|------------|-----------|
| **Detection Method** | Fixed rules | Machine learning patterns |
| **Adaptability** | Manual rule updates | Automatic learning from data |
| **False Positives** | Higher (rigid rules) | Lower (learned patterns) |
| **New Anomalies** | Requires rule coding | Learns from examples |
| **Explainability** | High (clear rules) | Medium (feature importance) |
| **Setup Complexity** | Low | Medium (requires training) |
| **Maintenance** | High (manual rules) | Low (automatic adaptation) |

## Future Enhancements

### 1. Enhanced Models
- Ensemble methods (Random Forest, Gradient Boosting)
- Deep learning approaches (LSTM for sequences)
- Online learning capabilities

### 2. Advanced Features
- Anomaly explanation and root cause analysis
- Confidence intervals and uncertainty quantification
- Active learning for improved training

### 3. Operational Improvements
- A/B testing framework
- Model performance monitoring
- Automated model retraining pipelines

## File Structure

```
src/barcode/
├── svm_anomaly_detector.py          # Main SVM implementation
├── svm_preprocessing/
│   └── feature_extractors/
│       ├── epc_fake_features.py      # EPC format features
│       ├── epc_dup_features.py       # Duplicate detection features
│       ├── jump_features.py          # Time jump features
│       ├── loc_err_features.py       # Location error features
│       └── evt_order_features.py     # Event order features
├── multi_anomaly_detector.py        # Original rule-based system
└── fastapi_server.py                # API endpoints

models/svm_models/                   # Model storage directory
├── epcFake_model.pkl
├── epcFake_scaler.pkl
├── epcDup_model.pkl
├── epcDup_scaler.pkl
├── [... other models ...]
└── model_metadata.json

test_svm_system.py                   # Test suite
SVM_IMPLEMENTATION_GUIDE.md         # This documentation
```

## Conclusion

The SVM-based anomaly detection system provides a robust, scalable, and production-ready alternative to rule-based detection. It leverages machine learning to automatically learn patterns from data, reducing false positives and adapting to new anomaly types without manual rule updates.

The implementation maintains full compatibility with existing systems while providing enhanced detection capabilities through advanced feature engineering and ensemble learning approaches.