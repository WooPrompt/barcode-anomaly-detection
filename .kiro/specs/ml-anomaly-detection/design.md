# Design Document: LSTM and SVM Models for Barcode Anomaly Detection

## Overview

This document explains HOW we will build the LSTM and SVM models for barcode anomaly detection. Think of this as the blueprint that shows exactly how to construct the machine learning system.

**High-Level Architecture:**
```
Input Data → Data Preprocessing → ML Models → Results Integration → API Response
     ↓              ↓                ↓              ↓              ↓
JSON from      Feature         LSTM + SVM      Combine with    Enhanced JSON
Backend        Engineering     Predictions     Rule-based      to Frontend
```

## Architecture

### System Components

1. **Data Preprocessing Pipeline** (`ml_preprocessing.py`)
   - Converts raw JSON data into ML-ready features
   - Handles temporal, categorical, and sequence features
   - Ensures data quality and consistency

2. **LSTM Model** (`lstm_anomaly_detector.py`)
   - Detects sequence-based anomalies
   - Predicts next expected events in barcode scanning sequences
   - Provides confidence scores for sequence violations

3. **SVM Model** (`svm_anomaly_detector.py`)
   - Detects statistical outliers in scanning patterns
   - Uses One-Class SVM for unsupervised anomaly detection
   - Identifies patterns that deviate from normal behavior

4. **Model Integration Layer** (`ml_integration.py`)
   - Combines LSTM, SVM, and rule-based results
   - Manages model loading and caching
   - Provides unified API interface

5. **Training Pipeline** (`model_training.py`)
   - Handles data splitting, model training, and validation
   - Implements proper ML practices (cross-validation, early stopping)
   - Saves trained models with metadata

## Components and Interfaces

### 1. Data Preprocessing Component

**Purpose:** Convert raw barcode data into features that ML models can understand.

**Input Interface:**
```python
def preprocess_for_ml(json_data: dict) -> dict:
    """
    Convert backend JSON to ML features
    
    Args:
        json_data: Raw JSON from backend API
        
    Returns:
        dict: Processed features for ML models
    """
```

**Feature Engineering Strategy:**

#### Temporal Features
```python
# From: "event_time": "2024-12-01 08:00:00"
# To: Multiple time-based features
{
    "hour": 8,           # Hour of day (0-23)
    "day_of_week": 4,    # Monday=0, Sunday=6
    "month": 12,         # Month (1-12)
    "is_weekend": False, # Boolean weekend flag
    "is_business_hours": True  # 9AM-5PM flag
}
```

**Why these features?** Anomalies might happen at unusual times (like midnight scans or weekend activity).

#### EPC Decomposition Features
```python
# From: "epc_code": "001.8804823.0000003.000001.20241201.000000001"
# To: Structured components
{
    "epc_header": "001",
    "epc_company": "8804823",
    "epc_product": "0000003", 
    "epc_lot": "000001",
    "epc_manufacture_date": "20241201",
    "epc_serial": "000000001",
    "manufacture_age_days": 45  # Days since manufacture
}
```

**Why these features?** Each part of EPC tells us something - company patterns, product types, age of products.

#### Location Enhancement Features
```python
# From: "location_id": 1
# To: Enhanced location data (using location_id_withGeospatial.csv)
{
    "location_id": 1,
    "latitude": 37.45,
    "longitude": 126.65,
    "location_type": "Factory",  # Factory/WMS/Wholesaler/Retailer
    "location_hierarchy": 0      # 0=Factory, 1=WMS, 2=Wholesaler, 3=Retailer
}
```

**Why these features?** Geographic and hierarchical information helps detect impossible movements.

#### Sequence Features (for LSTM)
```python
# From: List of events for same EPC
# To: Sequence representation
{
    "sequence_length": 5,
    "current_position": 3,        # Position in sequence (1-based)
    "previous_location": 1,       # Where it came from
    "previous_event_type": "Outbound",
    "time_since_last_scan": 2.5,  # Hours since previous scan
    "sequence_so_far": [1, 2, 3]  # Location sequence up to this point
}
```

**Why these features?** LSTM needs to understand the context and flow of events.

#### Missing Data Handling Strategy
```python
# Comprehensive missing data handling approach
def handle_missing_data(data):
    """
    Handle missing values with intelligent fallback strategies
    
    Strategy Priority:
    1. Skip critical missing data (event_time, epc_code)
    2. Interpolate from sequence context (location_id)
    3. Use domain defaults (business_step, event_type)
    4. Flag for special handling (partial sequences)
    """
    
    # Critical fields - cannot proceed without these
    if pd.isna(data.get('event_time')) or pd.isna(data.get('epc_code')):
        return None  # Skip this record entirely
    
    # Location interpolation from sequence
    if pd.isna(data.get('location_id')):
        data['location_id'] = infer_location_from_sequence_context(data)
        data['location_interpolated'] = True  # Flag for model awareness
    
    # Business step defaults based on location hierarchy
    if pd.isna(data.get('business_step')):
        data['business_step'] = map_location_to_business_step(data['location_id'])
        data['business_step_inferred'] = True
    
    # Event type defaults
    if pd.isna(data.get('event_type')):
        data['event_type'] = 'Unknown'
        data['event_type_missing'] = True  # Feature for ML models
    
    return data
```

**Why this approach?** Different missing data types require different strategies. Critical data (timestamps, EPC codes) cannot be guessed, but contextual data (location, business step) can often be inferred from surrounding events in the sequence.

#### Feature Scaling and Normalization Strategy
```python
# Comprehensive feature scaling for ML models
class FeatureScaler:
    def __init__(self):
        # Different scalers for different feature types
        self.temporal_scaler = StandardScaler()      # For time-based features
        self.location_scaler = MinMaxScaler()        # For coordinates (0-1 range)
        self.sequence_scaler = RobustScaler()        # For sequence features (handles outliers)
        
    def fit_transform_features(self, features_dict):
        """Scale all features to similar ranges for ML models"""
        scaled_features = {}
        
        # Temporal features: standardize (mean=0, std=1)
        temporal_features = ['hour', 'day_of_week', 'month', 'time_since_last_scan']
        if any(f in features_dict for f in temporal_features):
            temporal_data = np.array([[features_dict.get(f, 0) for f in temporal_features]])
            scaled_temporal = self.temporal_scaler.fit_transform(temporal_data)[0]
            for i, feature in enumerate(temporal_features):
                if feature in features_dict:
                    scaled_features[f'{feature}_scaled'] = scaled_temporal[i]
        
        # Location features: normalize to 0-1 range
        location_features = ['latitude', 'longitude', 'location_hierarchy']
        if any(f in features_dict for f in location_features):
            location_data = np.array([[features_dict.get(f, 0) for f in location_features]])
            scaled_location = self.location_scaler.fit_transform(location_data)[0]
            for i, feature in enumerate(location_features):
                if feature in features_dict:
                    scaled_features[f'{feature}_scaled'] = scaled_location[i]
        
        # Sequence features: robust scaling (handles outliers better)
        sequence_features = ['sequence_length', 'current_position', 'manufacture_age_days']
        if any(f in features_dict for f in sequence_features):
            sequence_data = np.array([[features_dict.get(f, 0) for f in sequence_features]])
            scaled_sequence = self.sequence_scaler.fit_transform(sequence_data)[0]
            for i, feature in enumerate(sequence_features):
                if feature in features_dict:
                    scaled_features[f'{feature}_scaled'] = scaled_sequence[i]
        
        # Keep original features for interpretability
        scaled_features.update(features_dict)
        return scaled_features
```

**Why different scalers?** Different feature types need different scaling approaches:
- **StandardScaler** for temporal features: Centers around mean, good for normally distributed data
- **MinMaxScaler** for location features: Ensures coordinates are in 0-1 range, preserves relationships
- **RobustScaler** for sequence features: Less sensitive to outliers (like very long sequences)

### 2. LSTM Model Component

**Purpose:** Detect anomalies in the sequence of barcode scans.

**Architecture Design:**
```python
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        # LSTM layers for sequence processing
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Attention mechanism for variable-length sequences
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        
        # Output layers for prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),  # Anomaly score (0-1)
            nn.Sigmoid()
        )
```

**Why this architecture?**
- **LSTM layers:** Good at remembering long-term patterns in sequences
- **Attention mechanism:** Handles variable-length sequences safely (no padding/truncating)
- **Dropout:** Prevents overfitting (memorizing training data)
- **Sigmoid output:** Gives probability score (0-1) that we can convert to 0-100

**Training Strategy:**
```python
# Create training examples from sequences
# For sequence: [Factory, WMS, Wholesaler, Retailer]
# Create examples:
# Input: [Factory] → Target: WMS (normal)
# Input: [Factory, WMS] → Target: Wholesaler (normal)  
# Input: [Factory, WMS, Wholesaler] → Target: Retailer (normal)

# During inference:
# Input: [Factory, WMS] → Predict: should be Wholesaler
# If actual is Retailer → Anomaly! (skipped Wholesaler)
```

**GPU Acceleration Design:**
```python
class LSTMTrainingManager:
    def __init__(self):
        # Automatic GPU detection and utilization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_gpu = torch.cuda.is_available()
        
        if self.use_gpu:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Optimize GPU memory usage
            torch.backends.cudnn.benchmark = True
        else:
            print("Using CPU for training")
    
    def train_model(self, model, train_loader, val_loader):
        """GPU-optimized training loop"""
        model = model.to(self.device)
        
        # Use mixed precision for faster GPU training
        if self.use_gpu:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(num_epochs):
            for batch in train_loader:
                # Move data to GPU
                inputs = batch['features'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                # Mixed precision forward pass
                if self.use_gpu:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
```

**Why GPU acceleration?** Training LSTM models can take 2 hours on GPU vs 20 hours on CPU. Mixed precision training further speeds up training while using less GPU memory.

### 3. SVM Model Component

**Purpose:** Detect statistical outliers in barcode scanning patterns.

**Architecture Design:**
```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class SVMAnomalyDetector:
    def __init__(self):
        # One-Class SVM for outlier detection
        self.svm = OneClassSVM(
            kernel='rbf',        # Radial basis function kernel
            gamma='scale',       # Automatic gamma selection
            nu=0.05             # Expected fraction of outliers (5%)
        )
        
        # Feature scaling (very important for SVM)
        self.scaler = StandardScaler()
        
    def fit(self, normal_data):
        """Train on normal data only"""
        # Scale features to 0-1 range
        scaled_data = self.scaler.fit_transform(normal_data)
        
        # Train SVM to learn "normal" boundary
        self.svm.fit(scaled_data)
        
    def predict_anomaly_score(self, data):
        """Return anomaly score (0-100)"""
        scaled_data = self.scaler.transform(data)
        
        # Get decision function score
        decision_score = self.svm.decision_function(scaled_data)
        
        # Convert to 0-100 probability
        # More negative = more anomalous
        anomaly_score = np.clip((-decision_score * 50) + 50, 0, 100)
        return anomaly_score
```

**Why One-Class SVM?**
- **Unsupervised:** We have lots of normal data, few anomaly examples
- **Robust:** Good at finding outliers in high-dimensional data
- **Interpretable:** Decision scores can be converted to probabilities

**Feature Selection for SVM:**
```python
# Features that work well with SVM:
svm_features = [
    'hour', 'day_of_week', 'is_weekend',           # Temporal patterns
    'location_hierarchy', 'latitude', 'longitude',  # Location patterns  
    'time_since_last_scan', 'sequence_length',     # Timing patterns
    'manufacture_age_days',                         # Product age patterns
    'location_transition_frequency'                 # How common this route is
]
```

**Hyperparameter Optimization Strategy:**
```python
class SVMHyperparameterOptimizer:
    def __init__(self):
        # Parameter grid for optimization
        self.param_grid = {
            'nu': [0.01, 0.05, 0.1, 0.2],           # Expected outlier fraction
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],  # Kernel coefficient
            'kernel': ['rbf', 'poly', 'sigmoid']     # Kernel types
        }
        
    def optimize_hyperparameters(self, training_data, cv_folds=5):
        """Find best hyperparameters using cross-validation"""
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import make_scorer
        
        # Custom scoring function for anomaly detection
        def anomaly_score(estimator, X):
            """Score based on decision function distribution"""
            decisions = estimator.decision_function(X)
            # Good model should have most data points with positive scores (normal)
            # and clear separation for outliers
            return np.mean(decisions > 0) - np.std(decisions)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            OneClassSVM(),
            self.param_grid,
            cv=cv_folds,
            scoring=make_scorer(anomaly_score),
            n_jobs=-1,  # Use all CPU cores
            verbose=1
        )
        
        # Fit and find best parameters
        grid_search.fit(training_data)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def create_optimized_model(self, best_params):
        """Create SVM model with optimized parameters"""
        return OneClassSVM(**best_params)
```

**Why hyperparameter optimization?** SVM performance is highly sensitive to parameters like `nu` (outlier fraction) and `gamma` (kernel width). Automatic optimization ensures we get the best possible anomaly detection performance for our specific data patterns.

### 4. Model Integration Component

**Purpose:** Combine results from LSTM, SVM, and existing rule-based detection.

**Integration Strategy:**
```python
class MLIntegration:
    def __init__(self):
        self.lstm_model = LSTMAnomalyDetector()
        self.svm_model = SVMAnomalyDetector()
        
        # Weights for combining different models
        self.model_weights = {
            'rule_based': 0.4,    # Existing rules are reliable
            'lstm': 0.35,         # LSTM good for sequences  
            'svm': 0.25          # SVM good for outliers
        }
    
    def detect_anomalies(self, json_data):
        """Main detection function"""
        # 1. Preprocess data
        features = preprocess_for_ml(json_data)
        
        # 2. Get predictions from all models
        lstm_scores = self.lstm_model.predict(features)
        svm_scores = self.svm_model.predict_anomaly_score(features)
        rule_scores = existing_rule_based_detection(json_data)
        
        # 3. Combine scores using weighted average
        combined_scores = (
            rule_scores * self.model_weights['rule_based'] +
            lstm_scores * self.model_weights['lstm'] + 
            svm_scores * self.model_weights['svm']
        )
        
        # 4. Create enhanced response
        return self.create_enhanced_response(
            json_data, rule_scores, lstm_scores, svm_scores, combined_scores
        )
    
    def detect_anomalies_parallel(self, json_data):
        """Parallel processing for real-time performance"""
        import concurrent.futures
        import threading
        
        # Thread-safe model loading
        with threading.Lock():
            if not hasattr(self, '_models_loaded'):
                self._load_models()
                self._models_loaded = True
        
        # Run all detection methods in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all tasks simultaneously
            rule_future = executor.submit(existing_rule_based_detection, json_data)
            lstm_future = executor.submit(self._lstm_predict_safe, json_data)
            svm_future = executor.submit(self._svm_predict_safe, json_data)
            
            # Collect results as they complete
            rule_scores = rule_future.result(timeout=1.0)  # 1 second timeout
            lstm_scores = lstm_future.result(timeout=1.5)  # 1.5 second timeout
            svm_scores = svm_future.result(timeout=1.0)    # 1 second timeout
        
        # Combine results
        combined_scores = self._combine_scores(rule_scores, lstm_scores, svm_scores)
        return self.create_enhanced_response(json_data, rule_scores, lstm_scores, svm_scores, combined_scores)
    
    def detect_anomalies_batch(self, batch_data, batch_size=100):
        """Efficient batch processing for large datasets"""
        results = []
        
        # Process in chunks to manage memory
        for i in range(0, len(batch_data), batch_size):
            chunk = batch_data[i:i + batch_size]
            
            # Vectorized preprocessing for the chunk
            chunk_features = self._preprocess_batch(chunk)
            
            # Batch predictions (much faster than individual predictions)
            lstm_batch_scores = self.lstm_model.predict_batch(chunk_features)
            svm_batch_scores = self.svm_model.predict_batch(chunk_features)
            
            # Process rule-based detection in parallel for the chunk
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                rule_futures = [executor.submit(existing_rule_based_detection, item) 
                              for item in chunk]
                rule_batch_scores = [f.result() for f in rule_futures]
            
            # Combine results for this chunk
            chunk_results = self._combine_batch_results(
                chunk, rule_batch_scores, lstm_batch_scores, svm_batch_scores
            )
            results.extend(chunk_results)
            
            # Memory cleanup after each chunk
            if i % (batch_size * 10) == 0:  # Every 1000 records
                self._cleanup_memory()
        
        return results
```

**Model Versioning and Hot-Swapping Strategy:**
```python
class ModelManager:
    def __init__(self):
        self.current_models = {}
        self.model_versions = {}
        self.model_lock = threading.RLock()
        
    def load_model_version(self, model_type, version):
        """Load a specific version of a model"""
        model_path = f"models/{model_type}_v{version}.pkl"
        metadata_path = f"models/{model_type}_v{version}_metadata.json"
        
        try:
            # Load model and metadata
            if model_type == "lstm":
                model = torch.load(model_path)
            else:  # SVM
                model = joblib.load(model_path)
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load {model_type} v{version}: {e}")
            return None, None
    
    def hot_swap_model(self, model_type, new_version):
        """Replace model without stopping service"""
        with self.model_lock:
            # Load new model
            new_model, metadata = self.load_model_version(model_type, new_version)
            if new_model is None:
                return False
                
            # Validate compatibility
            if not self._validate_model_compatibility(model_type, metadata):
                logger.error(f"Model {model_type} v{new_version} incompatible with current data schema")
                return False
            
            # Atomic swap
            old_version = self.model_versions.get(model_type, "unknown")
            self.current_models[model_type] = new_model
            self.model_versions[model_type] = new_version
            
            logger.info(f"Hot-swapped {model_type} from v{old_version} to v{new_version}")
            return True
    
    def _validate_model_compatibility(self, model_type, metadata):
        """Check if new model is compatible with current system"""
        # Check feature schema compatibility
        expected_features = self._get_current_feature_schema()
        model_features = metadata.get('feature_schema', [])
        
        # New model must support at least the current features
        missing_features = set(expected_features) - set(model_features)
        if missing_features:
            logger.error(f"New model missing features: {missing_features}")
            return False
            
        return True

class ModelRetrainingPipeline:
    def __init__(self):
        self.performance_monitor = ModelPerformanceMonitor()
        self.data_manager = TrainingDataManager()
        
    def should_retrain(self, model_type):
        """Determine if model needs retraining"""
        current_performance = self.performance_monitor.get_current_metrics(model_type)
        
        # Retrain if performance drops below threshold
        if current_performance['accuracy'] < 0.75:
            return True, "Performance degradation"
            
        # Retrain if data drift detected
        if self.performance_monitor.detect_data_drift(model_type):
            return True, "Data drift detected"
            
        # Retrain if model is too old
        model_age_days = self.performance_monitor.get_model_age_days(model_type)
        if model_age_days > 30:
            return True, "Model too old"
            
        return False, "No retraining needed"
    
    def trigger_retraining(self, model_type, reason):
        """Start automated retraining process"""
        logger.info(f"Starting retraining for {model_type}: {reason}")
        
        try:
            # Prepare fresh training data
            training_data = self.data_manager.prepare_training_data(
                start_date=datetime.now() - timedelta(days=90),
                end_date=datetime.now()
            )
            
            # Train new model version
            if model_type == "lstm":
                new_model = self._retrain_lstm(training_data)
            else:  # SVM
                new_model = self._retrain_svm(training_data)
            
            # Validate new model performance
            if self._validate_new_model(new_model, model_type):
                # Save new version
                new_version = self._get_next_version(model_type)
                self._save_model_with_metadata(new_model, model_type, new_version)
                
                # Hot-swap if validation passes
                model_manager.hot_swap_model(model_type, new_version)
                
                logger.info(f"Successfully retrained and deployed {model_type} v{new_version}")
                return True
            else:
                logger.warning(f"New {model_type} model failed validation, keeping current version")
                return False
                
        except Exception as e:
            logger.error(f"Retraining failed for {model_type}: {e}")
            return False
```

**Memory Management Strategy:**
```python
class MemoryManager:
    def __init__(self, max_memory_percent=80):
        self.max_memory_percent = max_memory_percent
        self.model_cache = {}
        self.last_used = {}
        
    def get_memory_usage(self):
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def should_unload_models(self):
        """Check if we need to free memory"""
        return self.get_memory_usage() > self.max_memory_percent
    
    def unload_least_recently_used(self):
        """Unload models that haven't been used recently"""
        if not self.should_unload_models():
            return
            
        # Sort models by last used time
        sorted_models = sorted(
            self.last_used.items(), 
            key=lambda x: x[1]
        )
        
        # Unload oldest models until memory is acceptable
        for model_key, last_used_time in sorted_models:
            if not self.should_unload_models():
                break
                
            if model_key in self.model_cache:
                del self.model_cache[model_key]
                logger.info(f"Unloaded {model_key} to free memory")
    
    def load_model_with_caching(self, model_type, version):
        """Load model with intelligent caching"""
        model_key = f"{model_type}_v{version}"
        
        # Check if already in cache
        if model_key in self.model_cache:
            self.last_used[model_key] = time.time()
            return self.model_cache[model_key]
        
        # Free memory if needed before loading
        self.unload_least_recently_used()
        
        # Load model
        model, metadata = ModelManager().load_model_version(model_type, version)
        if model is not None:
            self.model_cache[model_key] = (model, metadata)
            self.last_used[model_key] = time.time()
            
        return model, metadata
    
    def cleanup_memory(self):
        """Force cleanup of unused memory"""
        import gc
        
        # Clear Python garbage collector
        gc.collect()
        
        # Clear PyTorch cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Unload old models
        self.unload_least_recently_used()
```

**Why these design decisions?**
- **Hot-swapping:** Allows model updates without API downtime, critical for production systems
- **Automated retraining:** Ensures models stay current as data patterns change over time
- **Memory management:** Prevents system crashes from memory overload, especially important with large ML models
- **Model versioning:** Enables rollback if new models perform poorly, reduces deployment risk

**Response Format Enhancement:**
```python
# Enhanced API response with ML scores
{
    "EventHistory": [
        {
            "eventId": 301,
            "epcCode": "001.8804823.0000003.000001.20241201.000000001",
            
            # Existing rule-based results
            "epcFake": True,
            "epcFakeScore": 85,
            
            # New ML results
            "lstmSequenceScore": 72,      # LSTM confidence
            "svmOutlierScore": 68,        # SVM confidence  
            "mlCombinedScore": 75,        # Combined ML score
            "finalConfidence": 80,        # Overall confidence (rule + ML)
            
            # Explanation for users
            "mlExplanation": "LSTM detected unusual sequence pattern, SVM found statistical outlier"
        }
    ],
    
    # Enhanced summary statistics
    "mlSummaryStats": {
        "totalMLAnomalies": 3,
        "lstmDetections": 2,
        "svmDetections": 1,
        "ruleOnlyDetections": 5,
        "mlOnlyDetections": 1,
        "averageConfidence": 78.5
    }
}
```

## Data Models

### Training Data Schema
```python
# Structure for training data
training_data = {
    "sequences": [
        {
            "epc_code": "001.8804823...",
            "events": [
                {
                    "eventId": 301,
                    "location_id": 1,
                    "business_step": "Factory",
                    "event_type": "Outbound", 
                    "event_time": "2024-12-01 08:00:00",
                    "features": {...},  # Preprocessed features
                    "is_anomaly": False  # Label (for validation only)
                }
            ]
        }
    ],
    "metadata": {
        "total_sequences": 10000,
        "normal_sequences": 9500,
        "anomaly_sequences": 500,
        "date_range": "2024-01-01 to 2024-12-01"
    }
}
```

### Model Persistence Schema
```python
# How we save trained models
model_metadata = {
    "model_type": "LSTM",
    "version": "1.0",
    "training_date": "2024-12-15",
    "performance_metrics": {
        "precision": 0.85,
        "recall": 0.78,
        "f1_score": 0.81,
        "auc_roc": 0.89
    },
    "hyperparameters": {
        "hidden_size": 64,
        "num_layers": 2,
        "learning_rate": 0.001,
        "batch_size": 32
    },
    "feature_schema": ["hour", "location_id", "sequence_length", ...],
    "data_stats": {
        "training_samples": 50000,
        "validation_samples": 10000,
        "test_samples": 10000
    }
}
```

## Error Handling

### Data Quality Issues
```python
def handle_data_quality_issues(data):
    """Handle common data problems"""
    
    # Missing values
    if pd.isna(data['event_time']):
        # Skip this record - can't process without timestamp
        return None
        
    if pd.isna(data['location_id']):
        # Use default location or interpolate from sequence
        data['location_id'] = infer_location_from_sequence(data)
    
    # Invalid EPC codes
    if not validate_epc_format(data['epc_code']):
        # Flag as epcFake but still process for other anomalies
        data['epc_validation_failed'] = True
    
    # Extreme values
    if data['time_since_last_scan'] > 720:  # More than 30 days
        # Cap at reasonable maximum
        data['time_since_last_scan'] = 720
        
    return data
```

### Model Loading Failures
```python
def safe_model_loading():
    """Handle model loading failures gracefully"""
    try:
        lstm_model = torch.load('models/lstm_v1.0.pth')
        svm_model = joblib.load('models/svm_v1.0.pkl')
        return lstm_model, svm_model
        
    except FileNotFoundError:
        # Fall back to rule-based only
        logger.warning("ML models not found, using rule-based detection only")
        return None, None
        
    except Exception as e:
        # Log error but don't crash the system
        logger.error(f"Model loading failed: {e}")
        return None, None
```

### Performance Degradation
```python
def monitor_performance():
    """Monitor and handle performance issues"""
    
    # If inference takes too long, use caching
    @lru_cache(maxsize=1000)
    def cached_ml_prediction(data_hash):
        return ml_model.predict(data)
    
    # If memory usage is high, unload unused models
    if psutil.virtual_memory().percent > 85:
        unload_unused_models()
    
    # If accuracy drops, trigger retraining alert
    if current_accuracy < 0.7:
        send_retraining_alert()
```

## Testing Strategy

### Unit Testing
```python
# Test individual components
def test_preprocessing():
    """Test feature engineering"""
    sample_data = {...}
    features = preprocess_for_ml(sample_data)
    
    assert 'hour' in features
    assert 0 <= features['hour'] <= 23
    assert features['location_hierarchy'] in [0, 1, 2, 3]

def test_lstm_prediction():
    """Test LSTM model output"""
    mock_sequence = create_mock_sequence()
    score = lstm_model.predict_anomaly_score(mock_sequence)
    
    assert 0 <= score <= 100
    assert isinstance(score, float)
```

### Integration Testing
```python
def test_full_pipeline():
    """Test complete ML pipeline"""
    # Use real sample data
    sample_json = load_test_data('sample_barcode_data.json')
    
    # Run through full pipeline
    result = ml_integration.detect_anomalies(sample_json)
    
    # Verify output format matches API requirements
    assert 'EventHistory' in result
    assert 'mlSummaryStats' in result
    
    # Verify scores are reasonable
    for event in result['EventHistory']:
        if 'mlCombinedScore' in event:
            assert 0 <= event['mlCombinedScore'] <= 100
```

### Performance Testing
```python
def test_performance():
    """Test system performance under load"""
    
    # Test single prediction speed
    start_time = time.time()
    result = ml_integration.detect_anomalies(sample_data)
    single_prediction_time = time.time() - start_time
    
    assert single_prediction_time < 1.0  # Should be under 1 second
    
    # Test batch processing
    batch_data = create_batch_test_data(1000)  # 1000 records
    start_time = time.time()
    results = ml_integration.detect_anomalies_batch(batch_data)
    batch_time = time.time() - start_time
    
    assert batch_time < 30.0  # Should process 1000 records in under 30 seconds
```

This design provides a comprehensive, beginner-friendly blueprint for implementing LSTM and SVM models while integrating them safely with your existing system. The architecture is modular, testable, and handles real-world challenges like data quality issues and performance requirements.