#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM-based Anomaly Detection System for Supply Chain Barcodes
Uses 5 separate One-Class SVM models for different anomaly types.

Architecture:
- epcFake_svm: Malformed EPC detection (10 features)
- epcDup_svm: Duplicate scan detection (8 features)  
- locErr_svm: Location hierarchy violation (15 features)
- evtOrderErr_svm: Event sequence error (12 features)
- jump_svm: Impossible travel time (10 features)

Author: Data Analysis Team
Date: 2025-07-17
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, List, Any, Tuple
from datetime import datetime
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
import logging
warnings.filterwarnings('ignore')


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

# PyTorch CUDA GPU acceleration
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU acceleration enabled: {gpu_name}")
        print(f"GPU memory: {gpu_memory:.1f} GB")
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
    else:
        device = torch.device('cpu')
        print("CUDA not available, using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    device = torch.device('cpu')
    print("PyTorch not available, using CPU only")

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'svm_preprocessing'))

# Import preprocessing pipeline and feature extractors
try:
    from svm_preprocessing.pipeline import SVMPreprocessingPipeline
    from svm_preprocessing.data_manager import SVMDataManager
    from svm_preprocessing.feature_extractors.epc_fake_features import EPCFakeFeatureExtractor
    from svm_preprocessing.feature_extractors.epc_dup_features import EPCDupFeatureExtractor
    from svm_preprocessing.feature_extractors.jump_features import JumpFeatureExtractor
    from svm_preprocessing.feature_extractors.loc_err_features import LocationErrorFeatureExtractor
    from svm_preprocessing.feature_extractors.evt_order_features import EventOrderFeatureExtractor
    from svm_preprocessing.feature_extractors.jump_features_event_level import JumpFeatureExtractorEventLevel
    from svm_preprocessing.feature_extractors.loc_err_features_event_level import LocationErrorFeatureExtractorEventLevel
    PREPROCESSING_AVAILABLE = True
    print("Advanced preprocessing pipeline loaded successfully")
except ImportError as e:
    print(f"Warning: Could not import preprocessing pipeline: {e}")
    print("Falling back to basic feature extraction")
    PREPROCESSING_AVAILABLE = False
    
    # Fallback basic feature extractors
    class EPCFakeFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 10
        def get_feature_names(self): return [f"feature_{i}" for i in range(10)]
    class EPCDupFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 8
        def get_feature_names(self): return [f"feature_{i}" for i in range(8)]
    class JumpFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 10
        def get_feature_names(self): return [f"feature_{i}" for i in range(10)]
    class LocationErrorFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 15
        def get_feature_names(self): return [f"feature_{i}" for i in range(15)]
    class EventOrderFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 12
        def get_feature_names(self): return [f"feature_{i}" for i in range(12)]

# Import rule-based detector for training data generation
try:
    from multi_anomaly_detector import detect_anomalies_backend_format
except ImportError:
    try:
        from src.barcode.multi_anomaly_detector import detect_anomalies_backend_format
    except ImportError:
        print("Warning: Could not import rule-based detector")
        def detect_anomalies_backend_format(json_data):
            return '{"fileId": 1, "EventHistory": [], "epcAnomalyStats": [], "fileAnomalyStats": {"totalEvents": 0}}'

# Global SVM detector instance for CSV training
_csv_trained_detector = None


class SVMAnomalyDetector:
    """
    SVM-based anomaly detection system using 5 separate One-Class SVM models.
    Each model specializes in detecting one specific type of anomaly.
    """
    
    def __init__(self, model_dir: str = "models/svm_models", 
                 preprocessing_dir: str = "data/svm_training"):
        """Initialize SVM anomaly detector with advanced preprocessing pipeline"""
        self.model_dir = model_dir
        self.preprocessing_dir = preprocessing_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(preprocessing_dir, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # GPU configuration
        self.device = device if TORCH_AVAILABLE else torch.device('cpu')
        self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        
        # Optimize settings based on hardware
        if self.use_gpu:
            self.batch_size = 10000
            self.chunk_size = 100000
            self.logger.info(f"GPU mode enabled: batch_size={self.batch_size}, chunk_size={self.chunk_size}")
        else:
            self.batch_size = 5000
            self.chunk_size = 50000
            self.logger.info(f"CPU mode: batch_size={self.batch_size}, chunk_size={self.chunk_size}")
        
        # Initialize preprocessing pipeline if available
        if PREPROCESSING_AVAILABLE:
            self.preprocessing_pipeline = SVMPreprocessingPipeline(
                output_dir=preprocessing_dir,
                enable_logging=True
            )
            self.data_manager = SVMDataManager(preprocessing_dir)
            self.logger.info("Advanced preprocessing pipeline initialized")
        else:
            self.preprocessing_pipeline = None
            self.data_manager = None
            self.logger.warning("Using basic feature extraction")
        
        # Initialize feature extractors
        self.feature_extractors = {
            'epcFake': EPCFakeFeatureExtractor(),
            'epcDup': EPCDupFeatureExtractor(), 
            'locErr': LocationErrorFeatureExtractor(),
            'evtOrderErr': EventOrderFeatureExtractor(),
            'jump': JumpFeatureExtractor(),
            'jump_event': JumpFeatureExtractorEventLevel(),
            'locErr_event': LocationErrorFeatureExtractorEventLevel()
        }
        
        # Initialize models and scalers
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        
        # SVM hyperparameters (optimized for anomaly detection)
        self.svm_params = {
            'kernel': 'rbf',
            'gamma': 'scale',
            'nu': 0.1  # Expected fraction of outliers
            # Note: random_state not supported in older scikit-learn versions
                    }
        
        self._initialize_models()

    def _setup_logging(self):
        """Setup logging for SVM operations"""
        # Create logs directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Configure logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('logs/svm_detector.log'),
                    logging.StreamHandler()
                ]
            )
        self.logger = logging.getLogger(__name__)

    def _initialize_models(self):
        """Initialize SVM models and scalers for each anomaly type"""
        for anomaly_type in self.feature_extractors.keys():
            self.models[anomaly_type] = OneClassSVM(**self.svm_params)
            self.scalers[anomaly_type] = StandardScaler()
            self.model_metadata[anomaly_type] = {
                'trained': False,
                'training_date': None,
                'training_samples': 0,
                'feature_count': 0
            }

    def _process_single_file(self, df: pd.DataFrame, file_id: int) -> dict:
        """
        Process a single file_id and return its results in the required format.
        """
        event_history = []
        epc_anomaly_stats = {}

        epc_codes = df['epc_code'].unique()

        for epc_code in epc_codes:
            epc_group = df[df['epc_code'] == epc_code].copy()
            epc_anomaly_stats[epc_code] = {
                'epcCode': epc_code, 'totalEvents': 0, 'jumpCount': 0, 'evtOrderErrCount': 0,
                'epcFakeCount': 0, 'epcDupCount': 0, 'locErrCount': 0
            }

            # Event-level predictions
            for anomaly_type in ['jump_event', 'locErr_event']:
                if self.model_metadata.get(anomaly_type, {}).get('trained'):
                    extractor = self.feature_extractors[anomaly_type]
                    event_features = extractor.extract_features_per_event(epc_group)
                    
                    for i, features in enumerate(event_features):
                        score = self._predict_single_anomaly(anomaly_type, lambda: features)
                        if score > 0.5:
                            event_id = epc_group.iloc[i]['eventId']
                            base_anomaly_type = anomaly_type.split('_')[0]
                            epc_anomaly_stats[epc_code][f'{base_anomaly_type}Count'] += 1
                            
                            existing_event = next((e for e in event_history if e['eventId'] == event_id), None)
                            if existing_event:
                                existing_event[base_anomaly_type] = True
                                existing_event[f'{base_anomaly_type}Score'] = float(score * 100)
                            else:
                                event_history.append({
                                    'eventId': event_id,
                                    base_anomaly_type: True,
                                    f'{base_anomaly_type}Score': float(score * 100)
                                })

            # Sequence-level predictions
            for anomaly_type in ['epcFake', 'epcDup', 'evtOrderErr']:
                if self.model_metadata.get(anomaly_type, {}).get('trained'):
                    score = self._predict_single_anomaly(anomaly_type, 
                        lambda: self.feature_extractors[anomaly_type].extract_features(epc_code if anomaly_type == 'epcFake' else epc_group))
                    
                    if score > 0.5:
                        epc_anomaly_stats[epc_code][f'{anomaly_type}Count'] = 1
                        for _, event_row in epc_group.iterrows():
                            event_id = event_row['eventId']
                            existing_event = next((e for e in event_history if e['eventId'] == event_id), None)
                            if existing_event:
                                existing_event[anomaly_type] = True
                                existing_event[f'{anomaly_type}Score'] = float(score * 100)
                            else:
                                event_history.append({
                                    'eventId': event_id,
                                    anomaly_type: True,
                                    f'{anomaly_type}Score': float(score * 100)
                                })

            stats = epc_anomaly_stats[epc_code]
            stats['totalEvents'] = (stats['jumpCount'] + stats['evtOrderErrCount'] + 
                                  stats['epcFakeCount'] + stats['epcDupCount'] + stats['locErrCount'])

        epc_stats_list = [stats for stats in epc_anomaly_stats.values() if stats['totalEvents'] > 0]
        
        file_stats = {
            'totalEvents': sum(stats['totalEvents'] for stats in epc_stats_list),
            'jumpCount': sum(stats['jumpCount'] for stats in epc_stats_list),
            'evtOrderErrCount': sum(stats['evtOrderErrCount'] for stats in epc_stats_list),
            'epcFakeCount': sum(stats['epcFakeCount'] for stats in epc_stats_list),
            'epcDupCount': sum(stats['epcDupCount'] for stats in epc_stats_list),
            'locErrCount': sum(stats['locErrCount'] for stats in epc_stats_list)
        }
        
        return {
            "fileId": file_id,
            "EventHistory": event_history,
            "epcAnomalyStats": epc_stats_list,
            "fileAnomalyStats": file_stats
        }

    def _predict_single_anomaly(self, anomaly_type: str, feature_func) -> float:
        """Predict single anomaly type with GPU acceleration and return confidence score"""
        if not self.model_metadata.get(anomaly_type, {}).get('trained', False):
            return 0.0
        
        try:
            features = feature_func()
            features_array = np.array([features])
            
            # GPU-accelerated feature scaling if available
            if self.use_gpu:
                # Convert to tensor and move to GPU
                features_tensor = torch.tensor(features_array, dtype=torch.float32, device=self.device)
                
                # Apply scaling using pre-computed parameters
                if hasattr(self.scalers[anomaly_type], 'mean_'):
                    mean_tensor = torch.tensor(self.scalers[anomaly_type].mean_, device=self.device)
                    scale_tensor = torch.tensor(self.scalers[anomaly_type].scale_, device=self.device)
                    scaled_tensor = (features_tensor - mean_tensor) / scale_tensor
                    scaled_features = scaled_tensor.cpu().numpy()
                else:
                    # Fallback to CPU scaling
                    scaled_features = self.scalers[anomaly_type].transform(features_array)
            else:
                # Standard CPU scaling
                scaled_features = self.scalers[anomaly_type].transform(features_array)
            
            # Get prediction and decision score
            prediction = self.models[anomaly_type].predict(scaled_features)[0]
            decision_score = self.models[anomaly_type].decision_function(scaled_features)[0]
            
            # Convert to probability-like score (0-1)
            # SVM decision function returns distance from hyperplane
            # Negative values indicate anomalies
            if prediction == -1:  # Anomaly detected
                # Convert decision score to confidence (0-1)
                confidence = min(1.0, max(0.5, 1.0 + decision_score / abs(decision_score) * 0.5))
                return confidence
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error predicting {anomaly_type}: {e}")
            return 0.0

    def train_models(self, training_data: Dict = None, 
                    json_data_list: List[str] = None) -> Dict[str, Dict]:
        """
        Train all SVM models using One-Class SVM approach with GPU acceleration.
        
        Args:
            training_data: Pre-extracted training data
            json_data_list: Raw JSON data to generate training data from
            
        Returns:
            Training results for each model
        """
        if training_data is None:
            if json_data_list is None:
                raise ValueError("Either training_data or json_data_list must be provided")
            training_data = self.generate_training_data(json_data_list)
        
        print("Training SVM models...")
        training_results = {}
        
        for anomaly_type in self.feature_extractors.keys():
            if anomaly_type not in training_data or not training_data[anomaly_type]['features']:
                print(f"No training data for {anomaly_type}, skipping...")
                continue

            print(f"Training {anomaly_type} model...")
            
            features = np.array(training_data[anomaly_type]['features'])
            labels = np.array(training_data[anomaly_type]['labels'])
            
            if len(features) == 0:
                print(f"No training data for {anomaly_type}, skipping...")
                continue
            
            # For One-Class SVM, train only on normal data (label == 0)
            normal_features = features[labels == 0]
            
            # Validation: Ensure we have normal data and verify training labels
            unique_labels = np.unique(labels)
            print(f"{anomaly_type} - Training labels found: {unique_labels}")
            assert 0 in unique_labels, f"{anomaly_type}: No normal samples (label=0) found in training data"
            
            if len(normal_features) < 10:
                print(f"Insufficient normal data for {anomaly_type} ({len(normal_features)} samples), skipping...")
                continue
            
            # Validate no all-zero feature vectors
            zero_vectors = (normal_features == 0).all(axis=1).sum()
            if zero_vectors > 0:
                print(f"WARNING: {anomaly_type} has {zero_vectors} all-zero feature vectors")
                # Remove all-zero vectors
                non_zero_mask = ~(normal_features == 0).all(axis=1)
                normal_features = normal_features[non_zero_mask]
                if len(normal_features) < 10:
                    print(f"After removing zero vectors, insufficient data for {anomaly_type}, skipping...")
                    continue
            
            # GPU-accelerated feature scaling and training
            if self.use_gpu and len(normal_features) > 1000:
                print(f"GPU training {anomaly_type} with {len(normal_features)} samples...")
                
                # Convert to PyTorch tensors and move to GPU
                normal_tensor = torch.tensor(normal_features, dtype=torch.float32, device=self.device)
                
                # GPU-accelerated scaling
                mean = torch.mean(normal_tensor, dim=0)
                std = torch.std(normal_tensor, dim=0)
                std = torch.where(std == 0, torch.ones_like(std), std)  # Avoid division by zero
                scaled_tensor = (normal_tensor - mean) / std
                
                # Move back to CPU for scikit-learn (until cuML is available)
                scaled_features = scaled_tensor.cpu().numpy()
                
                # Fit scaler with GPU-computed parameters
                self.scalers[anomaly_type].mean_ = mean.cpu().numpy()
                self.scalers[anomaly_type].scale_ = std.cpu().numpy()
                self.scalers[anomaly_type].var_ = (std ** 2).cpu().numpy()
                self.scalers[anomaly_type].n_samples_seen_ = len(normal_features)
                
            else:
                print(f"CPU training {anomaly_type} with {len(normal_features)} samples...")
                # Standard CPU scaling
                self.scalers[anomaly_type].fit(normal_features)
                scaled_features = self.scalers[anomaly_type].transform(normal_features)
            
            # Train One-Class SVM (currently CPU-only, but data is pre-processed on GPU)
            self.models[anomaly_type].fit(scaled_features)
            
            # Evaluate on all data (normal + anomaly)
            all_scaled_features = self.scalers[anomaly_type].transform(features)
            predictions = self.models[anomaly_type].predict(all_scaled_features)
            
            # Convert SVM predictions (-1 = anomaly, 1 = normal) to binary (1 = anomaly, 0 = normal)
            predicted_anomalies = (predictions == -1).astype(int)
            
            # Calculate accuracy metrics
            from sklearn.metrics import classification_report, confusion_matrix
            accuracy = (predicted_anomalies == labels).mean()
            
            training_results[anomaly_type] = {
                'accuracy': accuracy,
                'normal_samples': len(normal_features),
                'total_samples': len(features),
                'anomaly_samples': len(features) - len(normal_features),
                'feature_dimensions': features.shape[1]
            }
            
            # Update metadata
            self.model_metadata[anomaly_type].update({
                'trained': True,
                'training_date': datetime.now().isoformat(),
                'training_samples': len(normal_features),
                'feature_count': features.shape[1]
            })
            
            print(f"{anomaly_type} - Accuracy: {accuracy:.3f}, Normal: {len(normal_features)}, Total: {len(features)}")
        
        # Save trained models
        self.save_models()
        
        return training_results

    def predict_anomalies(self, json_data: str) -> str:
        """
        Main prediction method for SVM-based anomaly detection.
        
        Args:
            json_data: JSON string with barcode scan data
            
        Returns:
            JSON string with detection results in backend format
        """
        try:
            input_data = json.loads(json_data)
            raw_df = pd.DataFrame(input_data['data'])
            
            if raw_df.empty:
                return json.dumps({
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
                }, indent=2, ensure_ascii=False)
            
            # Check if multiple file_ids exist
            file_ids = raw_df['file_id'].unique() if 'file_id' in raw_df.columns else [1]
            
            if len(file_ids) == 1:
                # Single file processing
                file_id = file_ids[0]
                result = self._process_single_file(raw_df, file_id)
                return json.dumps(convert_numpy_types(result), 
                                indent=2, ensure_ascii=False)
            else:
                # Multiple file processing
                all_results = []
                for file_id in file_ids:
                    file_df = raw_df[raw_df['file_id'] == file_id]
                    result = self._process_single_file(file_df, file_id)
                    all_results.append(result)
                
                return json.dumps(convert_numpy_types(all_results), indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Error in predict_anomalies: {e}")
            return json.dumps({"error": f"Prediction failed: {e}"}, indent=2, ensure_ascii=False)

    def generate_training_data(self, json_data_list: List[str]) -> Dict:
        """
        Generate training data from JSON input using rule-based detection as ground truth.
        """
        print("Generating training data from JSON inputs...")
        
        training_data = {}
        for anomaly_type in self.feature_extractors.keys():
            training_data[anomaly_type] = {
                'features': [],
                'labels': []
            }
        
        # Process each JSON input
        for i, json_data in enumerate(json_data_list):
            if i % 10 == 0:
                print(f"Processing batch {i+1}/{len(json_data_list)}")
            
            try:
                # Get rule-based labels as ground truth
                rule_results = detect_anomalies_backend_format(json_data)
                if isinstance(rule_results, str):
                    rule_results = json.loads(rule_results)
                
                if 'error' in rule_results:
                    continue
                
                # Parse input data
                input_data = json.loads(json_data)
                df = pd.DataFrame(input_data['data'])
                
                # Extract features for each anomaly type
                epc_codes = df['epc_code'].unique()
                
                for epc_code in epc_codes:
                    epc_group = df[df['epc_code'] == epc_code]
                    
                    # Check if this EPC has anomalies from rule-based detection
                    event_history = rule_results.get('EventHistory', [])
                    epc_event_ids = epc_group['eventId'].tolist()
                    epc_anomalies = [e for e in event_history if e.get('eventId') in epc_event_ids]
                    
                    # Extract features for each anomaly type
                    for anomaly_type in ['epcFake', 'epcDup', 'evtOrderErr']:
                        if anomaly_type in self.feature_extractors:
                            try:
                                if anomaly_type == 'epcFake':
                                    features = self.feature_extractors[anomaly_type].extract_features(epc_code)
                                else:
                                    features = self.feature_extractors[anomaly_type].extract_features(epc_group)
                                
                                # Determine label
                                has_anomaly = any(e.get(anomaly_type, False) for e in epc_anomalies)
                                label = 1 if has_anomaly else 0
                                
                                training_data[anomaly_type]['features'].append(features)
                                training_data[anomaly_type]['labels'].append(label)
                                
                            except Exception as e:
                                continue
                
            except Exception as e:
                continue
        
        # Print training data summary
        print("\nTraining data summary:")
        for anomaly_type in training_data:
            features = training_data[anomaly_type]['features']
            labels = training_data[anomaly_type]['labels']
            if len(features) > 0:
                anomaly_count = sum(labels)
                print(f"  {anomaly_type}: {len(features)} samples, {anomaly_count} anomalies ({anomaly_count/len(features)*100:.1f}%)")
            else:
                print(f"  {anomaly_type}: No training data")
        
        return training_data

    def save_models(self):
        """Save trained models to disk"""
        for anomaly_type in self.models.keys():
            if self.model_metadata.get(anomaly_type, {}).get('trained', False):
                # Save model
                model_path = os.path.join(self.model_dir, f"{anomaly_type}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models[anomaly_type], f)
                
                # Save scaler
                scaler_path = os.path.join(self.model_dir, f"{anomaly_type}_scaler.pkl")
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[anomaly_type], f)
        
        # Save metadata
        metadata_path = os.path.join(self.model_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
        
        print(f"Models saved to {self.model_dir}")
    
    def load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            # Load metadata
            metadata_path = os.path.join(self.model_dir, "model_metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            
            # Load models and scalers
            loaded_count = 0
            for anomaly_type in self.feature_extractors.keys():
                model_path = os.path.join(self.model_dir, f"{anomaly_type}_model.pkl")
                scaler_path = os.path.join(self.model_dir, f"{anomaly_type}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        self.models[anomaly_type] = pickle.load(f)
                    
                    with open(scaler_path, 'rb') as f:
                        self.scalers[anomaly_type] = pickle.load(f)
                    
                    self.model_metadata[anomaly_type]['trained'] = True
                    loaded_count += 1
            
            print(f"Loaded {loaded_count} trained models")
            return loaded_count > 0
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

# Global SVM detector instance
_svm_detector = None

def get_svm_detector() -> SVMAnomalyDetector:
    """Get or create global SVM detector instance"""
    global _svm_detector
    if _svm_detector is None:
        _svm_detector = SVMAnomalyDetector()
        _svm_detector.load_models()  # Try to load existing models
    return _svm_detector

def get_csv_trained_detector() -> SVMAnomalyDetector:
    """Get or create CSV-trained SVM detector instance"""
    global _csv_trained_detector
    if _csv_trained_detector is None:
        _csv_trained_detector = SVMAnomalyDetector()
        if not _csv_trained_detector.load_models():
            raise RuntimeError("No trained SVM models found. Please run CSV training first.")
    return _csv_trained_detector

def detect_anomalies_svm(json_data: str) -> str:
    """
    Main API function for SVM-based anomaly detection.
    
    Args:
        json_data: JSON string with barcode scan data
        
    Returns:
        JSON string with detection results in backend format
    """
    detector = get_csv_trained_detector()  # Use CSV-trained models
    return detector.predict_anomalies(json_data)

def train_svm_models(json_data_list: List[str]) -> Dict:
    """
    Train SVM models with provided training data.
    
    Args:
        json_data_list: List of JSON strings with barcode scan data
        
    Returns:
        Training results dictionary
    """
    detector = get_svm_detector()
    return detector.train_models(json_data_list=json_data_list)


if __name__ == "__main__":
    # Example usage
    print("SVM Anomaly Detection System")
    print("Available functions:")
    print("- detect_anomalies_svm(json_data): Predict anomalies")
    print("- train_svm_models(json_data_list): Train models")
    print("- get_svm_detector(): Get detector instance")
