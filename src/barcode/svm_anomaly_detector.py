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
warnings.filterwarnings('ignore')

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

# Import feature extractors with error handling
try:
    from svm_preprocessing.feature_extractors.epc_fake_features import EPCFakeFeatureExtractor
    from svm_preprocessing.feature_extractors.epc_dup_features import EPCDupFeatureExtractor
    from svm_preprocessing.feature_extractors.jump_features import JumpFeatureExtractor
    from svm_preprocessing.feature_extractors.loc_err_features import LocationErrorFeatureExtractor
    from svm_preprocessing.feature_extractors.evt_order_features import EventOrderFeatureExtractor
except ImportError as e:
    print(f"Warning: Could not import feature extractors: {e}")
    print("SVM functionality may be limited.")
    # Create dummy classes to prevent module import failure
    class EPCFakeFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 10
    class EPCDupFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 8
    class JumpFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 10
    class LocationErrorFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 15
    class EventOrderFeatureExtractor:
        def extract_features(self, *args): return [0.0] * 12

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
    
    def __init__(self, model_dir: str = "models/svm_models"):
        """Initialize SVM anomaly detector with GPU acceleration"""
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # GPU configuration
        self.device = device if TORCH_AVAILABLE else torch.device('cpu')
        self.use_gpu = TORCH_AVAILABLE and torch.cuda.is_available()
        
        # Optimize settings based on hardware
        if self.use_gpu:
            # GPU-optimized settings for RTX 4090
            self.batch_size = 10000  # Larger batches for GPU
            self.chunk_size = 100000  # Larger chunks
            print(f"GPU mode: batch_size={self.batch_size}, chunk_size={self.chunk_size}")
        else:
            # CPU-optimized settings
            self.batch_size = 5000
            self.chunk_size = 50000
            print(f"CPU mode: batch_size={self.batch_size}, chunk_size={self.chunk_size}")
        
        # Initialize feature extractors
        self.feature_extractors = {
            'epcFake': EPCFakeFeatureExtractor(),
            'epcDup': EPCDupFeatureExtractor(), 
            'locErr': LocationErrorFeatureExtractor(),
            'evtOrderErr': EventOrderFeatureExtractor(),
            'jump': JumpFeatureExtractor()
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
    
    def generate_training_data(self, json_data_list: List[str], 
                             save_training_data: bool = True) -> Dict[str, Dict]:
        """
        Generate training data using rule-based detection results.
        
        Args:
            json_data_list: List of JSON strings with barcode scan data
            save_training_data: Whether to save training data to disk
            
        Returns:
            Dictionary with training features and labels for each anomaly type
        """
        print("Generating training data from rule-based detection results...")
        
        training_data = {
            'epcFake': {'features': [], 'labels': []},
            'epcDup': {'features': [], 'labels': []},
            'locErr': {'features': [], 'labels': []},
            'evtOrderErr': {'features': [], 'labels': []},
            'jump': {'features': [], 'labels': []}
        }
        
        for i, json_data in enumerate(json_data_list):
            print(f"Processing dataset {i+1}/{len(json_data_list)}")
            
            try:
                # Get rule-based detection results
                rule_results = detect_anomalies_backend_format(json_data)
                rule_dict = json.loads(rule_results)
                
                # Parse input data
                input_data = json.loads(json_data)
                df = pd.DataFrame(input_data['data'])
                
                # Convert column names to match feature extractors
                if 'location_id' in df.columns:
                    df['reader_location'] = df['location_id'].astype(str)
                
                # Extract features for each anomaly type
                self._extract_training_features(df, rule_dict, training_data)
                
            except Exception as e:
                print(f"Error processing dataset {i+1}: {e}")
                continue
        
        # Print training data statistics
        for anomaly_type, data in training_data.items():
            normal_count = sum(1 for label in data['labels'] if label == 0)
            anomaly_count = sum(1 for label in data['labels'] if label == 1)
            print(f"{anomaly_type}: {normal_count} normal, {anomaly_count} anomaly samples")
        
        if save_training_data:
            self._save_training_data(training_data)
        
        return training_data
    
    def _extract_training_features(self, df: pd.DataFrame, 
                                 rule_results: Dict, 
                                 training_data: Dict):
        """Extract features and labels from rule-based results"""
        
        # 1. EPC Fake Features (per EPC code)
        epc_codes = df['epc_code'].unique()
        for epc_code in epc_codes:
            features = self.feature_extractors['epcFake'].extract_features(epc_code)
            
            # Check if this EPC was flagged as fake
            label = 0  # Normal
            for epc_stat in rule_results.get('epcAnomalyStats', []):
                if epc_stat['epcCode'] == epc_code and epc_stat['epcFakeCount'] > 0:
                    label = 1  # Anomaly
                    break
            
            training_data['epcFake']['features'].append(features)
            training_data['epcFake']['labels'].append(label)
        
        # 2. EPC Duplicate Features (per EPC sequence)
        for epc_code in epc_codes:
            epc_group = df[df['epc_code'] == epc_code].copy()
            features = self.feature_extractors['epcDup'].extract_features(epc_group)
            
            # Check if this EPC had duplicates
            label = 0  # Normal
            for epc_stat in rule_results.get('epcAnomalyStats', []):
                if epc_stat['epcCode'] == epc_code and epc_stat['epcDupCount'] > 0:
                    label = 1  # Anomaly
                    break
            
            training_data['epcDup']['features'].append(features)
            training_data['epcDup']['labels'].append(label)
        
        # 3. Location Error Features (per EPC sequence)
        for epc_code in epc_codes:
            epc_group = df[df['epc_code'] == epc_code].copy()
            features = self.feature_extractors['locErr'].extract_features(epc_group)
            
            # Check if this EPC had location errors
            label = 0  # Normal
            for epc_stat in rule_results.get('epcAnomalyStats', []):
                if epc_stat['epcCode'] == epc_code and epc_stat['locErrCount'] > 0:
                    label = 1  # Anomaly
                    break
            
            training_data['locErr']['features'].append(features)
            training_data['locErr']['labels'].append(label)
        
        # 4. Event Order Error Features (per EPC sequence)
        for epc_code in epc_codes:
            epc_group = df[df['epc_code'] == epc_code].copy()
            features = self.feature_extractors['evtOrderErr'].extract_features(epc_group)
            
            # Check if this EPC had event order errors
            label = 0  # Normal
            for epc_stat in rule_results.get('epcAnomalyStats', []):
                if epc_stat['epcCode'] == epc_code and epc_stat['evtOrderErrCount'] > 0:
                    label = 1  # Anomaly
                    break
            
            training_data['evtOrderErr']['features'].append(features)
            training_data['evtOrderErr']['labels'].append(label)
        
        # 5. Jump Features (per EPC sequence)
        for epc_code in epc_codes:
            epc_group = df[df['epc_code'] == epc_code].copy()
            features = self.feature_extractors['jump'].extract_features(epc_group)
            
            # Check if this EPC had jump anomalies
            label = 0  # Normal
            for epc_stat in rule_results.get('epcAnomalyStats', []):
                if epc_stat['epcCode'] == epc_code and epc_stat['jumpCount'] > 0:
                    label = 1  # Anomaly
                    break
            
            training_data['jump']['features'].append(features)
            training_data['jump']['labels'].append(label)
    
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
            print(f"Training {anomaly_type} model...")
            
            features = np.array(training_data[anomaly_type]['features'])
            labels = np.array(training_data[anomaly_type]['labels'])
            
            if len(features) == 0:
                print(f"No training data for {anomaly_type}, skipping...")
                continue
            
            # For One-Class SVM, train only on normal data (label == 0)
            normal_features = features[labels == 0]
            
            if len(normal_features) < 10:
                print(f"Insufficient normal data for {anomaly_type} ({len(normal_features)} samples), skipping...")
                continue
            
            # GPU-accelerated feature scaling and training
            if self.use_gpu and len(normal_features) > 1000:
                print(f"🎮 GPU training {anomaly_type} with {len(normal_features)} samples...")
                
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
                print(f"💻 CPU training {anomaly_type} with {len(normal_features)} samples...")
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
        Predict anomalies using trained SVM models.
        Supports multiple file_ids in single request.
        
        Args:
            json_data: JSON string with barcode scan data
            
        Returns:
            JSON string with detection results in backend format
            (separate result blocks for each file_id)
        """
        try:
            # Parse input data
            input_data = json.loads(json_data)
            df = pd.DataFrame(input_data['data'])
            
            # Convert column names to match feature extractors
            if 'location_id' in df.columns:
                df['reader_location'] = df['location_id'].astype(str)
            
            # Get unique file_ids
            file_ids = df['file_id'].unique() if 'file_id' in df.columns else [1]
            
            # Process each file_id separately and build multiple result blocks
            all_file_results = []
            
            for file_id in file_ids:
                file_df = df[df['file_id'] == file_id] if 'file_id' in df.columns else df
                file_result = self._process_single_file(file_df, int(file_id))
                all_file_results.append(file_result)
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
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
            
            all_file_results = convert_numpy_types(all_file_results)
            
            # Return array of file results for multi-file requests
            if len(all_file_results) == 1:
                # Single file - return normal format
                return json.dumps(all_file_results[0], ensure_ascii=False, indent=2)
            else:
                # Multiple files - return array of separate file result objects
                return json.dumps(all_file_results, ensure_ascii=False, indent=2)
            
        except Exception as e:
            raise Exception(f"SVM prediction error: {e}")
    
    def _process_single_file(self, df: pd.DataFrame, file_id: int) -> dict:
        """
        Process a single file_id and return its results in the required format.
        
        Args:
            df: DataFrame containing data for single file_id
            file_id: The file ID being processed
            
        Returns:
            Dictionary with results for this file_id
        """
        # Initialize result structures
        event_history = []
        epc_anomaly_stats = {}
        
        # Get unique EPC codes
        epc_codes = df['epc_code'].unique()
        
        # Process each EPC code
        for epc_code in epc_codes:
            epc_group = df[df['epc_code'] == epc_code].copy()
            epc_anomaly_stats[epc_code] = {
                'epcCode': epc_code,
                'totalEvents': 0,
                'jumpCount': 0,
                'evtOrderErrCount': 0,
                'epcFakeCount': 0,
                'epcDupCount': 0,
                'locErrCount': 0
            }
            
            # 1. EPC Fake Detection
            epc_fake_score = self._predict_single_anomaly('epcFake', 
                lambda: self.feature_extractors['epcFake'].extract_features(epc_code))
            
            if epc_fake_score > 0.5:
                epc_anomaly_stats[epc_code]['epcFakeCount'] = 1
                
                # Add to event history for each event of this EPC
                for _, event_row in epc_group.iterrows():
                    event_history.append({
                        'eventId': event_row['eventId'],
                        'epcFake': True,
                        'epcFakeScore': float(epc_fake_score * 100)
                    })
            
            # 2. EPC Duplicate Detection
            epc_dup_score = self._predict_single_anomaly('epcDup',
                lambda: self.feature_extractors['epcDup'].extract_features(epc_group))
            
            if epc_dup_score > 0.5:
                epc_anomaly_stats[epc_code]['epcDupCount'] = len(epc_group)
                
                # Add to event history for each event of this EPC
                for _, event_row in epc_group.iterrows():
                    existing_event = next((e for e in event_history if e['eventId'] == event_row['eventId']), None)
                    if existing_event:
                        existing_event['epcDup'] = True
                        existing_event['epcDupScore'] = float(epc_dup_score * 100)
                    else:
                        event_history.append({
                            'eventId': event_row['eventId'],
                            'epcDup': True,
                            'epcDupScore': float(epc_dup_score * 100)
                        })
            
            # 3. Location Error Detection
            loc_err_score = self._predict_single_anomaly('locErr',
                lambda: self.feature_extractors['locErr'].extract_features(epc_group))
            
            if loc_err_score > 0.5:
                epc_anomaly_stats[epc_code]['locErrCount'] = len(epc_group)
                
                # Add to event history for each event of this EPC
                for _, event_row in epc_group.iterrows():
                    existing_event = next((e for e in event_history if e['eventId'] == event_row['eventId']), None)
                    if existing_event:
                        existing_event['locErr'] = True
                        existing_event['locErrScore'] = float(loc_err_score * 100)
                    else:
                        event_history.append({
                            'eventId': event_row['eventId'],
                            'locErr': True,
                            'locErrScore': float(loc_err_score * 100)
                        })
            
            # 4. Event Order Error Detection
            evt_order_score = self._predict_single_anomaly('evtOrderErr',
                lambda: self.feature_extractors['evtOrderErr'].extract_features(epc_group))
            
            if evt_order_score > 0.5:
                epc_anomaly_stats[epc_code]['evtOrderErrCount'] = len(epc_group)
                
                # Add to event history for each event of this EPC
                for _, event_row in epc_group.iterrows():
                    existing_event = next((e for e in event_history if e['eventId'] == event_row['eventId']), None)
                    if existing_event:
                        existing_event['evtOrderErr'] = True
                        existing_event['evtOrderErrScore'] = float(evt_order_score * 100)
                    else:
                        event_history.append({
                            'eventId': event_row['eventId'],
                            'evtOrderErr': True,
                            'evtOrderErrScore': float(evt_order_score * 100)
                        })
            
            # 5. Jump Detection
            jump_score = self._predict_single_anomaly('jump',
                lambda: self.feature_extractors['jump'].extract_features(epc_group))
            
            if jump_score > 0.5:
                epc_anomaly_stats[epc_code]['jumpCount'] = len(epc_group)
                
                # Add to event history for each event of this EPC
                for _, event_row in epc_group.iterrows():
                    existing_event = next((e for e in event_history if e['eventId'] == event_row['eventId']), None)
                    if existing_event:
                        existing_event['jump'] = True
                        existing_event['jumpScore'] = float(jump_score * 100)
                    else:
                        event_history.append({
                            'eventId': event_row['eventId'],
                            'jump': True,
                            'jumpScore': float(jump_score * 100)
                        })
            
            # Calculate total events for this EPC
            stats = epc_anomaly_stats[epc_code]
            stats['totalEvents'] = (stats['jumpCount'] + stats['evtOrderErrCount'] + 
                                  stats['epcFakeCount'] + stats['epcDupCount'] + stats['locErrCount'])
        
        # Build final result structure
        epc_stats_list = [stats for stats in epc_anomaly_stats.values() if stats['totalEvents'] > 0]
        
        # Calculate file-level statistics
        file_stats = {
            'totalEvents': sum(stats['totalEvents'] for stats in epc_stats_list),
            'jumpCount': sum(stats['jumpCount'] for stats in epc_stats_list),
            'evtOrderErrCount': sum(stats['evtOrderErrCount'] for stats in epc_stats_list),
            'epcFakeCount': sum(stats['epcFakeCount'] for stats in epc_stats_list),
            'epcDupCount': sum(stats['epcDupCount'] for stats in epc_stats_list),
            'locErrCount': sum(stats['locErrCount'] for stats in epc_stats_list)
        }
        
        # Return standard format for single file
        return {
            "fileId": file_id,
            "EventHistory": event_history,
            "epcAnomalyStats": epc_stats_list,
            "fileAnomalyStats": file_stats
        }
    
    def _predict_single_anomaly(self, anomaly_type: str, feature_func) -> float:
        """Predict single anomaly type with GPU acceleration and return confidence score"""
        if not self.model_metadata[anomaly_type]['trained']:
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
    
    def save_models(self):
        """Save trained models to disk"""
        for anomaly_type in self.models.keys():
            if self.model_metadata[anomaly_type]['trained']:
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
            for anomaly_type in self.models.keys():
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
    
    def _save_training_data(self, training_data: Dict):
        """Save training data for future use"""
        training_data_path = os.path.join(self.model_dir, "training_data.pkl")
        with open(training_data_path, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"Training data saved to {training_data_path}")


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