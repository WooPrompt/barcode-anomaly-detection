"""
Data Manager for SVM Training Data Storage and Loading
Handles train/test splits, feature normalization, and data persistence
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

from .base_preprocessor import FeatureNormalizer, ImbalanceHandler


class SVMDataManager:
    """Manage SVM training data with EPC mapping and confidence tracking"""
    
    def __init__(self, output_dir: str = "data/svm_training"):
        self.output_dir = output_dir
        self.feature_normalizer = FeatureNormalizer()
        self.imbalance_handler = ImbalanceHandler()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Track data metadata
        self.data_metadata = {}
    
    def save_training_data(self, features: np.ndarray, labels: List[int], 
                          scores: List[float], epc_codes: List[str],
                          anomaly_type: str, feature_names: List[str] = None,
                          test_size: float = 0.2, random_state: int = 42,
                          apply_normalization: bool = True,
                          handle_imbalance: bool = True) -> Dict[str, Any]:
        """
        Save complete training data with train/test split and preprocessing
        
        Returns metadata about the saved data
        """
        
        # Validate input dimensions
        if len(features) != len(labels) or len(labels) != len(scores) or len(scores) != len(epc_codes):
            raise ValueError("Features, labels, scores, and epc_codes must have same length")
        
        # Create anomaly-specific directory
        anomaly_dir = os.path.join(self.output_dir, anomaly_type)
        os.makedirs(anomaly_dir, exist_ok=True)
        
        # Store original data
        original_features = features.copy()
        original_labels = np.array(labels)
        original_scores = np.array(scores)
        
        # Apply feature normalization
        if apply_normalization:
            features_normalized = self.feature_normalizer.fit_transform_features(
                features, anomaly_type
            )
            # Save scaler
            scaler_path = os.path.join(anomaly_dir, "feature_scaler.joblib")
            joblib.dump(self.feature_normalizer.get_scaler(anomaly_type), scaler_path)
        else:
            features_normalized = features
        
        # Handle class imbalance
        imbalance_metadata = {}
        if handle_imbalance:
            features_balanced, labels_balanced, imbalance_info = self.imbalance_handler.handle_imbalance(
                features_normalized, original_labels
            )
            imbalance_metadata = imbalance_info
        else:
            features_balanced = features_normalized
            labels_balanced = original_labels
        
        # Train/test split
        if len(features_balanced) > 1:
            X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
                features_balanced, labels_balanced, range(len(original_labels)),
                test_size=test_size, random_state=random_state, stratify=labels_balanced
            )
        else:
            # Handle single sample case
            X_train, X_test = features_balanced, np.array([])
            y_train, y_test = labels_balanced, np.array([])
            idx_train, idx_test = [0], []
        
        # Map indices back to original data for EPC tracking
        train_epc_codes = [epc_codes[i] for i in idx_train]
        test_epc_codes = [epc_codes[i] for i in idx_test] if len(idx_test) > 0 else []
        train_scores = [original_scores[i] for i in idx_train]
        test_scores = [original_scores[i] for i in idx_test] if len(idx_test) > 0 else []
        
        # Save training data
        np.save(os.path.join(anomaly_dir, "X_train.npy"), X_train)
        np.save(os.path.join(anomaly_dir, "y_train.npy"), y_train)
        np.save(os.path.join(anomaly_dir, "train_scores.npy"), train_scores)
        
        # Save test data (if exists)
        if len(X_test) > 0:
            np.save(os.path.join(anomaly_dir, "X_test.npy"), X_test)
            np.save(os.path.join(anomaly_dir, "y_test.npy"), y_test)
            np.save(os.path.join(anomaly_dir, "test_scores.npy"), test_scores)
        
        # Save EPC mappings
        epc_mapping = {
            'train': train_epc_codes,
            'test': test_epc_codes
        }
        with open(os.path.join(anomaly_dir, "epc_mapping.json"), 'w') as f:
            json.dump(epc_mapping, f)
        
        # Save feature names if provided
        if feature_names:
            with open(os.path.join(anomaly_dir, "feature_names.json"), 'w') as f:
                json.dump(feature_names, f)
        
        # Create comprehensive metadata
        metadata = {
            'anomaly_type': anomaly_type,
            'total_samples': len(original_labels),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_dimensions': features.shape[1],
            'class_distribution': {
                'original': {
                    'positive': int(np.sum(original_labels)),
                    'negative': int(len(original_labels) - np.sum(original_labels)),
                    'positive_ratio': float(np.mean(original_labels))
                },
                'train': {
                    'positive': int(np.sum(y_train)),
                    'negative': int(len(y_train) - np.sum(y_train)),
                    'positive_ratio': float(np.mean(y_train))
                }
            },
            'feature_preprocessing': {
                'normalization_applied': apply_normalization,
                'normalization_method': self.feature_normalizer.method if apply_normalization else None,
                'imbalance_handling': imbalance_metadata
            },
            'score_statistics': {
                'train': {
                    'mean': float(np.mean(train_scores)),
                    'std': float(np.std(train_scores)),
                    'min': float(np.min(train_scores)),
                    'max': float(np.max(train_scores))
                }
            },
            'data_paths': {
                'X_train': os.path.join(anomaly_dir, "X_train.npy"),
                'y_train': os.path.join(anomaly_dir, "y_train.npy"),
                'X_test': os.path.join(anomaly_dir, "X_test.npy") if len(X_test) > 0 else None,
                'y_test': os.path.join(anomaly_dir, "y_test.npy") if len(X_test) > 0 else None,
                'scaler': scaler_path if apply_normalization else None,
                'epc_mapping': os.path.join(anomaly_dir, "epc_mapping.json"),
                'feature_names': os.path.join(anomaly_dir, "feature_names.json") if feature_names else None
            },
            'split_config': {
                'test_size': test_size,
                'random_state': random_state,
                'stratified': True
            }
        }
        
        if len(test_scores) > 0:
            metadata['score_statistics']['test'] = {
                'mean': float(np.mean(test_scores)),
                'std': float(np.std(test_scores)),
                'min': float(np.min(test_scores)),
                'max': float(np.max(test_scores))
            }
        
        # Save metadata
        with open(os.path.join(anomaly_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Store in memory for quick access
        self.data_metadata[anomaly_type] = metadata
        
        return metadata
    
    def load_training_data(self, anomaly_type: str) -> Dict[str, Any]:
        """Load complete training data for an anomaly type"""
        anomaly_dir = os.path.join(self.output_dir, anomaly_type)
        
        if not os.path.exists(anomaly_dir):
            raise FileNotFoundError(f"No training data found for {anomaly_type}")
        
        # Load metadata
        metadata_path = os.path.join(anomaly_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load training data
        X_train = np.load(os.path.join(anomaly_dir, "X_train.npy"))
        y_train = np.load(os.path.join(anomaly_dir, "y_train.npy"))
        train_scores = np.load(os.path.join(anomaly_dir, "train_scores.npy"))
        
        # Load test data if exists
        X_test_path = os.path.join(anomaly_dir, "X_test.npy")
        if os.path.exists(X_test_path):
            X_test = np.load(X_test_path)
            y_test = np.load(os.path.join(anomaly_dir, "y_test.npy"))
            test_scores = np.load(os.path.join(anomaly_dir, "test_scores.npy"))
        else:
            X_test = y_test = test_scores = None
        
        # Load EPC mapping
        with open(os.path.join(anomaly_dir, "epc_mapping.json"), 'r') as f:
            epc_mapping = json.load(f)
        
        # Load feature names if exists
        feature_names_path = os.path.join(anomaly_dir, "feature_names.json")
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
        else:
            feature_names = None
        
        # Load scaler if exists
        scaler_path = os.path.join(anomaly_dir, "feature_scaler.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = None
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'epc_mapping': epc_mapping,
            'feature_names': feature_names,
            'scaler': scaler,
            'metadata': metadata
        }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of all saved training data"""
        summary = {
            'available_anomaly_types': [],
            'total_datasets': 0,
            'overall_statistics': {}
        }
        
        if not os.path.exists(self.output_dir):
            return summary
        
        anomaly_dirs = [d for d in os.listdir(self.output_dir) 
                       if os.path.isdir(os.path.join(self.output_dir, d))]
        
        dataset_stats = []
        
        for anomaly_type in anomaly_dirs:
            metadata_path = os.path.join(self.output_dir, anomaly_type, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                summary['available_anomaly_types'].append(anomaly_type)
                dataset_stats.append({
                    'anomaly_type': anomaly_type,
                    'total_samples': metadata['total_samples'],
                    'train_samples': metadata['train_samples'],
                    'test_samples': metadata['test_samples'],
                    'feature_dimensions': metadata['feature_dimensions'],
                    'positive_ratio': metadata['class_distribution']['original']['positive_ratio']
                })
        
        summary['total_datasets'] = len(dataset_stats)
        summary['datasets'] = dataset_stats
        
        if dataset_stats:
            summary['overall_statistics'] = {
                'total_samples': sum(d['total_samples'] for d in dataset_stats),
                'avg_positive_ratio': np.mean([d['positive_ratio'] for d in dataset_stats]),
                'feature_dimensions_range': {
                    'min': min(d['feature_dimensions'] for d in dataset_stats),
                    'max': max(d['feature_dimensions'] for d in dataset_stats)
                }
            }
        
        return summary
    
    def delete_training_data(self, anomaly_type: str):
        """Delete all training data for an anomaly type"""
        import shutil
        
        anomaly_dir = os.path.join(self.output_dir, anomaly_type)
        if os.path.exists(anomaly_dir):
            shutil.rmtree(anomaly_dir)
            
        if anomaly_type in self.data_metadata:
            del self.data_metadata[anomaly_type]
    
    def validate_data_integrity(self, anomaly_type: str) -> Dict[str, Any]:
        """Validate integrity of saved training data"""
        try:
            data = self.load_training_data(anomaly_type)
            validation_results = {
                'valid': True,
                'checks': {},
                'errors': []
            }
            
            # Check data consistency
            checks = {
                'X_train_shape_valid': len(data['X_train'].shape) == 2,
                'y_train_shape_valid': len(data['y_train'].shape) == 1,
                'train_samples_match': len(data['X_train']) == len(data['y_train']),
                'epc_mapping_train_match': len(data['epc_mapping']['train']) == len(data['y_train']),
                'scores_train_match': len(data['train_scores']) == len(data['y_train'])
            }
            
            if data['X_test'] is not None:
                checks.update({
                    'X_test_shape_valid': len(data['X_test'].shape) == 2,
                    'y_test_shape_valid': len(data['y_test'].shape) == 1,
                    'test_samples_match': len(data['X_test']) == len(data['y_test']),
                    'feature_dimensions_match': data['X_train'].shape[1] == data['X_test'].shape[1],
                    'epc_mapping_test_match': len(data['epc_mapping']['test']) == len(data['y_test']),
                    'scores_test_match': len(data['test_scores']) == len(data['y_test'])
                })
            
            validation_results['checks'] = checks
            
            # Collect errors
            failed_checks = [check for check, passed in checks.items() if not passed]
            if failed_checks:
                validation_results['valid'] = False
                validation_results['errors'] = failed_checks
            
            return validation_results
            
        except Exception as e:
            return {
                'valid': False,
                'checks': {},
                'errors': [f"Failed to load data: {str(e)}"]
            }