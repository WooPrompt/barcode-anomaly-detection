#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Feature Engineering Framework for Barcode Anomaly Detection
Author: Data Science Expert
Date: 2025-07-20
Context: Academic-grade feature extraction with vector space optimization

This framework implements comprehensive feature engineering for supply chain
barcode anomaly detection, focusing on temporal, spatial, and behavioral patterns.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import glob
from pathlib import Path

warnings.filterwarnings('ignore')

class BarcodeFeatureEngineer:
    """
    Comprehensive feature engineering class for barcode anomaly detection
    
    Implements academic-grade feature extraction with focus on:
    - Temporal patterns and time gap analysis
    - Spatial transitions and location sequence validation
    - Behavioral anomaly detection through statistical methods
    - Vector space optimization for downstream ML models
    """
    
    def __init__(self, data_path: str = "../../../data/raw/*.csv", 
                 output_dir: str = "results/feature_engineering"):
        """
        Initialize feature engineering framework
        
        Args:
            data_path: Path to raw CSV files
            output_dir: Directory for feature engineering outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.raw_data = None
        self.processed_data = None
        self.feature_catalog = {}
        self.dimensionality_stats = {}
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        print("=== ADVANCED FEATURE ENGINEERING FRAMEWORK ===")
        print(f"Data source: {data_path}")
        print(f"Output directory: {output_dir}")
        print("Focus: Temporal, Spatial, and Behavioral Pattern Recognition")
    
    def load_and_preprocess_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load raw data with preprocessing for feature engineering
        
        Args:
            sample_size: Optional sampling for computational efficiency
            
        Returns:
            Preprocessed DataFrame ready for feature extraction
        """
        print("\n=== DATA LOADING AND PREPROCESSING ===")
        
        # Load data files
        csv_files = glob.glob(self.data_path)
        data_list = []
        
        for file_path in csv_files:
            file_name = Path(file_path).stem
            df = pd.read_csv(file_path, encoding='utf-8-sig', sep='\t')
            df['source_file'] = file_name
            
            # Sample if specified
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
            
            data_list.append(df)
            print(f"Loaded {file_name}: {len(df):,} records")
        
        # Combine data
        self.raw_data = pd.concat(data_list, ignore_index=True)
        
        # Preprocessing
        self.processed_data = self.raw_data.copy()
        
        # Convert datetime columns
        datetime_cols = ['event_time', 'manufacture_date']
        for col in datetime_cols:
            if col in self.processed_data.columns:
                self.processed_data[col] = pd.to_datetime(
                    self.processed_data[col], errors='coerce'
                )
        
        # Sort by EPC and time for sequence analysis
        self.processed_data = self.processed_data.sort_values(
            ['epc_code', 'event_time']
        ).reset_index(drop=True)
        
        print(f"Combined dataset: {len(self.processed_data):,} records")
        print(f"Unique EPCs: {self.processed_data['epc_code'].nunique():,}")
        print(f"Date range: {self.processed_data['event_time'].min()} to {self.processed_data['event_time'].max()}")
        
        return self.processed_data
    
    def extract_temporal_features(self) -> pd.DataFrame:
        """
        Extract comprehensive temporal features for anomaly detection
        
        Returns:
            DataFrame with temporal features
        """
        print("\n=== TEMPORAL FEATURE EXTRACTION ===")
        
        df = self.processed_data.copy()
        
        # Basic temporal features
        df['hour'] = df['event_time'].dt.hour
        df['day_of_week'] = df['event_time'].dt.dayofweek
        df['day_of_month'] = df['event_time'].dt.day
        df['month'] = df['event_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_business_hours'] = df['hour'].between(8, 17).astype(int)
        
        # Time gap analysis (key for anomaly detection)
        df['prev_event_time'] = df.groupby('epc_code')['event_time'].shift(1)
        df['time_gap_seconds'] = (
            df['event_time'] - df['prev_event_time']
        ).dt.total_seconds()
        
        # Statistical time gap features
        df['time_gap_log'] = np.log1p(df['time_gap_seconds'].fillna(0))
        df['time_gap_zscore'] = df.groupby('epc_code')['time_gap_seconds'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # Velocity features (events per time unit)
        df['events_per_hour'] = df.groupby([
            'epc_code', 
            df['event_time'].dt.floor('H')
        ])['epc_code'].transform('count')
        
        df['events_per_day'] = df.groupby([
            'epc_code', 
            df['event_time'].dt.date
        ])['epc_code'].transform('count')
        
        # Temporal anomaly flags
        df['unusual_time_gap'] = (
            (df['time_gap_seconds'] > df['time_gap_seconds'].quantile(0.95)) |
            (df['time_gap_seconds'] < df['time_gap_seconds'].quantile(0.05))
        ).astype(int)
        
        df['night_scan'] = (df['hour'].between(0, 5) | df['hour'].between(22, 23)).astype(int)
        
        # Manufacturing-specific temporal features
        if 'manufacture_date' in df.columns:
            df['days_since_manufacture'] = (
                df['event_time'] - df['manufacture_date']
            ).dt.days
            
            df['scan_before_manufacture'] = (
                df['event_time'] < df['manufacture_date']
            ).astype(int)
        
        # Sequence position features
        df['scan_sequence_position'] = df.groupby('epc_code').cumcount() + 1
        df['is_first_scan'] = (df['scan_sequence_position'] == 1).astype(int)
        df['total_scans_for_epc'] = df.groupby('epc_code')['epc_code'].transform('count')
        df['scan_progress_ratio'] = df['scan_sequence_position'] / df['total_scans_for_epc']
        
        # Temporal clustering features
        df['time_cluster'] = pd.cut(
            df['hour'], 
            bins=[0, 6, 12, 18, 24], 
            labels=['Night', 'Morning', 'Afternoon', 'Evening']
        )
        
        self.feature_catalog['temporal'] = {
            'basic_time': ['hour', 'day_of_week', 'day_of_month', 'month'],
            'binary_flags': ['is_weekend', 'is_business_hours', 'night_scan', 'is_first_scan'],
            'time_gaps': ['time_gap_seconds', 'time_gap_log', 'time_gap_zscore'],
            'velocity': ['events_per_hour', 'events_per_day'],
            'sequence': ['scan_sequence_position', 'total_scans_for_epc', 'scan_progress_ratio'],
            'anomaly_flags': ['unusual_time_gap', 'scan_before_manufacture'],
            'manufacturing': ['days_since_manufacture']
        }
        
        print(f"Extracted {sum(len(v) for v in self.feature_catalog['temporal'].values())} temporal features")
        
        return df
    
    def extract_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract spatial and location transition features
        
        Args:
            df: DataFrame with temporal features
            
        Returns:
            DataFrame with spatial features added
        """
        print("\n=== SPATIAL FEATURE EXTRACTION ===")
        
        # Location sequence analysis
        df['prev_location_id'] = df.groupby('epc_code')['location_id'].shift(1)
        df['next_location_id'] = df.groupby('epc_code')['location_id'].shift(-1)
        
        # Location transition features
        df['location_changed'] = (
            df['location_id'] != df['prev_location_id']
        ).astype(int)
        
        df['location_backtrack'] = (
            (df['location_id'] == df.groupby('epc_code')['location_id'].shift(2)) &
            (df['location_id'] != df['prev_location_id'])
        ).astype(int)
        
        # Business step progression analysis
        business_step_order = {
            'Factory': 1, 'WMS': 2, 'Logistics_HUB': 3, 'Distribution': 4,
            'Retail': 5, 'Customer': 6
        }
        
        df['business_step_numeric'] = df['business_step'].map(business_step_order).fillna(0)
        df['prev_business_step_numeric'] = df.groupby('epc_code')['business_step_numeric'].shift(1)
        
        df['business_step_regression'] = (
            df['business_step_numeric'] < df['prev_business_step_numeric']
        ).astype(int)
        
        df['business_step_skip'] = (
            df['business_step_numeric'] - df['prev_business_step_numeric'] > 1
        ).astype(int)
        
        # Location frequency and rarity
        location_counts = df['location_id'].value_counts()
        df['location_frequency'] = df['location_id'].map(location_counts)
        df['location_rarity_score'] = 1 / df['location_frequency']
        
        # Hub type analysis
        df['hub_type_changed'] = (
            df['hub_type'] != df.groupby('epc_code')['hub_type'].shift(1)
        ).astype(int)
        
        # Geographic clustering (if geospatial data available)
        df['location_cluster'] = pd.qcut(
            df['location_id'], 
            q=5, 
            labels=['Cluster_1', 'Cluster_2', 'Cluster_3', 'Cluster_4', 'Cluster_5']
        )
        
        # Distance proxies (using location_id as proxy)
        df['location_distance_proxy'] = np.abs(
            df['location_id'] - df['prev_location_id']
        ).fillna(0)
        
        # Transition probability features
        transition_probs = self._calculate_transition_probabilities(df)
        df['transition_probability'] = df.apply(
            lambda row: transition_probs.get(
                (row['prev_location_id'], row['location_id']), 0
            ), axis=1
        )
        
        df['rare_transition'] = (df['transition_probability'] < 0.01).astype(int)
        
        self.feature_catalog['spatial'] = {
            'location_changes': ['location_changed', 'location_backtrack', 'hub_type_changed'],
            'business_progression': ['business_step_regression', 'business_step_skip'],
            'location_characteristics': ['location_frequency', 'location_rarity_score'],
            'transitions': ['location_distance_proxy', 'transition_probability', 'rare_transition'],
            'clustering': ['location_cluster']
        }
        
        print(f"Extracted {sum(len(v) for v in self.feature_catalog['spatial'].values())} spatial features")
        
        return df
    
    def _calculate_transition_probabilities(self, df: pd.DataFrame) -> Dict[Tuple[int, int], float]:
        """Calculate location transition probabilities for anomaly detection"""
        
        # Count transitions
        transitions = df.dropna(subset=['prev_location_id', 'location_id'])
        transition_counts = transitions.groupby(['prev_location_id', 'location_id']).size()
        
        # Calculate probabilities
        location_totals = transitions.groupby('prev_location_id').size()
        transition_probs = {}
        
        for (from_loc, to_loc), count in transition_counts.items():
            prob = count / location_totals[from_loc]
            transition_probs[(from_loc, to_loc)] = prob
        
        return transition_probs
    
    def extract_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract behavioral pattern features for anomaly detection
        
        Args:
            df: DataFrame with temporal and spatial features
            
        Returns:
            DataFrame with behavioral features added
        """
        print("\n=== BEHAVIORAL FEATURE EXTRACTION ===")
        
        # EPC-level aggregation features
        epc_stats = df.groupby('epc_code').agg({
            'location_id': ['nunique', 'count'],
            'time_gap_seconds': ['mean', 'std', 'min', 'max'],
            'business_step': 'nunique',
            'operator_id': 'nunique',
            'device_id': 'nunique'
        }).round(3)
        
        epc_stats.columns = [f'epc_{col[0]}_{col[1]}' for col in epc_stats.columns]
        df = df.merge(epc_stats, left_on='epc_code', right_index=True, how='left')
        
        # Operator and device behavior
        operator_stats = df.groupby('operator_id')['epc_code'].nunique()
        df['operator_epc_diversity'] = df['operator_id'].map(operator_stats)
        
        device_stats = df.groupby('device_id')['epc_code'].nunique()
        df['device_epc_diversity'] = df['device_id'].map(device_stats)
        
        # Scanning pattern features
        df['scans_per_location'] = df.groupby(['epc_code', 'location_id'])['epc_code'].transform('count')
        df['repeated_scan_location'] = (df['scans_per_location'] > 1).astype(int)
        
        # Product-specific features
        if 'product_name' in df.columns:
            product_stats = df.groupby('product_name').agg({
                'location_id': 'nunique',
                'time_gap_seconds': 'mean'
            })
            product_stats.columns = ['product_location_diversity', 'product_avg_time_gap']
            df = df.merge(product_stats, left_on='product_name', right_index=True, how='left')
        
        # Statistical outlier features
        numerical_cols = [
            'time_gap_seconds', 'location_frequency', 'events_per_hour'
        ]
        
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[f'{col}_outlier'] = (
                    (df[col] < lower_bound) | (df[col] > upper_bound)
                ).astype(int)
        
        # Entropy-based features
        df['location_entropy'] = df.groupby('epc_code')['location_id'].transform(
            lambda x: self._calculate_entropy(x)
        )
        
        df['time_entropy'] = df.groupby('epc_code')['hour'].transform(
            lambda x: self._calculate_entropy(x)
        )
        
        self.feature_catalog['behavioral'] = {
            'epc_aggregates': [col for col in df.columns if col.startswith('epc_')],
            'operator_device': ['operator_epc_diversity', 'device_epc_diversity'],
            'scanning_patterns': ['scans_per_location', 'repeated_scan_location'],
            'outlier_flags': [col for col in df.columns if col.endswith('_outlier')],
            'entropy_features': ['location_entropy', 'time_entropy']
        }
        
        print(f"Extracted {sum(len(v) for v in self.feature_catalog['behavioral'].values())} behavioral features")
        
        return df
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy for a series"""
        value_counts = series.value_counts(normalize=True)
        return -np.sum(value_counts * np.log2(value_counts + 1e-10))
    
    def create_feature_vectors(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create numerical feature vectors for ML models
        
        Args:
            df: DataFrame with all features
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        print("\n=== FEATURE VECTOR CREATION ===")
        
        # Select numerical features
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns and target-like columns
        exclude_cols = [
            'epc_code', 'location_id', 'operator_id', 'device_id',
            'epc_company', 'epc_product', 'epc_lot', 'epc_manufacture', 'epc_serial'
        ]
        
        feature_cols = [col for col in numerical_features if col not in exclude_cols]
        
        # Handle missing values
        feature_matrix = df[feature_cols].fillna(0)
        
        # Remove constant features
        constant_features = feature_matrix.columns[feature_matrix.nunique() <= 1].tolist()
        if constant_features:
            print(f"Removing {len(constant_features)} constant features")
            feature_matrix = feature_matrix.drop(columns=constant_features)
        
        feature_names = feature_matrix.columns.tolist()
        
        print(f"Created feature matrix: {feature_matrix.shape}")
        print(f"Final feature count: {len(feature_names)}")
        
        return feature_matrix.values, feature_names
    
    def apply_dimensionality_reduction(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Apply dimensionality reduction techniques for vector space optimization
        
        Args:
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary with dimensionality reduction results
        """
        print("\n=== DIMENSIONALITY REDUCTION ===")
        
        results = {}
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA Analysis
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Find optimal number of components (80% variance)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components_80 = np.argmax(cumvar >= 0.8) + 1
        
        # Reduced PCA
        pca_reduced = PCA(n_components=n_components_80)
        X_pca_reduced = pca_reduced.fit_transform(X_scaled)
        
        results['pca'] = {
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': cumvar,
            'n_components_80': n_components_80,
            'reduced_features': X_pca_reduced,
            'feature_importance': np.abs(pca_reduced.components_).mean(axis=0)
        }
        
        # t-SNE for visualization (sample for efficiency)
        if X_scaled.shape[0] > 5000:
            sample_idx = np.random.choice(X_scaled.shape[0], 5000, replace=False)
            X_sample = X_scaled[sample_idx]
        else:
            X_sample = X_scaled
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
        
        results['tsne'] = {
            'embedding': X_tsne,
            'sample_indices': sample_idx if X_scaled.shape[0] > 5000 else None
        }
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'pca_importance': results['pca']['feature_importance'],
            'variance': np.var(X_scaled, axis=0)
        }).sort_values('pca_importance', ascending=False)
        
        results['feature_importance'] = feature_importance
        
        self.dimensionality_stats = {
            'original_dimensions': X.shape[1],
            'reduced_dimensions': n_components_80,
            'variance_retained': float(cumvar[n_components_80 - 1]),
            'compression_ratio': X.shape[1] / n_components_80
        }
        
        print(f"Original dimensions: {X.shape[1]}")
        print(f"Reduced dimensions (80% variance): {n_components_80}")
        print(f"Compression ratio: {X.shape[1] / n_components_80:.2f}x")
        
        return results
    
    def run_complete_feature_engineering(self, sample_size: Optional[int] = 50000) -> Dict:
        """
        Execute complete feature engineering pipeline
        
        Args:
            sample_size: Optional sampling for computational efficiency
            
        Returns:
            Dictionary with all results
        """
        print("=== STARTING COMPLETE FEATURE ENGINEERING ===")
        
        # Load and preprocess data
        df = self.load_and_preprocess_data(sample_size)
        
        # Extract features
        df = self.extract_temporal_features()
        df = self.extract_spatial_features(df)
        df = self.extract_behavioral_features(df)
        
        # Create feature vectors
        X, feature_names = self.create_feature_vectors(df)
        
        # Apply dimensionality reduction
        reduction_results = self.apply_dimensionality_reduction(X, feature_names)
        
        # Compile results
        results = {
            'processed_data': df,
            'feature_matrix': X,
            'feature_names': feature_names,
            'feature_catalog': self.feature_catalog,
            'dimensionality_reduction': reduction_results,
            'dimensionality_stats': self.dimensionality_stats
        }
        
        # Save results
        self._save_results(results)
        
        print("\n=== FEATURE ENGINEERING COMPLETE ===")
        print(f"Total features extracted: {len(feature_names)}")
        print(f"Feature categories: {len(self.feature_catalog)}")
        print(f"Results saved to: {self.output_dir}")
        
        return results
    
    def _save_results(self, results: Dict):
        """Save feature engineering results"""
        
        # Save feature catalog
        with open(f"{self.output_dir}/feature_catalog.json", 'w') as f:
            json.dump(self.feature_catalog, f, indent=2)
        
        # Save dimensionality statistics
        with open(f"{self.output_dir}/dimensionality_stats.json", 'w') as f:
            json.dump(self.dimensionality_stats, f, indent=2, default=str)
        
        # Save feature importance
        results['dimensionality_reduction']['feature_importance'].to_csv(
            f"{self.output_dir}/feature_importance.csv", index=False
        )
        
        # Save processed data sample
        sample_data = results['processed_data'].head(10000)
        sample_data.to_csv(f"{self.output_dir}/processed_data_sample.csv", index=False)
        
        print("Results saved successfully!")


if __name__ == "__main__":
    # Initialize feature engineering framework
    engineer = BarcodeFeatureEngineer()
    
    # Run complete pipeline
    results = engineer.run_complete_feature_engineering(sample_size=50000)
    
    print("\nFeature Engineering Framework Ready for Anomaly Detection!")