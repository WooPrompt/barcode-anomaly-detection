#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Data Cleaning and Preprocessing Framework for Barcode Anomaly Detection
Author: Data Science Expert
Date: 2025-07-20
Context: Academic-grade data preprocessing with vector space optimization

This framework implements comprehensive data cleaning and preprocessing for supply chain
barcode anomaly detection, focusing on data quality, consistency, and ML readiness.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer
import json
import glob
from pathlib import Path
import re
from collections import Counter

warnings.filterwarnings('ignore')

class BarcodeDataCleaner:
    """
    Comprehensive data cleaning and preprocessing framework for barcode anomaly detection
    
    Implements academic-grade data preprocessing with focus on:
    - Missing value detection and domain-aware imputation
    - Data consistency validation and correction
    - Feature normalization and standardization for ML readiness
    - Categorical encoding preserving domain semantics
    - Data quality assurance and validation
    """
    
    def __init__(self, data_path: str = "../../../data/raw/*.csv", 
                 output_dir: str = "results/data_cleaning"):
        """
        Initialize data cleaning framework
        
        Args:
            data_path: Path to raw CSV files
            output_dir: Directory for cleaning outputs and reports
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.raw_data = None
        self.cleaned_data = None
        self.cleaning_log = {}
        self.quality_metrics = {}
        self.encoding_mappings = {}
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        print("=== ADVANCED DATA CLEANING & PREPROCESSING FRAMEWORK ===")
        print(f"Data source: {data_path}")
        print(f"Output directory: {output_dir}")
        print("Focus: Data Quality, Consistency, and ML Readiness")
    
    def load_and_profile_data(self) -> pd.DataFrame:
        """
        Load raw data and perform initial quality profiling
        
        Returns:
            Combined DataFrame with initial quality assessment
        """
        print("\n=== DATA LOADING AND INITIAL PROFILING ===")
        
        # Load data files
        csv_files = glob.glob(self.data_path)
        data_list = []
        file_profiles = {}
        
        for file_path in csv_files:
            file_name = Path(file_path).stem
            df = pd.read_csv(file_path, encoding='utf-8-sig', sep='\t')
            df['source_file'] = file_name
            
            # Initial data profiling
            file_profiles[file_name] = self._profile_single_file(df, file_name)
            
            data_list.append(df)
            print(f"Loaded {file_name}: {len(df):,} records, {len(df.columns)} columns")
        
        # Combine data
        self.raw_data = pd.concat(data_list, ignore_index=True)
        
        # Overall data profiling
        self.quality_metrics['initial_profile'] = self._profile_combined_data(self.raw_data)
        self.quality_metrics['file_profiles'] = file_profiles
        
        print(f"\nCombined dataset: {len(self.raw_data):,} records")
        print(f"Data quality score: {self.quality_metrics['initial_profile']['quality_score']:.2f}/10")
        
        return self.raw_data
    
    def _profile_single_file(self, df: pd.DataFrame, file_name: str) -> Dict:
        """Profile individual file for quality assessment"""
        
        profile = {
            'file_name': file_name,
            'record_count': len(df),
            'column_count': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'data_types': df.dtypes.value_counts().to_dict(),
            'unique_epcs': df['epc_code'].nunique() if 'epc_code' in df.columns else 0
        }
        
        return profile
    
    def _profile_combined_data(self, df: pd.DataFrame) -> Dict:
        """Profile combined dataset for overall quality assessment"""
        
        # Calculate quality score based on multiple factors
        completeness_score = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 10
        consistency_score = self._calculate_consistency_score(df)
        validity_score = self._calculate_validity_score(df)
        
        quality_score = (completeness_score + consistency_score + validity_score) / 3
        
        profile = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'completeness_score': completeness_score,
            'consistency_score': consistency_score,
            'validity_score': validity_score,
            'quality_score': quality_score,
            'data_types_distribution': df.dtypes.value_counts().to_dict(),
            'missing_summary': df.isnull().sum().to_dict(),
            'unique_values_summary': df.nunique().to_dict()
        }
        
        return profile
    
    def _calculate_consistency_score(self, df: pd.DataFrame) -> float:
        """Calculate data consistency score based on domain rules"""
        
        consistency_issues = 0
        total_checks = 0
        
        # Check 1: EPC format consistency
        if 'epc_code' in df.columns:
            total_checks += 1
            epc_pattern = r'^\d{3}\.\d{7}\.\d{7}\.\d{6}\.\d{8}\.\d{9}$'
            invalid_epcs = ~df['epc_code'].str.match(epc_pattern, na=False)
            if invalid_epcs.sum() > 0:
                consistency_issues += 1
        
        # Check 2: Temporal consistency
        if 'event_time' in df.columns:
            total_checks += 1
            try:
                event_times = pd.to_datetime(df['event_time'], errors='coerce')
                if event_times.isnull().sum() > 0:
                    consistency_issues += 1
            except:
                consistency_issues += 1
        
        # Check 3: Location ID consistency
        if 'location_id' in df.columns:
            total_checks += 1
            if df['location_id'].isnull().sum() > 0 or (df['location_id'] < 0).sum() > 0:
                consistency_issues += 1
        
        # Check 4: Business step ordering
        if 'business_step' in df.columns:
            total_checks += 1
            valid_steps = ['Factory', 'WMS', 'Logistics_HUB', 'Distribution', 'Retail', 'Customer']
            invalid_steps = ~df['business_step'].isin(valid_steps)
            if invalid_steps.sum() > 0:
                consistency_issues += 1
        
        consistency_score = ((total_checks - consistency_issues) / max(total_checks, 1)) * 10
        return consistency_score
    
    def _calculate_validity_score(self, df: pd.DataFrame) -> float:
        """Calculate data validity score based on business rules"""
        
        validity_issues = 0
        total_checks = 0
        
        # Check 1: Future timestamps (simulation context)
        if 'event_time' in df.columns:
            total_checks += 1
            try:
                event_times = pd.to_datetime(df['event_time'], errors='coerce')
                future_events = event_times > datetime.now()
                # For simulation data, some future events are expected
                if future_events.sum() > len(df) * 0.8:  # >80% future is suspicious
                    validity_issues += 1
            except:
                validity_issues += 1
        
        # Check 2: Reasonable time gaps
        if 'event_time' in df.columns:
            total_checks += 1
            try:
                df_sorted = df.sort_values(['epc_code', 'event_time'])
                time_gaps = df_sorted.groupby('epc_code')['event_time'].diff()
                time_gaps_seconds = time_gaps.dt.total_seconds()
                unreasonable_gaps = (time_gaps_seconds > 86400 * 365).sum()  # > 1 year
                if unreasonable_gaps > 0:
                    validity_issues += 1
            except:
                validity_issues += 1
        
        # Check 3: Product name consistency
        if 'product_name' in df.columns:
            total_checks += 1
            if df['product_name'].isnull().sum() > len(df) * 0.1:  # >10% missing
                validity_issues += 1
        
        validity_score = ((total_checks - validity_issues) / max(total_checks, 1)) * 10
        return validity_score
    
    def detect_and_handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive missing value detection and domain-aware imputation
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        print("\n=== MISSING VALUE DETECTION AND IMPUTATION ===")
        
        # Analyze missing value patterns
        missing_analysis = self._analyze_missing_patterns(df)
        self.cleaning_log['missing_value_analysis'] = missing_analysis
        
        # Apply domain-aware imputation strategies
        df_imputed = df.copy()
        
        # Strategy 1: Forward fill for temporal sequences
        temporal_cols = ['event_time', 'manufacture_date']
        for col in temporal_cols:
            if col in df_imputed.columns and df_imputed[col].isnull().sum() > 0:
                print(f"Forward filling temporal column: {col}")
                df_imputed[col] = df_imputed.groupby('epc_code')[col].fillna(method='ffill')
                
                # Backward fill remaining nulls
                df_imputed[col] = df_imputed.groupby('epc_code')[col].fillna(method='bfill')
        
        # Strategy 2: Mode imputation for categorical variables
        categorical_cols = ['business_step', 'event_type', 'hub_type', 'scan_location']
        for col in categorical_cols:
            if col in df_imputed.columns and df_imputed[col].isnull().sum() > 0:
                print(f"Mode imputation for categorical column: {col}")
                mode_value = df_imputed[col].mode().iloc[0] if not df_imputed[col].mode().empty else 'Unknown'
                df_imputed[col] = df_imputed[col].fillna(mode_value)
        
        # Strategy 3: KNN imputation for numerical features
        numerical_cols = ['location_id', 'operator_id', 'device_id']
        for col in numerical_cols:
            if col in df_imputed.columns and df_imputed[col].isnull().sum() > 0:
                print(f"KNN imputation for numerical column: {col}")
                
                # Create feature matrix for KNN
                feature_cols = [c for c in ['location_id', 'operator_id', 'device_id'] 
                               if c in df_imputed.columns and c != col]
                
                if feature_cols:
                    imputer = KNNImputer(n_neighbors=5)
                    impute_data = df_imputed[feature_cols + [col]]
                    imputed_values = imputer.fit_transform(impute_data)
                    df_imputed[col] = imputed_values[:, -1]
        
        # Strategy 4: Business rule-based imputation
        df_imputed = self._apply_business_rule_imputation(df_imputed)
        
        # Log imputation results
        imputation_summary = {
            'original_missing': df.isnull().sum().to_dict(),
            'final_missing': df_imputed.isnull().sum().to_dict(),
            'imputation_methods': {
                'temporal_forward_fill': temporal_cols,
                'categorical_mode': categorical_cols,
                'numerical_knn': numerical_cols,
                'business_rules': 'Applied'
            }
        }
        
        self.cleaning_log['imputation_summary'] = imputation_summary
        
        print(f"Missing values reduced from {df.isnull().sum().sum():,} to {df_imputed.isnull().sum().sum():,}")
        
        return df_imputed
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze missing value patterns for informed imputation strategy"""
        
        missing_analysis = {}
        
        # Missing value counts and percentages
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        missing_analysis['by_column'] = {
            'counts': missing_counts.to_dict(),
            'percentages': missing_percentages.to_dict()
        }
        
        # Missing value patterns (combinations)
        missing_patterns = df.isnull().apply(lambda x: ''.join(x.astype(int).astype(str)), axis=1)
        pattern_counts = missing_patterns.value_counts().head(10)
        
        missing_analysis['patterns'] = pattern_counts.to_dict()
        
        # Missing value correlation
        if missing_counts.sum() > 0:
            missing_corr = df.isnull().corr()
            missing_analysis['correlations'] = missing_corr.to_dict()
        
        # Missingness by source file
        if 'source_file' in df.columns:
            missing_by_source = df.groupby('source_file').apply(
                lambda x: x.isnull().sum() / len(x) * 100
            )
            missing_analysis['by_source_file'] = missing_by_source.to_dict()
        
        return missing_analysis
    
    def _apply_business_rule_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply domain-specific business rules for imputation"""
        
        # Rule 1: If scan_location is missing but location_id exists, infer from location mapping
        if 'scan_location' in df.columns and 'location_id' in df.columns:
            location_mapping = df.dropna(subset=['scan_location', 'location_id']).groupby('location_id')['scan_location'].first()
            
            missing_scan_location = df['scan_location'].isnull() & df['location_id'].notnull()
            df.loc[missing_scan_location, 'scan_location'] = df.loc[missing_scan_location, 'location_id'].map(location_mapping)
        
        # Rule 2: If business_step is missing, infer from hub_type
        if 'business_step' in df.columns and 'hub_type' in df.columns:
            hub_to_step_mapping = {
                'Factory': 'Factory',
                'WMS': 'WMS', 
                'Logistics_HUB': 'Logistics_HUB',
                'Distribution': 'Distribution',
                'Retail': 'Retail'
            }
            
            missing_business_step = df['business_step'].isnull() & df['hub_type'].notnull()
            df.loc[missing_business_step, 'business_step'] = df.loc[missing_business_step, 'hub_type'].map(
                lambda x: next((step for hub, step in hub_to_step_mapping.items() if hub in str(x)), 'Unknown')
            )
        
        # Rule 3: If manufacture_date is missing, use median by product
        if 'manufacture_date' in df.columns and 'product_name' in df.columns:
            missing_mfg_date = df['manufacture_date'].isnull()
            if missing_mfg_date.sum() > 0:
                product_median_dates = df.groupby('product_name')['manufacture_date'].median()
                df.loc[missing_mfg_date, 'manufacture_date'] = df.loc[missing_mfg_date, 'product_name'].map(product_median_dates)
        
        return df
    
    def detect_and_correct_inconsistencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and correct data inconsistencies based on domain knowledge
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with inconsistencies corrected
        """
        print("\n=== CONSISTENCY DETECTION AND CORRECTION ===")
        
        df_corrected = df.copy()
        inconsistency_log = {}
        
        # 1. EPC format validation and correction
        if 'epc_code' in df_corrected.columns:
            epc_issues = self._correct_epc_format(df_corrected)
            inconsistency_log['epc_format'] = epc_issues
        
        # 2. Temporal consistency validation
        if 'event_time' in df_corrected.columns:
            temporal_issues = self._correct_temporal_inconsistencies(df_corrected)
            inconsistency_log['temporal'] = temporal_issues
        
        # 3. Location consistency validation
        location_issues = self._correct_location_inconsistencies(df_corrected)
        inconsistency_log['location'] = location_issues
        
        # 4. Business process consistency
        process_issues = self._correct_business_process_inconsistencies(df_corrected)
        inconsistency_log['business_process'] = process_issues
        
        # 5. Cross-field consistency validation
        cross_field_issues = self._correct_cross_field_inconsistencies(df_corrected)
        inconsistency_log['cross_field'] = cross_field_issues
        
        self.cleaning_log['inconsistency_corrections'] = inconsistency_log
        
        total_corrections = sum(len(issues.get('corrections', [])) for issues in inconsistency_log.values())
        print(f"Applied {total_corrections} consistency corrections")
        
        return df_corrected
    
    def _correct_epc_format(self, df: pd.DataFrame) -> Dict:
        """Correct EPC format inconsistencies"""
        
        epc_issues = {'invalid_format': [], 'corrections': []}
        
        if 'epc_code' not in df.columns:
            return epc_issues
        
        # Expected EPC format: 001.XXXXXXX.XXXXXXX.XXXXXX.XXXXXXXX.XXXXXXXXX
        epc_pattern = r'^(\d{3})\.(\d{7})\.(\d{7})\.(\d{6})\.(\d{8})\.(\d{9})$'
        
        for idx, epc in df['epc_code'].items():
            if pd.isna(epc):
                continue
                
            epc_str = str(epc)
            
            # Check if EPC matches expected format
            if not re.match(epc_pattern, epc_str):
                epc_issues['invalid_format'].append({
                    'index': idx,
                    'original_epc': epc_str,
                    'issue': 'Invalid format'
                })
                
                # Attempt automatic correction
                corrected_epc = self._attempt_epc_correction(epc_str)
                if corrected_epc != epc_str:
                    df.loc[idx, 'epc_code'] = corrected_epc
                    epc_issues['corrections'].append({
                        'index': idx,
                        'original': epc_str,
                        'corrected': corrected_epc
                    })
        
        return epc_issues
    
    def _attempt_epc_correction(self, epc_str: str) -> str:
        """Attempt to correct common EPC format issues"""
        
        # Remove whitespace and convert to uppercase
        cleaned_epc = epc_str.strip().upper()
        
        # If missing dots, try to add them based on expected positions
        if '.' not in cleaned_epc and len(cleaned_epc) >= 40:
            # Insert dots at expected positions: 3, 10, 17, 23, 31
            corrected = f"{cleaned_epc[:3]}.{cleaned_epc[3:10]}.{cleaned_epc[10:17]}.{cleaned_epc[17:23]}.{cleaned_epc[23:31]}.{cleaned_epc[31:]}"
            return corrected
        
        # If has wrong separators, replace with dots
        for separator in ['-', '_', ' ', ',']:
            if separator in cleaned_epc:
                corrected = cleaned_epc.replace(separator, '.')
                return corrected
        
        return epc_str  # Return original if no correction possible
    
    def _correct_temporal_inconsistencies(self, df: pd.DataFrame) -> Dict:
        """Correct temporal data inconsistencies"""
        
        temporal_issues = {'invalid_timestamps': [], 'corrections': []}
        
        datetime_cols = ['event_time', 'manufacture_date', 'expiry_date']
        
        for col in datetime_cols:
            if col not in df.columns:
                continue
                
            # Convert to datetime and identify invalid entries
            try:
                original_values = df[col].copy()
                df[col] = pd.to_datetime(df[col], errors='coerce')
                
                # Find entries that became NaT (invalid)
                invalid_mask = df[col].isnull() & original_values.notnull()
                
                for idx in df[invalid_mask].index:
                    temporal_issues['invalid_timestamps'].append({
                        'index': idx,
                        'column': col,
                        'original_value': original_values.loc[idx]
                    })
                
                # Attempt correction for invalid timestamps
                if col == 'event_time':
                    # For event_time, interpolate based on sequence
                    df[col] = df.groupby('epc_code')[col].apply(
                        lambda x: x.interpolate(method='time')
                    )
                
            except Exception as e:
                temporal_issues['processing_error'] = str(e)
        
        return temporal_issues
    
    def _correct_location_inconsistencies(self, df: pd.DataFrame) -> Dict:
        """Correct location-related inconsistencies"""
        
        location_issues = {'invalid_locations': [], 'corrections': []}
        
        # Check for negative location IDs
        if 'location_id' in df.columns:
            negative_locations = df['location_id'] < 0
            if negative_locations.sum() > 0:
                location_issues['invalid_locations'].extend([
                    {'index': idx, 'issue': 'Negative location_id', 'value': val}
                    for idx, val in df[negative_locations]['location_id'].items()
                ])
                
                # Correct negative location IDs (take absolute value)
                df.loc[negative_locations, 'location_id'] = df.loc[negative_locations, 'location_id'].abs()
                location_issues['corrections'].append('Converted negative location_id to positive')
        
        # Check location-scan_location consistency
        if 'location_id' in df.columns and 'scan_location' in df.columns:
            # Build location mapping from non-null pairs
            valid_pairs = df.dropna(subset=['location_id', 'scan_location'])
            location_mapping = valid_pairs.groupby('location_id')['scan_location'].first()
            
            # Check for inconsistent mappings
            for loc_id, group in df.groupby('location_id'):
                unique_scan_locations = group['scan_location'].dropna().unique()
                if len(unique_scan_locations) > 1:
                    location_issues['invalid_locations'].append({
                        'location_id': loc_id,
                        'issue': 'Multiple scan_location values',
                        'values': unique_scan_locations.tolist()
                    })
        
        return location_issues
    
    def _correct_business_process_inconsistencies(self, df: pd.DataFrame) -> Dict:
        """Correct business process flow inconsistencies"""
        
        process_issues = {'invalid_sequences': [], 'corrections': []}
        
        if 'business_step' not in df.columns:
            return process_issues
        
        # Define valid business step order
        step_order = {
            'Factory': 1,
            'WMS': 2, 
            'Logistics_HUB': 3,
            'Distribution': 4,
            'Retail': 5,
            'Customer': 6
        }
        
        # Check for backward movements in business process
        df['step_order'] = df['business_step'].map(step_order)
        df_sorted = df.sort_values(['epc_code', 'event_time'])
        
        for epc, group in df_sorted.groupby('epc_code'):
            prev_step = 0
            for idx, row in group.iterrows():
                current_step = row['step_order']
                if pd.notna(current_step) and current_step < prev_step:
                    process_issues['invalid_sequences'].append({
                        'epc_code': epc,
                        'index': idx,
                        'issue': 'Backward business step movement',
                        'from_step': prev_step,
                        'to_step': current_step
                    })
                
                if pd.notna(current_step):
                    prev_step = current_step
        
        # Remove temporary column
        df.drop('step_order', axis=1, inplace=True)
        
        return process_issues
    
    def _correct_cross_field_inconsistencies(self, df: pd.DataFrame) -> Dict:
        """Correct inconsistencies across multiple fields"""
        
        cross_field_issues = {'inconsistencies': [], 'corrections': []}
        
        # Check event_time vs manufacture_date consistency
        if 'event_time' in df.columns and 'manufacture_date' in df.columns:
            # Event time should be after manufacture date
            invalid_timing = (df['event_time'] < df['manufacture_date']) & df['event_time'].notnull() & df['manufacture_date'].notnull()
            
            if invalid_timing.sum() > 0:
                cross_field_issues['inconsistencies'].extend([
                    {
                        'index': idx,
                        'issue': 'Event time before manufacture date',
                        'event_time': row['event_time'],
                        'manufacture_date': row['manufacture_date']
                    }
                    for idx, row in df[invalid_timing].iterrows()
                ])
        
        # Check hub_type vs business_step consistency
        if 'hub_type' in df.columns and 'business_step' in df.columns:
            expected_combinations = {
                ('Factory', 'Factory'),
                ('WMS', 'WMS'),
                ('Logistics_HUB', 'Logistics_HUB'),
                ('Distribution', 'Distribution'),
                ('Retail', 'Retail')
            }
            
            actual_combinations = set(zip(df['hub_type'].fillna(''), df['business_step'].fillna('')))
            unexpected_combinations = actual_combinations - expected_combinations
            
            if unexpected_combinations:
                cross_field_issues['inconsistencies'].append({
                    'issue': 'Unexpected hub_type/business_step combinations',
                    'combinations': list(unexpected_combinations)
                })
        
        return cross_field_issues
    
    def normalize_and_standardize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply normalization and standardization for ML readiness
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with normalized/standardized features
        """
        print("\n=== FEATURE NORMALIZATION AND STANDARDIZATION ===")
        
        df_normalized = df.copy()
        normalization_log = {}
        
        # Identify numerical columns for normalization
        numerical_cols = df_normalized.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns from normalization
        id_cols = ['location_id', 'operator_id', 'device_id', 'epc_company', 'epc_product', 
                  'epc_lot', 'epc_manufacture', 'epc_serial']
        numerical_cols = [col for col in numerical_cols if col not in id_cols]
        
        if numerical_cols:
            # Strategy 1: StandardScaler for features with normal distribution
            normal_features = self._identify_normal_features(df_normalized, numerical_cols)
            if normal_features:
                scaler = StandardScaler()
                df_normalized[normal_features] = scaler.fit_transform(df_normalized[normal_features])
                
                # Store scaler for future use
                self.encoding_mappings['standard_scaler'] = {
                    'scaler': scaler,
                    'features': normal_features
                }
                
                normalization_log['standard_scaled'] = normal_features
                print(f"Standard scaling applied to {len(normal_features)} features")
            
            # Strategy 2: RobustScaler for features with outliers
            outlier_features = [col for col in numerical_cols if col not in normal_features]
            if outlier_features:
                robust_scaler = RobustScaler()
                df_normalized[outlier_features] = robust_scaler.fit_transform(df_normalized[outlier_features])
                
                # Store scaler for future use
                self.encoding_mappings['robust_scaler'] = {
                    'scaler': robust_scaler,
                    'features': outlier_features
                }
                
                normalization_log['robust_scaled'] = outlier_features
                print(f"Robust scaling applied to {len(outlier_features)} features")
        
        # Strategy 3: MinMax scaling for bounded features
        bounded_features = self._identify_bounded_features(df_normalized)
        if bounded_features:
            minmax_scaler = MinMaxScaler()
            df_normalized[bounded_features] = minmax_scaler.fit_transform(df_normalized[bounded_features])
            
            # Store scaler for future use
            self.encoding_mappings['minmax_scaler'] = {
                'scaler': minmax_scaler,
                'features': bounded_features
            }
            
            normalization_log['minmax_scaled'] = bounded_features
            print(f"MinMax scaling applied to {len(bounded_features)} features")
        
        self.cleaning_log['normalization'] = normalization_log
        
        return df_normalized
    
    def _identify_normal_features(self, df: pd.DataFrame, numerical_cols: List[str]) -> List[str]:
        """Identify features that follow normal distribution for standard scaling"""
        
        normal_features = []
        
        for col in numerical_cols:
            if col not in df.columns or df[col].isnull().all():
                continue
                
            # Sample for large datasets
            sample_data = df[col].dropna().sample(min(5000, len(df[col].dropna())), random_state=42)
            
            # Shapiro-Wilk test for normality
            if len(sample_data) >= 3:
                try:
                    _, p_value = stats.shapiro(sample_data)
                    if p_value > 0.05:  # Normal distribution
                        normal_features.append(col)
                except:
                    pass  # Skip if test fails
        
        return normal_features
    
    def _identify_bounded_features(self, df: pd.DataFrame) -> List[str]:
        """Identify features that are naturally bounded (percentages, ratios, etc.)"""
        
        bounded_features = []
        
        # Look for features that might be percentages or ratios
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                if col.lower() in ['percentage', 'ratio', 'rate', 'probability']:
                    bounded_features.append(col)
                elif df[col].min() >= 0 and df[col].max() <= 1:
                    bounded_features.append(col)
        
        return bounded_features
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables preserving domain semantics
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with encoded categorical variables
        """
        print("\n=== CATEGORICAL VARIABLE ENCODING ===")
        
        df_encoded = df.copy()
        encoding_log = {}
        
        # Identify categorical columns
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
        
        # Remove columns that should not be encoded
        exclude_cols = ['epc_code', 'source_file']
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        for col in categorical_cols:
            unique_values = df_encoded[col].nunique()
            
            if unique_values <= 2:
                # Binary encoding for binary categorical variables
                df_encoded, encoding_info = self._binary_encode(df_encoded, col)
                encoding_log[col] = {'method': 'binary', 'info': encoding_info}
                
            elif unique_values <= 10:
                # One-hot encoding for low cardinality categorical variables
                df_encoded, encoding_info = self._onehot_encode(df_encoded, col)
                encoding_log[col] = {'method': 'onehot', 'info': encoding_info}
                
            elif col in ['scan_location', 'product_name']:
                # Label encoding for high cardinality but ordinal nature
                df_encoded, encoding_info = self._label_encode(df_encoded, col)
                encoding_log[col] = {'method': 'label', 'info': encoding_info}
                
            else:
                # Target encoding or frequency encoding for very high cardinality
                df_encoded, encoding_info = self._frequency_encode(df_encoded, col)
                encoding_log[col] = {'method': 'frequency', 'info': encoding_info}
        
        # Special encoding for ordinal variables
        ordinal_cols = ['business_step']
        for col in ordinal_cols:
            if col in df_encoded.columns:
                df_encoded, encoding_info = self._ordinal_encode(df_encoded, col)
                encoding_log[col] = {'method': 'ordinal', 'info': encoding_info}
        
        self.cleaning_log['categorical_encoding'] = encoding_log
        
        print(f"Encoded {len(encoding_log)} categorical variables")
        
        return df_encoded
    
    def _binary_encode(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, Dict]:
        """Binary encode for binary categorical variables"""
        
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            # Create binary mapping
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df[f'{col}_encoded'] = df[col].map(mapping)
            
            # Store mapping for future use
            self.encoding_mappings[f'{col}_binary'] = mapping
            
            encoding_info = {'mapping': mapping, 'new_column': f'{col}_encoded'}
            
            # Drop original column
            df.drop(col, axis=1, inplace=True)
            
        else:
            encoding_info = {'error': f'Expected 2 unique values, found {len(unique_vals)}'}
        
        return df, encoding_info
    
    def _onehot_encode(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, Dict]:
        """One-hot encode for low cardinality categorical variables"""
        
        # Create dummy variables
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
        
        # Add dummy columns to dataframe
        df = pd.concat([df, dummies], axis=1)
        
        # Store column names for future use
        dummy_cols = dummies.columns.tolist()
        self.encoding_mappings[f'{col}_onehot'] = dummy_cols
        
        encoding_info = {'dummy_columns': dummy_cols}
        
        # Drop original column
        df.drop(col, axis=1, inplace=True)
        
        return df, encoding_info
    
    def _label_encode(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, Dict]:
        """Label encode for high cardinality categorical variables"""
        
        label_encoder = LabelEncoder()
        
        # Handle missing values
        df_temp = df[col].fillna('Missing')
        
        # Fit and transform
        encoded_values = label_encoder.fit_transform(df_temp)
        df[f'{col}_encoded'] = encoded_values
        
        # Store encoder for future use
        self.encoding_mappings[f'{col}_label'] = {
            'encoder': label_encoder,
            'classes': label_encoder.classes_.tolist()
        }
        
        encoding_info = {
            'classes': label_encoder.classes_.tolist(),
            'new_column': f'{col}_encoded'
        }
        
        # Drop original column
        df.drop(col, axis=1, inplace=True)
        
        return df, encoding_info
    
    def _frequency_encode(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, Dict]:
        """Frequency encode for very high cardinality categorical variables"""
        
        # Calculate frequency mapping
        freq_mapping = df[col].value_counts().to_dict()
        
        # Apply frequency encoding
        df[f'{col}_freq_encoded'] = df[col].map(freq_mapping)
        
        # Store mapping for future use
        self.encoding_mappings[f'{col}_frequency'] = freq_mapping
        
        encoding_info = {
            'frequency_mapping': freq_mapping,
            'new_column': f'{col}_freq_encoded'
        }
        
        # Drop original column
        df.drop(col, axis=1, inplace=True)
        
        return df, encoding_info
    
    def _ordinal_encode(self, df: pd.DataFrame, col: str) -> Tuple[pd.DataFrame, Dict]:
        """Ordinal encode for ordered categorical variables"""
        
        if col == 'business_step':
            # Define business step order
            order_mapping = {
                'Factory': 1,
                'WMS': 2,
                'Logistics_HUB': 3,
                'Distribution': 4,
                'Retail': 5,
                'Customer': 6
            }
        else:
            # Generic ordinal encoding based on natural order
            unique_vals = sorted(df[col].dropna().unique())
            order_mapping = {val: idx for idx, val in enumerate(unique_vals)}
        
        # Apply ordinal encoding
        df[f'{col}_ordinal'] = df[col].map(order_mapping)
        
        # Store mapping for future use
        self.encoding_mappings[f'{col}_ordinal'] = order_mapping
        
        encoding_info = {
            'order_mapping': order_mapping,
            'new_column': f'{col}_ordinal'
        }
        
        # Drop original column
        df.drop(col, axis=1, inplace=True)
        
        return df, encoding_info
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive data quality validation after cleaning
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            Data quality validation report
        """
        print("\n=== DATA QUALITY VALIDATION ===")
        
        validation_report = {}
        
        # 1. Completeness validation
        missing_counts = df.isnull().sum()
        completeness_score = (1 - missing_counts.sum() / (len(df) * len(df.columns))) * 100
        
        validation_report['completeness'] = {
            'score': completeness_score,
            'missing_by_column': missing_counts.to_dict(),
            'total_missing': missing_counts.sum()
        }
        
        # 2. Consistency validation
        consistency_score = self._calculate_consistency_score(df)
        validation_report['consistency'] = {
            'score': consistency_score
        }
        
        # 3. Validity validation
        validity_score = self._calculate_validity_score(df)
        validation_report['validity'] = {
            'score': validity_score
        }
        
        # 4. Uniqueness validation
        duplicate_count = df.duplicated().sum()
        uniqueness_score = (1 - duplicate_count / len(df)) * 100
        
        validation_report['uniqueness'] = {
            'score': uniqueness_score,
            'duplicate_rows': duplicate_count
        }
        
        # 5. Distribution validation
        distribution_stats = self._validate_feature_distributions(df)
        validation_report['distributions'] = distribution_stats
        
        # 6. Overall quality score
        overall_score = (completeness_score + consistency_score + validity_score + uniqueness_score) / 4
        validation_report['overall_quality_score'] = overall_score
        
        print(f"Data quality validation complete. Overall score: {overall_score:.2f}/100")
        
        return validation_report
    
    def _validate_feature_distributions(self, df: pd.DataFrame) -> Dict:
        """Validate feature distributions for anomaly detection suitability"""
        
        distribution_stats = {}
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            if df[col].isnull().all():
                continue
                
            col_data = df[col].dropna()
            
            stats_dict = {
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'q25': float(col_data.quantile(0.25)),
                'q75': float(col_data.quantile(0.75)),
                'outlier_count': int(((col_data < col_data.quantile(0.25) - 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25))) |
                                    (col_data > col_data.quantile(0.75) + 1.5 * (col_data.quantile(0.75) - col_data.quantile(0.25)))).sum())
            }
            
            distribution_stats[col] = stats_dict
        
        return distribution_stats
    
    def run_complete_cleaning_pipeline(self) -> pd.DataFrame:
        """
        Execute complete data cleaning and preprocessing pipeline
        
        Returns:
            Cleaned and processed DataFrame ready for ML
        """
        print("=== STARTING COMPLETE DATA CLEANING PIPELINE ===")
        
        # Step 1: Load and profile data
        raw_data = self.load_and_profile_data()
        
        # Step 2: Handle missing values
        data_after_imputation = self.detect_and_handle_missing_values(raw_data)
        
        # Step 3: Correct inconsistencies
        data_after_correction = self.detect_and_correct_inconsistencies(data_after_imputation)
        
        # Step 4: Normalize and standardize
        data_after_normalization = self.normalize_and_standardize_features(data_after_correction)
        
        # Step 5: Encode categorical variables
        data_after_encoding = self.encode_categorical_variables(data_after_normalization)
        
        # Step 6: Final validation
        final_quality_report = self.validate_data_quality(data_after_encoding)
        
        # Store cleaned data
        self.cleaned_data = data_after_encoding
        self.quality_metrics['final_quality'] = final_quality_report
        
        # Save results
        self._save_cleaning_results()
        
        print("\n=== DATA CLEANING PIPELINE COMPLETE ===")
        print(f"Original records: {len(raw_data):,}")
        print(f"Final records: {len(self.cleaned_data):,}")
        print(f"Final quality score: {final_quality_report['overall_quality_score']:.2f}/100")
        print(f"Results saved to: {self.output_dir}")
        
        return self.cleaned_data
    
    def _save_cleaning_results(self):
        """Save cleaning results and documentation"""
        
        # Save cleaned data sample
        sample_data = self.cleaned_data.head(10000)
        sample_data.to_csv(f"{self.output_dir}/cleaned_data_sample.csv", index=False)
        
        # Save cleaning log
        with open(f"{self.output_dir}/cleaning_log.json", 'w') as f:
            json.dump(self.cleaning_log, f, indent=2, default=str)
        
        # Save quality metrics
        with open(f"{self.output_dir}/quality_metrics.json", 'w') as f:
            json.dump(self.quality_metrics, f, indent=2, default=str)
        
        # Save encoding mappings
        encoding_mappings_serializable = {}
        for key, value in self.encoding_mappings.items():
            if 'scaler' in str(type(value)) or 'encoder' in str(type(value)):
                # Skip non-serializable objects, save metadata instead
                encoding_mappings_serializable[key] = str(type(value))
            else:
                encoding_mappings_serializable[key] = value
        
        with open(f"{self.output_dir}/encoding_mappings.json", 'w') as f:
            json.dump(encoding_mappings_serializable, f, indent=2, default=str)
        
        print("Cleaning results saved successfully!")


if __name__ == "__main__":
    # Initialize data cleaning framework
    cleaner = BarcodeDataCleaner()
    
    # Run complete pipeline
    cleaned_data = cleaner.run_complete_cleaning_pipeline()
    
    print("\nData Cleaning Framework Ready for Feature Engineering and ML!")