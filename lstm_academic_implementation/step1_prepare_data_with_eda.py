#!/usr/bin/env python3
"""
Step 1: Data Preparation Using EDA Insights
Uses pre-existing EDA analysis to guide data preprocessing
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path

# Add source paths
sys.path.append('src')
sys.path.append('lstm_academic_implementation/src')

from lstm_data_preprocessor import LSTMDataPreprocessor
from lstm_critical_fixes import AdaptiveDimensionalityReducer

from pathlib import Path
import pandas as pd
import json

def load_eda_insights():
    """Load insights from pre-existing EDA analysis and print them."""
    
    insights = {}

    # Get base project directory
    base_dir = Path(__file__).resolve().parent.parent  # Adjust if needed
    results_dir = base_dir / 'src' / 'barcode' / 'EDA' / 'results'

    # 1. Data quality report (TXT)
    quality_file = results_dir / 'data_quality_report.txt'
    if quality_file.exists():
        with open(quality_file, 'r', encoding='utf-8') as f:
            content = f.read()
            insights['data_quality'] = content
            print("\n📋 Data Quality Report:")
            print(content)
    else:
        print("⚠️ data_quality_report.txt not found.")

    # 2. Statistical analysis (JSON)
    stats_file = results_dir / 'statistical_analysis.json'
    if stats_file.exists():
        with open(stats_file, 'r', encoding='utf-8') as f:
            stats = json.load(f)
            insights['statistics'] = stats
            print("\n📊 Statistical Analysis:")
            print(json.dumps(stats, indent=2))
    else:
        print("⚠️ statistical_analysis.json not found.")

    # 3. Correlation matrix (CSV)
    corr_file = results_dir / 'correlation_matrix.csv'
    if corr_file.exists():
        df_corr = pd.read_csv(corr_file, index_col=0)
        insights['correlations'] = df_corr
        print("\n🔗 Correlation Matrix:")
        print(df_corr)
    else:
        print("⚠️ correlation_matrix.csv not found.")

    return insights



def prepare_lstm_data():
    """Prepare data for LSTM training using EDA guidance"""
    
    print("Starting LSTM data preparation with EDA insights")
    
    # Step 1: Load EDA insights
    eda_insights = load_eda_insights()
    
    # Step 2: Load raw CSV files (as specified in principle.llm.txt)
    csv_files = [
        'data/raw/icn.csv',
        'data/raw/kum.csv', 
        'data/raw/ygs.csv',
        'data/raw/hws.csv'
    ]
    
    print(f"Loading {len(csv_files)} raw CSV files...")
    
    # Check file existence
    existing_files = []
    for file in csv_files:
        if os.path.exists(file):
            existing_files.append(file)
            file_size = os.path.getsize(file) / (1024*1024)  # MB
            print(f"   Found {file} ({file_size:.1f} MB)")
        else:
            print(f"   Missing {file}")
    
    if not existing_files:
        print("ERROR: No raw CSV files found! Please check data/raw/ directory")
        return False
    
    # Step 3: Initialize preprocessor with production settings
    preprocessor = LSTMDataPreprocessor(
        test_ratio=0.2,
        buffer_days=7,
        random_state=42
    )
    
    # Step 4: Load and validate data
    print("Loading and validating raw barcode data...")
    try:
        raw_data = preprocessor.load_and_validate_data(existing_files)
        print(f"Successfully loaded {len(raw_data):,} barcode scan records")
        
        # Data summary
        print(f"Data Summary:")
        print(f"   Date range: {raw_data['event_time'].min()} to {raw_data['event_time'].max()}")
        print(f"   Unique EPCs: {raw_data['epc_code'].nunique():,}")
        print(f"   Unique locations: {raw_data['location_id'].nunique()}")
        print(f"   Business steps: {raw_data['business_step'].unique().tolist()}")
        
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        return False
    
    # Step 5: Apply EDA-guided feature engineering
    print("Applying EDA-guided feature engineering...")
    
    # Use EDA insights to guide feature engineering
    if 'correlations' in eda_insights:
        print("Using correlation insights from EDA for feature selection")
        high_corr_features = []
        corr_matrix = eda_insights['correlations']
        
        # Find highly correlated features (>0.8)
        for col in corr_matrix.columns:
            for idx in corr_matrix.index:
                if col != idx and abs(corr_matrix.loc[idx, col]) > 0.8:
                    high_corr_features.append(f"{col}-{idx}")
        
        if high_corr_features:
            print(f"   Found {len(high_corr_features)} highly correlated feature pairs from EDA")
    
    # Extract features using preprocessor
    raw_data = preprocessor.extract_temporal_features(raw_data)
    raw_data = preprocessor.extract_spatial_features(raw_data)
    raw_data = preprocessor.extract_behavioral_features(raw_data)
    
    # Step 6: Generate labels for training
    print("Generating anomaly labels...")
    raw_data = preprocessor.generate_labels_from_rules(raw_data)
    
    # Check label distribution
    anomaly_types = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
    print("Label distribution:")
    for anomaly_type in anomaly_types:
        if anomaly_type in raw_data.columns:
            count = raw_data[anomaly_type].sum()
            rate = count / len(raw_data) * 100
            print(f"   {anomaly_type}: {count:,} positive ({rate:.2f}%)")
    
    # Step 7: Apply critical fixes for feature redundancy
    print("Applying critical fixes for feature optimization...")
    
    # Use adaptive dimensionality reducer
    dim_reducer = AdaptiveDimensionalityReducer()
    feature_cols = [col for col in raw_data.columns 
                   if raw_data[col].dtype in ['float64', 'int64'] 
                   and col not in ['epc_code'] + anomaly_types]
    
    if len(feature_cols) > 10:
        X_features = raw_data[feature_cols].fillna(0)
        analysis_results = dim_reducer.analyze_feature_redundancy(X_features, feature_cols)
        
        print(f"Feature Analysis Results:")
        print(f"   Total features: {analysis_results['total_features']}")
        print(f"   High VIF features: {analysis_results['high_vif_features_count']}")
        print(f"   PCA recommended: {analysis_results['pca_recommended']}")
        print(f"   Decision: {analysis_results['decision_rationale']}")
    
    # Step 8: EPC-aware temporal split
    print("Performing EPC-aware temporal split...")
    train_data, test_data = preprocessor.epc_aware_temporal_split(raw_data)
    
    print(f"Training data: {len(train_data):,} records ({len(train_data['epc_code'].unique()):,} EPCs)")
    print(f"Testing data: {len(test_data):,} records ({len(test_data['epc_code'].unique()):,} EPCs)")
    
    # Verify no EPC overlap
    train_epcs = set(train_data['epc_code'].unique())
    test_epcs = set(test_data['epc_code'].unique())
    overlap = train_epcs.intersection(test_epcs)
    
    if overlap:
        print(f"ERROR: EPC overlap detected: {len(overlap)} EPCs in both sets!")
        return False
    else:
        print("SUCCESS: No EPC overlap - data split is valid")
    
    # Step 9: Generate sequences for LSTM
    print("Generating LSTM sequences...")
    from lstm_data_preprocessor import AdaptiveLSTMSequenceGenerator
    
    sequence_generator = AdaptiveLSTMSequenceGenerator(
        base_sequence_length=15,
        min_length=5,
        max_length=25
    )
    
    train_sequences, train_labels, train_metadata = sequence_generator.generate_sequences(train_data)
    test_sequences, test_labels, test_metadata = sequence_generator.generate_sequences(test_data)
    
    print(f"Training sequences: {len(train_sequences):,}")
    print(f"Testing sequences: {len(test_sequences):,}")
    
    if len(train_sequences) == 0:
        print("ERROR: No training sequences generated! Check data quality.")
        return False
    
    # Step 10: Save prepared data
    print("Saving prepared data for LSTM training...")
    
    # Create output directory
    output_dir = Path('lstm_academic_implementation')
    output_dir.mkdir(exist_ok=True)
    
    # Save DataFrames
    train_data.to_csv(output_dir / 'prepared_train_data.csv', index=False)
    test_data.to_csv(output_dir / 'prepared_test_data.csv', index=False)
    
    # Save sequences
    np.save(output_dir / 'train_sequences.npy', train_sequences)
    np.save(output_dir / 'train_labels.npy', train_labels)
    np.save(output_dir / 'test_sequences.npy', test_sequences)
    np.save(output_dir / 'test_labels.npy', test_labels)
    
    # Save metadata
    with open(output_dir / 'train_metadata.json', 'w') as f:
        json.dump(train_metadata, f, indent=2, default=str)
    
    # Save EDA-enhanced preprocessing report
    preprocessing_report = preprocessor.create_preprocessing_report()
    preprocessing_report['eda_insights_used'] = {
        'data_quality_analyzed': 'data_quality' in eda_insights,
        'correlations_analyzed': 'correlations' in eda_insights,
        'statistics_analyzed': 'statistics' in eda_insights
    }
    
    with open(output_dir / 'preprocessing_report_with_eda.json', 'w') as f:
        json.dump(preprocessing_report, f, indent=2, default=str)
    
    print("Data preparation complete!")
    print("Files created:")
    print("   prepared_train_data.csv")
    print("   prepared_test_data.csv")
    print("   train_sequences.npy") 
    print("   train_labels.npy")
    print("   test_sequences.npy")
    print("   test_labels.npy")
    print("   preprocessing_report_with_eda.json")
    
    return True

if __name__ == "__main__":
    success = prepare_lstm_data()
    if success:
        print("\nSUCCESS: Ready for Step 2: LSTM Training")
        print("Run: python lstm_academic_implementation/step2_train_lstm_model.py")
    else:
        print("\nERROR: Data preparation failed. Please check the errors above.")