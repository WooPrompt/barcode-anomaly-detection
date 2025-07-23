#!/usr/bin/env python3
"""
Quick Threshold Fix - Immediate Solution
Purpose: Apply reduced thresholds to generate realistic anomaly rates for LSTM training

Author: Data Analyst Team
Date: 2025-07-22
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

# Add source paths  
sys.path.append('src')
sys.path.append('lstm_academic_implementation/src')

from lstm_data_preprocessor import LSTMDataPreprocessor

class QuickThresholdFixer:
    """
    Applies emergency threshold reduction to generate training data
    """
    
    def __init__(self):
        # Reduced thresholds (50-80% reduction from original business rules)
        self.emergency_thresholds = {
            'epcFake': 15,        # Reduced from 60
            'epcDup': 10,         # Reduced from 30  
            'locErr': 12,         # Reduced from 40
            'evtOrderErr': 8,     # Reduced from 35
            'jump': 15            # Reduced from 50
        }
        
        print("🚨 EMERGENCY THRESHOLD REDUCTION APPLIED")
        print("   Original → Emergency:")
        print("   epcFake: 60 → 15 (75% reduction)")
        print("   epcDup: 30 → 10 (67% reduction)")
        print("   locErr: 40 → 12 (70% reduction)")
        print("   evtOrderErr: 35 → 8 (77% reduction)")
        print("   jump: 50 → 15 (70% reduction)")
        
    def apply_emergency_fix_to_lstm_data_prep(self) -> bool:
        """
        Apply emergency thresholds to LSTM data preparation
        """
        
        print("\n🔧 Applying Emergency Fix to LSTM Data Preparation...")
        
        try:
            # Step 1: Load and prepare data with emergency thresholds
            data_files = [
                'data/raw/icn.csv',
                'data/raw/kum.csv', 
                'data/raw/ygs.csv',
                'data/raw/hws.csv'
            ]
            
            # Check file existence
            existing_files = [f for f in data_files if Path(f).exists()]
            if not existing_files:
                print("❌ No data files found!")
                return False
            
            # Initialize preprocessor
            preprocessor = LSTMDataPreprocessor(
                test_ratio=0.2,
                buffer_days=7,
                random_state=42
            )
            
            # Load data
            print("📊 Loading data...")
            raw_data = preprocessor.load_and_validate_data(existing_files)
            print(f"   Loaded {len(raw_data):,} records")
            
            # Apply emergency threshold-based labeling
            print("🏷️ Applying emergency threshold labeling...")
            labeled_data = self.apply_emergency_labeling(raw_data)
            
            # Check label distribution
            print("\n📊 Emergency Threshold Label Distribution:")
            anomaly_types = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
            total_anomalies = 0
            
            for anomaly_type in anomaly_types:
                if anomaly_type in labeled_data.columns:
                    count = labeled_data[anomaly_type].sum()
                    rate = count / len(labeled_data) * 100
                    total_anomalies += count
                    print(f"   {anomaly_type}: {count:,} ({rate:.2f}%)")
            
            print(f"   TOTAL ANOMALIES: {total_anomalies:,} ({total_anomalies/len(labeled_data)*100:.2f}%)")
            
            # Save emergency-labeled data
            output_dir = Path('lstm_academic_implementation/emergency_data')
            output_dir.mkdir(exist_ok=True)
            
            labeled_data.to_csv(output_dir / 'emergency_labeled_data.csv', index=False)
            
            # Save emergency configuration
            emergency_config = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'emergency_thresholds': self.emergency_thresholds,
                'label_distribution': {
                    anomaly_type: {
                        'count': int(labeled_data[anomaly_type].sum()),
                        'rate': float(labeled_data[anomaly_type].sum() / len(labeled_data))
                    }
                    for anomaly_type in anomaly_types if anomaly_type in labeled_data.columns
                },
                'total_records': len(labeled_data),
                'total_anomaly_rate': float(total_anomalies / len(labeled_data))
            }
            
            with open(output_dir / 'emergency_config.json', 'w') as f:
                json.dump(emergency_config, f, indent=2)
            
            print(f"\n✅ Emergency fix applied successfully!")
            print(f"   📁 Data saved: {output_dir}/")
            print(f"   📋 Config saved: emergency_config.json")
            
            return True
            
        except Exception as e:
            print(f"❌ Emergency fix failed: {e}")
            return False
    
    def apply_emergency_labeling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply emergency threshold-based labeling"""
        
        from barcode.multi_anomaly_detector import (
            calculate_epc_fake_score,
            calculate_duplicate_score, 
            calculate_location_error_score,
            calculate_event_order_score,
            preprocess_scan_data
        )
        
        print("   🔄 Processing anomaly detection with emergency thresholds...")
        
        df_processed = preprocess_scan_data(df.copy())
        
        # Initialize anomaly columns
        for anomaly_type in self.emergency_thresholds.keys():
            df_processed[anomaly_type] = 0
        
        # Process by EPC (sample for performance)
        unique_epcs = df_processed['epc_code'].unique()
        sample_size = min(5000, len(unique_epcs))  # Process subset for speed
        epc_sample = np.random.choice(unique_epcs, sample_size, replace=False)
        
        print(f"   📊 Processing {sample_size:,} EPCs...")
        
        processed_count = 0
        for epc_code in epc_sample:
            if processed_count % 1000 == 0 and processed_count > 0:
                print(f"      Processed {processed_count:,} EPCs...")
            
            epc_data = df_processed[df_processed['epc_code'] == epc_code]
            epc_indices = epc_data.index
            
            # 1. EPC Fake Detection
            fake_score = calculate_epc_fake_score(epc_code)
            if fake_score >= self.emergency_thresholds['epcFake']:
                df_processed.loc[epc_indices, 'epcFake'] = 1
            
            # 2. Other anomaly types (simplified for speed)
            if len(epc_data) > 1:
                epc_sorted = epc_data.sort_values('event_time')
                
                # Location error
                location_sequence = epc_sorted['scan_location'].tolist()
                loc_score = calculate_location_error_score(location_sequence)
                if loc_score >= self.emergency_thresholds['locErr']:
                    df_processed.loc[epc_indices, 'locErr'] = 1
                
                # Event order error
                event_sequence = epc_sorted['event_type'].tolist()
                event_score = calculate_event_order_score(event_sequence)
                if event_score >= self.emergency_thresholds['evtOrderErr']:
                    df_processed.loc[epc_indices, 'evtOrderErr'] = 1
                
                # Duplicate detection (simplified)
                duplicate_timestamps = epc_sorted['event_time'].duplicated().sum()
                if duplicate_timestamps > 0:
                    df_processed.loc[epc_indices, 'epcDup'] = 1
                
                # Jump detection (simplified)
                time_diffs = []
                for i in range(1, len(epc_sorted)):
                    try:
                        time_diff = (pd.to_datetime(epc_sorted.iloc[i]['event_time']) - 
                                   pd.to_datetime(epc_sorted.iloc[i-1]['event_time'])).total_seconds() / 3600
                        time_diffs.append(abs(time_diff))
                    except:
                        continue
                
                if time_diffs:
                    max_gap = max(time_diffs)
                    min_gap = min(time_diffs)
                    if min_gap < 0.1 or max_gap > 24 * 7:  # Very short or very long gaps
                        # Simple scoring
                        jump_score = min(50, max_gap / 24 * 10) if max_gap > 24 else 30 if min_gap < 0.1 else 0
                        if jump_score >= self.emergency_thresholds['jump']:
                            df_processed.loc[epc_indices, 'jump'] = 1
            
            processed_count += 1
        
        print(f"   ✅ Processed {processed_count:,} EPCs")
        
        return df_processed

def main():
    """Run quick threshold fix"""
    
    print("🚨 QUICK THRESHOLD FIX - EMERGENCY MODE")
    print("=" * 50)
    print("Purpose: Generate realistic anomaly rates for LSTM training")
    print("Method: Reduce thresholds by 67-77% from business rules")
    print("=" * 50)
    
    fixer = QuickThresholdFixer()
    success = fixer.apply_emergency_fix_to_lstm_data_prep()
    
    if success:
        print("\n🎉 EMERGENCY FIX SUCCESSFUL!")
        print("\n📋 Next Steps:")
        print("   1. Review generated anomaly rates")
        print("   2. Run LSTM training with emergency-labeled data")
        print("   3. Validate model performance")
        print("   4. Consider threshold refinement based on results")
        
        print("\n💡 Files Generated:")
        print("   📊 lstm_academic_implementation/emergency_data/emergency_labeled_data.csv")
        print("   ⚙️ lstm_academic_implementation/emergency_data/emergency_config.json")
    else:
        print("\n💥 Emergency fix failed!")

if __name__ == "__main__":
    main()