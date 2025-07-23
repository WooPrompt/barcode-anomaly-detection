#!/usr/bin/env python3
"""
Data Quality Investigation Tool
Purpose: Understand why no anomalies are detected in your datasets

Author: Data Analyst Team  
Date: 2025-07-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add source paths
sys.path.append('src')
from barcode.multi_anomaly_detector import (
    calculate_epc_fake_score, 
    calculate_duplicate_score,
    calculate_location_error_score,
    calculate_event_order_score,
    preprocess_scan_data
)

class DataQualityInvestigator:
    """
    Investigates why anomaly detection is finding 0% anomalies
    
    Key Questions:
    1. Are EPC codes too uniform/perfect?
    2. Are there any actual violations in the data?
    3. Are thresholds too high for this dataset?
    4. Is the data synthetic/artificially clean?
    """
    
    def __init__(self):
        self.investigation_results = {}
        
    def investigate_epc_patterns(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Investigate EPC code patterns and potential fake indicators"""
        
        print(f"\n🔍 Investigating EPC Patterns in {dataset_name}...")
        
        results = {
            'total_epcs': len(df),
            'unique_epcs': df['epc_code'].nunique(),
            'epc_samples': [],
            'format_analysis': {},
            'fake_scores_distribution': []
        }
        
        # Sample EPC codes for manual inspection
        sample_epcs = df['epc_code'].drop_duplicates().head(20).tolist()
        results['epc_samples'] = sample_epcs
        
        print(f"   📊 Total Records: {results['total_epcs']:,}")
        print(f"   📊 Unique EPCs: {results['unique_epcs']:,}")
        print(f"   📋 Sample EPC Codes:")
        for i, epc in enumerate(sample_epcs[:10]):
            fake_score = calculate_epc_fake_score(epc)
            results['fake_scores_distribution'].append(fake_score)
            print(f"      {i+1:2d}. {epc} (score: {fake_score})")
        
        # Analyze EPC format patterns
        epc_parts_analysis = {}
        for epc in sample_epcs:
            parts = epc.split('.')
            if len(parts) == 6:
                epc_parts_analysis['valid_structure_count'] = epc_parts_analysis.get('valid_structure_count', 0) + 1
                
                # Analyze each part
                header, company, product, lot, date, serial = parts
                epc_parts_analysis.setdefault('headers', set()).add(header)
                epc_parts_analysis.setdefault('companies', set()).add(company)
                epc_parts_analysis.setdefault('dates', set()).add(date)
        
        results['format_analysis'] = {
            'valid_structures': epc_parts_analysis.get('valid_structure_count', 0),
            'unique_headers': len(epc_parts_analysis.get('headers', set())),
            'unique_companies': len(epc_parts_analysis.get('companies', set())),
            'unique_dates': len(epc_parts_analysis.get('dates', set())),
            'headers': list(epc_parts_analysis.get('headers', set())),
            'companies': list(epc_parts_analysis.get('companies', set()))
        }
        
        print(f"   ✅ Valid EPC Structures: {results['format_analysis']['valid_structures']}/20")
        print(f"   🏢 Unique Companies: {results['format_analysis']['companies']}")
        print(f"   📅 Unique Dates: {results['format_analysis']['unique_dates']}")
        
        return results
    
    def investigate_temporal_patterns(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Investigate time-based patterns and potential violations"""
        
        print(f"\n⏰ Investigating Temporal Patterns in {dataset_name}...")
        
        df['event_time'] = pd.to_datetime(df['event_time'])
        
        results = {
            'date_range': {
                'start': df['event_time'].min(),
                'end': df['event_time'].max(),
                'span_days': (df['event_time'].max() - df['event_time'].min()).days
            },
            'time_gaps_analysis': {},
            'duplicate_timestamps': 0,
            'suspicious_patterns': []
        }
        
        print(f"   📅 Date Range: {results['date_range']['start']} to {results['date_range']['end']}")
        print(f"   📊 Span: {results['date_range']['span_days']} days")
        
        # Check for duplicate timestamps
        duplicate_times = df[df.duplicated(['epc_code', 'event_time'], keep=False)]
        results['duplicate_timestamps'] = len(duplicate_times)
        
        if results['duplicate_timestamps'] > 0:
            print(f"   ⚠️ Found {results['duplicate_timestamps']} records with duplicate EPC+timestamp")
            
            # Sample duplicate cases
            sample_duplicates = duplicate_times.groupby(['epc_code', 'event_time']).size().head(5)
            print("   📋 Sample Duplicate Cases:")
            for (epc, time), count in sample_duplicates.items():
                print(f"      EPC: {epc}, Time: {time}, Count: {count}")
        else:
            print("   ✅ No duplicate EPC+timestamp combinations found")
        
        # Analyze time gaps for space-time jumps
        epc_sample = df['epc_code'].value_counts().head(10).index
        time_gaps = []
        
        for epc in epc_sample:
            epc_data = df[df['epc_code'] == epc].sort_values('event_time')
            if len(epc_data) > 1:
                for i in range(1, len(epc_data)):
                    time_gap = (epc_data.iloc[i]['event_time'] - 
                              epc_data.iloc[i-1]['event_time']).total_seconds() / 3600
                    time_gaps.append(time_gap)
        
        if time_gaps:
            results['time_gaps_analysis'] = {
                'mean_gap_hours': np.mean(time_gaps),
                'min_gap_hours': np.min(time_gaps),
                'max_gap_hours': np.max(time_gaps),
                'suspicious_gaps': len([g for g in time_gaps if g < 0.1])  # < 6 minutes
            }
            
            print(f"   ⏱️ Average time gap: {results['time_gaps_analysis']['mean_gap_hours']:.2f} hours")
            print(f"   ⏱️ Suspicious short gaps: {results['time_gaps_analysis']['suspicious_gaps']}")
        
        return results
    
    def investigate_location_patterns(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Investigate location-based patterns and hierarchy violations"""
        
        print(f"\n🗺️ Investigating Location Patterns in {dataset_name}...")
        
        results = {
            'unique_locations': df['scan_location'].nunique(),
            'unique_business_steps': df['business_step'].nunique(),
            'location_samples': df['scan_location'].value_counts().head(10).to_dict(),
            'business_step_samples': df['business_step'].value_counts().to_dict(),
            'location_transitions': {}
        }
        
        print(f"   📍 Unique Locations: {results['unique_locations']}")
        print(f"   🏢 Unique Business Steps: {results['unique_business_steps']}")
        print(f"   📋 Top Locations:")
        for location, count in list(results['location_samples'].items())[:5]:
            print(f"      {location}: {count:,} records")
        
        print(f"   📋 Business Steps:")
        for step, count in results['business_step_samples'].items():
            print(f"      {step}: {count:,} records")
        
        # Check for location hierarchy violations
        epc_sample = df['epc_code'].value_counts().head(5).index
        hierarchy_violations = 0
        
        for epc in epc_sample:
            epc_data = df[df['epc_code'] == epc].sort_values('event_time')
            location_sequence = epc_data['scan_location'].tolist()
            
            loc_score = calculate_location_error_score(location_sequence)
            if loc_score > 0:
                hierarchy_violations += 1
                print(f"   ⚠️ EPC {epc}: Location violation score {loc_score}")
        
        results['hierarchy_violations_found'] = hierarchy_violations
        print(f"   📊 Hierarchy violations found: {hierarchy_violations}/5 sampled EPCs")
        
        return results
    
    def calculate_detailed_anomaly_scores(self, df: pd.DataFrame, dataset_name: str) -> dict:
        """Calculate detailed anomaly scores with distribution analysis"""
        
        print(f"\n🎯 Calculating Detailed Anomaly Scores for {dataset_name}...")
        
        df_processed = preprocess_scan_data(df.copy())
        
        anomaly_scores = {
            'epcFake': [],
            'epcDup': [],
            'locErr': [],
            'evtOrderErr': [],
            'jump': []
        }
        
        # Sample analysis (first 1000 EPCs for performance)
        epc_sample = df_processed['epc_code'].unique()[:1000]
        
        for epc_code in epc_sample:
            epc_group = df_processed[df_processed['epc_code'] == epc_code].sort_values('event_time')
            
            # Calculate all scores
            fake_score = calculate_epc_fake_score(epc_code)
            anomaly_scores['epcFake'].append(fake_score)
            
            # Duplicate score
            max_dup_score = 0
            for timestamp, time_group in epc_group.groupby('event_time'):
                dup_score = calculate_duplicate_score(epc_code, time_group)
                max_dup_score = max(max_dup_score, dup_score)
            anomaly_scores['epcDup'].append(max_dup_score)
            
            # Location and event scores
            location_sequence = epc_group['scan_location'].tolist()
            event_sequence = epc_group['event_type'].tolist()
            
            loc_score = calculate_location_error_score(location_sequence)
            event_score = calculate_event_order_score(event_sequence)
            
            anomaly_scores['locErr'].append(loc_score)
            anomaly_scores['evtOrderErr'].append(event_score)
            
            # Simple jump score
            if len(epc_group) > 1:
                time_diffs = []
                for i in range(1, len(epc_group)):
                    time_diff = (pd.to_datetime(epc_group.iloc[i]['event_time']) - 
                               pd.to_datetime(epc_group.iloc[i-1]['event_time'])).total_seconds() / 3600
                    time_diffs.append(max(0, time_diff))
                
                if time_diffs and len(time_diffs) > 1:
                    mean_diff = np.mean(time_diffs)
                    std_diff = np.std(time_diffs)
                    if std_diff > 0:
                        max_z = max([abs(t - mean_diff) / std_diff for t in time_diffs])
                        jump_score = min(100, max_z * 20) if max_z > 2 else 0
                    else:
                        jump_score = 0
                else:
                    jump_score = 0
            else:
                jump_score = 0
                
            anomaly_scores['jump'].append(jump_score)
        
        # Calculate statistics
        results = {}
        for anomaly_type, scores in anomaly_scores.items():
            scores_array = np.array(scores)
            results[anomaly_type] = {
                'count': len(scores),
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'median': float(np.median(scores)),
                'non_zero_count': int(np.sum(scores_array > 0)),
                'non_zero_rate': float(np.sum(scores_array > 0) / len(scores)),
                'percentiles': {
                    '90': float(np.percentile(scores, 90)),
                    '95': float(np.percentile(scores, 95)),
                    '99': float(np.percentile(scores, 99))
                }
            }
            
            print(f"   {anomaly_type}:")
            print(f"      Mean: {results[anomaly_type]['mean']:.2f}")
            print(f"      Max: {results[anomaly_type]['max']:.2f}")
            print(f"      Non-zero: {results[anomaly_type]['non_zero_count']}/{len(scores)} ({results[anomaly_type]['non_zero_rate']:.1%})")
        
        return results
    
    def generate_recommendations(self) -> dict:
        """Generate actionable recommendations based on investigation"""
        
        print(f"\n📋 Generating Data Quality Recommendations...")
        
        recommendations = {
            'data_characteristics': 'SYNTHETIC_OR_HIGHLY_CLEAN',
            'recommended_actions': [],
            'threshold_adjustments': {},
            'next_steps': []
        }
        
        # Analyze across all datasets
        all_fake_scores = []
        all_non_zero_rates = {}
        
        for dataset_name, results in self.investigation_results.items():
            if 'anomaly_scores' in results:
                for anomaly_type, stats in results['anomaly_scores'].items():
                    all_non_zero_rates.setdefault(anomaly_type, []).append(stats['non_zero_rate'])
                    if anomaly_type == 'epcFake':
                        all_fake_scores.extend([stats['mean'], stats['max']])
        
        # Generate specific recommendations
        avg_non_zero_rates = {k: np.mean(v) for k, v in all_non_zero_rates.items()}
        
        if all(rate < 0.01 for rate in avg_non_zero_rates.values()):
            recommendations['data_characteristics'] = 'EXTREMELY_CLEAN_OR_SYNTHETIC'
            recommendations['recommended_actions'] = [
                "1. VERIFY DATA AUTHENTICITY: Check if data is synthetic/simulated",
                "2. LOWER THRESHOLDS: Reduce business rule thresholds by 50-80%",
                "3. ADD NOISE: Inject controlled noise to create realistic anomaly rates",
                "4. USE DIFFERENT DATASETS: Test with production/real-world data"
            ]
            
            # Suggest threshold reductions
            recommendations['threshold_adjustments'] = {
                'epcFake': 'Reduce from 60 to 20-30',
                'epcDup': 'Reduce from 30 to 10-15',
                'locErr': 'Reduce from 40 to 15-25',
                'evtOrderErr': 'Reduce from 35 to 10-20',
                'jump': 'Reduce from 50 to 20-30'
            }
        
        recommendations['next_steps'] = [
            "1. Run with reduced thresholds to generate training data",
            "2. Manually review sample 'anomalies' to validate quality",
            "3. Consider using percentile-based thresholds (top 1-5%)",
            "4. Test LSTM training with available data even if clean"
        ]
        
        return recommendations
    
    def run_full_investigation(self, data_files: list) -> dict:
        """Run complete investigation on all datasets"""
        
        print("🔍 STARTING COMPREHENSIVE DATA QUALITY INVESTIGATION")
        print("=" * 60)
        
        for file_path in data_files:
            if not Path(file_path).exists():
                continue
                
            dataset_name = Path(file_path).stem
            print(f"\n📊 INVESTIGATING DATASET: {dataset_name.upper()}")
            print("-" * 40)
            
            # Load data
            df = pd.read_csv(file_path, sep='\t')
            sample_size = min(5000, len(df))  # Smaller sample for investigation
            df_sample = df.sample(n=sample_size, random_state=42)
            
            # Run all investigations
            dataset_results = {
                'epc_patterns': self.investigate_epc_patterns(df_sample, dataset_name),
                'temporal_patterns': self.investigate_temporal_patterns(df_sample, dataset_name),
                'location_patterns': self.investigate_location_patterns(df_sample, dataset_name),
                'anomaly_scores': self.calculate_detailed_anomaly_scores(df_sample, dataset_name)
            }
            
            self.investigation_results[dataset_name] = dataset_results
        
        # Generate final recommendations
        recommendations = self.generate_recommendations()
        
        print("\n" + "=" * 60)
        print("📋 INVESTIGATION COMPLETE - RECOMMENDATIONS")
        print("=" * 60)
        print(f"Data Characteristic: {recommendations['data_characteristics']}")
        print("\nRecommended Actions:")
        for action in recommendations['recommended_actions']:
            print(f"  {action}")
        
        print("\nSuggested Threshold Adjustments:")
        for anomaly_type, adjustment in recommendations['threshold_adjustments'].items():
            print(f"  {anomaly_type}: {adjustment}")
        
        return {
            'investigation_results': self.investigation_results,
            'recommendations': recommendations
        }

def main():
    """Run data quality investigation"""
    
    investigator = DataQualityInvestigator()
    
    data_files = [
        'data/raw/icn.csv',
        'data/raw/kum.csv', 
        'data/raw/ygs.csv',
        'data/raw/hws.csv'
    ]
    
    results = investigator.run_full_investigation(data_files)
    
    # Save results
    import json
    with open('data_quality_investigation_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Investigation complete!")
    print(f"📋 Report saved: data_quality_investigation_report.json")

if __name__ == "__main__":
    main()