#!/usr/bin/env python3
"""
Threshold Sensitivity Analyzer
Purpose: Test different threshold levels to find optimal settings for your clean data

Author: Data Analyst Team
Date: 2025-07-22
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add source paths
sys.path.append('src')
from barcode.multi_anomaly_detector import calculate_epc_fake_score

class ThresholdSensitivityAnalyzer:
    """
    Tests multiple threshold levels to find the sweet spot for your data
    """
    
    def __init__(self):
        self.results = {}
    
    def test_threshold_ranges(self, df: pd.DataFrame, dataset_name: str):
        """Test different threshold ranges to find optimal detection rates"""
        
        print(f"🔧 Testing Threshold Sensitivity for {dataset_name}...")
        
        # Sample EPC codes for testing
        sample_epcs = df['epc_code'].drop_duplicates().head(1000).tolist()
        
        # Calculate fake scores for all samples
        fake_scores = [calculate_epc_fake_score(epc) for epc in sample_epcs]
        
        # Test different threshold ranges
        threshold_ranges = {
            'very_strict': np.arange(0, 20, 2),      # 0-18
            'strict': np.arange(0, 40, 4),           # 0-36  
            'moderate': np.arange(0, 60, 6),         # 0-54
            'lenient': np.arange(0, 80, 8),          # 0-72
            'very_lenient': np.arange(0, 100, 10)    # 0-90
        }
        
        threshold_results = {}
        
        for category, thresholds in threshold_ranges.items():
            category_results = []
            
            for threshold in thresholds:
                anomaly_count = sum(1 for score in fake_scores if score >= threshold)
                anomaly_rate = anomaly_count / len(fake_scores)
                
                category_results.append({
                    'threshold': float(threshold),
                    'anomaly_count': int(anomaly_count),
                    'anomaly_rate': float(anomaly_rate),
                    'detection_rate_percent': float(anomaly_rate * 100)
                })
            
            threshold_results[category] = category_results
            
            # Show best candidates for each category
            best_candidate = max(category_results, key=lambda x: x['anomaly_rate'] if x['anomaly_rate'] < 0.1 else 0)
            print(f"   {category}: Best threshold {best_candidate['threshold']} → {best_candidate['detection_rate_percent']:.2f}% detection")
        
        return {
            'dataset': dataset_name,
            'sample_size': len(sample_epcs),
            'score_distribution': {
                'mean': float(np.mean(fake_scores)),
                'std': float(np.std(fake_scores)),
                'min': float(np.min(fake_scores)),
                'max': float(np.max(fake_scores)),
                'median': float(np.median(fake_scores))
            },
            'threshold_analysis': threshold_results
        }
    
    def recommend_optimal_thresholds(self) -> dict:
        """Recommend optimal thresholds based on sensitivity analysis"""
        
        print("\n🎯 Calculating Optimal Threshold Recommendations...")
        
        # Target detection rates
        target_rates = {
            'epcFake': 0.005,      # 0.5%
            'epcDup': 0.025,       # 2.5%
            'locErr': 0.015,       # 1.5%
            'evtOrderErr': 0.020,  # 2.0%
            'jump': 0.008          # 0.8%
        }
        
        recommendations = {}
        
        for dataset_name, results in self.results.items():
            dataset_recommendations = {}
            
            # For epcFake (we have actual data)
            threshold_analysis = results['threshold_analysis']
            
            # Find threshold that gives closest to target rate for epcFake
            best_threshold = None
            best_diff = float('inf')
            
            for category, thresholds in threshold_analysis.items():
                for threshold_data in thresholds:
                    rate_diff = abs(threshold_data['anomaly_rate'] - target_rates['epcFake'])
                    if rate_diff < best_diff:
                        best_diff = rate_diff
                        best_threshold = threshold_data['threshold']
                        best_rate = threshold_data['anomaly_rate']
            
            dataset_recommendations['epcFake'] = {
                'recommended_threshold': best_threshold,
                'expected_rate': best_rate,
                'target_rate': target_rates['epcFake']
            }
            
            # For other anomalies, scale proportionally
            scaling_factor = best_threshold / 60.0 if best_threshold else 0.3  # Current business threshold
            
            dataset_recommendations['epcDup'] = {
                'recommended_threshold': round(30 * scaling_factor),
                'target_rate': target_rates['epcDup']
            }
            dataset_recommendations['locErr'] = {
                'recommended_threshold': round(40 * scaling_factor),
                'target_rate': target_rates['locErr']
            }
            dataset_recommendations['evtOrderErr'] = {
                'recommended_threshold': round(35 * scaling_factor),
                'target_rate': target_rates['evtOrderErr']
            }
            dataset_recommendations['jump'] = {
                'recommended_threshold': round(50 * scaling_factor),
                'target_rate': target_rates['jump']
            }
            
            recommendations[dataset_name] = dataset_recommendations
            
            print(f"\n   📊 {dataset_name} Recommendations:")
            for anomaly_type, rec in dataset_recommendations.items():
                print(f"      {anomaly_type}: {rec['recommended_threshold']} (target: {rec['target_rate']:.1%})")
        
        return recommendations

def main():
    """Run threshold sensitivity analysis"""
    
    print("🔧 THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 50)
    
    analyzer = ThresholdSensitivityAnalyzer()
    
    data_files = [
        'data/raw/icn.csv',
        'data/raw/kum.csv', 
        'data/raw/ygs.csv',
        'data/raw/hws.csv'
    ]
    
    for file_path in data_files:
        if not Path(file_path).exists():
            continue
            
        dataset_name = Path(file_path).stem
        print(f"\n📊 Analyzing {dataset_name}...")
        
        # Load data
        df = pd.read_csv(file_path, sep='\t')
        sample_df = df.sample(n=min(2000, len(df)), random_state=42)
        
        # Run sensitivity analysis
        results = analyzer.test_threshold_ranges(sample_df, dataset_name)
        analyzer.results[dataset_name] = results
    
    # Generate recommendations
    recommendations = analyzer.recommend_optimal_thresholds()
    
    # Save results
    import json
    output_data = {
        'analysis_results': analyzer.results,
        'threshold_recommendations': recommendations,
        'metadata': {
            'purpose': 'Find optimal thresholds for clean datasets',
            'target_detection_rates': '0.5-2.5% per anomaly type'
        }
    }
    
    with open('threshold_sensitivity_analysis.json', 'w') as f:
        json.dump(output_data, f, indent=2, default=str)
    
    print(f"\n✅ Sensitivity analysis complete!")
    print(f"📋 Results saved: threshold_sensitivity_analysis.json")
    
    print(f"\n🎯 SUMMARY RECOMMENDATIONS:")
    for dataset, recs in recommendations.items():
        print(f"   {dataset}:")
        for anomaly_type, rec in recs.items():
            print(f"      {anomaly_type}: {rec['recommended_threshold']}")

if __name__ == "__main__":
    main()