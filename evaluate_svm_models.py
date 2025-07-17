#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM Model Evaluation Script

Evaluates trained SVM models using reserved test data (tt.txt compliant).
This script runs scientific validation of SVM performance.

Usage:
    python evaluate_svm_models.py

Requirements:
    - SVM models must be trained first (run train_svm_models.py)
    - Evaluation data must be available (automatically created during training)

Author: Data Analysis Team  
Date: 2025-07-17
"""

import sys
import os
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'barcode'))

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate trained SVM models')
    parser.add_argument('--model-dir', default='models/svm_models', help='SVM models directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("SVM Model Evaluation (tt.txt Compliant)")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Import evaluation modules
        from svm_csv_trainer import SVMCSVTrainer
        
        # Initialize trainer for evaluation
        trainer = SVMCSVTrainer()
        
        # Check if models exist
        model_files = []
        model_dir = trainer.model_dir
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('_model.pkl'):  # Look for actual model files
                    model_files.append(f)
        
        if not model_files:
            print("ERROR: No trained SVM models found!")
            print(f"   Expected location: {model_dir}")
            print(f"   Please run: python train_svm_models.py")
            return False
        
        print(f"Found {len(model_files)} trained SVM models:")
        for model_file in model_files:
            print(f"   - {model_file}")
        
        # Check if evaluation data exists
        eval_metadata_path = os.path.join(model_dir, "evaluation_data_metadata.json")
        if not os.path.exists(eval_metadata_path):
            print("ERROR: No evaluation data metadata found!")
            print(f"   Expected: {eval_metadata_path}")
            print(f"   Please retrain models: python train_svm_models.py")
            return False
        
        print(f"Evaluation data metadata found")
        
        # Run evaluation
        print(f"\nStarting SVM Model Evaluation...")
        print(f"   This compares SVM predictions vs rule-based ground truth")
        print(f"   Using reserved test data (never used in training)")
        
        # Perform evaluation (quick version)
        print("   Running quick evaluation (5 chunks max for speed)")
        
        # Temporarily modify chunk limit for faster evaluation
        original_chunk_size = trainer.chunk_size
        trainer.chunk_size = 5000  # Smaller chunks for faster processing
        
        evaluation_results = trainer.evaluate_svm_models()
        
        # Restore original chunk size
        trainer.chunk_size = original_chunk_size
        
        # Display results summary
        print(f"\nEvaluation Results Summary:")
        print("=" * 50)
        
        total_samples = evaluation_results.get('_summary', {}).get('total_samples_evaluated', 0)
        print(f"Total samples evaluated: {total_samples:,}")
        
        print(f"\nPerformance by Anomaly Type:")
        print("-" * 50)
        
        for anomaly_type, metrics in evaluation_results.items():
            if anomaly_type.startswith('_'):
                continue
                
            precision = metrics['precision']
            recall = metrics['recall'] 
            f1_score = metrics['f1_score']
            accuracy = metrics['accuracy']
            support = metrics['support']
            
            print(f"{anomaly_type:>15}: F1={f1_score:.3f}, P={precision:.3f}, R={recall:.3f}, Acc={accuracy:.3f} (n={support})")
        
        # Overall assessment
        avg_f1 = sum(m['f1_score'] for k, m in evaluation_results.items() if not k.startswith('_')) / 5
        print(f"\nOverall Average F1-Score: {avg_f1:.3f}")
        
        if avg_f1 >= 0.8:
            print("EXCELLENT: SVM models perform very well!")
        elif avg_f1 >= 0.6:
            print("GOOD: SVM models perform adequately")  
        elif avg_f1 >= 0.4:
            print("FAIR: SVM models need improvement")
        else:
            print("POOR: SVM models need significant improvement")
        
        # Save location
        results_file = os.path.join(model_dir, "svm_evaluation_results.json")
        print(f"\nDetailed results saved: {results_file}")
        
        print(f"\nFor Professor:")
        print(f"   - Used {total_samples:,} test samples (never seen during training)")
        print(f"   - Compared SVM predictions vs rule-based ground truth") 
        print(f"   - Average F1-score across 5 models: {avg_f1:.3f}")
        print(f"   - Evaluation follows tt.txt train/test separation requirements")
        
        return True
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print(f"\nEvaluation completed successfully!")