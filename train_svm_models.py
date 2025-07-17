#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVM Model Training Script

Trains SVM models using large CSV datasets from data/raw/
This is the main script to run for initial training and periodic retraining.

Usage:
    python train_svm_models.py

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
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train SVM models from CSV data')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--chunk-size', type=int, default=15000, help='Chunk size for processing')
    parser.add_argument('--retrain', action='store_true', help='Force retrain existing models')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze CSV data without training')
    
    args = parser.parse_args()
    
    print("🚀 SVM Model Training Pipeline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Chunk size: {args.chunk_size:,}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # Import training modules
        from svm_csv_trainer import SVMCSVTrainer
        
        # Initialize trainer
        trainer = SVMCSVTrainer(data_dir=args.data_dir)
        trainer.chunk_size = args.chunk_size
        
        if args.analyze_only:
            # Only analyze CSV data
            print("📊 Analyzing CSV data...")
            analysis = trainer.analyze_csv_data()
            
            stats = analysis['file_stats']
            memory_est = analysis['memory_estimates']
            
            print("\n📈 Analysis Results:")
            print(f"  CSV files found: {stats['total_files']}")
            print(f"  Total estimated rows: {stats['estimated_total_rows']:,}")
            print(f"  Total file size: {sum(stats['file_sizes'].values()):.1f} MB")
            print(f"  Estimated memory usage: {memory_est['total_memory_mb']:.1f} MB")
            print(f"  Recommended chunk size: {memory_est['recommended_chunk_size']:,}")
            
            print("\n📁 File Details:")
            for file_info in stats['files']:
                print(f"  {file_info['name']}: {file_info['size_mb']:.1f} MB, ~{file_info['estimated_rows']:,} rows")
            
            return True
        
        # Check if models already exist
        model_dir = "models/svm_models"
        models_exist = os.path.exists(os.path.join(model_dir, "model_metadata.json"))
        
        if models_exist and not args.retrain:
            print("⚠️  Trained models already exist!")
            print("   Use --retrain flag to force retraining")
            print("   Use --analyze-only to analyze data without training")
            
            response = input("\nDo you want to retrain anyway? (y/N): ")
            if response.lower() != 'y':
                print("Training cancelled.")
                return False
        
        # Start training
        print("🤖 Starting SVM model training...")
        
        if args.retrain and models_exist:
            results = trainer.retrain_models()
        else:
            results = trainer.train_models_from_csv()
        
        # Display results
        print("\n" + "=" * 60)
        print("✅ Training Completed Successfully!")
        print("=" * 60)
        
        print(f"📊 Training Summary:")
        print(f"  Duration: {results['training_duration_minutes']:.1f} minutes")
        print(f"  CSV files used: {len(results['csv_files_used'])}")
        print(f"  Models trained: {len(results['training_results'])}")
        
        print(f"\n🎯 Model Performance:")
        for model_name, metrics in results['training_results'].items():
            print(f"  {model_name}:")
            print(f"    - Accuracy: {metrics['accuracy']:.3f}")
            print(f"    - Training samples: {metrics['normal_samples']:,}")
            print(f"    - Features: {metrics['feature_dimensions']}")
        
        if results.get('validation_results'):
            val_results = results['validation_results']
            print(f"\n🔍 Validation Results:")
            print(f"  Predictions: {val_results['successful_predictions']}/{val_results['total_predictions']}")
            print(f"  Avg prediction time: {val_results.get('average_prediction_time', 0):.3f}s")
            
            anomaly_rates = val_results.get('anomaly_detection_rates', {})
            if any(anomaly_rates.values()):
                print(f"  Anomalies detected:")
                for anomaly_type, count in anomaly_rates.items():
                    if count > 0:
                        print(f"    - {anomaly_type}: {count}")
        
        print(f"\n💾 Models saved to: {model_dir}")
        print("🎉 Ready for API usage!")
        
        # Show next steps
        print(f"\n📋 Next Steps:")
        print("1. Start FastAPI server:")
        print("   python fastapi_server.py")
        print("2. Test SVM endpoint:")
        print("   POST http://localhost:8000/api/v1/barcode-anomaly-detect/svm")
        print("3. Check API documentation:")
        print("   http://localhost:8000/docs")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install pandas numpy scikit-learn tqdm psutil")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("Make sure CSV files exist in data/raw/ directory")
        return False
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)