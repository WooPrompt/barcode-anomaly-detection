#!/usr/bin/env python3
"""
SVM Preprocessing Pipeline CLI Entry Point
Provides command-line interface for running the complete SVM preprocessing pipeline
"""

import sys
import os
import argparse
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from barcode.svm_preprocessing.pipeline import SVMPreprocessingPipeline
from barcode.svm_preprocessing.config import load_config, get_default_config, get_production_config
from barcode.svm_preprocessing.batch_processor import BatchProcessor


def main():
    parser = argparse.ArgumentParser(
        description='SVM Preprocessing Pipeline for Barcode Anomaly Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with CSV input
  python run_svm_preprocessing.py --input data/raw_scans.csv --output data/svm_training

  # Use custom configuration
  python run_svm_preprocessing.py --input data/raw_scans.csv --config custom_config.json

  # Process specific anomaly types only
  python run_svm_preprocessing.py --input data/raw_scans.csv --anomaly-types epcFake epcDup

  # Enable batch processing for large datasets
  python run_svm_preprocessing.py --input data/large_dataset.csv --batch-size 5000

  # Optimize thresholds for target positive ratios
  python run_svm_preprocessing.py --input data/raw_scans.csv --optimize-thresholds

  # Production mode (optimized settings)
  python run_svm_preprocessing.py --input data/raw_scans.csv --production
        """
    )
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input CSV file with raw barcode scan data')
    
    # Optional arguments
    parser.add_argument('--output', '-o', default='data/svm_training',
                       help='Output directory for training data (default: data/svm_training)')
    
    parser.add_argument('--config', '-c',
                       help='Configuration file path (JSON)')
    
    parser.add_argument('--anomaly-types', nargs='+',
                       choices=['epcFake', 'epcDup', 'evtOrderErr', 'locErr', 'jump'],
                       help='Specific anomaly types to process (default: all)')
    
    parser.add_argument('--batch-size', type=int,
                       help='Batch size for memory-efficient processing')
    
    parser.add_argument('--optimize-thresholds', action='store_true',
                       help='Optimize thresholds for balanced class distribution')
    
    parser.add_argument('--production', action='store_true',
                       help='Use production-optimized configuration')
    
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save processed data to disk (for testing)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze data and generate recommendations, do not process')
    
    parser.add_argument('--memory-estimate', action='store_true',
                       help='Estimate memory requirements and exit')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
            print(f"✓ Loaded configuration from {args.config}")
        elif args.production:
            config = get_production_config()
            print("✓ Using production-optimized configuration")
        else:
            config = get_default_config()
            print("✓ Using default configuration")
        
        # Override config with command line arguments
        if args.output:
            config.output_dir = args.output
        
        if args.anomaly_types:
            config.anomaly_types = args.anomaly_types
        
        if args.batch_size:
            config.performance.enable_batch_processing = True
            config.performance.batch_size = args.batch_size
        
        # Load input data
        print(f"📂 Loading data from {args.input}...")
        
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file {args.input} not found")
            return 1
        
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df)} records from {args.input}")
        
        # Memory estimation
        if args.memory_estimate:
            print("🧠 Estimating memory requirements...")
            
            feature_dimensions = {
                'epcFake': config.features.epc_fake_dimensions,
                'epcDup': config.features.epc_dup_dimensions,
                'evtOrderErr': config.features.evt_order_dimensions,
                'locErr': config.features.loc_err_dimensions,
                'jump': config.features.jump_dimensions
            }
            
            # Estimate number of unique EPCs
            estimated_epcs = df['epc_code'].nunique() if 'epc_code' in df.columns else len(df) // 5
            
            batch_processor = BatchProcessor()
            memory_estimates = batch_processor.estimate_memory_requirements(
                estimated_epcs, feature_dimensions
            )
            
            print("💾 Memory Requirements Estimate:")
            for anomaly_type, estimate in memory_estimates.items():
                if anomaly_type != 'overall':
                    print(f"  {anomaly_type}: {estimate['total_memory_mb']:.1f} MB")
            
            overall = memory_estimates['overall']
            print(f"\n📊 Overall:")
            print(f"  Total memory needed: {overall['total_memory_all_types_mb']:.1f} MB")
            print(f"  System memory: {overall['system_memory_mb']:.1f} MB")
            print(f"  Available memory: {overall['available_memory_mb']:.1f} MB")
            print(f"  Recommended batch size: {overall['recommended_batch_size']}")
            
            return 0
        
        # Initialize pipeline
        pipeline = SVMPreprocessingPipeline(
            output_dir=config.output_dir,
            label_thresholds={
                'epcFake': config.labels.epc_fake_threshold,
                'epcDup': config.labels.epc_dup_threshold,
                'evtOrderErr': config.labels.evt_order_threshold,
                'locErr': config.labels.loc_err_threshold,
                'jump': config.labels.jump_threshold
            },
            enable_logging=True
        )
        
        print(f"🚀 Initialized SVM preprocessing pipeline")
        print(f"   Target anomaly types: {config.anomaly_types}")
        print(f"   Output directory: {config.output_dir}")
        
        # Optimize thresholds if requested
        if args.optimize_thresholds:
            print("🎯 Optimizing thresholds for balanced classes...")
            
            target_ratios = config.labels.target_positive_ratios
            optimized_thresholds = pipeline.optimize_thresholds(df, target_ratios)
            
            print("✓ Optimized thresholds:")
            for anomaly_type, threshold in optimized_thresholds.items():
                print(f"  {anomaly_type}: {threshold}")
            
            # Update pipeline thresholds
            pipeline.label_generator.update_thresholds(optimized_thresholds)
        
        # Process data
        if args.analyze_only:
            print("📊 Analyzing data (no processing)...")
            
            # Run minimal processing for analysis
            results = pipeline.process_data(
                df, 
                anomaly_types=config.anomaly_types,
                save_data=False,
                batch_size=config.performance.batch_size if config.performance.enable_batch_processing else None
            )
            
            # Generate analysis
            analysis = pipeline.analyze_preprocessing_results(results)
            
            print("\n📈 Data Analysis Results:")
            print(json.dumps(analysis, indent=2))
            
        else:
            print("⚙️ Processing data...")
            start_time = datetime.now()
            
            # Run full preprocessing pipeline
            results = pipeline.process_data(
                df,
                anomaly_types=config.anomaly_types,
                save_data=not args.no_save,
                batch_size=config.performance.batch_size if config.performance.enable_batch_processing else None
            )
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"✅ Processing completed in {processing_time:.1f} seconds")
            
            # Print summary
            if '_summary' in results:
                summary = results['_summary']
                print(f"\n📋 Processing Summary:")
                print(f"  Successful types: {len(summary['successful_types'])}")
                print(f"  Failed types: {len(summary['failed_types'])}")
                
                if summary['failed_types']:
                    print(f"  ❌ Failed: {', '.join(summary['failed_types'])}")
                
                if 'aggregate_statistics' in summary:
                    stats = summary['aggregate_statistics']
                    print(f"  Total samples: {stats['total_samples_across_types']}")
                    print(f"  Average positive ratio: {stats['avg_positive_ratio']:.3f}")
            
            # Print individual results
            for anomaly_type, result in results.items():
                if anomaly_type.startswith('_'):
                    continue
                
                if 'error' in result:
                    print(f"❌ {anomaly_type}: {result['error']}")
                else:
                    summary = result['summary']
                    print(f"✓ {anomaly_type}: {summary['total_samples']} samples, "
                          f"{summary['positive_ratio']:.3f} positive ratio")
            
            # Generate and display analysis
            if not args.no_save:
                analysis = pipeline.analyze_preprocessing_results(results)
                
                if analysis['recommendations']:
                    print(f"\n💡 Recommendations:")
                    for rec in analysis['recommendations']:
                        print(f"  • {rec}")
                
                # Save analysis to file
                analysis_path = os.path.join(config.output_dir, 'analysis_report.json')
                os.makedirs(config.output_dir, exist_ok=True)
                with open(analysis_path, 'w') as f:
                    json.dump(analysis, f, indent=2)
                print(f"\n📄 Analysis report saved to {analysis_path}")
        
        print("\n🎉 SVM preprocessing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Processing interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)