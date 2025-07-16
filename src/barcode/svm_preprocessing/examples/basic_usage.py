"""
Basic Usage Example - Start Here!

This is the SIMPLEST way to get started with SVM preprocessing.
Perfect for beginners who want to see the system working end-to-end
without getting lost in the details.

🎯 What this example does:
1. Creates sample barcode scan data
2. Runs the complete preprocessing pipeline
3. Shows you the training data that gets created
4. Explains what each step accomplishes

🚀 Run this first to see the magic happen!
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from barcode.svm_preprocessing.pipeline_05.pipeline_runner import SimpleRunner


def create_sample_data() -> pd.DataFrame:
    """
    Create realistic sample barcode scan data for demonstration.
    
    This simulates what real barcode scan data looks like in a supply chain.
    Each row represents one scan event with EPC code, time, location, etc.
    """
    print("📝 Creating sample barcode scan data...")
    
    sample_data = []
    
    # Create 10 different products (EPCs) with various journey patterns
    for product_id in range(1, 11):
        epc_code = f"001.8804823.{product_id:07d}.123456.20240101.{product_id:09d}"
        
        # Each product has 2-6 scan events in their supply chain journey
        num_events = np.random.randint(2, 7)
        
        for event_num in range(num_events):
            # Create realistic timestamps (events spread over days)
            base_time = pd.Timestamp('2024-01-01 09:00:00')
            time_offset = pd.Timedelta(days=event_num, hours=np.random.randint(0, 8))
            event_time = base_time + time_offset
            
            # Choose event type based on supply chain stage
            if event_num == 0:
                event_type = 'Aggregation'      # Production
                location = '화성공장'
            elif event_num < num_events - 1:
                event_type = 'WMS_Inbound'       # Logistics
                location = f'물류센터_{event_num}'
            else:
                event_type = 'POS_Retail'       # Retail
                location = '마트_A점'
            
            sample_data.append({
                'epc_code': epc_code,
                'event_time': event_time.strftime('%Y-%m-%d %H:%M:%S'),
                'event_type': event_type,
                'reader_location': location,
                'business_step': event_type.split('_')[0]
            })
    
    # Add some anomalous cases for demonstration
    
    # 1. Fake EPC (bad format)
    sample_data.append({
        'epc_code': 'INVALID.FORMAT.EPC',
        'event_time': '2024-01-05 10:00:00',
        'event_type': 'Aggregation',
        'reader_location': '화성공장',
        'business_step': 'Aggregation'
    })
    
    # 2. Duplicate scan (same EPC, same time, different location - impossible!)
    duplicate_epc = "001.8804823.9999999.123456.20240101.999999999"
    for location in ['서울물류센터', '부산물류센터']:  # Same time, different places
        sample_data.append({
            'epc_code': duplicate_epc,
            'event_time': '2024-01-06 14:30:00',  # SAME TIME
            'event_type': 'WMS_Inbound',
            'reader_location': location,
            'business_step': 'WMS'
        })
    
    # 3. Time jump (huge gap between events)
    jump_epc = "001.8804823.8888888.123456.20240101.888888888"
    sample_data.extend([
        {
            'epc_code': jump_epc,
            'event_time': '2024-01-01 09:00:00',
            'event_type': 'Aggregation',
            'reader_location': '화성공장',
            'business_step': 'Aggregation'
        },
        {
            'epc_code': jump_epc,
            'event_time': '2024-06-01 15:00:00',  # 5 MONTHS LATER!
            'event_type': 'POS_Retail',
            'reader_location': '마트_B점',
            'business_step': 'POS'
        }
    ])
    
    df = pd.DataFrame(sample_data)
    print(f"✅ Created {len(df)} scan events for {df['epc_code'].nunique()} unique products")
    
    return df


def run_basic_preprocessing(sample_df: pd.DataFrame) -> dict:
    """
    Run the complete SVM preprocessing pipeline on sample data.
    
    This demonstrates the full workflow from raw data to SVM-ready features.
    """
    print("\\n🚀 Starting SVM preprocessing pipeline...")
    print("=" * 50)
    
    # Initialize the simple runner (handles all complexity for you)
    runner = SimpleRunner()
    
    # Run preprocessing - this does EVERYTHING:
    # 1. Clean the raw data
    # 2. Group scans by EPC code
    # 3. Extract features for each anomaly type
    # 4. Generate labels using rule-based scoring
    # 5. Normalize features for SVM
    # 6. Handle class imbalance
    # 7. Split into train/test sets
    # 8. Save everything to disk
    
    results = runner.process_data(sample_df)
    
    print("\\n✅ Preprocessing completed successfully!")
    return results


def explore_results(results: dict):
    """
    Explore and understand the preprocessing results.
    
    This shows you what the system created and how to interpret it.
    """
    print("\\n🔍 Exploring Results...")
    print("=" * 50)
    
    print(f"\\n📊 Summary of processed data:")
    print(f"   {'Anomaly Type':<15} {'Samples':<8} {'Features':<10} {'Positives':<10} {'Ratio':<8}")
    print(f"   {'-'*15:<15} {'-'*8:<8} {'-'*10:<10} {'-'*10:<10} {'-'*8:<8}")
    
    total_samples = 0
    total_features = 0
    
    for anomaly_type, data in results.items():
        if anomaly_type.startswith('_'):  # Skip metadata
            continue
        
        features = data['features']
        labels = data['labels']
        
        samples = len(labels)
        feature_count = features.shape[1] if len(features) > 0 else 0
        positive_count = sum(labels)
        positive_ratio = positive_count / samples if samples > 0 else 0
        
        print(f"   {anomaly_type:<15} {samples:<8} {feature_count:<10} {positive_count:<10} {positive_ratio:<8.1%}")
        
        total_samples += samples
        total_features += feature_count
    
    print(f"   {'-'*15:<15} {'-'*8:<8} {'-'*10:<10} {'-'*10:<10} {'-'*8:<8}")
    print(f"   {'TOTAL':<15} {total_samples:<8} {total_features:<10}")
    
    # Show some actual features
    print(f"\\n🔬 Sample features (first anomaly type):")
    first_anomaly = list(results.keys())[0]
    first_data = results[first_anomaly]
    
    if len(first_data['features']) > 0:
        sample_features = first_data['features'][0]  # First sample
        feature_names = first_data.get('feature_names', [f'feature_{i}' for i in range(len(sample_features))])
        
        print(f"   Example from '{first_anomaly}':")
        for name, value in zip(feature_names[:5], sample_features[:5]):  # Show first 5
            print(f"      {name}: {value:.3f}")
        
        if len(sample_features) > 5:
            print(f"      ... and {len(sample_features)-5} more features")
    
    # Explain what happens next
    print(f"\\n🎯 What you can do next:")
    print(f"   1. Train SVM models using this preprocessed data")
    print(f"   2. Evaluate model performance on the test sets")
    print(f"   3. Use trained models to predict anomalies in new data")
    print(f"   4. Analyze which features are most important")
    
    # Show file locations
    print(f"\\n📁 Data saved to:")
    if '_summary' in results and 'data_directory' in results['_summary']:
        data_dir = results['_summary']['data_directory']
        print(f"   Directory: {data_dir}")
        print(f"   Files: *_X_train.npy, *_y_train.npy, *_X_test.npy, etc.")
    else:
        print(f"   Check the output directory specified in the pipeline")


def main():
    """
    Main function that runs the complete basic usage example.
    
    This is what gets executed when you run this file directly.
    """
    print("🎓 SVM Preprocessing - Basic Usage Example")
    print("=" * 60)
    print()
    print("This example will:")
    print("  1. Create realistic sample barcode scan data")
    print("  2. Run the complete SVM preprocessing pipeline")
    print("  3. Show you the results and explain what happened")
    print("  4. Guide you on next steps")
    print()
    input("Press Enter to start...")
    
    try:
        # Step 1: Create sample data
        sample_df = create_sample_data()
        
        # Show a preview of the data
        print(f"\\n👀 Preview of sample data:")
        print(sample_df.head(3).to_string(index=False))
        print(f"   ... and {len(sample_df)-3} more rows")
        
        # Step 2: Run preprocessing
        results = run_basic_preprocessing(sample_df)
        
        # Step 3: Explore results
        explore_results(results)
        
        print(f"\\n🎉 Basic usage example completed successfully!")
        print(f"\\n📚 Next steps for learning:")
        print(f"   • Try advanced_usage.py for more control")
        print(f"   • Explore individual components in ../01_core/")
        print(f"   • Read tutorials in ../tutorials/")
        print(f"   • Experiment with your own data")
        
    except Exception as e:
        print(f"\\n❌ Error occurred: {e}")
        print(f"\\n🔧 Troubleshooting tips:")
        print(f"   • Make sure you're running from the correct directory")
        print(f"   • Check that all required packages are installed")
        print(f"   • Try the test suite first: python -m pytest tests/")
        
        # Show traceback for debugging
        import traceback
        print(f"\\n🐛 Full error details:")
        traceback.print_exc()


if __name__ == "__main__":
    main()