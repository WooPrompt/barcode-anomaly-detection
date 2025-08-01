import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from datetime import datetime
import numpy as np

# Add project path for imports
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

# Import rule-based detection system
try:
    from src.barcode.multi_anomaly_detector import detect_anomalies_backend_format
    print("Successfully imported rule-based detection system")
except ImportError as e:
    print(f"Failed to import rule-based detection: {e}")
    sys.exit(1)

def load_and_merge_data():
    """Load and merge all CSV files from data/raw/"""
    raw_data_dir = os.path.join(project_root, "data", "raw")
    
    df_list = []
    for filename in os.listdir(raw_data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(raw_data_dir, filename)
            df = pd.read_csv(file_path, sep='\t')
            df["source_file"] = filename
            df_list.append(df)
            print(f"Loaded {filename}: {len(df)} rows")
    
    # Merge all dataframes
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f" Total merged data: {len(merged_df)} rows from {len(df_list)} files")
    
    return merged_df

def analyze_anomaly_distribution(df):
    """Analyze anomaly distribution using rule-based detection"""
    print("\n ANOMALY DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Debug: Check data format and columns
    print(f" DataFrame columns: {list(df.columns)}")
    print(f" DataFrame shape: {df.shape}")
    print(f" Sample data:")
    print(df.head(2))
    
    # Add dummy eventId column (required by backend detection system)
    # Note: CSV files don't have eventId - it's generated by backend in real operation
    df_with_eventid = df.copy()
    df_with_eventid['eventId'] = range(1, len(df) + 1)
    
    # Add file_id column if missing (required by detection system)
    if 'file_id' not in df_with_eventid.columns:
        df_with_eventid['file_id'] = 1
        print(" Added missing file_id column (set to 1)")
    
    # Debug: Check required columns for detection
    required_cols = ['eventId', 'epc_code', 'location_id', 'business_step', 'event_type', 'event_time', 'file_id']
    missing_cols = [col for col in required_cols if col not in df_with_eventid.columns]
    if missing_cols:
        print(f" Missing required columns: {missing_cols}")
    else:
        print(" All required columns present")
    
    # Convert to format expected by detection system (JSON string)
    detection_input = {
        "data": df_with_eventid.to_dict('records')
    }
    detection_json = json.dumps(detection_input)
    
    print(f" Sample JSON input (first record):")
    print(json.dumps(detection_input["data"][0], indent=2))
    
    # Apply rule-based detection
    print(" Applying rule-based anomaly detection...")
    try:
        results_raw = detect_anomalies_backend_format(detection_json)
        
        # Parse JSON string if function returns string
        if isinstance(results_raw, str):
            results = json.loads(results_raw)
        else:
            results = results_raw
        
        # Extract anomaly statistics
        event_history = results.get('EventHistory', [])
        epc_stats = results.get('epcAnomalyStats', [])
        file_stats = results.get('fileAnomalyStats', {})
        
        print(f" Total events with anomalies: {len(event_history)}")
        print(f" Total EPC codes with anomalies: {len(epc_stats)}")
        
        # Anomaly type distribution
        anomaly_types = ['epcFake', 'epcDup', 'locErr', 'evtOrderErr', 'jump']
        anomaly_counts = {}
        
        for anomaly_type in anomaly_types:
            count = file_stats.get(f'{anomaly_type}Count', 0)
            anomaly_counts[anomaly_type] = count
            print(f"  {anomaly_type}: {count} occurrences")
        
        return results, anomaly_counts
        
    except Exception as e:
        print(f" Error in anomaly detection: {e}")
        return None, {}

def calculate_outlier_ratios(anomaly_counts, total_events):
    """Calculate outlier ratios for SVM nu parameter estimation"""
    print(f"\n OUTLIER RATIO ANALYSIS (for SVM nu parameter)")
    print("=" * 50)
    
    outlier_ratios = {}
    
    for anomaly_type, count in anomaly_counts.items():
        ratio = count / total_events if total_events > 0 else 0
        outlier_ratios[anomaly_type] = ratio
        
        # Recommend nu parameter (should be <= outlier ratio)
        recommended_nu = min(ratio * 0.8, 0.1) if ratio > 0 else 0.01
        
        print(f"{anomaly_type}:")
        print(f"  Events with anomaly: {count}")
        print(f"  Outlier ratio: {ratio:.4f} ({ratio*100:.2f}%)")
        print(f"  Recommended nu: {recommended_nu:.4f}")
        print()
    
    return outlier_ratios

def plot_anomaly_distribution(anomaly_counts, outlier_ratios):
    """Create visualizations for anomaly distribution"""
    plt.style.use('seaborn-v0_8')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Anomaly Count Bar Chart
    anomaly_types = list(anomaly_counts.keys())
    counts = list(anomaly_counts.values())
    
    bars1 = ax1.bar(anomaly_types, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    ax1.set_title('Anomaly Count Distribution', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Anomalies')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. Outlier Ratio Pie Chart
    ratios = [ratio for ratio in outlier_ratios.values() if ratio > 0]
    labels = [anomaly_type for anomaly_type, ratio in outlier_ratios.items() if ratio > 0]
    
    if ratios:
        ax2.pie(ratios, labels=labels, autopct='%1.2f%%', startangle=90)
        ax2.set_title('Outlier Ratio Distribution', fontsize=14, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No anomalies detected', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Outlier Ratio Distribution', fontsize=14, fontweight='bold')
    
    # 3. Recommended Nu Parameters
    nu_values = []
    for anomaly_type, ratio in outlier_ratios.items():
        recommended_nu = min(ratio * 0.8, 0.1) if ratio > 0 else 0.01
        nu_values.append(recommended_nu)
    
    bars3 = ax3.bar(anomaly_types, nu_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
    ax3.set_title('Recommended Nu Parameters for SVM', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Nu Value')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, max(nu_values) * 1.2 if nu_values else 0.1)
    
    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # 4. Outlier Ratio vs Nu Parameter Comparison
    x_pos = np.arange(len(anomaly_types))
    width = 0.35
    
    bars4a = ax4.bar(x_pos - width/2, list(outlier_ratios.values()), width, 
                     label='Outlier Ratio', color='lightblue', alpha=0.7)
    bars4b = ax4.bar(x_pos + width/2, nu_values, width,
                     label='Recommended Nu', color='orange', alpha=0.7)
    
    ax4.set_title('Outlier Ratio vs Recommended Nu Parameter', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Ratio / Nu Value')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(anomaly_types, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(project_root, "src", "barcode", "analysis", "anomaly_distribution_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f" Visualization saved to: {output_path}")
    plt.show()

def analyze_temporal_patterns(df, results):
    """Analyze temporal patterns in anomalies"""
    print(f"\n TEMPORAL PATTERN ANALYSIS")
    print("=" * 50)
    
    if not results or 'EventHistory' not in results:
        print(" No anomaly results available for temporal analysis")
        return
    
    # Convert event_time to datetime
    df['event_time'] = pd.to_datetime(df['event_time'])
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.day_name()
    df['date'] = df['event_time'].dt.date
    
    # Create temporal analysis plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Hourly distribution
    hourly_counts = df['hour'].value_counts().sort_index()
    ax1.plot(hourly_counts.index, hourly_counts.values, marker='o')
    ax1.set_title('Event Distribution by Hour of Day')
    ax1.set_xlabel('Hour')
    ax1.set_ylabel('Number of Events')
    ax1.grid(True, alpha=0.3)
    
    # Daily distribution
    daily_counts = df['day_of_week'].value_counts()
    ax2.bar(daily_counts.index, daily_counts.values)
    ax2.set_title('Event Distribution by Day of Week')
    ax2.set_ylabel('Number of Events')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save temporal analysis
    temporal_output_path = os.path.join(project_root, "src", "barcode", "analysis", "temporal_patterns.png")
    plt.savefig(temporal_output_path, dpi=300, bbox_inches='tight')
    print(f" Temporal analysis saved to: {temporal_output_path}")
    plt.show()

def analyze_geographic_patterns(df):
    """Analyze geographic distribution of events"""
    print(f"\n GEOGRAPHIC PATTERN ANALYSIS")
    print("=" * 50)
    
    # Location distribution
    location_counts = df['scan_location'].value_counts()
    print(" Events by Location:")
    for location, count in location_counts.head(10).items():
        print(f"  {location}: {count} events")
    
    # Factory distribution  
    factory_counts = df['source_file'].value_counts()
    print(f"\n Events by Factory:")
    for factory, count in factory_counts.items():
        print(f"  {factory}: {count} events")
    
    # Create geographic visualization
    plt.figure(figsize=(12, 8))
    
    # Factory distribution pie chart
    plt.subplot(2, 2, 1)
    plt.pie(factory_counts.values, labels=factory_counts.index, autopct='%1.1f%%')
    plt.title('Event Distribution by Factory')
    
    # Top 10 locations bar chart
    plt.subplot(2, 2, 2)
    top_locations = location_counts.head(10)
    plt.barh(range(len(top_locations)), top_locations.values)
    plt.yticks(range(len(top_locations)), top_locations.index)
    plt.title('Top 10 Locations by Event Count')
    plt.xlabel('Number of Events')
    
    plt.tight_layout()
    
    # Save geographic analysis
    geo_output_path = os.path.join(project_root, "src", "barcode", "analysis", "geographic_patterns.png")
    plt.savefig(geo_output_path, dpi=300, bbox_inches='tight')
    print(f" Geographic analysis saved to: {geo_output_path}")
    plt.show()

def main():
    """Main analysis function"""
    print(" BARCODE ANOMALY DETECTION - DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_and_merge_data()
    total_events = len(df)
    
    print(f"\n DATASET OVERVIEW")
    print("-" * 30)
    print(f"Total events: {total_events:,}")
    print(f"Date range: {df['event_time'].min()} to {df['event_time'].max()}")
    print(f"Unique EPC codes: {df['epc_code'].nunique():,}")
    print(f"Unique locations: {df['scan_location'].nunique()}")
    
    # Analyze anomalies
    results, anomaly_counts = analyze_anomaly_distribution(df)
    
    if anomaly_counts:
        # Calculate outlier ratios for nu parameter
        outlier_ratios = calculate_outlier_ratios(anomaly_counts, total_events)
        
        # Create visualizations
        plot_anomaly_distribution(anomaly_counts, outlier_ratios)
        
        # Temporal analysis
        analyze_temporal_patterns(df, results)
        
        # Geographic analysis
        analyze_geographic_patterns(df)
        
        # Summary for presentation
        print(f"\n SUMMARY FOR PRESENTATION")
        print("=" * 40)
        print(f"• Dataset: {total_events:,} events across {df['source_file'].nunique()} factories")
        print(f"• Anomaly rate: {sum(anomaly_counts.values())/total_events*100:.2f}%")
        print(f"• Most common anomaly: {max(anomaly_counts.keys(), key=anomaly_counts.get)}")
        print(f"• Recommended nu parameters range: {min([min(outlier_ratios.values())*0.8, 0.01]):.4f} - {max(outlier_ratios.values())*0.8:.4f}")
        
    else:
        print(" No anomalies detected in the dataset")

if __name__ == "__main__":
    main()