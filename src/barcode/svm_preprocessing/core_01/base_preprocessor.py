"""
Base Preprocessor - The Foundation of Data Processing

This is the FIRST component you should understand. It handles the basic tasks
that every machine learning project needs:

1. Clean messy data (remove nulls, fix formats)
2. Group related records together (EPC sequences)
3. Load reference data (geography, transitions)

🎓 Learning Goals:
- Understand how raw data becomes clean data
- Learn about EPC grouping and time sorting
- See how reference data is loaded and used

🔧 What it does:
- Takes raw CSV data with barcode scans
- Fixes data quality issues
- Groups scans by EPC code
- Sorts events by time within each EPC
- Loads supporting data (locations, transitions)
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

# Import the existing preprocessing function
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from multi_anomaly_detector import preprocess_scan_data, load_csv_data


class BasePreprocessor:
    """
    The foundation class for all data preprocessing.
    
    This class handles the basic data cleaning and grouping that every
    anomaly detection model needs. Think of it as the "janitor" that
    cleans up messy data before the "detectives" (feature extractors)
    can analyze it.
    
    Example:
        >>> preprocessor = BasePreprocessor()
        >>> clean_data = preprocessor.clean_data(raw_df)
        >>> epc_groups = preprocessor.group_by_epc(clean_data)
    """
    
    def __init__(self):
        """
        Initialize the preprocessor and load reference data.
        
        The reference data includes:
        - Geographic information (countries, regions)
        - Location transitions (how products move)
        - Location hierarchy (warehouse levels)
        """
        print(" Loading reference data...")
        
        # REUSE existing function from multi_anomaly_detector.py (lines 572-604)
        self.geo_df, self.transition_stats, self.location_mapping = load_csv_data()
        
        print(f" Loaded {len(self.geo_df)} geographic records")
        print(f" Loaded {len(self.transition_stats)} transition patterns")
        print(f" Loaded {len(self.location_mapping)} location mappings")
    
    def clean_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean raw barcode scan data.
        
        This is the first step in any data processing pipeline. Raw data
        from sensors and scanners is often messy and needs cleaning.
        
        Args:
            raw_df: Raw dataframe with columns like epc_code, event_time, etc.
            
        Returns:
            Clean dataframe ready for feature extraction
            
        Example:
            >>> raw_data = pd.read_csv("messy_scans.csv")
            >>> clean_data = preprocessor.clean_data(raw_data)
            >>> print(f"Cleaned {len(clean_data)} records")
        """
        print(f"🧹 Cleaning {len(raw_df)} raw records...")
        
        # REUSE existing function from multi_anomaly_detector.py (lines 364-390)
        clean_df = preprocess_scan_data(raw_df)
        
        print(f"✅ Cleaned data: {len(clean_df)} records remaining")
        return clean_df
    
    def group_by_epc(self, clean_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group scan events by EPC code and sort by time.
        
        Each EPC code represents one product's journey through the supply chain.
        We need to group all events for each EPC together and sort them by time
        to understand the sequence of what happened.
        
        Args:
            clean_df: Clean dataframe from clean_data()
            
        Returns:
            Dictionary mapping EPC code to its sorted event sequence
            
        Example:
            >>> epc_groups = preprocessor.group_by_epc(clean_data)
            >>> print(f"Found {len(epc_groups)} unique products")
            >>> 
            >>> # Look at one product's journey
            >>> sample_epc = list(epc_groups.keys())[0]
            >>> journey = epc_groups[sample_epc]
            >>> print(f"EPC {sample_epc} had {len(journey)} events")
        """
        print(f"📦 Grouping events by EPC code...")
        
        epc_groups = {}
        
        # Group by EPC code and sort each group by time
        for epc_code, group in clean_df.groupby('epc_code'):
            # Sort by event_time to get chronological order
            sorted_group = group.sort_values('event_time').reset_index(drop=True)
            epc_groups[epc_code] = sorted_group
        
        print(f"✅ Created {len(epc_groups)} EPC groups")
        
        # Show some statistics
        event_counts = [len(group) for group in epc_groups.values()]
        print(f"📊 Average events per EPC: {np.mean(event_counts):.1f}")
        print(f"📊 Max events per EPC: {max(event_counts)}")
        print(f"📊 Min events per EPC: {min(event_counts)}")
        
        return epc_groups
    
    def analyze_data_quality(self, epc_groups: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze the quality of grouped data.
        
        This helps you understand your data before training models.
        Good data quality leads to better model performance.
        
        Args:
            epc_groups: EPC groups from group_by_epc()
            
        Returns:
            Dictionary with quality metrics and recommendations
            
        Example:
            >>> quality = preprocessor.analyze_data_quality(epc_groups)
            >>> print(f"Data quality score: {quality['overall_score']}/10")
            >>> for rec in quality['recommendations']:
            ...     print(f"💡 {rec}")
        """
        print("🔍 Analyzing data quality...")
        
        # Basic statistics
        total_epcs = len(epc_groups)
        total_events = sum(len(group) for group in epc_groups.values())
        event_counts = [len(group) for group in epc_groups.values()]
        
        # Quality checks
        single_event_epcs = sum(1 for count in event_counts if count == 1)
        very_long_sequences = sum(1 for count in event_counts if count > 20)
        
        # Time span analysis
        time_spans = []
        for group in epc_groups.values():
            if len(group) > 1:
                group_sorted = group.sort_values('event_time')
                start_time = pd.to_datetime(group_sorted.iloc[0]['event_time'])
                end_time = pd.to_datetime(group_sorted.iloc[-1]['event_time'])
                span_hours = (end_time - start_time).total_seconds() / 3600
                time_spans.append(span_hours)
        
        # Location diversity
        all_locations = set()
        for group in epc_groups.values():
            all_locations.update(group['reader_location'].unique())
        
        # Generate quality score (0-10)
        quality_score = 10.0
        recommendations = []
        
        if single_event_epcs / total_epcs > 0.3:
            quality_score -= 2
            recommendations.append(f"High single-event EPCs ({single_event_epcs/total_epcs:.1%}). Consider collecting more scan data.")
        
        if very_long_sequences > 0:
            quality_score -= 1
            recommendations.append(f"{very_long_sequences} EPCs have >20 events. Check for duplicate scans.")
        
        if len(time_spans) > 0 and np.mean(time_spans) < 1:
            quality_score -= 1
            recommendations.append("Very short time spans detected. Events might be too close together.")
        
        if len(all_locations) < 5:
            quality_score -= 1
            recommendations.append(f"Only {len(all_locations)} unique locations. Supply chain might be too simple.")
        
        quality_report = {
            'overall_score': max(0, quality_score),
            'total_epcs': total_epcs,
            'total_events': total_events,
            'avg_events_per_epc': np.mean(event_counts),
            'single_event_epcs': single_event_epcs,
            'very_long_sequences': very_long_sequences,
            'unique_locations': len(all_locations),
            'avg_time_span_hours': np.mean(time_spans) if time_spans else 0,
            'recommendations': recommendations
        }
        
        print(f"✅ Data quality analysis complete")
        print(f"📊 Overall quality score: {quality_report['overall_score']:.1f}/10")
        
        return quality_report
    
    def get_sample_epc_journey(self, epc_groups: Dict[str, pd.DataFrame], 
                              epc_code: str = None) -> pd.DataFrame:
        """
        Get a sample EPC journey for learning and debugging.
        
        This is helpful when you want to understand what the data looks like
        or debug issues with specific products.
        
        Args:
            epc_groups: EPC groups from group_by_epc()
            epc_code: Specific EPC to examine (optional)
            
        Returns:
            DataFrame showing one product's complete journey
            
        Example:
            >>> journey = preprocessor.get_sample_epc_journey(epc_groups)
            >>> print("Sample product journey:")
            >>> for i, event in journey.iterrows():
            ...     print(f"{event['event_time']}: {event['event_type']} at {event['reader_location']}")
        """
        if epc_code and epc_code in epc_groups:
            sample_epc = epc_code
        else:
            # Pick an EPC with a reasonable number of events for demonstration
            suitable_epcs = [epc for epc, group in epc_groups.items() 
                           if 3 <= len(group) <= 10]
            if suitable_epcs:
                sample_epc = suitable_epcs[0]
            else:
                sample_epc = list(epc_groups.keys())[0]
        
        journey = epc_groups[sample_epc].copy()
        
        print(f"📦 Sample EPC Journey: {sample_epc}")
        print(f"🕒 Events: {len(journey)}")
        print(f"🌍 Locations: {journey['reader_location'].nunique()}")
        print(f"📅 Time span: {journey['event_time'].min()} to {journey['event_time'].max()}")
        
        return journey
    
    def get_reference_data(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """
        Get the loaded reference data for use by other components.
        
        Returns:
            Tuple of (geographic_data, transition_stats, location_mapping)
            
        Example:
            >>> geo_df, transitions, locations = preprocessor.get_reference_data()
            >>> print(f"Reference data available for {len(locations)} locations")
        """
        return self.geo_df, self.transition_stats, self.location_mapping
    
    def process_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Complete preprocessing workflow for a file.
        
        This is a convenience method that runs the entire preprocessing
        pipeline in one step. Perfect for beginners who want to get started quickly.
        
        Args:
            file_path: Path to CSV file with raw scan data
            
        Returns:
            Tuple of (clean_dataframe, epc_groups_dictionary)
            
        Example:
            >>> clean_data, epc_groups = preprocessor.process_file("raw_scans.csv")
            >>> print(f"Processed {len(epc_groups)} products")
        """
        print(f" Starting complete preprocessing for {file_path}")
        
        # Load raw data
        print(" Loading raw data...")
        raw_df = pd.read_csv(file_path)
        print(f" Loaded {len(raw_df)} raw records")
        
        # Clean data
        clean_df = self.clean_data(raw_df)
        
        # Group by EPC
        epc_groups = self.group_by_epc(clean_df)
        
        # Analyze quality
        quality = self.analyze_data_quality(epc_groups)
        
        print(f" Preprocessing complete!")
        print(f" Quality Score: {quality['overall_score']:.1f}/10")
        
        if quality['recommendations']:
            print(" Recommendations:")
            for rec in quality['recommendations']:
                print(f"  • {rec}")
        
        return clean_df, epc_groups


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of BasePreprocessor.
    
    This shows how to use the preprocessor step by step.
    Run this file directly to see it in action!
    """
    
    print("🎓 BasePreprocessor Example")
    print("=" * 50)
    
    # Create sample data for demonstration
    print(" Creating sample data...")
    sample_data = []
    for i in range(50):
        epc_code = f"001.8804823.1234567.123456.20240101.{i:09d}"
        for j in range(np.random.randint(2, 6)):  # 2-5 events per EPC
            sample_data.append({
                'epc_code': epc_code,
                'event_time': f"2024-01-{j+1:02d} {10+j:02d}:00:00",
                'event_type': np.random.choice(['production', 'logistics', 'retail']),
                'reader_location': f"Location_{j+1}"
            })
    
    sample_df = pd.DataFrame(sample_data)
    
    # Initialize preprocessor
    print("\n Initializing preprocessor...")
    preprocessor = BasePreprocessor()
    
    # Clean data
    print("\n Cleaning data...")
    clean_df = preprocessor.clean_data(sample_df)
    
    # Group by EPC
    print("\n Grouping by EPC...")
    epc_groups = preprocessor.group_by_epc(clean_df)
    
    # Analyze quality
    print("\n Analyzing quality...")
    quality = preprocessor.analyze_data_quality(epc_groups)
    
    # Show sample journey
    print("\n Sample journey...")
    journey = preprocessor.get_sample_epc_journey(epc_groups)
    print(journey[['event_time', 'event_type', 'reader_location']])
    
    print("\n Example complete!")
    print("Next step: Learn about SequenceProcessor in sequence_processor.py")