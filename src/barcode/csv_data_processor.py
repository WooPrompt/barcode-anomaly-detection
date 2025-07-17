#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Data Processor for SVM Training

Handles large CSV files from data/raw/ for SVM model training.
Supports tab-separated format with memory-efficient chunk processing.

Author: Data Analysis Team
Date: 2025-07-17
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Iterator, Any
from datetime import datetime
import logging
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, desc="Processing", **kwargs):
        print(f"{desc}...")
        return iterable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVDataProcessor:
    """Process large CSV files for SVM training with memory efficiency"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.training_dir = os.path.join(data_dir, "training_data")
        
        # Create directories
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(os.path.join(self.training_dir, "features"), exist_ok=True)
        os.makedirs(os.path.join(self.training_dir, "labels"), exist_ok=True)
        
        # CSV schema mapping
        self.csv_columns = [
            'scan_location', 'location_id', 'hub_type', 'business_step', 
            'event_type', 'operator_id', 'device_id', 'epc_code', 
            'epc_header', 'epc_company', 'epc_product', 'epc_lot', 
            'epc_manufacture', 'epc_serial', 'product_name', 'event_time', 
            'manufacture_date', 'expiry_date'
        ]
        
        # Required columns for SVM training
        self.required_columns = [
            'location_id', 'business_step', 'event_type', 'epc_code', 
            'event_time', 'scan_location'
        ]
    
    def get_csv_files(self) -> List[str]:
        """Get all CSV files from raw directory"""
        csv_files = []
        for file in os.listdir(self.raw_dir):
            if file.endswith('.csv'):
                csv_files.append(os.path.join(self.raw_dir, file))
        
        logger.info(f"Found {len(csv_files)} CSV files: {[os.path.basename(f) for f in csv_files]}")
        return csv_files
    
    def read_csv_chunk(self, file_path: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
        """Read CSV file in chunks to handle large files"""
        logger.info(f"Reading {os.path.basename(file_path)} in chunks of {chunk_size}")
        
        try:
            chunk_iter = pd.read_csv(
                file_path,
                sep='\t',  # Tab-separated as specified
                names=self.csv_columns,
                header=0,  # Assume first row is header
                chunksize=chunk_size,
                dtype={
                    'location_id': 'Int64',
                    'epc_code': 'str',
                    'business_step': 'str',
                    'event_type': 'str',
                    'scan_location': 'str'
                },
                parse_dates=['event_time', 'manufacture_date', 'expiry_date'],
                low_memory=False
            )
            
            for chunk_idx, chunk in enumerate(chunk_iter):
                # Basic data cleaning
                chunk = self._clean_chunk(chunk)
                
                if not chunk.empty:
                    logger.debug(f"Processed chunk {chunk_idx + 1}, shape: {chunk.shape}")
                    yield chunk
                    
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            raise
    
    def _clean_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate chunk data"""
        # Remove rows with missing required columns
        initial_size = len(chunk)
        chunk = chunk.dropna(subset=self.required_columns)
        
        # Convert data types
        chunk['location_id'] = chunk['location_id'].astype(str)
        chunk['epc_code'] = chunk['epc_code'].astype(str)
        
        # Filter out invalid EPC codes (basic validation)
        chunk = chunk[chunk['epc_code'].str.len() > 10]
        
        # Sort by event_time for sequence analysis
        chunk = chunk.sort_values(['epc_code', 'event_time']).reset_index(drop=True)
        
        cleaned_size = len(chunk)
        if cleaned_size < initial_size:
            logger.debug(f"Cleaned chunk: {initial_size} -> {cleaned_size} rows")
        
        return chunk
    
    def chunk_to_json_format(self, chunk: pd.DataFrame, file_id: int) -> Dict[str, Any]:
        """Convert CSV chunk to JSON format expected by rule-based detector"""
        events = []
        
        for idx, row in chunk.iterrows():
            event = {
                "eventId": int(idx + file_id * 100000),  # Unique event ID
                "epc_code": str(row['epc_code']),
                "location_id": int(row['location_id']) if pd.notna(row['location_id']) else 0,
                "business_step": str(row['business_step']),
                "event_type": str(row['event_type']),
                "event_time": row['event_time'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['event_time']) else "",
                "file_id": file_id
            }
            events.append(event)
        
        return {"data": events}
    
    def split_train_validation(self, csv_files: List[str], 
                             train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
        """Split CSV files into training and validation sets"""
        np.random.seed(42)  # For reproducible splits
        
        total_files = len(csv_files)
        train_count = int(total_files * train_ratio)
        
        # Shuffle files
        shuffled_files = csv_files.copy()
        np.random.shuffle(shuffled_files)
        
        train_files = shuffled_files[:train_count]
        val_files = shuffled_files[train_count:]
        
        logger.info(f"Split {total_files} files: {len(train_files)} train, {len(val_files)} validation")
        return train_files, val_files
    
    def get_file_stats(self, csv_files: List[str]) -> Dict[str, Any]:
        """Get statistics about CSV files"""
        stats = {
            'total_files': len(csv_files),
            'file_sizes': {},
            'estimated_total_rows': 0,
            'files': []
        }
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            # Estimate rows (rough estimate: 100 bytes per row)
            estimated_rows = int(os.path.getsize(file_path) / 100)
            
            stats['file_sizes'][file_name] = file_size
            stats['estimated_total_rows'] += estimated_rows
            stats['files'].append({
                'name': file_name,
                'size_mb': round(file_size, 2),
                'estimated_rows': estimated_rows
            })
        
        return stats
    
    def process_csv_for_training(self, csv_files: List[str], 
                               chunk_size: int = 10000) -> Iterator[Tuple[Dict, str]]:
        """Process CSV files and yield training data chunks"""
        logger.info(f"Processing {len(csv_files)} CSV files for training")
        
        for file_idx, file_path in enumerate(csv_files):
            file_name = os.path.basename(file_path)
            logger.info(f"Processing file {file_idx + 1}/{len(csv_files)}: {file_name}")
            
            try:
                chunk_count = 0
                for chunk in self.read_csv_chunk(file_path, chunk_size):
                    if not chunk.empty:
                        # Convert to JSON format
                        json_data = self.chunk_to_json_format(chunk, file_idx)
                        json_str = json.dumps(json_data)
                        
                        chunk_count += 1
                        yield json_data, f"{file_name}_chunk_{chunk_count}"
                
                logger.info(f"Completed {file_name}: {chunk_count} chunks processed")
                
            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}")
                continue
    
    def save_training_batch(self, training_data: List[Dict], 
                          batch_name: str) -> str:
        """Save training batch to disk"""
        batch_path = os.path.join(self.training_dir, "features", f"{batch_name}.json")
        
        with open(batch_path, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved training batch: {batch_path}")
        return batch_path
    
    def estimate_memory_usage(self, csv_files: List[str], 
                            chunk_size: int = 10000) -> Dict[str, float]:
        """Estimate memory usage for processing"""
        stats = self.get_file_stats(csv_files)
        
        # Rough estimates
        avg_row_size_bytes = 500  # Estimated bytes per row
        chunk_memory_mb = (chunk_size * avg_row_size_bytes) / (1024 * 1024)
        
        # Feature extraction memory (rough estimate)
        feature_memory_mb = chunk_memory_mb * 2  # Features + original data
        
        # Total memory needed
        total_memory_mb = feature_memory_mb + 500  # Base overhead
        
        return {
            'chunk_size': chunk_size,
            'chunk_memory_mb': round(chunk_memory_mb, 2),
            'feature_memory_mb': round(feature_memory_mb, 2),
            'total_memory_mb': round(total_memory_mb, 2),
            'recommended_chunk_size': self._recommend_chunk_size(),
            'total_estimated_rows': stats['estimated_total_rows']
        }
    
    def _recommend_chunk_size(self) -> int:
        """Recommend optimal chunk size based on available memory"""
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            if available_memory_gb > 16:
                return 50000  # Large chunks for high memory
            elif available_memory_gb > 8:
                return 25000  # Medium chunks
            else:
                return 10000  # Small chunks for limited memory
        except:
            return 10000  # Default safe size


# Utility functions
def process_all_csv_files(data_dir: str = "data", 
                         chunk_size: int = 10000) -> Dict[str, Any]:
    """Process all CSV files and return summary"""
    processor = CSVDataProcessor(data_dir)
    
    # Get CSV files
    csv_files = processor.get_csv_files()
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {processor.raw_dir}")
    
    # Get statistics
    stats = processor.get_file_stats(csv_files)
    memory_est = processor.estimate_memory_usage(csv_files, chunk_size)
    
    # Split train/validation
    train_files, val_files = processor.split_train_validation(csv_files)
    
    return {
        'csv_files': csv_files,
        'train_files': train_files,
        'validation_files': val_files,
        'file_stats': stats,
        'memory_estimates': memory_est,
        'processor': processor
    }


if __name__ == "__main__":
    # Example usage
    processor = CSVDataProcessor()
    
    # Get file information
    csv_files = processor.get_csv_files()
    stats = processor.get_file_stats(csv_files)
    memory_est = processor.estimate_memory_usage(csv_files)
    
    print("CSV Processing Information:")
    print(f"Files found: {stats['total_files']}")
    print(f"Estimated total rows: {stats['estimated_total_rows']:,}")
    print(f"Recommended chunk size: {memory_est['recommended_chunk_size']:,}")
    print(f"Estimated memory usage: {memory_est['total_memory_mb']:.1f} MB")