"""
Sequence Processor - Smart Handling of Variable-Length Sequences

This is the SECOND core component to understand. Machine learning models
(especially SVM) need fixed-length input vectors, but real-world data
has variable-length sequences. This component solves that problem intelligently.

🎓 Learning Goals:
- Understand why sequence length matters for SVM
- Learn different strategies for handling variable lengths
- See how to preserve temporal information while standardizing length

🔧 What it does:
- Takes variable-length sequences (1-20 events per EPC)
- Converts them to fixed-length vectors (exactly 10 elements)
- Uses smart padding/truncation to preserve important information
- Avoids meaningless zero-padding that hurts model performance

🧠 Key Insight:
The user feedback emphasized: "Don't cut sequences mindlessly!"
This processor implements statistical analysis to make smart decisions.
"""

import numpy as np
import statistics
from typing import List, Dict, Any, Union


class SequenceProcessor:
    """
    Intelligently process variable-length sequences for SVM training.
    
    Real-world problem: Product journeys have different lengths
    - Some products: 2 events (production → retail)
    - Other products: 15 events (complex supply chain)
    - SVM needs: exactly the same length for all products
    
    Our solution: Smart processing that preserves important patterns
    while standardizing length.
    
    Example:
        >>> processor = SequenceProcessor()
        >>> short_seq = [1.0, 2.0, 3.0]  # Too short
        >>> fixed_seq = processor.process_sequence(short_seq, target_length=5)
        >>> print(fixed_seq)  # [1.0, 2.0, 3.0, 3.0, 3.0] (smart padding)
    """
    
    def __init__(self):
        """
        Initialize the sequence processor with smart defaults.
        
        Key parameters based on user feedback:
        - truncation_threshold: Don't cut more than 3% of data
        - max_sequence_length: Supply chain max is typically 10 steps
        """
        # From txt.txt feedback: "3% 이상 자르면 안자르고 보간함"
        self.truncation_threshold = 0.03  # Don't truncate more than 3%
        self.max_sequence_length = 10     # Typical supply chain maximum
        
        print("🔧 SequenceProcessor initialized")
        print(f"   📏 Max truncation: {self.truncation_threshold*100}%")
        print(f"   📐 Target length: {self.max_sequence_length}")
    
    def process_sequence(self, sequence: List[float], target_length: int, 
                        sequence_type: str = 'general') -> List[float]:
        """
        Convert any sequence to exactly target_length elements.
        
        This is the main function that implements intelligent sequence processing
        based on statistical analysis and user feedback.
        
        Args:
            sequence: Original sequence (any length)
            target_length: Desired length (usually 10)
            sequence_type: Type hint for processing strategy
                         'temporal' = time-based, preserve continuity
                         'binary' = 0/1 values, use majority value
                         'statistical' = use mean for padding
                         'general' = use median (safest default)
        
        Returns:
            Fixed-length sequence with preserved patterns
            
        Example:
            >>> # Too short - needs padding
            >>> short = [1.0, 2.0, 3.0]
            >>> padded = processor.process_sequence(short, 5, 'temporal')
            >>> print(padded)  # [1.0, 2.0, 3.0, 3.0, 3.0]
            >>>
            >>> # Too long - needs smart truncation/interpolation
            >>> long = list(range(20))  # [0, 1, 2, ..., 19]
            >>> compressed = processor.process_sequence(long, 10)
            >>> print(len(compressed))  # 10
        """
        if not sequence:
            print("⚠️ Empty sequence provided, returning zeros")
            return [0.0] * target_length
        
        if len(sequence) == target_length:
            return sequence.copy()  # Perfect length, no changes needed
        
        print(f"🔄 Processing sequence: {len(sequence)} → {target_length}")
        
        # Case 1: Sequence too short - needs padding
        if len(sequence) < target_length:
            return self._intelligent_padding(sequence, target_length, sequence_type)
        
        # Case 2: Sequence too long - needs truncation or interpolation
        else:
            return self._intelligent_truncation(sequence, target_length)
    
    def _intelligent_padding(self, sequence: List[float], target_length: int, 
                           sequence_type: str) -> List[float]:
        """
        Add elements to short sequences using context-aware strategies.
        
        From user feedback: "무조건 0 패딩 금지" (No mindless zero padding!)
        Each sequence type gets a different padding strategy that makes sense.
        
        Args:
            sequence: Short sequence needing padding
            target_length: Target length
            sequence_type: Type of data for appropriate padding
            
        Returns:
            Padded sequence with meaningful values
        """
        pad_count = target_length - len(sequence)
        print(f"   📏 Padding {pad_count} elements using '{sequence_type}' strategy")
        
        if sequence_type == 'binary':
            # For 0/1 sequences, use the majority value
            ones = sequence.count(1.0)
            zeros = sequence.count(0.0)
            pad_value = 1.0 if ones > zeros else 0.0
            print(f"   📊 Binary padding: using {pad_value} (majority value)")
            
        elif sequence_type == 'temporal':
            # For time-based sequences, continue with last value
            pad_value = sequence[-1]
            print(f"   🕒 Temporal padding: using {pad_value} (last value)")
            
        elif sequence_type == 'statistical':
            # For general numeric sequences, use mean
            pad_value = statistics.mean(sequence)
            print(f"   📈 Statistical padding: using {pad_value:.3f} (mean)")
            
        else:  # 'general' or unknown
            # Safest default: use median (robust to outliers)
            pad_value = statistics.median(sequence)
            print(f"   📊 General padding: using {pad_value:.3f} (median)")
        
        result = sequence.copy() + [pad_value] * pad_count
        return result
    
    def _intelligent_truncation(self, sequence: List[float], target_length: int) -> List[float]:
        """
        Reduce long sequences using statistical decision-making.
        
        From user feedback: "3퍼 넘으면 안자르고 보간함" 
        (If cutting >3%, use interpolation instead)
        
        Args:
            sequence: Long sequence needing reduction
            target_length: Target length
            
        Returns:
            Reduced sequence preserving important patterns
        """
        truncation_ratio = (len(sequence) - target_length) / len(sequence)
        
        print(f"   ✂️ Truncation needed: {truncation_ratio:.1%}")
        
        if truncation_ratio > self.truncation_threshold:
            # Too much data would be lost - use interpolation
            print(f"   🔄 Using interpolation (>{self.truncation_threshold*100}% threshold)")
            return self._interpolate_sequence(sequence, target_length)
        else:
            # Small truncation is OK - just take first N elements
            print(f"   ✂️ Using simple truncation (<{self.truncation_threshold*100}% threshold)")
            return sequence[:target_length]
    
    def _interpolate_sequence(self, sequence: List[float], target_length: int) -> List[float]:
        """
        Use linear interpolation to compress long sequences.
        
        This preserves the overall shape and trends of the sequence
        while reducing its length. Much better than truncation for
        preserving temporal patterns.
        
        Args:
            sequence: Long sequence
            target_length: Desired length
            
        Returns:
            Interpolated sequence with preserved trends
        """
        if target_length >= len(sequence):
            return sequence.copy()
        
        print(f"   🎯 Interpolating from {len(sequence)} to {target_length} points")
        
        # Create evenly spaced indices for sampling
        original_indices = np.linspace(0, len(sequence) - 1, target_length)
        interpolated = []
        
        for target_idx in original_indices:
            # Find the two closest original points
            lower_idx = int(np.floor(target_idx))
            upper_idx = min(lower_idx + 1, len(sequence) - 1)
            
            if lower_idx == upper_idx:
                # Exact match
                interpolated.append(sequence[lower_idx])
            else:
                # Linear interpolation between two points
                weight = target_idx - lower_idx
                value = sequence[lower_idx] * (1 - weight) + sequence[upper_idx] * weight
                interpolated.append(value)
        
        return interpolated
    
    def analyze_sequence_lengths(self, sequences: List[List[float]]) -> Dict[str, Any]:
        """
        Analyze a collection of sequences to understand length distribution.
        
        This helps you make informed decisions about target lengths and
        processing strategies for your specific dataset.
        
        Args:
            sequences: List of variable-length sequences
            
        Returns:
            Statistical analysis of sequence lengths
            
        Example:
            >>> sequences = [[1, 2], [1, 2, 3, 4], [1]]  # Different lengths
            >>> stats = processor.analyze_sequence_lengths(sequences)
            >>> print(f"Average length: {stats['mean_length']:.1f}")
            >>> print(f"Recommended target: {stats['recommended_target']}")
        """
        if not sequences:
            return {'error': 'No sequences provided'}
        
        lengths = [len(seq) for seq in sequences]
        
        analysis = {
            'total_sequences': len(sequences),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': statistics.mean(lengths),
            'median_length': statistics.median(lengths),
            'std_length': statistics.stdev(lengths) if len(lengths) > 1 else 0,
        }
        
        # Calculate percentiles for better understanding
        analysis['percentiles'] = {
            '25th': np.percentile(lengths, 25),
            '50th': np.percentile(lengths, 50),  # Same as median
            '75th': np.percentile(lengths, 75),
            '90th': np.percentile(lengths, 90),
            '95th': np.percentile(lengths, 95)
        }
        
        # Recommend target length based on data
        # Use 90th percentile to capture most sequences without excessive padding
        recommended_target = int(np.ceil(analysis['percentiles']['90th']))
        analysis['recommended_target'] = min(recommended_target, self.max_sequence_length)
        
        # Estimate processing impact
        padding_needed = sum(1 for length in lengths if length < analysis['recommended_target'])
        truncation_needed = sum(1 for length in lengths if length > analysis['recommended_target'])
        
        analysis['processing_impact'] = {
            'sequences_needing_padding': padding_needed,
            'sequences_needing_truncation': truncation_needed,
            'padding_ratio': padding_needed / len(lengths),
            'truncation_ratio': truncation_needed / len(lengths)
        }
        
        print(f"📊 Sequence Length Analysis:")
        print(f"   📏 Range: {analysis['min_length']} - {analysis['max_length']}")
        print(f"   📈 Average: {analysis['mean_length']:.1f}")
        print(f"   🎯 Recommended target: {analysis['recommended_target']}")
        print(f"   📊 {padding_needed} need padding, {truncation_needed} need truncation")
        
        return analysis
    
    def demonstrate_processing_strategies(self) -> None:
        """
        Show examples of different processing strategies.
        
        This is a learning function that demonstrates how different
        sequence types are handled. Great for understanding the logic!
        """
        print("🎓 Sequence Processing Demonstration")
        print("=" * 50)
        
        # Example sequences
        examples = {
            'binary_sequence': [1, 0, 1, 1, 0],  # Binary classification results
            'temporal_sequence': [1.2, 1.5, 1.8, 2.1],  # Time-based measurements
            'count_sequence': [5, 8, 12, 15, 18, 22, 25],  # Counting events
            'score_sequence': [0.1, 0.3, 0.7, 0.9]  # Probability scores
        }
        
        target_length = 7
        
        for seq_type, sequence in examples.items():
            print(f"\\n📝 {seq_type.replace('_', ' ').title()}:")
            print(f"   Original: {sequence} (length: {len(sequence)})")
            
            if 'binary' in seq_type:
                processed = self.process_sequence(sequence, target_length, 'binary')
            elif 'temporal' in seq_type:
                processed = self.process_sequence(sequence, target_length, 'temporal')
            elif 'score' in seq_type:
                processed = self.process_sequence(sequence, target_length, 'statistical')
            else:
                processed = self.process_sequence(sequence, target_length, 'general')
            
            print(f"   Processed: {[round(x, 2) for x in processed]} (length: {len(processed)})")
        
        # Demonstrate interpolation with long sequence
        print(f"\\n📈 Interpolation Example:")
        long_sequence = list(range(20))  # [0, 1, 2, ..., 19]
        print(f"   Original: {long_sequence[:5]}...{long_sequence[-5:]} (length: {len(long_sequence)})")
        
        interpolated = self.process_sequence(long_sequence, 8, 'general')
        print(f"   Interpolated: {[round(x, 1) for x in interpolated]} (length: {len(interpolated)})")
        
        print("\\n✅ Demonstration complete!")


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of SequenceProcessor.
    
    This shows how to use the sequence processor with different types of data.
    Run this file directly to see it in action!
    """
    
    print("🎓 SequenceProcessor Example")
    print("=" * 50)
    
    # Initialize processor
    processor = SequenceProcessor()
    
    # Run demonstration
    processor.demonstrate_processing_strategies()
    
    # Test with realistic EPC event sequences
    print("\\n🧪 Real-world EPC Examples:")
    print("-" * 30)
    
    # Simulate different EPC journey lengths
    epc_journeys = [
        [1, 2],                    # Short journey: production → retail
        [1, 2, 3, 4],             # Medium journey: production → logistics → distribution → retail
        [1, 2, 2, 3, 3, 4, 4],    # Long journey with duplicates
        list(range(1, 16))         # Very long journey (15 steps)
    ]
    
    # Analyze the collection
    analysis = processor.analyze_sequence_lengths(epc_journeys)
    
    # Process each journey to target length
    target = analysis['recommended_target']
    print(f"\\n🎯 Processing all journeys to length {target}:")
    
    for i, journey in enumerate(epc_journeys):
        processed = processor.process_sequence(journey, target, 'temporal')
        print(f"   EPC {i+1}: {journey} → {[round(x, 1) for x in processed]}")
    
    print("\\n✅ Example complete!")
    print("Next step: Learn about FeatureNormalizer in feature_normalizer.py")