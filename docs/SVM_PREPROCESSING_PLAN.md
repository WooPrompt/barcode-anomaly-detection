# SVM Data Preprocessing Plan for 5 Anomaly Types

## 📋 Project Overview

Transform existing rule-based anomaly detection into training data for 5 separate SVM models:
- **epcFake**: EPC format violations
- **epcDup**: Duplicate scan detection  
- **locErr**: Location hierarchy violations
- **evtOrderErr**: Event order violations
- **jump**: Impossible travel times

**Goal**: Create modular, maintainable preprocessing pipeline that reuses existing rule-based logic.

## 🏗️ Directory Structure

```
src/barcode/svm_preprocessing/
├── __init__.py
├── base_preprocessor.py          # Core preprocessing (reuse existing)
├── feature_extractors/
│   ├── __init__.py
│   ├── epc_fake_features.py      # EPC format features
│   ├── epc_dup_features.py       # Duplicate scan features  
│   ├── loc_err_features.py       # Location hierarchy features
│   ├── evt_order_features.py     # Event order features
│   └── jump_features.py          # Time jump features
├── label_generators/
│   ├── __init__.py
│   └── rule_based_labels.py      # Convert scores to labels
├── data_manager.py               # Data storage and loading
├── pipeline.py                   # Main orchestration
└── config.py                     # Configuration management
```

## 🔄 Data Flow Pipeline

```
Raw Data → Base Preprocessing → Feature Extraction → Label Generation → SVM Training Data
    ↓              ↓                    ↓                  ↓              ↓
JSON/CSV    preprocess_scan_data()   5 extractors    rule_based_labels   X_train, y_train
```

## 🔗 Reused Functions from multi_anomaly_detector.py

| Function | Lines | Used in Phase | Purpose |
|----------|-------|---------------|---------|
| `preprocess_scan_data()` | 364-390 | Phase 1 | Core data cleaning & normalization |
| `load_csv_data()` | 572-604 | Phase 1 | Load reference datasets |
| `validate_epc_parts()` | 49-73 | Phase 2.1 | EPC format feature extraction |
| `validate_manufacture_date()` | 75-96 | Phase 2.1 | Date validation features |
| `classify_event_type()` | 216-247 | Phase 2.2 | Event sequence features |
| `get_location_hierarchy_level()` | 293-312 | Phase 2.3 | Location hierarchy features |
| `calculate_epc_fake_score()` | 98-161 | Phase 3 | Generate epcFake labels |
| `calculate_duplicate_score()` | 163-191 | Phase 3 | Generate epcDup labels |
| `calculate_location_error_score()` | 314-343 | Phase 3 | Generate locErr labels |
| `calculate_event_order_score()` | 249-279 | Phase 3 | Generate evtOrderErr labels |
| `calculate_time_jump_score()` | 193-214 | Phase 3 | Generate jump labels |

**Total Reuse**: 11 functions, ~400 lines of existing, tested code

##  Implementation Phases

### Phase 1: Base Infrastructure + Intelligent Sequence Processing

**File: `base_preprocessor.py`**

```python
from multi_anomaly_detector import preprocess_scan_data, load_csv_data
import numpy as np
from typing import Dict, List

class BasePreprocessor:
    """Reuse existing preprocessing logic with intelligent sequence handling"""
    
    def __init__(self):
        # REUSE: lines 572-604 (load_csv_data function)
        self.geo_df, self.transition_stats, self.location_mapping = load_csv_data()
    
    def preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """REUSE: lines 364-390 (preprocess_scan_data function)"""
        return preprocess_scan_data(raw_df)
    
    def get_epc_groups(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group by EPC for sequence analysis"""
        return {epc: group.sort_values('event_time').reset_index(drop=True) 
                for epc, group in df.groupby('epc_code')}
```

**File: `base_preprocessor.py` (Enhanced with Sequence Processor)**

```python
class SequenceProcessor:
    """ NEW: Intelligent sequence length adjustment for SVM features"""
    
    def __init__(self):
        self.truncation_threshold = 0.03  # 3% 이상 자르면 위험
    
    def process_sequence(self, sequence: List[float], target_length: int, 
                        sequence_type: str = 'general') -> List[float]:
        """통계적 판단 기반 시퀀스 처리"""
        
        if len(sequence) == target_length:
            return sequence
        
        elif len(sequence) < target_length:
            # 패딩 필요 - 상황별 지능적 패딩
            return self._intelligent_padding(sequence, target_length, sequence_type)
        
        else:
            # 절단 고려 - 신중한 판단
            return self._intelligent_truncation(sequence, target_length, sequence_type)
    
    def _intelligent_padding(self, sequence: List[float], target_length: int, 
                           sequence_type: str) -> List[float]:
        """ 상황별 패딩 전략 (무조건 0 패딩 금지)"""
        padding_needed = target_length - len(sequence)
        
        if len(sequence) == 0:
            # 완전 빈 시퀀스 → 0으로 패딩 (불가피)
            return [0.0] * target_length
        
        # 시퀀스 특성 분석
        if sequence_type == 'binary':
            # 이진 시퀀스 → 가장 빈도 높은 값으로 패딩
            pad_value = 1.0 if sequence.count(1.0) > sequence.count(0.0) else 0.0
            
        elif sequence_type == 'temporal':
            # 시간 시퀀스 → 마지막 값 유지 (시간 연속성)
            pad_value = sequence[-1]
            
        elif sequence_type == 'categorical':
            # 범주형 → 가장 빈번한 값
            pad_value = max(set(sequence), key=sequence.count)
            
        elif self._has_trend(sequence):
            # 트렌드 있음 → 마지막 값 또는 추세 연장
            pad_value = sequence[-1]
            
        else:
            # 기본: 중앙값으로 패딩 (이상치에 덜 민감)
            pad_value = np.median(sequence)
        
        return sequence + [pad_value] * padding_needed
    
    def _intelligent_truncation(self, sequence: List[float], target_length: int, 
                              sequence_type: str) -> List[float]:
        """ 통계적 판단 기반 절단 (3% 룰 적용)"""
        truncation_ratio = (len(sequence) - target_length) / len(sequence)
        
        #  3% 이상 절단 위험 → 보간 사용
        if truncation_ratio > self.truncation_threshold:
            print(f"  Truncation {truncation_ratio*100:.1f}% > 3% threshold, using interpolation")
            return self._interpolate_sequence(sequence, target_length)
        
        # 절단 안전 → 어느 부분을 자를지 지능적 판단
        if sequence_type == 'temporal' or self._has_important_end(sequence):
            # 시간 시퀀스 또는 끝 부분 중요 → 앞쪽 자르기
            return sequence[-target_length:]
            
        elif self._has_important_start(sequence):
            # 시작 부분 중요 → 뒤쪽 자르기
            return sequence[:target_length]
            
        else:
            # 균등 샘플링 (정보 손실 최소화)
            indices = np.linspace(0, len(sequence)-1, target_length, dtype=int)
            return [sequence[i] for i in indices]
    
    def _interpolate_sequence(self, sequence: List[float], target_length: int) -> List[float]:
        """보간을 통한 길이 조정 (정보 손실 최소화)"""
        if target_length >= len(sequence):
            return sequence
        
        # 선형 보간으로 정보 보존
        x_old = np.linspace(0, 1, len(sequence))
        x_new = np.linspace(0, 1, target_length)
        interpolated = np.interp(x_new, x_old, sequence)
        
        return interpolated.tolist()
    
    def _has_trend(self, sequence: List[float]) -> bool:
        """시퀀스에 트렌드가 있는지 판단"""
        if len(sequence) < 3:
            return False
        
        # 선형 상관관계로 트렌드 검출
        x = np.arange(len(sequence))
        correlation = np.corrcoef(x, sequence)[0, 1]
        return abs(correlation) > 0.5
    
    def _has_important_start(self, sequence: List[float]) -> bool:
        """시작 부분이 중요한지 판단"""
        start_portion = sequence[:len(sequence)//3]
        return np.std(start_portion) > np.std(sequence) * 0.8
    
    def _has_important_end(self, sequence: List[float]) -> bool:
        """끝 부분이 중요한지 판단"""
        end_portion = sequence[-len(sequence)//3:]
        return np.std(end_portion) > np.std(sequence) * 0.8
```

## 🎯 **Phase 1 Strategy Explained**

### **What Phase 1 Actually Does:**

**1. Foundation Layer (5 minutes)**
```python
# 기본 wrapper 생성
preprocessor = BasePreprocessor()  
# → loads geo_df, transition_stats automatically
# → provides clean interface to existing functions
```

**2. Smart Sequence Handling (NEW - 30 minutes)**
```python
# 기존 문제: 무조건 0 패딩
sequence = [1, 0, 1]  # length 3
padded = sequence + [0, 0]  # → [1, 0, 1, 0, 0] (의미없는 0들)

# 🔥 새로운 방법: 지능적 패딩
processor = SequenceProcessor()
padded = processor.process_sequence(sequence, 5, 'binary')
# → [1, 0, 1, 1, 1] (가장 빈도 높은 값으로 패딩)
```

**3. Reuse Guarantee (5 minutes)**
```python
# 모든 기존 함수 그대로 사용
df_clean = preprocessor.preprocess(raw_df)  # lines 364-390
geo_df = preprocessor.geo_df                # lines 572-604
groups = preprocessor.get_epc_groups(df_clean)  # NEW grouping logic
```

### **Why This Foundation is Critical:**

1. **Zero Risk**: 기존 working code 건드리지 않음
2. **Smart Sequence**: 시퀀스 정보 손실 방지
3. **Standard Interface**: 모든 다음 phase에서 동일하게 사용
4. **Statistical Intelligence**: 3% 룰, 트렌드 분석, 보간법 적용

### **Total Phase 1 Time: 45 minutes**
- Base wrapper: 5분  
- Sequence processor: 30분
- Testing: 10분

**Ready for this enhanced Phase 1?**

### Phase 2: Feature Extractors

#### 2.1 EPC Fake Features (Enhanced)
**File: `feature_extractors/epc_fake_features.py`**

```python
from multi_anomaly_detector import validate_epc_parts, validate_manufacture_date
import numpy as np

class EpcFakeFeatureExtractor:
    """Extract numerical features from EPC format validation with FIXED dimensions"""
    
    # 🔥 IMPROVEMENT 6: Fixed feature dimensions
    FEATURE_DIMENSION = 10  # Always return exactly 10 features
    
    def extract_features(self, epc_code: str) -> List[float]:
        parts = epc_code.strip().split('.')
        validations = validate_epc_parts(parts)  # REUSE: lines 49-73
        
        # Calculate all possible features
        raw_features = [
            # Structure features (4 features)
            len(parts),                                    # Number of parts
            float(validations.get('structure', False)),   # Valid structure (0/1)
            float(validations.get('header', False)),      # Valid header (0/1)
            float(validations.get('company', False)),     # Valid company (0/1)
            
            # Component validation features (3 features)
            float(validations.get('product', False)),     # Valid product (0/1)
            float(validations.get('lot', False)),         # Valid lot (0/1)
            float(validations.get('serial', False)),      # Valid serial (0/1)
            
            # Date features (1 feature)
            self._extract_date_features(parts[4] if len(parts) > 4 else ''),
            
            # Statistical features (2 features)
            self._calculate_epc_entropy(epc_code),         # Information entropy
            self._calculate_component_length_ratio(parts) # Length ratios
        ]
        
        # 🔥 IMPROVEMENT 6: Ensure exactly FEATURE_DIMENSION features
        return self._normalize_to_fixed_length(raw_features)
    
    def _normalize_to_fixed_length(self, features: List[float]) -> List[float]:
        """Ensure exactly FEATURE_DIMENSION features through padding/truncation"""
        if len(features) > self.FEATURE_DIMENSION:
            # Truncate if too many
            return features[:self.FEATURE_DIMENSION]
        elif len(features) < self.FEATURE_DIMENSION:
            # Pad with zeros if too few
            padding = [0.0] * (self.FEATURE_DIMENSION - len(features))
            return features + padding
        else:
            return features
    
    def _extract_date_features(self, date_string: str) -> float:
        """Extract date-related features"""
        date_valid, error_type = validate_manufacture_date(date_string)  # REUSE: lines 75-96
        if date_valid:
            return 1.0
        elif error_type == 'future_date':
            return -1.0
        elif error_type == 'too_old':
            return 0.5
        else:
            return 0.0
    
    def _calculate_epc_entropy(self, epc_code: str) -> float:
        """Calculate information entropy of EPC code"""
        if not epc_code:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in epc_code:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        total_chars = len(epc_code)
        entropy = 0.0
        for count in char_counts.values():
            prob = count / total_chars
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy / 8.0  # Normalize to [0,1] range
    
    def _calculate_component_length_ratio(self, parts: List[str]) -> float:
        """Calculate ratio of actual vs expected component lengths"""
        if len(parts) != 6:
            return 0.0
        
        expected_lengths = [3, 7, 7, 6, 8, 9]  # header, company, product, lot, date, serial
        actual_lengths = [len(part) for part in parts]
        
        # Calculate similarity ratio
        matches = sum(1 for actual, expected in zip(actual_lengths, expected_lengths) 
                     if actual == expected)
        return matches / len(expected_lengths)
```
```

#### 2.2 Event Order Features
**File: `feature_extractors/evt_order_features.py`**

```python
from multi_anomaly_detector import classify_event_type

class EventOrderFeatureExtractor:
    """Extract numerical features from event sequence analysis"""
    
    def extract_features(self, epc_group: pd.DataFrame) -> List[float]:
        event_sequence = epc_group['event_type'].tolist()
        
        # Use updated classify_event_type function (returns direction, step)
        directions, steps = [], []
        for event in event_sequence:
            direction, step = classify_event_type(event)  # REUSE: lines 216-247
            directions.append(direction)
            steps.append(step)
        
        features = [
            # Sequence length features
            len(event_sequence),                          # Total events
            len(set(directions)),                         # Unique directions
            len(set(steps)),                             # Unique steps
            
            # Direction distribution (percentages)
            directions.count('inbound') / len(directions) if directions else 0,
            directions.count('outbound') / len(directions) if directions else 0,
            directions.count('other') / len(directions) if directions else 0,
            
            # Pattern features
            self._count_consecutive_patterns(directions, steps),
            self._calculate_sequence_entropy(event_sequence),
            
            # Temporal features
            self._extract_time_intervals(epc_group['event_time'])
        ]
        return features
```

#### 2.3 Location Hierarchy Features
**File: `feature_extractors/loc_err_features.py`**

```python
from multi_anomaly_detector import get_location_hierarchy_level

class LocationErrorFeatureExtractor:
    """Extract numerical features from location sequence analysis"""
    
    def extract_features(self, epc_group: pd.DataFrame) -> List[float]:
        location_sequence = epc_group['scan_location'].tolist()
        
        # Use existing hierarchy function
        hierarchy_levels = [get_location_hierarchy_level(loc) 
                          for loc in location_sequence]  # REUSE: lines 293-312
        
        features = [
            # Hierarchy progression features
            len(set(hierarchy_levels)),                   # Unique levels visited
            max(hierarchy_levels) - min(hierarchy_levels) if hierarchy_levels else 0,  # Range
            
            # Transition features
            self._count_backward_transitions(hierarchy_levels),   # Backward moves
            self._count_level_skips(hierarchy_levels),           # Skipped levels
            
            # Distribution features (percentages)
            hierarchy_levels.count(0) / len(hierarchy_levels) if hierarchy_levels else 0,   # Factory %
            hierarchy_levels.count(1) / len(hierarchy_levels) if hierarchy_levels else 0,   # Logistics %
            hierarchy_levels.count(2) / len(hierarchy_levels) if hierarchy_levels else 0,   # Wholesale %
            hierarchy_levels.count(3) / len(hierarchy_levels) if hierarchy_levels else 0,   # Retail %
            hierarchy_levels.count(99) / len(hierarchy_levels) if hierarchy_levels else 0,  # Unknown %
            
            # Statistical features
            np.std(hierarchy_levels) if len(hierarchy_levels) > 1 else 0
        ]
        return features
```

#### 2.4 Duplicate Features
**File: `feature_extractors/epc_dup_features.py`**

```python
class EpcDupFeatureExtractor:
    """Extract numerical features from duplicate scan analysis"""
    
    def extract_features(self, epc_group: pd.DataFrame) -> List[float]:
        features = [
            # Time clustering features
            self._count_time_clusters(epc_group['event_time_rounded']),
            self._max_events_per_timestamp(epc_group['event_time_rounded']),
            
            # Location diversity features
            self._location_diversity_at_same_time(epc_group),
            self._max_locations_per_timestamp(epc_group),
            
            # Statistical features
            len(epc_group),                               # Total events for this EPC
            epc_group['scan_location'].nunique(),        # Unique locations
            self._calculate_time_variance(epc_group['event_time'])
        ]
        return features
```

#### 2.5 Jump Features
**File: `feature_extractors/jump_features.py`**

```python
class JumpFeatureExtractor:
    """Extract numerical features from travel time analysis"""
    
    def extract_features(self, epc_group: pd.DataFrame) -> List[float]:
        if len(epc_group) < 2:
            return [0] * 10  # No transitions to analyze
        
        time_diffs = []
        for i in range(1, len(epc_group)):
            current_time = pd.to_datetime(epc_group.iloc[i]['event_time'])
            prev_time = pd.to_datetime(epc_group.iloc[i-1]['event_time'])
            time_diff_hours = (current_time - prev_time).total_seconds() / 3600
            time_diffs.append(time_diff_hours)
        
        features = [
            # Time difference statistics
            np.mean(time_diffs) if time_diffs else 0,     # Average time between events
            np.std(time_diffs) if len(time_diffs) > 1 else 0,  # Time variance
            min(time_diffs) if time_diffs else 0,         # Minimum time gap
            max(time_diffs) if time_diffs else 0,         # Maximum time gap
            
            # Anomaly indicators
            sum(1 for td in time_diffs if td < 0),        # Negative time count
            sum(1 for td in time_diffs if td > 24),       # > 24 hour gaps
            
            # Geographic features (if available)
            self._calculate_distance_features(epc_group),
            self._calculate_speed_features(epc_group),
            
            # Statistical features
            len([td for td in time_diffs if abs(td) > 2 * np.std(time_diffs)]) if len(time_diffs) > 1 else 0
        ]
        return features
```

### Phase 3: Label Generation

**File: `label_generators/rule_based_labels.py`**

```python
from multi_anomaly_detector import (
    calculate_epc_fake_score,         # REUSE: lines 98-161
    calculate_duplicate_score,        # REUSE: lines 163-191
    calculate_location_error_score,   # REUSE: lines 314-343
    calculate_event_order_score,      # REUSE: lines 249-279
    calculate_time_jump_score         # REUSE: lines 193-214
)

class RuleBasedLabelGenerator:
    """Convert existing rule-based scores to binary labels with confidence tracking"""
    
    def __init__(self, threshold: int = 50):
        self.threshold = threshold
    
    def generate_labels_with_scores(self, epc_code: str, epc_group: pd.DataFrame) -> Dict[str, Tuple[int, float]]:
        """🔥 IMPROVEMENT 2: Convert rule scores to binary labels WITH confidence scores"""
        results = {}
        
        # Calculate scores using existing functions
        epc_fake_score = calculate_epc_fake_score(epc_code)
        epc_dup_score = self._get_max_duplicate_score(epc_group)
        loc_err_score = calculate_location_error_score(epc_group['scan_location'].tolist())
        evt_order_score = calculate_event_order_score(epc_group['event_type'].tolist())
        jump_score = self._get_max_jump_score(epc_group)
        
        # Return (binary_label, confidence_score) tuples
        results['epcFake'] = (1 if epc_fake_score > self.threshold else 0, float(epc_fake_score))
        results['epcDup'] = (1 if epc_dup_score > self.threshold else 0, float(epc_dup_score))
        results['locErr'] = (1 if loc_err_score > self.threshold else 0, float(loc_err_score))
        results['evtOrderErr'] = (1 if evt_order_score > self.threshold else 0, float(evt_order_score))
        results['jump'] = (1 if jump_score > self.threshold else 0, float(jump_score))
        
        return results
    
    def generate_labels(self, epc_code: str, epc_group: pd.DataFrame) -> Dict[str, int]:
        """Legacy method for backward compatibility"""
        results = self.generate_labels_with_scores(epc_code, epc_group)
        return {anomaly_type: label for anomaly_type, (label, score) in results.items()}
    
    def _get_max_duplicate_score(self, epc_group: pd.DataFrame) -> int:
        """Find maximum duplicate score across all timestamps"""
        max_score = 0
        for timestamp, time_group in epc_group.groupby('event_time_rounded'):
            score = calculate_duplicate_score(epc_group.iloc[0]['epc_code'], time_group)
            max_score = max(max_score, score)
        return max_score
```

### Phase 4A: SVM-Specific Components

**File: `svm_preprocessing/feature_normalizer.py`**

```python
# 🔥 IMPROVEMENT 8: Feature normalization/scaling for SVM
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import numpy as np

class FeatureNormalizer:
    """SVM-optimized feature normalization"""
    
    def __init__(self, method: str = 'robust'):
        self.method = method
        self.scalers = {}
    
    def fit_transform_features(self, X: np.ndarray, anomaly_type: str) -> np.ndarray:
        """Fit scaler and transform features for training"""
        if self.method == 'robust':
            # 🔥 RobustScaler is less sensitive to outliers (critical for anomaly data)
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        self.scalers[anomaly_type] = scaler  # Save for prediction time
        return X_scaled
    
    def transform_features(self, X: np.ndarray, anomaly_type: str) -> np.ndarray:
        """Transform features using fitted scaler (for prediction)"""
        if anomaly_type not in self.scalers:
            raise ValueError(f"Scaler for {anomaly_type} not found. Call fit_transform_features first.")
        return self.scalers[anomaly_type].transform(X)
    
    def save_scalers(self, output_dir: str):
        """Save fitted scalers for prediction time"""
        for anomaly_type, scaler in self.scalers.items():
            scaler_path = f"{output_dir}/{anomaly_type}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
    
    def load_scalers(self, model_dir: str):
        """Load saved scalers"""
        import os
        for anomaly_type in ['epcFake', 'epcDup', 'evtOrderErr', 'locErr', 'jump']:
            scaler_path = f"{model_dir}/{anomaly_type}_scaler.joblib"
            if os.path.exists(scaler_path):
                self.scalers[anomaly_type] = joblib.load(scaler_path)
```

**File: `svm_preprocessing/imbalance_handler.py`**

```python
# 🔥 IMPROVEMENT 7: Class imbalance handling
import numpy as np
from typing import Tuple, Dict, Optional

class ImbalanceHandler:
    """Handle severe class imbalance in anomaly detection data"""
    
    def __init__(self, strategy: str = 'smote', min_samples_per_class: int = 50):
        self.strategy = strategy
        self.min_samples_per_class = min_samples_per_class
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray, 
                        anomaly_type: str) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
        """Apply imbalance handling strategy"""
        
        positive_count = np.sum(y)
        negative_count = len(y) - positive_count
        
        print(f"{anomaly_type}: {negative_count} normal, {positive_count} anomaly "
              f"({positive_count/len(y)*100:.1f}% positive)")
        
        # If severely imbalanced, apply strategy
        if positive_count < self.min_samples_per_class or positive_count / len(y) < 0.05:
            
            if self.strategy == 'smote' and positive_count >= 2:
                return self._apply_smote(X, y)
            
            elif self.strategy == 'weighted':
                return self._apply_class_weights(X, y)
            
            elif self.strategy == 'threshold_tuning':
                return self._prepare_threshold_tuning(X, y)
            
            else:
                # Fallback to weighted approach
                return self._apply_class_weights(X, y)
        
        return X, y, None
    
    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, None]:
        """Apply SMOTE oversampling"""
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y)-1))
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            print(f"   SMOTE applied: {len(X)} → {len(X_resampled)} samples")
            return X_resampled, y_resampled, None
            
        except ImportError:
            print("   SMOTE not available, falling back to weighted approach")
            return self._apply_class_weights(X, y)
        except Exception as e:
            print(f"   SMOTE failed ({e}), falling back to weighted approach")
            return self._apply_class_weights(X, y)
    
    def _apply_class_weights(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Calculate class weights for balanced training"""
        pos_count = np.sum(y)
        neg_count = len(y) - pos_count
        
        if pos_count == 0 or neg_count == 0:
            return X, y, None
        
        # Calculate balanced weights
        pos_weight = len(y) / (2 * pos_count)
        neg_weight = len(y) / (2 * neg_count)
        
        class_weights = {0: neg_weight, 1: pos_weight}
        print(f"   Class weights: normal={neg_weight:.2f}, anomaly={pos_weight:.2f}")
        
        return X, y, class_weights
    
    def _prepare_threshold_tuning(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Prepare metadata for threshold tuning"""
        thresholds = [30, 40, 50, 60, 70]
        metadata = {'threshold_candidates': thresholds}
        print(f"   Threshold tuning mode: will test {thresholds}")
        
        return X, y, metadata
```

### Phase 4B: Data Management (Enhanced)

**File: `data_manager.py`**

```python
from sklearn.model_selection import train_test_split
from .feature_normalizer import FeatureNormalizer
from .imbalance_handler import ImbalanceHandler

class SVMDataManager:
    """Handle data storage and loading for SVM training with enhanced tracking"""
    
    def __init__(self, output_dir: str = "data/svm_training"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 🔥 IMPROVEMENT 7 & 8: Initialize SVM-specific components
        self.normalizer = FeatureNormalizer()
        self.imbalance_handler = ImbalanceHandler()
    
    def save_training_data(self, features: np.ndarray, labels: np.ndarray, 
                          scores: np.ndarray, epc_codes: List[str], anomaly_type: str,
                          test_size: float = 0.2):
        """Save features, labels, scores, and EPC mapping with SVM optimizations"""
        
        print(f"🔥 Processing {anomaly_type} with SVM optimizations...")
        
        # 🔥 IMPROVEMENT 8: Feature normalization (CRITICAL for SVM)
        print(f"   Normalizing features...")
        features_normalized = self.normalizer.fit_transform_features(features, anomaly_type)
        
        # 🔥 IMPROVEMENT 7: Handle class imbalance
        print(f"   Handling class imbalance...")
        features_balanced, labels_balanced, class_weights = self.imbalance_handler.handle_imbalance(
            features_normalized, labels, anomaly_type)
        
        # 🔥 IMPROVEMENT 1: EPC mapping for debugging/interpretation
        epc_path = f"{self.output_dir}/{anomaly_type}_epc_codes.npy"
        if len(features_balanced) != len(epc_codes):
            # SMOTE was applied, need to extend EPC codes
            repeat_factor = len(features_balanced) // len(epc_codes)
            epc_codes_extended = np.repeat(epc_codes, repeat_factor)[:len(features_balanced)]
            np.save(epc_path, epc_codes_extended)
        else:
            np.save(epc_path, np.array(epc_codes))
        
        # 🔥 IMPROVEMENT 2: Save confidence scores alongside binary labels
        scores_path = f"{self.output_dir}/{anomaly_type}_scores.npy"
        if len(features_balanced) != len(scores):
            # SMOTE was applied, need to extend scores
            repeat_factor = len(features_balanced) // len(scores)
            scores_extended = np.repeat(scores, repeat_factor)[:len(features_balanced)]
            np.save(scores_path, scores_extended)
        else:
            np.save(scores_path, scores)
        
        # 🔥 IMPROVEMENT 3: Automatic train/test split
        if len(features) > 10:  # Only split if we have enough samples
            X_train, X_test, y_train, y_test, scores_train, scores_test, epc_train, epc_test = \
                train_test_split(features, labels, scores, epc_codes, 
                               test_size=test_size, random_state=42, stratify=labels)
            
            # Save train sets
            np.save(f"{self.output_dir}/{anomaly_type}_X_train.npy", X_train)
            np.save(f"{self.output_dir}/{anomaly_type}_y_train.npy", y_train)
            np.save(f"{self.output_dir}/{anomaly_type}_scores_train.npy", scores_train)
            np.save(f"{self.output_dir}/{anomaly_type}_epc_train.npy", np.array(epc_train))
            
            # Save test sets
            np.save(f"{self.output_dir}/{anomaly_type}_X_test.npy", X_test)
            np.save(f"{self.output_dir}/{anomaly_type}_y_test.npy", y_test)
            np.save(f"{self.output_dir}/{anomaly_type}_scores_test.npy", scores_test)
            np.save(f"{self.output_dir}/{anomaly_type}_epc_test.npy", np.array(epc_test))
        else:
            # Too few samples, save as full dataset
            X_train, y_train, scores_train, epc_train = features, labels, scores, epc_codes
            np.save(f"{self.output_dir}/{anomaly_type}_X_train.npy", X_train)
            np.save(f"{self.output_dir}/{anomaly_type}_y_train.npy", y_train)
            np.save(f"{self.output_dir}/{anomaly_type}_scores_train.npy", scores_train)
            np.save(f"{self.output_dir}/{anomaly_type}_epc_train.npy", np.array(epc_train))
        
        # Legacy format for backward compatibility
        np.save(f"{self.output_dir}/{anomaly_type}_features.npy", features)
        np.save(f"{self.output_dir}/{anomaly_type}_labels.npy", labels)
        
        # Enhanced metadata with confidence info
        metadata = {
            'anomaly_type': anomaly_type,
            'feature_count': features.shape[1],
            'sample_count': features.shape[0],
            'positive_samples': int(np.sum(labels)),
            'negative_samples': int(np.sum(1 - labels)),
            'train_samples': len(y_train) if len(features) > 10 else len(features),
            'test_samples': len(y_test) if len(features) > 10 else 0,
            'score_stats': {
                'mean_positive_score': float(np.mean(scores[labels == 1])) if np.any(labels == 1) else 0,
                'mean_negative_score': float(np.mean(scores[labels == 0])) if np.any(labels == 0) else 0,
                'borderline_cases': int(np.sum((scores >= 45) & (scores <= 55)))  # Near threshold
            },
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{self.output_dir}/{anomaly_type}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def load_training_data(self, anomaly_type: str, split: str = 'train') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load training data with EPC codes and scores"""
        suffix = f"_{split}" if split in ['train', 'test'] else ""
        
        X = np.load(f"{self.output_dir}/{anomaly_type}_X{suffix}.npy")
        y = np.load(f"{self.output_dir}/{anomaly_type}_y{suffix}.npy")
        scores = np.load(f"{self.output_dir}/{anomaly_type}_scores{suffix}.npy")
        epc_codes = np.load(f"{self.output_dir}/{anomaly_type}_epc{suffix}.npy")
        
        return X, y, scores, epc_codes
    
    def analyze_predictions(self, anomaly_type: str, y_pred: np.ndarray, split: str = 'test'):
        """Analyze SVM predictions with EPC mapping for debugging"""
        X, y_true, scores, epc_codes = self.load_training_data(anomaly_type, split)
        
        # Find misclassified cases
        misclassified = y_pred != y_true
        
        analysis = {
            'total_samples': len(y_true),
            'misclassified_count': int(np.sum(misclassified)),
            'accuracy': float(np.mean(y_pred == y_true)),
            'misclassified_epcs': epc_codes[misclassified].tolist(),
            'misclassified_scores': scores[misclassified].tolist(),
            'borderline_misclassified': int(np.sum(misclassified & (scores >= 45) & (scores <= 55)))
        }
        
        return analysis
```

### Phase 4C: SVM Inference Engine

**File: `svm_preprocessing/inference_engine.py`**

```python
# 🔥 IMPROVEMENT 10: Complete SVM inference pipeline
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
from .feature_normalizer import FeatureNormalizer
from ..multi_anomaly_detector import preprocess_scan_data

class SVMInferenceEngine:
    """Production-ready SVM inference with complete preprocessing pipeline"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.models = {}
        self.normalizer = FeatureNormalizer()
        self.extractors = {}
        self.feature_dimensions = {
            'epcFake': 10, 'epcDup': 8, 'evtOrderErr': 12, 'locErr': 15, 'jump': 10
        }
        self._load_trained_models()
    
    def _load_trained_models(self):
        """Load all trained SVM models and scalers"""
        import os
        
        for anomaly_type in ['epcFake', 'epcDup', 'evtOrderErr', 'locErr', 'jump']:
            # Load SVM model
            model_path = f"{self.model_dir}/{anomaly_type}_svm.joblib"
            if os.path.exists(model_path):
                self.models[anomaly_type] = joblib.load(model_path)
                print(f"✅ Loaded {anomaly_type} SVM model")
            else:
                print(f"⚠️  {anomaly_type} SVM model not found")
        
        # Load feature scalers
        self.normalizer.load_scalers(self.model_dir)
        
        # Initialize feature extractors (same config as training)
        self._initialize_extractors()
    
    def _initialize_extractors(self):
        """Initialize feature extractors with same config as training"""
        from .feature_extractors.epc_fake_features import EpcFakeFeatureExtractor
        from .feature_extractors.epc_dup_features import EpcDupFeatureExtractor
        from .feature_extractors.evt_order_features import EventOrderFeatureExtractor
        from .feature_extractors.loc_err_features import LocationErrorFeatureExtractor
        from .feature_extractors.jump_features import JumpFeatureExtractor
        
        self.extractors = {
            'epcFake': EpcFakeFeatureExtractor(),
            'epcDup': EpcDupFeatureExtractor(),
            'evtOrderErr': EventOrderFeatureExtractor(),
            'locErr': LocationErrorFeatureExtractor(),
            'jump': JumpFeatureExtractor()
        }
    
    def predict_anomalies(self, epc_code: str, epc_group: pd.DataFrame) -> Dict[str, float]:
        """Predict anomaly probabilities for a single EPC"""
        predictions = {}
        
        for anomaly_type, model in self.models.items():
            try:
                # 1. Extract features (same as training)
                if anomaly_type == 'epcFake':
                    features = self.extractors[anomaly_type].extract_features(epc_code)
                else:
                    features = self.extractors[anomaly_type].extract_features(epc_group)
                
                # 2. Ensure correct dimensions
                expected_dim = self.feature_dimensions[anomaly_type]
                if len(features) != expected_dim:
                    print(f"⚠️  {anomaly_type}: Expected {expected_dim} features, got {len(features)}")
                    features = self._pad_or_truncate(features, expected_dim)
                
                # 3. Normalize features (same scaler as training)
                features_array = np.array(features).reshape(1, -1)
                features_normalized = self.normalizer.transform_features(features_array, anomaly_type)
                
                # 4. SVM prediction
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(features_normalized)[0][1]  # Anomaly probability
                else:
                    # For models without probability, use decision function
                    decision = model.decision_function(features_normalized)[0]
                    prob = 1.0 / (1.0 + np.exp(-decision))  # Sigmoid transformation
                
                predictions[anomaly_type] = float(prob)
                
            except Exception as e:
                print(f"❌ Error predicting {anomaly_type}: {e}")
                predictions[anomaly_type] = 0.0
        
        return predictions
    
    def _pad_or_truncate(self, features: List[float], target_length: int) -> List[float]:
        """Ensure features have correct length"""
        if len(features) > target_length:
            return features[:target_length]
        elif len(features) < target_length:
            return features + [0.0] * (target_length - len(features))
        return features
    
    def batch_predict(self, df: pd.DataFrame) -> List[Dict]:
        """Batch prediction for multiple EPCs"""
        results = []
        
        # Use same preprocessing as training
        df_clean = preprocess_scan_data(df)
        
        for epc_code, epc_group in df_clean.groupby('epc_code'):
            epc_group = epc_group.sort_values('event_time').reset_index(drop=True)
            
            # Get SVM predictions
            svm_predictions = self.predict_anomalies(epc_code, epc_group)
            
            # Find primary anomaly and confidence
            if svm_predictions:
                primary_anomaly = max(svm_predictions.items(), key=lambda x: x[1])
                max_score = primary_anomaly[1]
                primary_type = primary_anomaly[0]
            else:
                max_score = 0.0
                primary_type = 'none'
            
            result = {
                'epc_code': epc_code,
                'svm_predictions': svm_predictions,
                'max_anomaly_probability': max_score,
                'primary_anomaly_type': primary_type,
                'is_anomaly': max_score > 0.5,  # 50% threshold
                'confidence': 'high' if max_score > 0.8 else 'medium' if max_score > 0.3 else 'low'
            }
            
            results.append(result)
        
        return results
    
    def compare_with_rules(self, df: pd.DataFrame) -> List[Dict]:
        """Compare SVM predictions with rule-based results"""
        from ..multi_anomaly_detector import detect_multi_anomalies_enhanced, load_csv_data
        
        # Get rule-based predictions
        geo_df, transition_stats, _ = load_csv_data()
        rule_results = detect_multi_anomalies_enhanced(df, transition_stats, geo_df)
        
        # Get SVM predictions
        svm_results = self.batch_predict(df)
        
        # Compare results
        comparison = []
        for svm_result in svm_results:
            epc_code = svm_result['epc_code']
            
            # Find corresponding rule-based result
            rule_result = next((r for r in rule_results if r['epcCode'] == epc_code), None)
            
            comparison.append({
                'epc_code': epc_code,
                'svm_primary': svm_result['primary_anomaly_type'],
                'svm_confidence': svm_result['max_anomaly_probability'],
                'rule_primary': rule_result['primaryAnomaly'] if rule_result else 'none',
                'rule_types': rule_result['anomalyTypes'] if rule_result else [],
                'agreement': (svm_result['primary_anomaly_type'] == rule_result['primaryAnomaly']) if rule_result else False
            })
        
        return comparison
```

### Phase 4D: Complete SVM Workflow

**File: `svm_preprocessing/svm_workflow.py`**

```python
# 🔥 Complete train → predict workflow
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import pandas as pd
import numpy as np

class SVMWorkflow:
    """End-to-end SVM workflow: preprocessing → training → evaluation → prediction"""
    
    def __init__(self, data_dir: str = "data/svm_training", model_dir: str = "models/svm"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
    
    def train_all_models(self, evaluate: bool = True):
        """Train SVM models for all anomaly types"""
        from .data_manager import SVMDataManager
        
        data_manager = SVMDataManager(self.data_dir)
        results = {}
        
        for anomaly_type in ['epcFake', 'epcDup', 'evtOrderErr', 'locErr', 'jump']:
            print(f"\n🚀 Training {anomaly_type} SVM...")
            
            try:
                # Load training data
                X_train, y_train, scores_train, epc_train = data_manager.load_training_data(anomaly_type, 'train')
                
                if len(X_train) == 0:
                    print(f"   ⚠️  No training data for {anomaly_type}")
                    continue
                
                # Train SVM with probability estimates
                model = SVC(
                    probability=True,
                    class_weight='balanced',  # Handle remaining imbalance
                    kernel='rbf',             # Good default for anomaly detection
                    gamma='scale',            # Auto-tune gamma
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                
                # Save model
                model_path = f"{self.model_dir}/{anomaly_type}_svm.joblib"
                joblib.dump(model, model_path)
                print(f"   ✅ Model saved: {model_path}")
                
                # Evaluate if test data exists
                if evaluate:
                    try:
                        X_test, y_test, scores_test, epc_test = data_manager.load_training_data(anomaly_type, 'test')
                        evaluation = self._evaluate_model(model, X_test, y_test, anomaly_type)
                        results[anomaly_type] = evaluation
                    except:
                        print(f"   ⚠️  No test data for evaluation")
                        results[anomaly_type] = {'status': 'trained_no_test'}
                
            except Exception as e:
                print(f"   ❌ Training failed: {e}")
                results[anomaly_type] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def _evaluate_model(self, model, X_test, y_test, anomaly_type):
        """Evaluate trained model"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.0  # Only one class in test set
        
        evaluation = {
            'status': 'success',
            'test_samples': len(y_test),
            'positive_samples': int(np.sum(y_test)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_score': float(auc),
            'accuracy': float(np.mean(y_pred == y_test))
        }
        
        print(f"   📊 {anomaly_type} Results: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return evaluation
    
    def predict_new_data(self, input_csv: str, output_csv: str = None):
        """Apply trained models to new data"""
        from .inference_engine import SVMInferenceEngine
        
        # Load new data
        df = pd.read_csv(input_csv)
        print(f"📊 Loading {len(df)} events from {input_csv}")
        
        # Initialize inference engine
        engine = SVMInferenceEngine(self.model_dir)
        
        # Get predictions
        predictions = engine.batch_predict(df)
        
        # Convert to DataFrame for easy analysis
        results_df = pd.DataFrame(predictions)
        
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"💾 Results saved to {output_csv}")
        
        # Print summary
        anomaly_count = sum(1 for r in predictions if r['is_anomaly'])
        print(f"📈 Summary: {anomaly_count}/{len(predictions)} EPCs flagged as anomalies "
              f"({anomaly_count/len(predictions)*100:.1f}%)")
        
        return predictions
```

### Phase 5: Main Pipeline

**File: `pipeline.py`**

```python
class SVMPreprocessingPipeline:
    """Main orchestration pipeline for SVM data preprocessing"""
    
    def __init__(self):
        self.preprocessor = BasePreprocessor()
        self.extractors = {
            'epcFake': EpcFakeFeatureExtractor(),
            'epcDup': EpcDupFeatureExtractor(), 
            'locErr': LocationErrorFeatureExtractor(),
            'evtOrderErr': EventOrderFeatureExtractor(),
            'jump': JumpFeatureExtractor()
        }
        self.label_generator = RuleBasedLabelGenerator()
        self.data_manager = SVMDataManager()
    
    def process_data(self, raw_df: pd.DataFrame) -> Dict[str, Tuple]:
        """Main processing pipeline"""
        print("Starting SVM data preprocessing...")
        
        # 1. Base preprocessing (REUSE existing preprocess_scan_data - lines 364-390)
        print("Step 1: Base preprocessing...")
        df_clean = self.preprocessor.preprocess(raw_df)
        epc_groups = self.preprocessor.get_epc_groups(df_clean)
        print(f"Processed {len(epc_groups)} unique EPCs")
        
        # 2. Extract features and labels per EPC
        print("Step 2: Feature extraction and label generation...")
        all_features = {anomaly_type: [] for anomaly_type in self.extractors.keys()}
        all_labels = {anomaly_type: [] for anomaly_type in self.extractors.keys()}
        
        # Enhanced data collection with EPC mapping and confidence scores
        all_epc_codes = {anomaly_type: [] for anomaly_type in self.extractors.keys()}
        all_scores = {anomaly_type: [] for anomaly_type in self.extractors.keys()}
        
        for epc_code, epc_group in epc_groups.items():
            # Generate labels with confidence scores (IMPROVEMENT 2)
            labels_with_scores = self.label_generator.generate_labels_with_scores(epc_code, epc_group)
            
            # Extract features for each anomaly type
            for anomaly_type, extractor in self.extractors.items():
                if anomaly_type == 'epcFake':
                    features = extractor.extract_features(epc_code)
                else:
                    features = extractor.extract_features(epc_group)
                
                label, score = labels_with_scores[anomaly_type]
                
                all_features[anomaly_type].append(features)
                all_labels[anomaly_type].append(label)
                all_scores[anomaly_type].append(score)  # IMPROVEMENT 2: Save confidence
                all_epc_codes[anomaly_type].append(epc_code)  # IMPROVEMENT 1: EPC mapping
        
        # 3. Convert to numpy arrays and save with enhancements
        print("Step 3: Saving enhanced training data...")
        results = {}
        for anomaly_type in self.extractors.keys():
            X = np.array(all_features[anomaly_type])
            y = np.array(all_labels[anomaly_type])
            scores = np.array(all_scores[anomaly_type])
            epc_codes = all_epc_codes[anomaly_type]
            
            # IMPROVEMENTS 1, 2, 3: Save with EPC mapping, scores, and train/test split
            self.data_manager.save_training_data(X, y, scores, epc_codes, anomaly_type)
            results[anomaly_type] = (X, y)
            
            print(f"{anomaly_type}: {X.shape[0]} samples, {X.shape[1]} features, "
                  f"{np.sum(y)} positive samples ({np.mean(y)*100:.1f}%)")
        
        return results
```

### Phase 6: Configuration Management (Enhanced)

**File: `config.py`**

```python
# 🔥 IMPROVEMENT 4: Runtime performance optimization configuration
SVM_CONFIG = {
    'feature_extraction': {
        'epcFake': {
            'enable_entropy': True,
            'enable_ratios': True,
            'enable_date_features': True,
            'runtime_safe': {  # Lightweight features for real-time API
                'enable_entropy': False,  # Expensive calculation
                'enable_ratios': True,
                'enable_date_features': True
            }
        },
        'evtOrderErr': {
            'temporal_features': True,
            'sequence_depth': 10,
            'enable_entropy': True,
            'runtime_safe': {
                'temporal_features': False,  # Time interval calculations are slow
                'sequence_depth': 5,         # Reduce depth for speed
                'enable_entropy': False
            }
        },
        'locErr': {
            'hierarchy_features': True,
            'geographic_features': False,
            'enable_statistics': True,
            'runtime_safe': {
                'hierarchy_features': True,  # Fast hierarchy lookup
                'geographic_features': False,
                'enable_statistics': False   # Std dev calculations can be slow
            }
        },
        'epcDup': {
            'time_window_seconds': 1,
            'location_features': True,
            'clustering_features': True,
            'runtime_safe': {
                'time_window_seconds': 1,
                'location_features': True,
                'clustering_features': False  # Complex clustering is expensive
            }
        }, 
        'jump': {
            'statistical_features': True,
            'z_score_threshold': 2,
            'geographic_features': False,
            'runtime_safe': {
                'statistical_features': False,  # Statistical calculations are slow
                'z_score_threshold': 2,
                'geographic_features': False
            }
        }
    },
    'label_generation': {
        'threshold': 50,
        'class_balance': 'auto',
        'save_confidence_scores': True
    },
    'data_management': {
        'output_dir': 'data/svm_training',
        'save_metadata': True,
        'compress': False,
        'auto_train_test_split': True,
        'test_size': 0.2
    },
    'runtime': {
        'safe_mode': False,  # Set to True for real-time API usage
        'max_processing_time_ms': 1000,  # Alert if processing takes longer
        'enable_performance_logging': True
    }
}

def get_config(runtime_safe: bool = False):
    """Get configuration optimized for runtime or training"""
    config = SVM_CONFIG.copy()
    
    if runtime_safe:
        # Replace feature configs with runtime_safe versions
        for anomaly_type, settings in config['feature_extraction'].items():
            if 'runtime_safe' in settings:
                config['feature_extraction'][anomaly_type].update(settings['runtime_safe'])
        
        config['runtime']['safe_mode'] = True
    
    return config
```

### Phase 7: CLI Entry Point

**File: `scripts/run_pipeline.py`**

```python
#!/usr/bin/env python3
"""
🔥 IMPROVEMENT 5: CLI entry point for easy pipeline execution
Usage: python scripts/run_pipeline.py --input data/raw.csv --output data/svm_training
"""

import argparse
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from barcode.svm_preprocessing.pipeline import SVMPreprocessingPipeline
from barcode.svm_preprocessing.config import get_config

def main():
    parser = argparse.ArgumentParser(description='Run SVM preprocessing pipeline')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', default='data/svm_training', help='Output directory')
    parser.add_argument('--runtime-safe', action='store_true', 
                       help='Use runtime-safe features (faster, less accurate)')
    parser.add_argument('--threshold', type=int, default=50, 
                       help='Anomaly score threshold for labeling')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0.0-1.0)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        return 1
    
    print(f"🚀 Starting SVM preprocessing pipeline...")
    print(f"📁 Input: {args.input}")
    print(f"📁 Output: {args.output}")
    print(f"⚡ Runtime safe: {args.runtime_safe}")
    
    try:
        # Load data
        print("📊 Loading input data...")
        raw_df = pd.read_csv(args.input)
        print(f"   Loaded {len(raw_df)} records")
        
        # Configure pipeline
        config = get_config(runtime_safe=args.runtime_safe)
        config['data_management']['output_dir'] = args.output
        config['label_generation']['threshold'] = args.threshold
        config['data_management']['test_size'] = args.test_size
        
        # Run pipeline
        pipeline = SVMPreprocessingPipeline(config=config)
        results = pipeline.process_data(raw_df)
        
        # Summary
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Results summary:")
        for anomaly_type, (X, y) in results.items():
            positive_rate = (y.sum() / len(y) * 100) if len(y) > 0 else 0
            print(f"   {anomaly_type}: {X.shape[0]} samples, {X.shape[1]} features, "
                  f"{y.sum()} positives ({positive_rate:.1f}%)")
        
        print(f"\n📁 Data saved to: {args.output}")
        print(f"🔬 Ready for SVM training!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())
```

**File: `scripts/test_pipeline.py`**

```python
#!/usr/bin/env python3
"""Quick test harness for pipeline validation"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from barcode.svm_preprocessing.pipeline import SVMPreprocessingPipeline

def create_test_data():
    """Create synthetic test data"""
    test_data = []
    
    # Normal EPC
    test_data.append({
        'epc_code': '001.8804823.0000001.000001.20240701.000000001',
        'event_time': '2024-07-01 09:00:00',
        'scan_location': '화성공장',
        'event_type': 'Aggregation',
        'business_step': 'Factory'
    })
    
    # Fake EPC
    test_data.append({
        'epc_code': 'invalid.format.epc',
        'event_time': '2024-07-01 10:00:00', 
        'scan_location': '서울물류센터',
        'event_type': 'WMS_Inbound',
        'business_step': 'WMS'
    })
    
    # Duplicate scan (same EPC, same time, different location)
    test_data.extend([
        {
            'epc_code': '001.8804823.0000002.000001.20240701.000000002',
            'event_time': '2024-07-01 11:00:00',
            'scan_location': '부산공장',
            'event_type': 'Aggregation',
            'business_step': 'Factory'
        },
        {
            'epc_code': '001.8804823.0000002.000001.20240701.000000002',
            'event_time': '2024-07-01 11:00:00',  # Same time!
            'scan_location': '인천공장',           # Different location!
            'event_type': 'Aggregation',
            'business_step': 'Factory'
        }
    ])
    
    return pd.DataFrame(test_data)

def main():
    print("🧪 Running pipeline test...")
    
    # Create test data
    test_df = create_test_data()
    print(f"📊 Created {len(test_df)} test records")
    
    # Run pipeline
    pipeline = SVMPreprocessingPipeline()
    results = pipeline.process_data(test_df)
    
    # Validate results
    print("✅ Test results:")
    for anomaly_type, (X, y) in results.items():
        print(f"   {anomaly_type}: {X.shape[0]} samples, {X.shape[1]} features, {y.sum()} anomalies")
        
        # Basic validation
        assert X.shape[0] > 0, f"No samples for {anomaly_type}"
        assert X.shape[1] > 0, f"No features for {anomaly_type}"
        assert len(y) == X.shape[0], f"Feature/label mismatch for {anomaly_type}"
    
    print("🎉 All tests passed!")

if __name__ == '__main__':
    main()
```

## 🔥 Critical Improvements Added

Based on feedback from `docs/txt.txt`, the plan now includes **10 critical enhancements** (5 original + 5 additional SVM-specific):

### 1. **EPC-Feature Mapping** 🎯
- **Problem**: "누가 문제였는지 모르면 해석이 불가능함"
- **Solution**: Save `epc_codes.npy` alongside features for debugging
- **Files**: `data_manager.py`, `pipeline.py`

### 2. **Label Confidence Scores** 📊
- **Problem**: "신뢰도 없는 이진 분류는 위험함" (49점 vs 51점 구분 불가)
- **Solution**: Save original scores with binary labels: `(label, score)` tuples
- **Files**: `rule_based_labels.py`, `data_manager.py`

### 3. **Train/Test Split** 🧪
- **Problem**: "학습 잘 됐는지 평가할 방법 없음"
- **Solution**: Automatic `train_test_split()` with stratification
- **Files**: `data_manager.py`

### 4. **Runtime Performance** ⚡
- **Problem**: "실시간 시스템에서 죽을 수 있음" (무거운 feature 계산)
- **Solution**: `runtime_safe` config mode with lightweight features
- **Files**: `config.py`, all feature extractors

### 5. **CLI Entry Point** 🖥️
- **Problem**: "실제로 돌려보기 너무 불편함"
- **Solution**: `scripts/run_pipeline.py` for easy execution
- **Files**: `scripts/run_pipeline.py`, `scripts/test_pipeline.py`

## 🚨 Additional SVM-Specific Critical Issues

### 6. **Fixed Feature Dimensions** 🔧
- **Problem**: "각 Feature Extractor가 다른 길이의 벡터 반환" → SVM 훈련 불가능
- **Solution**: All extractors return fixed-length vectors with padding/truncation
- **Impact**: **🔴 필수** - Without this, SVM training will fail

### 7. **Class Imbalance Handling** ⚖️
- **Problem**: "epcFake: 정상 95% vs 이상 5%" → SVM이 항상 정상 예측
- **Solution**: SMOTE oversampling, class weights, threshold tuning
- **Impact**: **🔴 필수** - Critical for real-world performance

### 8. **Feature Normalization/Scaling** 📏
- **Problem**: "SVM은 feature scale에 매우 민감함" (0.001 vs 23847.5)
- **Solution**: StandardScaler/RobustScaler for all features
- **Impact**: **🔴 필수** - SVM performance degrades severely without this

### 9. **Memory Efficiency** 🧠
- **Problem**: "수백만 EPC × 5개 anomaly type = OOM" → 메모리 부족으로 크래시
- **Solution**: Batch processing with temporary files
- **Impact**: **🟡 성능** - Required for large datasets

### 10. **Inference Engine** 🔮
- **Problem**: "전처리 파이프라인만 있고, 실제 SVM 예측 연동 불명확"
- **Solution**: Complete train → predict workflow with model loading
- **Impact**: **🔴 필수** - Without this, trained models are unusable

## 🚀 Updated Implementation Priority

### **Critical Path (Must Do First)**
1. **Phase 1**: Base infrastructure (1-2 hours)
2. **Phase 2**: Enhanced feature extractors with FIXED dimensions (3-4 hours each) **🔴 CRITICAL**
3. **Phase 3**: Enhanced label generation with confidence (1.5 hours)
4. **Phase 4A**: SVM-specific components (3 hours) **🔴 CRITICAL**
   - Feature normalizer (ESSENTIAL for SVM)
   - Class imbalance handler (ESSENTIAL for real performance)
5. **Phase 4B**: Enhanced data management with SVM optimizations (2 hours)
6. **Phase 4C**: SVM inference engine (2 hours) **🔴 CRITICAL**
7. **Phase 4D**: Complete SVM workflow (1 hour)

### **Enhanced Features (High Value)**
8. **Phase 5**: Pipeline integration with all improvements (2 hours)
9. **Phase 6**: Runtime-optimized configuration (1 hour)
10. **Phase 7**: CLI scripts and testing (1 hour)
11. **Phase 8**: Memory optimization (batch processing) (2.5 hours) **🟡 For large datasets**

### **Total Effort**: ~20-25 hours for production-ready SVM system

## 📊 **Critical Dependencies**

- **Phases 2, 4A must be completed** before any SVM training will work
- **Phase 4C required** for any real-world predictions
- **Phase 8 only needed** for datasets with >100K EPCs

## ✅ Key Benefits

- **Modular**: Each anomaly type is independent
- **Reusable**: Leverages ALL existing rule-based logic  
- **Refinable**: Easy to modify individual components
- **Manageable**: Clear separation of concerns
- **Testable**: Each module can be tested independently
- **Scalable**: Easy to add new feature types or anomaly types

## 🔧 Usage Example

```python
# Simple usage
pipeline = SVMPreprocessingPipeline()
raw_data = pd.read_csv('data/raw/scan_data.csv')
training_data = pipeline.process_data(raw_data)

# Load for SVM training
X_epc, y_epc = pipeline.data_manager.load_training_data('epcFake')
X_dup, y_dup = pipeline.data_manager.load_training_data('epcDup')
# ... etc for all 5 types

# Train individual SVM models
from sklearn.svm import OneClassSVM
svm_epc = OneClassSVM().fit(X_epc[y_epc == 0])  # Train on normal samples
svm_dup = OneClassSVM().fit(X_dup[y_dup == 0])  # Train on normal samples
```

## 📋 Next Steps

1. Create the directory structure
2. Implement Phase 1 (base infrastructure)
3. Start with one feature extractor (recommend `epcFake` first - simplest)
4. Test with small dataset
5. Iterate and refine

**Ready to start implementation?**