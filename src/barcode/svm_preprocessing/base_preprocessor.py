"""
Base Preprocessor with Intelligent Sequence Processing for SVM Training
Reuses existing functions from multi_anomaly_detector.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_anomaly_detector import preprocess_scan_data, load_csv_data
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import statistics


class SequenceProcessor:
    """Intelligent sequence length adjustment for SVM features"""
    
    def __init__(self):
        self.truncation_threshold = 0.03  # 3% 이상 자르면 위험
        self.max_sequence_length = 10  # 최대 시퀀스 길이
    
    def process_sequence(self, sequence: List[float], target_length: int, 
                        sequence_type: str = 'general') -> List[float]:
        """통계적 판단 기반 시퀀스 처리"""
        if len(sequence) == target_length:
            return sequence
        
        # 시퀀스가 더 짧은 경우 - 패딩
        if len(sequence) < target_length:
            return self._intelligent_padding(sequence, target_length, sequence_type)
        
        # 시퀀스가 더 긴 경우 - 잘라야 할지 판단
        truncation_ratio = (len(sequence) - target_length) / len(sequence)
        
        if truncation_ratio > self.truncation_threshold:
            # 3% 이상 자르면 위험 -> 보간 사용
            return self._interpolate_sequence(sequence, target_length)
        else:
            # 적게 자르는 경우 -> 끝에서 자르기
            return sequence[:target_length]
    
    def _intelligent_padding(self, sequence: List[float], target_length: int, 
                           sequence_type: str) -> List[float]:
        """상황별 패딩 전략 (무조건 0 패딩 금지)"""
        if not sequence:
            return [0.0] * target_length
        
        pad_count = target_length - len(sequence)
        
        if sequence_type == 'binary':
            # 이진 시퀀스의 경우 다수값으로 패딩
            pad_value = 1.0 if sequence.count(1.0) > sequence.count(0.0) else 0.0
        elif sequence_type == 'temporal':
            # 시간 연속성 유지를 위해 마지막 값으로 패딩
            pad_value = sequence[-1]
        elif sequence_type == 'statistical':
            # 통계적 특성 유지를 위해 평균값으로 패딩
            pad_value = statistics.mean(sequence)
        else:
            # 일반적인 경우 중앙값 사용
            pad_value = statistics.median(sequence)
        
        return sequence + [pad_value] * pad_count
    
    def _interpolate_sequence(self, sequence: List[float], target_length: int) -> List[float]:
        """시퀀스를 보간하여 target_length로 축소"""
        if target_length >= len(sequence):
            return sequence
        
        # 선형 보간을 위한 인덱스 계산
        step = (len(sequence) - 1) / (target_length - 1)
        interpolated = []
        
        for i in range(target_length):
            if i == target_length - 1:
                interpolated.append(sequence[-1])
            else:
                idx = i * step
                lower_idx = int(idx)
                upper_idx = min(lower_idx + 1, len(sequence) - 1)
                
                if lower_idx == upper_idx:
                    interpolated.append(sequence[lower_idx])
                else:
                    # 선형 보간
                    weight = idx - lower_idx
                    value = sequence[lower_idx] * (1 - weight) + sequence[upper_idx] * weight
                    interpolated.append(value)
        
        return interpolated
    
    def analyze_sequence_statistics(self, sequences: List[List[float]]) -> Dict[str, Any]:
        """시퀀스들의 통계적 특성 분석"""
        lengths = [len(seq) for seq in sequences]
        
        return {
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': statistics.mean(lengths),
            'median_length': statistics.median(lengths),
            'std_length': statistics.stdev(lengths) if len(lengths) > 1 else 0,
            'length_distribution': {
                'q25': np.percentile(lengths, 25),
                'q75': np.percentile(lengths, 75),
                'q90': np.percentile(lengths, 90),
                'q95': np.percentile(lengths, 95)
            }
        }


class BasePreprocessor:
    """Base preprocessing class that reuses existing logic"""
    
    def __init__(self):
        # REUSE: lines 572-604 (load_csv_data function)
        self.geo_df, self.transition_stats, self.location_mapping = load_csv_data()
        self.sequence_processor = SequenceProcessor()
    
    def preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """REUSE: lines 364-390 (preprocess_scan_data function)"""
        return preprocess_scan_data(raw_df)
    
    def get_epc_groups(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Group by EPC for sequence analysis"""
        return {epc: group.sort_values('event_time').reset_index(drop=True) 
                for epc, group in df.groupby('epc_code')}
    
    def analyze_dataset_sequences(self, df: pd.DataFrame) -> Dict[str, Any]:
        """데이터셋의 시퀀스 특성 분석"""
        epc_groups = self.get_epc_groups(df)
        
        # 각 EPC별 이벤트 개수 수집
        event_counts = [len(group) for group in epc_groups.values()]
        
        return self.sequence_processor.analyze_sequence_statistics([
            list(range(count)) for count in event_counts
        ])
    
    def get_reference_data(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Reference data accessor"""
        return self.geo_df, self.transition_stats, self.location_mapping


class FeatureNormalizer:
    """SVM용 feature 정규화"""
    
    def __init__(self, method: str = 'robust'):
        self.method = method
        self.scalers = {}
    
    def fit_transform_features(self, X: np.ndarray, anomaly_type: str) -> np.ndarray:
        """Feature 정규화 적용 및 scaler 저장"""
        if self.method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X)
        self.scalers[anomaly_type] = scaler
        return X_scaled
    
    def transform_features(self, X: np.ndarray, anomaly_type: str) -> np.ndarray:
        """예측시 사용 (이미 fit된 scaler 사용)"""
        if anomaly_type not in self.scalers:
            raise ValueError(f"Scaler for {anomaly_type} not fitted yet")
        return self.scalers[anomaly_type].transform(X)
    
    def get_scaler(self, anomaly_type: str):
        """Scaler 객체 반환"""
        return self.scalers.get(anomaly_type)


class ImbalanceHandler:
    """클래스 불균형 해결을 위한 전략들"""
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray, 
                        strategy: str = 'smote', min_samples: int = 50) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """클래스 불균형 처리"""
        
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        # 최소 샘플 수 체크
        if min(counts) < min_samples:
            strategy = 'weighted'  # SMOTE 불가시 weighted로 폴백
        
        if strategy == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42, k_neighbors=min(min(counts)-1, 5))
                X_resampled, y_resampled = smote.fit_resample(X, y)
                return X_resampled, y_resampled, {'method': 'smote', 'original_distribution': class_distribution}
            except Exception as e:
                # SMOTE 실패시 weighted로 폴백
                strategy = 'weighted'
        
        if strategy == 'weighted':
            # class_weight 계산
            pos_weight = len(y) / (2 * np.sum(y)) if np.sum(y) > 0 else 1.0
            neg_weight = len(y) / (2 * (len(y) - np.sum(y))) if np.sum(y) < len(y) else 1.0
            class_weights = {0: neg_weight, 1: pos_weight}
            
            return X, y, {'method': 'weighted', 'class_weights': class_weights, 'original_distribution': class_distribution}
        
        elif strategy == 'threshold_tuning':
            # 다양한 임계값 테스트를 위한 메타데이터
            return X, y, {'method': 'threshold_tuning', 'thresholds': [30, 40, 50, 60, 70], 'original_distribution': class_distribution}
        
        else:
            return X, y, {'method': 'none', 'original_distribution': class_distribution}