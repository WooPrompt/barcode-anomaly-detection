# SVM Preprocessing for Barcode Anomaly Detection

This module provides a beginner-friendly, modular system for preprocessing barcode scan data for SVM-based anomaly detection.

## 📁 Directory Structure (Learning Path)

```
svm_preprocessing/
│
├── 01_core/                 # 🧠 Core Components (Start Here!)
│   ├── base_preprocessor.py     # Basic data cleaning and grouping
│   ├── sequence_processor.py    # Smart sequence handling
│   └── feature_normalizer.py    # Feature scaling for SVM
│
├── 02_features/             # 🔍 Feature Extraction (One per anomaly type)
│   ├── base_feature_extractor.py   # Common feature extraction patterns
│   ├── epc_fake_extractor.py       # EPC format validation features
│   ├── epc_dup_extractor.py        # Duplicate detection features
│   ├── event_order_extractor.py    # Event sequence features
│   ├── location_error_extractor.py # Location hierarchy features
│   └── time_jump_extractor.py      # Time anomaly features
│
├── 03_labels/               # 🏷️ Label Generation (Convert scores to training labels)
│   ├── base_label_generator.py     # Common labeling patterns
│   ├── rule_based_generator.py     # Use existing rule-based scores
│   └── threshold_optimizer.py      # Optimize classification thresholds
│
├── 04_data/                 # 💾 Data Management (Save/Load training data)
│   ├── data_manager.py             # Save and load processed data
│   ├── train_test_splitter.py      # Smart train/test splitting
│   └── batch_processor.py          # Memory-efficient processing
│
├── 05_pipeline/             # 🚀 Pipeline Assembly (Put it all together)
│   ├── preprocessing_pipeline.py   # Main processing pipeline
│   ├── config_manager.py           # Configuration management
│   └── pipeline_runner.py          # Easy-to-use pipeline runner
│
├── examples/                # 📖 Example Usage
│   ├── basic_usage.py              # Simple example
│   ├── advanced_usage.py           # Advanced features
│   └── custom_features.py          # How to add new features
│
└── tutorials/               # 🎓 Step-by-Step Tutorials
    ├── 01_understanding_features.md
    ├── 02_creating_labels.md
    ├── 03_training_data_prep.md
    └── 04_full_pipeline.md
```

## 🚀 Quick Start

### 1. Basic Usage (Beginner)
```python
from svm_preprocessing.examples.basic_usage import run_basic_preprocessing
result = run_basic_preprocessing("data/raw_scans.csv")
```

### 2. Step-by-Step (Learning)
```python
# Step 1: Clean and group data
from svm_preprocessing.01_core.base_preprocessor import BasePreprocessor
preprocessor = BasePreprocessor()
clean_data = preprocessor.process("data/raw_scans.csv")

# Step 2: Extract features for one anomaly type
from svm_preprocessing.02_features.epc_fake_extractor import EPCFakeExtractor
extractor = EPCFakeExtractor()
features = extractor.extract_features(clean_data)

# Step 3: Generate labels
from svm_preprocessing.03_labels.rule_based_generator import RuleBasedGenerator
generator = RuleBasedGenerator()
labels = generator.generate_labels(clean_data, "epcFake")

# Step 4: Save training data
from svm_preprocessing.04_data.data_manager import DataManager
manager = DataManager()
manager.save_training_data(features, labels, "epcFake")
```

### 3. Full Pipeline (Production)
```python
from svm_preprocessing.05_pipeline.pipeline_runner import PipelineRunner
runner = PipelineRunner()
results = runner.run_full_pipeline("data/raw_scans.csv", "output/svm_training")
```

## 🎯 Learning Objectives

1. **Understanding Data Flow**: Follow numbered directories (01→02→03→04→05)
2. **Modular Design**: Each component has a single responsibility
3. **Easy Testing**: Each module can be tested independently
4. **Gradual Complexity**: Start simple, add complexity as needed
5. **Real Examples**: Working code examples for every component

## 📚 Tutorial Path

1. Start with `tutorials/01_understanding_features.md`
2. Try `examples/basic_usage.py`
3. Read component documentation in order (01_core → 02_features → ...)
4. Experiment with `examples/custom_features.py`
5. Build your own pipeline using components

## 🔧 Key Features

- ✅ **Fixed Feature Dimensions**: Each anomaly type has consistent feature vector size
- ✅ **Class Imbalance Handling**: SMOTE and weighted approaches
- ✅ **Feature Normalization**: RobustScaler for SVM optimization
- ✅ **Memory Efficiency**: Batch processing for large datasets
- ✅ **EPC Tracking**: Maintain mapping between features and original EPC codes
- ✅ **Confidence Preservation**: Keep original rule-based scores alongside binary labels
- ✅ **Smart Sequence Processing**: Intelligent padding/truncation for sequences
- ✅ **Train/Test Splitting**: Automatic data splitting with stratification

## 🎓 For Beginners

If you're new to SVM or machine learning preprocessing:

1. **Start Here**: `examples/basic_usage.py`
2. **Learn Concepts**: `tutorials/01_understanding_features.md`
3. **Hands-On**: Follow the numbered directories in order
4. **Ask Questions**: Each file has detailed docstrings and comments

## 🚀 For Advanced Users

If you want to customize or extend the system:

1. **Configuration**: `05_pipeline/config_manager.py`
2. **Custom Features**: `examples/custom_features.py`
3. **Batch Processing**: `04_data/batch_processor.py`
4. **Pipeline Customization**: `05_pipeline/preprocessing_pipeline.py`

Happy Learning! 🎉