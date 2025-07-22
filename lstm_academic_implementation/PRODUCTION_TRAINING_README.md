# 🏭 Production LSTM Training Guide - ML Engineering Perspective

**Target Audience**: ML Engineers, Data Scientists, MLOps Engineers  
**Training Time**: 2-7 days for production-grade models  
**Hardware Requirements**: GPU with ≥16GB VRAM, 64GB+ system RAM

---

## 🎯 **Why Production Training Takes Days, Not Hours**

### **Training Complexity Factors**

| Factor | Demo Training | **Production Training** | Impact on Duration |
|--------|---------------|------------------------|-------------------|
| **Dataset Size** | 1K sequences | **10M+ sequences** | 1000x longer |
| **Model Size** | 32 hidden units | **512+ hidden units** | 16x more parameters |
| **Batch Size** | 32 samples | **256-1024 samples** | Memory-bound training |
| **Hyperparameter Search** | None | **Bayesian optimization** | 50-100 trials |
| **Cross-validation** | Single split | **5-fold CV** | 5x training iterations |
| **Early Stopping Patience** | 10 epochs | **50+ epochs** | Conservative stopping |

### **Production Training Pipeline Duration**

```
Total Production Training Time: 2-7 days
├── Data Preparation: 4-8 hours
├── Hyperparameter Optimization: 1-3 days  
├── Final Model Training: 12-24 hours
├── Model Validation: 4-8 hours
└── Production Testing: 4-8 hours
```

---

## 🔬 **Phase 1: Data Preparation (4-8 hours)**

### **Large-Scale Data Processing**

```python
# Real production data volumes
PRODUCTION_CONFIG = {
    'expected_data_size': '10M+ barcode scans',
    'unique_epcs': '1M+ unique EPCs', 
    'time_range': '2+ years historical data',
    'processing_time': '4-8 hours on 64-core machine'
}
```

### **Advanced Feature Engineering Pipeline**

```python
# production_data_prep.py
class ProductionDataPreprocessor:
    """
    Production-scale data preprocessing with distributed computing
    
    Key ML Engineering Concepts:
    - Distributed computing: Process data across multiple cores/machines
    - Memory mapping: Handle datasets larger than RAM
    - Feature stores: Cache computed features for reuse
    - Data lineage: Track feature transformations for reproducibility
    """
    
    def __init__(self):
        self.dask_client = None  # Distributed computing framework
        self.feature_store = None  # MLflow or Feast for feature caching
        self.data_validator = None  # Great Expectations for data quality
    
    def distributed_feature_engineering(self, data_partitions):
        """
        Distributed feature engineering using Dask
        
        ML Engineering Concepts:
        - Partitioning: Split data across workers for parallel processing
        - Map-reduce: Apply transformations in parallel, then combine
        - Lazy evaluation: Build computation graph before execution
        """
        
        # Temporal features with distributed computing
        temporal_features = data_partitions.map_partitions(
            self.extract_temporal_features,
            meta=('temporal', 'f8')
        )
        
        # Spatial features with geospatial computations
        spatial_features = data_partitions.map_partitions(
            self.extract_spatial_features,
            meta=('spatial', 'f8')  
        )
        
        # Behavioral features with statistical aggregations
        behavioral_features = data_partitions.groupby('epc_code').apply(
            self.extract_behavioral_features,
            meta=('behavioral', 'f8')
        )
        
        return temporal_features, spatial_features, behavioral_features
    
    def advanced_feature_selection(self, features, targets):
        """
        Production feature selection with multiple algorithms
        
        ML Engineering Concepts:
        - mRMR: Minimum Redundancy Maximum Relevance
        - Recursive Feature Elimination: Iteratively remove features
        - SHAP-based selection: Use explainability for feature importance
        - Statistical tests: Chi-square, mutual information
        """
        
        from sklearn.feature_selection import (
            SelectKBest, RFE, mutual_info_classif
        )
        from mrmr import mrmr_classif
        
        # Multiple feature selection methods
        selectors = {
            'mrmr': mrmr_classif,
            'mutual_info': SelectKBest(mutual_info_classif, k=50),
            'rfe': RFE(estimator=RandomForestClassifier(), n_features_to_select=50)
        }
        
        selected_features = {}
        for name, selector in selectors.items():
            selected_features[name] = selector.fit_transform(features, targets)
        
        # Ensemble feature selection (voting)
        return self.ensemble_feature_selection(selected_features)
```

### **Production Data Quality Validation**

```python
# Data quality checks with Great Expectations
def validate_production_data_quality(data):
    """
    Production data quality validation
    
    ML Engineering Concepts:
    - Data drift detection: Distribution changes over time
    - Schema validation: Ensure consistent data structure  
    - Completeness checks: Missing value analysis
    - Consistency checks: Cross-field validation rules
    """
    
    import great_expectations as ge
    
    # Create expectation suite
    suite = ge.DataContext().create_expectation_suite('barcode_data_v1')
    
    # Core data quality expectations
    expectations = [
        # Completeness expectations
        suite.expect_column_to_exist('epc_code'),
        suite.expect_column_values_to_not_be_null('epc_code'),
        
        # Format expectations  
        suite.expect_column_values_to_match_regex(
            'epc_code', 
            r'^[0-9]{3}\.[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'
        ),
        
        # Range expectations
        suite.expect_column_values_to_be_between('time_gap_seconds', 0, 86400),
        
        # Distribution expectations
        suite.expect_column_mean_to_be_between('location_entropy', 0, 4),
        
        # Uniqueness expectations
        suite.expect_column_pair_values_a_to_be_greater_than_b(
            'event_time_max', 'event_time_min'
        )
    ]
    
    # Validate against production data
    validation_results = suite.validate(data)
    
    if not validation_results.success:
        raise DataQualityError(f"Data quality validation failed: {validation_results}")
    
    return validation_results
```

---

## 🧠 **Phase 2: Hyperparameter Optimization (1-3 days)**

### **Bayesian Hyperparameter Search**

```python
# hyperparameter_optimization.py
import optuna
from optuna.integration import PyTorchLightningPruningCallback

class ProductionHyperparameterOptimizer:
    """
    Production hyperparameter optimization with Optuna
    
    ML Engineering Concepts:
    - Bayesian optimization: Use Gaussian processes to guide search
    - Multi-objective optimization: Balance accuracy vs latency vs memory
    - Pruning: Early stopping for unpromising trials
    - Distributed search: Parallel hyperparameter trials
    """
    
    def __init__(self, n_trials=100, n_jobs=8):
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        
        # Create distributed study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            storage='postgresql://localhost/optuna_studies'  # Distributed storage
        )
    
    def objective(self, trial):
        """
        Hyperparameter optimization objective function
        
        Search Space:
        - Architecture: hidden_size, num_layers, attention_heads
        - Regularization: dropout, weight_decay, label_smoothing
        - Training: learning_rate, batch_size, scheduler_type
        - Loss function: focal_loss_alpha, focal_loss_gamma
        """
        
        # Suggest hyperparameters
        params = {
            # Architecture hyperparameters
            'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512, 768]),
            'num_layers': trial.suggest_int('num_layers', 2, 6),
            'attention_heads': trial.suggest_categorical('attention_heads', [4, 8, 16]),
            
            # Regularization hyperparameters
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-2),
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2),
            
            # Training hyperparameters
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512]),
            'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'onecycle', 'plateau']),
            
            # Loss function hyperparameters
            'focal_alpha': trial.suggest_float('focal_alpha', 0.1, 0.5),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
        }
        
        # Train model with suggested hyperparameters
        val_auc = self.train_and_validate(params, trial)
        
        return val_auc
    
    def multi_objective_optimization(self):
        """
        Multi-objective optimization balancing accuracy, latency, and memory
        
        ML Engineering Concepts:
        - Pareto optimization: Find trade-offs between competing objectives
        - Model efficiency: Balance accuracy vs computational cost
        - Production constraints: Memory and latency budgets
        """
        
        def multi_objective(trial):
            params = self.suggest_hyperparameters(trial)
            
            # Train model
            val_auc, inference_latency, model_size = self.train_and_profile(params)
            
            # Return multiple objectives
            return val_auc, -inference_latency, -model_size  # Maximize AUC, minimize latency/size
        
        # Multi-objective study
        study = optuna.create_study(
            directions=['maximize', 'maximize', 'maximize'],  # AUC up, latency down, size down
            sampler=optuna.samplers.NSGAIISampler()
        )
        
        study.optimize(multi_objective, n_trials=self.n_trials)
        
        # Get Pareto-optimal solutions
        pareto_front = study.best_trials
        
        return pareto_front
```

### **Advanced Cross-Validation Strategy**

```python
# production_cross_validation.py
class ProductionCrossValidator:
    """
    Production cross-validation with temporal and spatial awareness
    
    ML Engineering Concepts:
    - Temporal cross-validation: Respect time ordering in validation
    - Spatial cross-validation: Prevent spatial leakage across locations
    - Nested CV: Hyperparameter search within cross-validation
    - Stratified sampling: Maintain class distribution across folds
    """
    
    def temporal_block_cv(self, data, n_splits=5):
        """
        Temporal block cross-validation for time series data
        
        Prevents future information leakage by ensuring training data
        always comes before validation data in time.
        """
        
        # Sort by time
        data_sorted = data.sort_values('event_time')
        
        # Create temporal blocks
        block_size = len(data_sorted) // n_splits
        
        for i in range(n_splits):
            # Training: all data before current block
            train_end = i * block_size
            train_indices = data_sorted.index[:train_end]
            
            # Validation: current block
            val_start = train_end
            val_end = (i + 1) * block_size
            val_indices = data_sorted.index[val_start:val_end]
            
            yield train_indices, val_indices
    
    def epc_aware_stratified_cv(self, data, n_splits=5):
        """
        EPC-aware stratified cross-validation
        
        Ensures no EPC appears in both train and validation sets
        while maintaining anomaly class distribution.
        """
        
        from sklearn.model_selection import StratifiedKFold
        
        # Group by EPC and calculate anomaly rates
        epc_groups = data.groupby('epc_code').agg({
            'epcFake': 'max',
            'epcDup': 'max', 
            'locErr': 'max',
            'evtOrderErr': 'max',
            'jump': 'max'
        }).reset_index()
        
        # Create stratification labels (multi-label to single label)
        epc_groups['anomaly_pattern'] = (
            epc_groups['epcFake'].astype(str) +
            epc_groups['epcDup'].astype(str) +
            epc_groups['locErr'].astype(str) +
            epc_groups['evtOrderErr'].astype(str) +
            epc_groups['jump'].astype(str)
        )
        
        # Stratified split on EPCs
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_epc_idx, val_epc_idx in skf.split(
            epc_groups['epc_code'], 
            epc_groups['anomaly_pattern']
        ):
            train_epcs = epc_groups.iloc[train_epc_idx]['epc_code']
            val_epcs = epc_groups.iloc[val_epc_idx]['epc_code']
            
            train_indices = data[data['epc_code'].isin(train_epcs)].index
            val_indices = data[data['epc_code'].isin(val_epcs)].index
            
            yield train_indices, val_indices
```

---

## 🚀 **Phase 3: Production Model Training (12-24 hours)**

### **Distributed Training Setup**

```python
# distributed_training.py
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

class ProductionLSTMTrainer:
    """
    Production LSTM trainer with distributed training support
    
    ML Engineering Concepts:
    - Data parallelism: Split batches across multiple GPUs
    - Model parallelism: Split large models across GPUs  
    - Gradient accumulation: Simulate larger batch sizes
    - Mixed precision: Use FP16 for memory efficiency
    - Gradient clipping: Prevent exploding gradients
    """
    
    def __init__(self, world_size=4, backend='nccl'):
        self.world_size = world_size  # Number of GPUs
        self.backend = backend  # Communication backend
        
    def setup_distributed_training(self, rank, world_size):
        """Setup distributed training environment"""
        
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        
        # Initialize process group
        dist.init_process_group(
            backend=self.backend,
            rank=rank,
            world_size=world_size
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
        
    def create_distributed_model(self, model, rank):
        """Wrap model for distributed training"""
        
        # Move model to GPU
        model = model.to(rank)
        
        # Wrap with DistributedDataParallel
        ddp_model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=True  # For complex architectures
        )
        
        return ddp_model
    
    def train_with_mixed_precision(self, model, train_loader, optimizer, scaler):
        """
        Training loop with automatic mixed precision
        
        ML Engineering Concepts:
        - FP16 training: Use half-precision for memory efficiency
        - Loss scaling: Prevent gradient underflow in FP16
        - Gradient accumulation: Simulate larger effective batch sizes
        """
        
        model.train()
        total_loss = 0.0
        accumulation_steps = 4  # Gradient accumulation
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                predictions, attention_weights = model(sequences)
                loss = self.criterion(predictions, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
        
        return total_loss / len(train_loader)
```

### **Advanced Training Techniques**

```python
# advanced_training_techniques.py
class AdvancedTrainingTechniques:
    """
    Advanced ML training techniques for production models
    
    ML Engineering Concepts:
    - Curriculum learning: Start with easy examples, progress to hard
    - Self-supervised pre-training: Learn representations without labels
    - Knowledge distillation: Transfer knowledge from large to small models
    - Adversarial training: Improve robustness to input perturbations
    """
    
    def curriculum_learning(self, train_loader, model, difficulty_scorer):
        """
        Curriculum learning: gradually increase training difficulty
        
        Start training with easier examples (short sequences, clear patterns)
        and gradually introduce harder examples (long sequences, ambiguous patterns).
        """
        
        # Score difficulty of each sample
        difficulties = []
        for batch in train_loader:
            sequences, labels = batch
            difficulty_scores = difficulty_scorer(sequences, labels)
            difficulties.extend(difficulty_scores)
        
        # Sort by difficulty (easy to hard)
        sorted_indices = np.argsort(difficulties)
        
        # Create curriculum schedule
        curriculum_schedule = {
            'epochs_1_10': sorted_indices[:len(sorted_indices)//4],    # 25% easiest
            'epochs_11_20': sorted_indices[:len(sorted_indices)//2],   # 50% easiest  
            'epochs_21_30': sorted_indices[:3*len(sorted_indices)//4], # 75% easiest
            'epochs_31_plus': sorted_indices                           # All data
        }
        
        return curriculum_schedule
    
    def self_supervised_pretraining(self, unlabeled_sequences):
        """
        Self-supervised pre-training for better representations
        
        Pre-train the model on unlabeled data using reconstruction tasks
        before fine-tuning on the anomaly detection task.
        """
        
        # Masked sequence modeling (like BERT for sequences)
        def mask_sequence_task(sequences):
            masked_sequences = sequences.clone()
            mask_prob = 0.15
            
            # Randomly mask time steps
            mask = torch.rand(sequences.shape[:2]) < mask_prob
            masked_sequences[mask] = 0  # Mask with zeros
            
            return masked_sequences, mask
        
        # Contrastive learning task
        def contrastive_task(sequences):
            # Create augmented versions of sequences
            aug1 = self.augment_sequence(sequences, noise_level=0.1)
            aug2 = self.augment_sequence(sequences, noise_level=0.1)
            
            return aug1, aug2
        
        # Pre-training loop
        for epoch in range(50):  # Pre-train for 50 epochs
            for batch in unlabeled_sequences:
                # Masked modeling task
                masked_seq, mask = mask_sequence_task(batch)
                reconstructed = model(masked_seq)
                reconstruction_loss = F.mse_loss(reconstructed[mask], batch[mask])
                
                # Contrastive task  
                aug1, aug2 = contrastive_task(batch)
                repr1 = model.get_representation(aug1)
                repr2 = model.get_representation(aug2)
                contrastive_loss = self.contrastive_loss(repr1, repr2)
                
                # Combined pre-training loss
                pretrain_loss = reconstruction_loss + 0.1 * contrastive_loss
                
                # Backward pass
                pretrain_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
    def knowledge_distillation(self, student_model, teacher_model, train_loader):
        """
        Knowledge distillation: train smaller model using larger model's knowledge
        
        ML Engineering Concepts:
        - Teacher-student paradigm: Large accurate model teaches small fast model
        - Soft targets: Use teacher's probability distributions, not just predictions
        - Temperature scaling: Soften probability distributions for better knowledge transfer
        """
        
        def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0, alpha=0.5):
            # Soft targets from teacher
            teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
            student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
            
            # Distillation loss (KL divergence)
            distill_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
            distill_loss *= (temperature ** 2)
            
            # Standard loss on true labels
            standard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
            
            # Combined loss
            return alpha * distill_loss + (1 - alpha) * standard_loss
        
        # Training loop
        teacher_model.eval()  # Teacher in eval mode
        student_model.train()  # Student in train mode
        
        for batch in train_loader:
            sequences, labels = batch
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits, _ = teacher_model(sequences)
            
            # Get student predictions
            student_logits, _ = student_model(sequences)
            
            # Calculate distillation loss
            loss = distillation_loss(student_logits, teacher_logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
```

---

## 📊 **Phase 4: Model Validation & Testing (4-8 hours)**

### **Comprehensive Model Evaluation**

```python
# production_model_evaluation.py
class ProductionModelEvaluator:
    """
    Production model evaluation with statistical rigor
    
    ML Engineering Concepts:
    - Statistical significance testing: Ensure results aren't due to chance
    - Confidence intervals: Quantify uncertainty in performance estimates
    - Power analysis: Determine minimum effect sizes we can detect
    - Multiple comparison correction: Adjust p-values for multiple tests
    """
    
    def bootstrap_confidence_intervals(self, y_true, y_pred, metric_func, n_bootstrap=1000):
        """
        Bootstrap confidence intervals for performance metrics
        
        Provides uncertainty estimates for model performance by resampling
        the test set many times and computing metric distributions.
        """
        
        bootstrap_scores = []
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap resample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metric on bootstrap sample
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        
        # Calculate confidence intervals
        ci_lower = np.percentile(bootstrap_scores, 2.5)
        ci_upper = np.percentile(bootstrap_scores, 97.5)
        mean_score = np.mean(bootstrap_scores)
        
        return mean_score, ci_lower, ci_upper
    
    def statistical_significance_testing(self, model_a_scores, model_b_scores):
        """
        Statistical significance testing between two models
        
        Uses paired t-test with multiple comparison correction
        to determine if performance differences are statistically significant.
        """
        
        from scipy import stats
        from statsmodels.stats.multitest import multipletests
        
        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(model_a_scores, model_b_scores)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(model_a_scores) + np.var(model_b_scores)) / 2)
        cohens_d = (np.mean(model_a_scores) - np.mean(model_b_scores)) / pooled_std
        
        # Power analysis
        from statsmodels.stats.power import ttest_power
        power = ttest_power(cohens_d, len(model_a_scores), alpha=0.05, alternative='two-sided')
        
        return {
            't_statistic': t_statistic,
            'p_value': p_value,
            'cohens_d': cohens_d,
            'statistical_power': power,
            'significant': p_value < 0.05,
            'interpretation': self._interpret_effect_size(cohens_d)
        }
    
    def cross_validation_with_confidence(self, model, X, y, cv_folds=5):
        """
        Cross-validation with statistical confidence assessment
        
        Performs k-fold cross-validation and provides confidence intervals
        and significance tests for the performance estimates.
        """
        
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import roc_auc_score, make_scorer
        
        # Define scoring function
        auc_scorer = make_scorer(roc_auc_score, average='macro', multi_class='ovr')
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=auc_scorer)
        
        # Statistical analysis
        mean_score = np.mean(cv_scores)
        std_score = np.std(cv_scores)
        se_score = std_score / np.sqrt(cv_folds)  # Standard error
        
        # 95% confidence interval
        ci_lower = mean_score - 1.96 * se_score
        ci_upper = mean_score + 1.96 * se_score
        
        # One-sample t-test against null hypothesis (AUC = 0.5)
        t_stat, p_value = stats.ttest_1samp(cv_scores, 0.5)
        
        return {
            'mean_cv_score': mean_score,
            'std_cv_score': std_score,
            'confidence_interval_95': (ci_lower, ci_upper),
            'individual_fold_scores': cv_scores.tolist(),
            't_statistic_vs_random': t_stat,
            'p_value_vs_random': p_value,
            'significantly_better_than_random': p_value < 0.05
        }
```

---

## 🏭 **Phase 5: Production Deployment Pipeline (4-8 hours)**

### **Model Serving Infrastructure**

```python
# production_model_serving.py
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from prometheus_client import Counter, Histogram, generate_latest

class ProductionModelServer:
    """
    Production model serving with monitoring and observability
    
    ML Engineering Concepts:
    - Model serving: Deploy models as REST APIs
    - Load balancing: Distribute requests across multiple model instances
    - Monitoring: Track latency, throughput, error rates
    - A/B testing: Compare different model versions
    - Canary deployment: Gradual rollout of new models
    """
    
    def __init__(self):
        self.app = FastAPI(title="LSTM Anomaly Detection API")
        self.model = None
        
        # Monitoring metrics
        self.request_count = Counter('model_requests_total', 'Total model requests')
        self.request_duration = Histogram('model_request_duration_seconds', 'Request duration')
        self.error_count = Counter('model_errors_total', 'Total model errors')
        
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes with monitoring"""
        
        @self.app.post("/predict")
        @self.request_duration.time()
        async def predict(self, request: PredictionRequest):
            """Main prediction endpoint with monitoring"""
            
            self.request_count.inc()
            
            try:
                # Input validation
                self.validate_input(request)
                
                # Model inference
                prediction = await self.model_inference(request)
                
                # Output validation
                self.validate_output(prediction)
                
                return prediction
                
            except Exception as e:
                self.error_count.inc()
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check(self):
            """Health check endpoint"""
            
            return {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "version": "1.0.0"
            }
        
        @self.app.get("/metrics")
        async def metrics(self):
            """Prometheus metrics endpoint"""
            
            return generate_latest()
    
    async def model_inference(self, request):
        """Async model inference for high throughput"""
        
        # Convert request to model input
        sequences = self.preprocess_request(request)
        
        # Model inference (non-blocking)
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            predictions = await loop.run_in_executor(
                None, 
                self.model.forward, 
                sequences
            )
        
        # Convert predictions to response format
        response = self.postprocess_predictions(predictions)
        
        return response

class ModelVersionManager:
    """
    Model version management for A/B testing and canary deployments
    
    ML Engineering Concepts:
    - Model registry: Centralized storage for model versions
    - Feature flags: Control which model version is served
    - Traffic splitting: Route percentage of traffic to different models
    - Performance monitoring: Compare model versions in production
    """
    
    def __init__(self):
        self.models = {}  # Version -> Model mapping
        self.traffic_split = {}  # Version -> Traffic percentage
        self.performance_metrics = {}  # Version -> Metrics
    
    def register_model_version(self, version, model_path, traffic_percentage=0):
        """Register new model version"""
        
        # Load model
        model = torch.jit.load(model_path)
        model.eval()
        
        # Register in version manager
        self.models[version] = model
        self.traffic_split[version] = traffic_percentage
        
        logger.info(f"Registered model version {version} with {traffic_percentage}% traffic")
    
    def route_request(self, request_id):
        """Route request to appropriate model version based on traffic split"""
        
        import random
        
        # Deterministic routing based on request ID
        random.seed(hash(request_id))
        routing_value = random.random()
        
        cumulative_percentage = 0
        for version, percentage in self.traffic_split.items():
            cumulative_percentage += percentage / 100.0
            if routing_value <= cumulative_percentage:
                return version
        
        # Default to latest version
        return max(self.models.keys())
    
    def canary_deployment(self, new_version, canary_percentage=5):
        """Gradual canary deployment of new model version"""
        
        # Start with small percentage
        self.traffic_split[new_version] = canary_percentage
        
        # Monitor performance for specified duration
        asyncio.create_task(self.monitor_canary(new_version, duration_hours=2))
    
    async def monitor_canary(self, version, duration_hours):
        """Monitor canary deployment and auto-rollback if needed"""
        
        start_time = time.time()
        
        while time.time() - start_time < duration_hours * 3600:
            # Check performance metrics
            metrics = self.get_version_metrics(version)
            
            # Auto-rollback conditions
            if (metrics['error_rate'] > 0.05 or  # >5% error rate
                metrics['p95_latency'] > 100 or   # >100ms p95 latency
                metrics['auc_score'] < 0.8):      # <0.8 AUC score
                
                logger.warning(f"Auto-rollback triggered for version {version}")
                self.traffic_split[version] = 0
                break
            
            await asyncio.sleep(300)  # Check every 5 minutes
        
        # If monitoring passed, increase traffic
        if self.traffic_split[version] > 0:
            self.traffic_split[version] = min(50, self.traffic_split[version] * 2)
```

---

## 📈 **Production Training Configuration Example**

```python
# production_training_config.py
PRODUCTION_TRAINING_CONFIG = {
    # Data configuration
    'data': {
        'expected_size_gb': 50,
        'num_epcs': 1000000,
        'num_events': 10000000,
        'time_range_months': 24,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15
    },
    
    # Model architecture
    'model': {
        'hidden_size': 512,
        'num_layers': 4,
        'attention_heads': 16,
        'dropout': 0.2,
        'num_classes': 5,
        'sequence_length': 25
    },
    
    # Training configuration
    'training': {
        'batch_size': 256,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'max_epochs': 200,
        'patience': 50,
        'gradient_clip_norm': 1.0,
        'label_smoothing': 0.1,
        'mixed_precision': True
    },
    
    # Hyperparameter search
    'hyperparameter_search': {
        'n_trials': 100,
        'timeout_hours': 72,
        'pruning_enabled': True,
        'distributed_search': True
    },
    
    # Infrastructure
    'infrastructure': {
        'num_gpus': 4,
        'gpu_memory_gb': 32,
        'system_memory_gb': 128,
        'num_cpu_cores': 64,
        'storage_type': 'nvme_ssd'
    },
    
    # Expected timeline
    'timeline': {
        'data_preparation_hours': 6,
        'hyperparameter_search_hours': 72,
        'final_training_hours': 18,
        'validation_hours': 6,
        'deployment_hours': 4,
        'total_days': 4.5
    }
}
```

---

## 🎯 **Key Production Training Differences**

| Aspect | Research/Demo | **Production Training** |
|--------|---------------|------------------------|
| **Dataset Size** | 1K-10K samples | **10M+ samples** |
| **Training Time** | 30 minutes | **2-7 days** |
| **Model Complexity** | 100K parameters | **10M+ parameters** |
| **Validation Strategy** | Single holdout | **5-fold CV + temporal validation** |
| **Hyperparameter Search** | Manual tuning | **Bayesian optimization (100+ trials)** |
| **Infrastructure** | Single GPU | **Multi-GPU distributed training** |
| **Monitoring** | Basic metrics | **Production monitoring + A/B testing** |
| **Robustness Testing** | None | **Adversarial + noise robustness** |
| **Deployment** | Local script | **Containerized microservice** |

The key insight is that **production ML engineering is fundamentally different** from research or demo code. The complexity comes from:

1. **Scale**: 1000x more data requiring distributed processing
2. **Robustness**: Must handle real-world edge cases and adversarial inputs  
3. **Performance**: Sub-10ms inference with 99.9% uptime requirements
4. **Monitoring**: Full observability with automatic rollback capabilities
5. **Compliance**: Data privacy, model governance, and audit trails

This is why production ML training takes days/weeks, not hours!