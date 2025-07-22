# 🚀 LSTM Barcode Anomaly Detection - User Guide

**Think of this as a Smart Security Guard for Your Supply Chain**

This system is like having a super-smart security guard that watches millions of barcode scans and instantly spots when something fishy is happening - like fake products, shipping errors, or impossible logistics jumps.

---

## 🤔 What Does This System Do?

Imagine you're running a huge warehouse where millions of products with barcodes move around every day. Sometimes bad things happen:

- **Fake Products** sneak into your supply chain 
- **Duplicate Scans** happen by mistake
- **Products go to wrong locations** 
- **Events happen in wrong order** (like a product being "delivered" before it was "shipped")
- **Impossible jumps** happen (like a product teleporting 1000 miles in 5 minutes)

Our AI system watches all these barcode scans and **automatically catches these problems in real-time** - faster than any human could!

---

## 📋 What You Need Before Starting

### Step 1: Check Your Computer
You need a reasonably modern computer. Think of it like checking if your car can handle a road trip:

- **Windows 10/11, Mac, or Linux** (like checking you have a car)
- **At least 8GB of memory** (like making sure your car has enough gas)
- **Some free disk space (at least 10GB)** (like having room for luggage)

### Step 2: Install Python
Python is like the "engine" that runs our AI system:

1. Go to [python.org](https://python.org)
2. Download Python 3.11 or newer
3. Install it (just click "Next" through the installer)
4. **Important:** Check the box that says "Add Python to PATH" during installation

### Step 3: Install the AI Tools
Open your command prompt (like opening the hood of your car) and type these commands one by one:

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib seaborn
pip install statsmodels scipy psutil
```

**What these do:**
- `torch` = The main AI brain
- `pandas` = Helps organize data like Excel but for AI
- `scikit-learn` = Basic AI tools 
- `matplotlib` = Makes pretty charts
- `statsmodels` = Advanced statistics tools

---

## 🗂️ Setting Up Your Data

### What Kind of Data Do You Need?

Your data should be a CSV file (like Excel) with these columns:

| Column Name | What It Means | Example |
|-------------|---------------|---------|
| `epc_code` | Unique barcode ID | `001.8804823.1293291.010001.20250722.000001` |
| `event_time` | When scan happened | `2025-07-22 10:30:15` |
| `location_id` | Where it was scanned | `WAREHOUSE_A_DOCK_3` |
| `business_step` | What stage in supply chain | `Factory`, `WMS`, `Retail` |
| `scan_location` | Specific scanner | `SCANNER_001` |
| `event_type` | Type of scan | `Observation`, `Aggregation` |
| `operator_id` | Who did the scan | `EMPLOYEE_123` |

### Example Data File

Create a file called `my_barcode_data.csv`:
```csv
epc_code,event_time,location_id,business_step,scan_location,event_type,operator_id
001.8804823.1293291.010001.20250722.000001,2025-07-22 08:00:00,FACTORY_001,Factory,SCAN_A,Observation,OP_001
001.8804823.1293291.010001.20250722.000001,2025-07-22 10:15:00,WAREHOUSE_001,WMS,SCAN_B,Observation,OP_002
001.8804823.1293291.010001.20250722.000001,2025-07-22 14:30:00,DISTRIBUTION_001,Distribution,SCAN_C,Observation,OP_003
```

---

## 🎯 Step-by-Step: Training Your AI Model

Training is like teaching the AI to recognize bad patterns by showing it lots of examples.

### Step 1: Prepare Your Data

Create a simple Python script called `step1_prepare_data.py`:

```python
# This script teaches the AI what normal vs suspicious looks like
from src.lstm_data_preprocessor import LSTMDataPreprocessor

print("🔄 Starting data preparation...")
print("Think of this like teaching a security guard what to look for")

# Create our data teacher
preprocessor = LSTMDataPreprocessor()

# Load your data files (put your CSV files in a list)
csv_files = [
    'my_barcode_data.csv',
    # Add more files here if you have them
]

print("📂 Loading your barcode scan data...")
data = preprocessor.load_and_validate_data(csv_files)
print(f"✅ Loaded {len(data):,} barcode scans")

# The AI learns patterns from timing, locations, and sequences
print("🧠 Teaching AI about timing patterns...")
data = preprocessor.extract_temporal_features(data)

print("📍 Teaching AI about location patterns...")  
data = preprocessor.extract_spatial_features(data)

print("🔍 Teaching AI about behavior patterns...")
data = preprocessor.extract_behavioral_features(data)

# Create labels (mark which scans are suspicious)
print("🏷️ Creating labels for suspicious activities...")
data = preprocessor.generate_labels_from_rules(data)

# Split data: 80% for training, 20% for testing
print("📊 Splitting data for training and testing...")
train_data, test_data = preprocessor.epc_aware_temporal_split(data)

print(f"📚 Training data: {len(train_data):,} scans")
print(f"🧪 Testing data: {len(test_data):,} scans")

# Save the prepared data
train_data.to_csv('prepared_train_data.csv', index=False)
test_data.to_csv('prepared_test_data.csv', index=False)

print("✅ Data preparation complete!")
print("Your AI security guard is ready for training!")
```

Run this script:
```bash
python step1_prepare_data.py
```

### Step 2: Train the AI Model

Create `step2_train_model.py`:

```python
# This actually trains your AI security guard
import torch
import pandas as pd
import numpy as np
from src.lstm_data_preprocessor import AdaptiveLSTMSequenceGenerator
from src.production_lstm_model import ProductionLSTM, LSTMTrainer

print("🎓 Starting AI training...")
print("This is like sending your security guard to training academy")

# Load prepared data
print("📖 Loading prepared training data...")
train_data = pd.read_csv('prepared_train_data.csv')
test_data = pd.read_csv('prepared_test_data.csv')

# Convert data to sequences (like showing the AI short movies of barcode scans)
print("🎬 Converting data to sequence format...")
sequence_generator = AdaptiveLSTMSequenceGenerator()

train_sequences, train_labels, train_metadata = sequence_generator.generate_sequences(train_data)
test_sequences, test_labels, test_metadata = sequence_generator.generate_sequences(test_data)

print(f"🎬 Created {len(train_sequences):,} training sequences")
print(f"🧪 Created {len(test_sequences):,} testing sequences")

# Create the AI model (the "brain" of your security guard)
print("🧠 Creating AI model...")
model = ProductionLSTM(
    input_size=11,      # Number of features the AI looks at
    hidden_size=64,     # How much the AI can "remember"
    num_classes=5       # 5 types of suspicious activities to detect
)

# Create the trainer (like the training instructor)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainer = LSTMTrainer(model, device=device)

print(f"💻 Using device: {device}")
print("⚡ GPU found - training will be faster!" if device.type == 'cuda' else "💻 Using CPU - training will be slower")

# Set up training parameters
trainer.setup_training(
    learning_rate=0.001,    # How fast the AI learns (not too fast, not too slow)
    weight_decay=0.0001,    # Prevents overfitting (memorizing instead of learning)
    max_epochs=50           # Maximum number of times to see all data
)

# Convert to PyTorch format
train_sequences = torch.FloatTensor(train_sequences)
train_labels = torch.FloatTensor(train_labels)
test_sequences = torch.FloatTensor(test_sequences) 
test_labels = torch.FloatTensor(test_labels)

# Create data loaders (feed data to AI in small batches)
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(train_sequences, train_labels)
test_dataset = TensorDataset(test_sequences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("🎯 Starting training process...")
print("The AI will learn by seeing examples over and over")
print("Like showing a security guard thousands of security camera videos")

# Train the model
training_results = trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    epochs=50,
    patience=10  # Stop early if AI stops improving
)

print("🎉 Training complete!")
print(f"🏆 Best performance: {training_results['best_val_auc']:.3f}")
print("(1.000 = perfect, 0.500 = random guessing)")

# Save the trained model
torch.save(model.state_dict(), 'trained_lstm_model.pt')
print("💾 Model saved as 'trained_lstm_model.pt'")
print("Your AI security guard is now ready for duty!")
```

Run the training:
```bash
python step2_train_model.py
```

**What to expect:**
- Training will take 30 minutes to 2 hours depending on your computer
- You'll see progress updates showing how well the AI is learning
- The AI will automatically stop when it's done learning

---

## 🔍 Step 3: Using Your Trained Model

Create `step3_use_model.py`:

```python
# This shows how to use your trained AI security guard
from src.lstm_inferencer import LSTMInferencer, InferenceRequest
import json

print("🤖 Loading your trained AI security guard...")

# Load the trained model
inferencer = LSTMInferencer(
    model_path='trained_lstm_model.pt',
    enable_explanations=True  # Get explanations for predictions
)

print("✅ AI security guard is online and ready!")

# Example: Check some barcode scans for suspicious activity
test_events = [
    {
        'event_time': '2025-07-22T10:00:00Z',
        'location_id': 'WAREHOUSE_001', 
        'business_step': 'Factory',
        'scan_location': 'SCAN_A',
        'event_type': 'Observation',
        'operator_id': 'OP_001'
    },
    {
        'event_time': '2025-07-22T10:05:00Z',  # Only 5 minutes later
        'location_id': 'WAREHOUSE_999',         # 500 miles away (suspicious!)
        'business_step': 'Distribution',
        'scan_location': 'SCAN_Z', 
        'event_type': 'Observation',
        'operator_id': 'OP_999'
    }
]

# Create a request to check these scans
request = InferenceRequest(
    epc_code='001.8804823.1293291.010001.20250722.000001',
    events=test_events,
    request_id='test_001'
)

print("🔍 Analyzing barcode scans for suspicious activity...")

# Get prediction from AI
response = inferencer.predict(request)

print(f"\n📊 Analysis Results for EPC: {response.epc_code}")
print(f"⏱️  Analysis took: {response.processing_time_ms:.1f} milliseconds")
print(f"🚨 Overall Risk Score: {response.overall_risk_score:.2f} (0=safe, 1=very suspicious)")

if response.predictions:
    print(f"\n⚠️  Suspicious Activities Detected: {len(response.predictions)}")
    
    for pred in response.predictions:
        risk_emoji = "🟢" if pred.risk_level == "low" else "🟡" if pred.risk_level == "medium" else "🔴"
        print(f"  {risk_emoji} {pred.anomaly_type}: {pred.confidence:.1%} confidence ({pred.risk_level} risk)")
        
        # Explain what the AI found suspicious
        if 'threshold' in pred.explanation:
            print(f"     💡 AI detected this because confidence ({pred.confidence:.1%}) > threshold ({pred.explanation['threshold']:.1%})")
else:
    print("✅ No suspicious activity detected - everything looks normal!")

# Show what the AI was paying attention to
if response.explainability and 'note' not in response.explainability:
    print(f"\n🧠 What the AI was focusing on:")
    for anomaly_type, explanation in response.explainability.items():
        if 'top_contributing_features' in explanation:
            print(f"  For {anomaly_type}:")
            for feature, importance in explanation['top_contributing_features'][:3]:
                print(f"    📈 {feature}: {importance:.3f} importance")

print(f"\n📋 Full analysis saved to JSON format")
json_result = inferencer.to_json_schema(response)
with open('analysis_result.json', 'w') as f:
    json.dump(json_result, f, indent=2)
```

Run the analysis:
```bash
python step3_use_model.py
```

---

## 📊 Step 4: Evaluating How Good Your Model Is

Create `step4_evaluate_model.py`:

```python
# This checks how good your AI security guard really is
import torch
import pandas as pd
import numpy as np
from src.lstm_evaluation import ComprehensiveLSTMEvaluator
from src.production_lstm_model import ProductionLSTM
from src.lstm_data_preprocessor import AdaptiveLSTMSequenceGenerator

print("📊 Starting comprehensive model evaluation...")
print("This is like giving your security guard a final exam")

# Load test data
print("📖 Loading test data...")
test_data = pd.read_csv('prepared_test_data.csv')

# Convert to sequences
sequence_generator = AdaptiveLSTMSequenceGenerator()
test_sequences, test_labels, _ = sequence_generator.generate_sequences(test_data)

# Load the trained model
print("🤖 Loading trained model...")
model = ProductionLSTM(input_size=11, hidden_size=64, num_classes=5)
model.load_state_dict(torch.load('trained_lstm_model.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Convert to tensors
X_test = torch.FloatTensor(test_sequences)
y_test = test_labels

# Create evaluator
evaluator = ComprehensiveLSTMEvaluator()

print("🧪 Running comprehensive evaluation...")
print("This tests your AI on data it has never seen before")

# Run full evaluation
results = evaluator.evaluate_model(
    model=model,
    X_test=X_test, 
    y_test=y_test,
    device=device,
    model_name="My_LSTM_Security_Guard"
)

print("\n🎯 EVALUATION RESULTS")
print("=" * 50)

# Overall Performance
auc_score = results['standard_metrics'].get('auc_macro', 0)
print(f"🏆 Overall Performance Score: {auc_score:.1%}")

if auc_score >= 0.9:
    print("   🌟 EXCELLENT! Your AI is performing amazingly well!")
elif auc_score >= 0.8:
    print("   ✅ GOOD! Your AI is performing well!")
elif auc_score >= 0.7:
    print("   🟡 OKAY! Your AI is decent but could be better!")
else:
    print("   🔴 NEEDS WORK! Your AI needs more training or better data!")

# Explain what the score means
print(f"\n📖 What this score means:")
print(f"   • 1.00 (100%) = Perfect detection, never makes mistakes")
print(f"   • 0.90 (90%)  = Excellent, catches 90% of problems correctly") 
print(f"   • 0.80 (80%)  = Good, catches 80% of problems correctly")
print(f"   • 0.70 (70%)  = Okay, catches 70% of problems correctly")
print(f"   • 0.50 (50%)  = Random guessing, not useful")

# Cost Analysis
if 'cost_sensitive_metrics' in results:
    cost_reduction = results['cost_sensitive_metrics'].get('overall', {}).get('average_cost_reduction_percent', 0)
    print(f"\n💰 Business Impact:")
    print(f"   Cost Reduction: {cost_reduction:.1f}%")
    
    if cost_reduction > 30:
        print("   🤑 EXCELLENT cost savings for your business!")
    elif cost_reduction > 15:
        print("   💵 GOOD cost savings for your business!")
    else:
        print("   💸 Modest cost savings, but still helpful!")

# Robustness Test
if 'noise_robustness' in results:
    robustness = results['noise_robustness'].get('robustness_summary', {})
    robust_5_percent = robustness.get('robust_to_5_percent_noise', False)
    
    print(f"\n🛡️  Reliability Test:")
    if robust_5_percent:
        print("   ✅ PASSED! Your AI works well even with messy/imperfect data")
    else:
        print("   ⚠️  CAUTION! Your AI might struggle with messy/imperfect data")

# Individual Problem Detection
print(f"\n🔍 How well it detects each type of problem:")
per_class = results.get('per_class_analysis', {})

problem_names = {
    'epcFake': 'Fake Products',
    'epcDup': 'Duplicate Scans', 
    'locErr': 'Wrong Locations',
    'evtOrderErr': 'Wrong Order Events',
    'jump': 'Impossible Jumps'
}

for problem_code, problem_name in problem_names.items():
    if problem_code in per_class:
        score = per_class[problem_code].get('f1_score', 0)
        print(f"   📦 {problem_name}: {score:.1%}")
        
        if score >= 0.8:
            emoji = "🌟"
        elif score >= 0.6:
            emoji = "✅"
        elif score >= 0.4:
            emoji = "🟡"
        else:
            emoji = "🔴"
        print(f"      {emoji} Detection quality")

# Generate readable report
print(f"\n📄 Generating detailed report...")
report = evaluator.generate_evaluation_report("My_LSTM_Security_Guard")

with open('evaluation_report.txt', 'w') as f:
    f.write(report)

print(f"📄 Detailed report saved to 'evaluation_report.txt'")
print(f"📊 Full results saved to 'evaluation_results/' folder")

print(f"\n🎉 Evaluation complete!")
print(f"Your AI security guard has been thoroughly tested and is ready for real-world use!")
```

Run the evaluation:
```bash
python step4_evaluate_model.py
```

---

## 🚨 Real-World Usage: Setting Up Monitoring

Create `step5_monitor_live.py` for real-world monitoring:

```python
# This runs your AI security guard in real-time
from src.lstm_inferencer import LSTMInferencer, InferenceRequest
import time
import json

print("🚨 Starting Real-Time Barcode Monitoring System")
print("Your AI security guard is now on duty!")

# Load your trained model
inferencer = LSTMInferencer(
    model_path='trained_lstm_model.pt',
    enable_explanations=True
)

def check_barcode_events(epc_code, events):
    """Check if barcode events are suspicious"""
    
    request = InferenceRequest(
        epc_code=epc_code,
        events=events,
        request_id=f"live_{int(time.time())}"
    )
    
    response = inferencer.predict(request)
    
    # Alert if high risk detected
    if response.overall_risk_score > 0.7:  # High risk threshold
        print(f"🚨 HIGH RISK ALERT! 🚨")
        print(f"EPC: {epc_code}")
        print(f"Risk Score: {response.overall_risk_score:.1%}")
        
        for pred in response.predictions:
            print(f"  ⚠️  {pred.anomaly_type}: {pred.confidence:.1%} confidence")
        
        # Here you could send email, Slack message, etc.
        # send_alert_email(response)
        # send_slack_notification(response)
        
    elif response.overall_risk_score > 0.3:  # Medium risk
        print(f"🟡 Medium risk detected for {epc_code}: {response.overall_risk_score:.1%}")
        
    else:
        print(f"✅ {epc_code}: Normal activity")
    
    return response

# Example: Monitor some live events
print("\n🔍 Monitoring live barcode scans...")

# In real life, you would connect this to your barcode scanning system
# For demo, we'll simulate some scans
sample_events = [
    {
        'event_time': '2025-07-22T14:00:00Z',
        'location_id': 'WAREHOUSE_001',
        'business_step': 'WMS',
        'scan_location': 'SCAN_DOCK_A',
        'event_type': 'Observation', 
        'operator_id': 'OP_005'
    },
    {
        'event_time': '2025-07-22T14:02:00Z',  # 2 minutes later
        'location_id': 'FACTORY_500',          # Different location (could be suspicious)
        'business_step': 'Factory',            # Going backwards in process (suspicious!)
        'scan_location': 'SCAN_LINE_99',
        'event_type': 'Observation',
        'operator_id': 'OP_999'
    }
]

result = check_barcode_events(
    epc_code='001.8804823.1293291.010001.20250722.000001',
    events=sample_events
)

print(f"\n📊 System Health Check:")
health = inferencer.get_health_status()
print(f"   🤖 Model Status: {'✅ Healthy' if health['status'] == 'healthy' else '❌ Unhealthy'}")
print(f"   ⚡ Device: {health['device']}")
print(f"   📈 Total Requests: {health['inference_stats']['total_requests']}")
print(f"   ⏱️  Average Response Time: {health['inference_stats']['average_latency_ms']:.1f}ms")

print(f"\n🎯 Your AI security guard is working perfectly!")
print(f"Connect this to your real barcode scanning system to monitor 24/7!")
```

---

## 🆘 Troubleshooting Common Issues

### Problem: "Python is not recognized"
**Solution:** You need to install Python and add it to your PATH
1. Reinstall Python from python.org
2. Check "Add Python to PATH" during installation

### Problem: "No module named 'torch'"
**Solution:** Install PyTorch
```bash
pip install torch torchvision torchaudio
```

### Problem: Training is very slow
**Solutions:**
- **Use a smaller dataset** for testing (like 10,000 scans instead of 1 million)
- **Reduce epochs** from 50 to 20 in the training script
- **Use a computer with a GPU** (graphics card) for much faster training

### Problem: Low accuracy scores
**Solutions:**
- **Get more data** - AI needs lots of examples to learn well
- **Improve data quality** - Remove scans with missing information
- **Adjust the model** - Try different settings in the training script

### Problem: "CUDA out of memory"
**Solution:** Reduce batch size in training script:
```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Changed from 32 to 16
```

---

## 📈 Understanding Your Results

### Performance Scores Explained

| Score Range | What It Means | Real-World Example |
|-------------|---------------|-------------------|
| **90-100%** | Excellent - Professional grade | Like a top security expert who catches almost everything |
| **80-89%** | Good - Production ready | Like a well-trained security guard who catches most problems |
| **70-79%** | Okay - Needs improvement | Like a new security guard who misses some things |
| **60-69%** | Poor - More training needed | Like a security guard who needs more practice |
| **Below 60%** | Very poor - Start over | Like random guessing - not useful |

### Cost Reduction Explained

- **30%+ cost reduction** = Your AI saves significant money by catching problems early
- **15-30% cost reduction** = Good savings, pays for itself
- **5-15% cost reduction** = Modest but worthwhile savings  
- **Below 5%** = Might not be worth the effort

### Risk Levels Explained

- **🔴 High Risk (80%+)** = Definitely investigate immediately
- **🟡 Medium Risk (60-80%)** = Probably investigate soon  
- **🟢 Low Risk (30-60%)** = Keep an eye on it
- **✅ Normal (Below 30%)** = Everything looks fine

---

## 🎯 Quick Start Checklist

1. **✅ Install Python 3.11+** from python.org
2. **✅ Install required packages** using pip commands above
3. **✅ Prepare your CSV data** with required columns
4. **✅ Run step1_prepare_data.py** to clean and organize data
5. **✅ Run step2_train_model.py** to train your AI (takes 30min-2hrs)
6. **✅ Run step3_use_model.py** to test predictions
7. **✅ Run step4_evaluate_model.py** to check performance
8. **✅ Run step5_monitor_live.py** for real-time monitoring

**Total time needed:** 3-6 hours depending on your computer and data size

---

## 🤝 Getting Help

**If something doesn't work:**

1. **Check the error message** - it usually tells you what's wrong
2. **Try Google searching** the exact error message  
3. **Check your data format** - make sure CSV has all required columns
4. **Start with smaller data** - try 1,000 scans instead of 1 million
5. **Ask for help** from your IT team or a Python-savvy colleague

**Common beginner mistakes:**
- Forgetting to install Python properly
- Using the wrong CSV column names
- Running out of computer memory with too much data
- Not having enough training data (need at least 10,000 scans)

---

## 🎉 Congratulations!

You now have your own **AI-powered barcode security guard** that can:

- ✅ **Monitor millions of scans** automatically
- ✅ **Detect 5 types of suspicious activity** in real-time  
- ✅ **Explain why** something looks suspicious
- ✅ **Save money** by catching problems early
- ✅ **Work 24/7** without getting tired

**Your supply chain just got a lot smarter and safer!** 🚀