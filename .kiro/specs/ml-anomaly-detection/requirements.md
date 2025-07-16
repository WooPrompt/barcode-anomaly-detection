# Requirements Document: LSTM and SVM Models for Barcode Anomaly Detection

## Introduction

This document explains what we need to build: **machine learning models** that can detect barcode anomalies automatically. Think of it like teaching a computer to spot suspicious patterns in barcode scanning data.

**Why do we need this?**
- Your current system uses **rules** (like "if EPC format is wrong, it's fake")
- But some anomalies are **subtle patterns** that rules can't catch
- Machine learning can **learn from examples** and find hidden patterns

**What are LSTM and SVM?**
- **LSTM**: A type of AI that's good at understanding **sequences** (like "what should happen next in the scanning process?")
- **SVM**: A type of AI that's good at finding **outliers** (like "this scanning pattern is weird compared to normal ones")

## Requirements

### Requirement 1: Data Preparation (Making Data Ready for AI)

**Why do we need this?** 
Raw barcode data is messy and computers can't understand it directly. We need to clean it and convert it into numbers that AI models can process.

**What we need to do:**

1. **Extract time information** from scan timestamps
   - **Why?** Scanning patterns might be different at night vs. day, weekdays vs. weekends
   - **Example:** "2024-07-15 14:30:00" becomes hour=14, day_of_week=Monday

2. **Break down EPC codes** into meaningful parts
   - **Why?** Each part of the EPC (company, product, date) tells us something important
   - **Example:** "001.8804823.0000001.000001.20240701.000000001" becomes 6 separate features

3. **Enrich location data** with additional features
   - **Why?** We have location_id (1,2,3,4) but we can add more useful information like coordinates and location type
   - **Example:** location_id=1 becomes location_id=1, latitude=37.45, longitude=126.65, location_type="Factory"

4. **Create sequence features** for LSTM
   - **Why?** LSTM needs to see the order of events (what happened before/after)
   - **Example:** [Factory→WMS→Retail] becomes a sequence of numbers

5. **Handle missing data**
   - **Why?** Sometimes scan data is incomplete, and we need to decide what to do
   - **Example:** If location is missing, should we guess it or skip that record?

6. **Scale numbers** to similar ranges
   - **Why?** AI works better when all numbers are in similar ranges (like 0-1)
   - **Example:** Convert latitude (37.45) and time_hours (5.2) to similar scales

### Requirement 2: LSTM Model (Learning What Should Happen Next)

**Why do we need LSTM?**
Think of LSTM like a smart assistant who watches barcode scanning patterns and learns "what normally happens next." For example, after a product is scanned at a factory, it usually goes to a warehouse, then to a store. If it suddenly appears at a different factory, that's suspicious!

**What LSTM will do:**

1. **Learn normal sequences** from historical data
   - **Why?** We need to teach the AI what "normal" looks like first
   - **Example:** Factory→Warehouse→Store is normal, Factory→Store (skipping warehouse) might be suspicious

2. **Predict what should happen next** for new scans
   - **Why?** If we can predict the next step, we can spot when reality doesn't match
   - **Example:** After "Factory scan", LSTM predicts "Warehouse scan should be next"

3. **Use sliding windows** to create training examples
   - **Why?** We need to show LSTM many examples of "what came before" and "what came after"
   - **Example:** From sequence [A,B,C,D,E], create training pairs: [A,B]→C, [B,C]→D, [C,D]→E

4. **Give confidence scores** (0-100%) to match your current system
   - **Why?** Your current rule-based system already gives scores like epcFakeScore=85, so LSTM should do the same
   - **Example:** Current: `"epcFakeScore": 85`, New: `"lstmSequenceScore": 95` - both use the same 0-100 scale

5. **Handle different sequence lengths** safely
   - **Why?** Some products have 3 scans, others have 10 scans - LSTM needs consistent input size
   - **The Problem:** Padding with zeros is dangerous because it adds fake data, truncating loses real data
   - **Better Solutions:** 
     - Use **variable-length LSTM** (PyTorch supports this naturally)
     - Or **group by sequence length** (train separate models for short/medium/long sequences)
     - Or use **attention mechanisms** that can handle any length
   - **Example:** Instead of forcing all sequences to be length 5, let LSTM handle sequences of length 3, 7, 12 naturally

6. **Use GPU if available** for faster training
   - **Why?** LSTM training can be slow on CPU, much faster on GPU
   - **Example:** Training time: 2 hours on GPU vs 20 hours on CPU

### Requirement 3: SVM Model (Finding Weird Patterns)

**Why do we need SVM?**
Think of SVM like a security guard who learns what "normal people" look like, then spots anyone who looks suspicious. SVM learns from thousands of normal barcode scans, then can identify scans that are "weird" compared to the normal ones.

**What SVM will do:**

1. **Learn what "normal" looks like** from good data
   - **Why?** We need to show SVM thousands of examples of normal scanning behavior
   - **Example:** Normal scans happen during business hours, follow logical routes, have reasonable timing

2. **Spot outliers** in new data
   - **Why?** Once SVM knows "normal", it can spot anything that's different
   - **Example:** "This scan pattern is only 5% similar to normal patterns - it's suspicious!"

3. **Use only normal data** for training (unsupervised learning)
   - **Why?** We don't have many examples of anomalies, but we have lots of normal data
   - **Example:** Train on 100,000 normal scans, then use it to find the rare anomalies

4. **Optimize settings** automatically
   - **Why?** SVM has settings (nu, gamma) that need to be tuned for best performance
   - **Example:** Try different settings and pick the ones that work best on test data

5. **Scale all features** to similar ranges
   - **Why?** SVM gets confused if some numbers are huge (like timestamps) and others are small (like location IDs)
   - **Example:** Convert all features to 0-1 range before training

6. **Measure how well it works**
   - **Why?** We need to know if SVM is actually good at finding anomalies
   - **Example:** "SVM correctly identifies 85% of real anomalies with only 5% false alarms"

### Requirement 4: Combining All Models Together

**Why do we need this?**
You already have rule-based detection (like "EPC format is wrong"). Now we're adding LSTM and SVM. We need to combine all three approaches to get the best results - like having three different security guards checking the same area.

**What we need to do:**

1. **Combine results** from all three approaches
   - **Why?** Each method is good at finding different types of problems
   - **Example:** Rules find format errors (100% sure), LSTM finds sequence problems (80% sure), SVM finds weird patterns (60% sure)

2. **Add ML confidence** to rule-based findings
   - **Why?** Rules say "yes/no", but ML can say "how confident are we?"
   - **Example:** Rule says "EPC is fake", ML adds "and I'm 95% confident this is correct"

3. **Flag ML-only discoveries**
   - **Why?** Sometimes ML finds problems that rules miss
   - **Example:** "Rules didn't catch this, but ML thinks it's 70% likely to be suspicious"

4. **Return all scores** in API response
   - **Why?** Users want to see both rule-based and ML-based confidence
   - **Example:** `{"rule_anomaly": true, "lstm_confidence": 85, "svm_confidence": 72}`

5. **Handle conflicts** between models
   - **Why?** Sometimes models disagree - we need rules for what to do
   - **Example:** If rules say "normal" but ML says "anomaly", trust the higher confidence score

### Requirement 5: Training and Testing the Models

**Why do we need this?**
Just like students need to study and take tests, AI models need to learn from data and be tested to make sure they're actually good at their job. We need to be scientific about this!

**What we need to do:**

1. **Split data** into three parts (70% training, 15% validation, 15% testing)
   - **Why?** We need separate data to train, tune, and test our models
   - **Example:** Use 70,000 scans to teach, 15,000 to tune settings, 15,000 to final test

2. **Stop training** when the model stops improving
   - **Why?** Sometimes models "memorize" training data instead of learning patterns (overfitting)
   - **Example:** If validation accuracy stops improving for 10 rounds, stop training

3. **Measure how good** the models are
   - **Why?** We need numbers to know if our models are actually useful
   - **Example:** "Model catches 85% of real anomalies and only gives 5% false alarms"

4. **Save model information** when we save the trained model
   - **Why?** Later we need to remember how well it worked and what settings we used
   - **Example:** Save "trained on 2024-07-15, accuracy=85%, settings: learning_rate=0.001"

5. **Check compatibility** when loading saved models
   - **Why?** Make sure old models still work with new data formats
   - **Example:** If we add new features, check if old model can handle them

6. **Support retraining** when we get new data
   - **Why?** Models need updates as patterns change over time
   - **Example:** Retrain monthly with new scan data to keep models current

### Requirement 6: Making It Work in Real-Time

**Why do we need this?**
Your current FastAPI system needs to work with the new ML models. When someone uploads barcode data, they want results quickly (within seconds), not after waiting 10 minutes for AI to think!

**What we need to do:**

1. **Run all checks** at the same time
   - **Why?** Instead of doing rules first, then LSTM, then SVM (slow), do them all together (fast)
   - **Example:** Like having 3 people check different things simultaneously instead of one person doing everything

2. **Speed up slow responses** with smart caching
   - **Why?** If the same barcode data is checked multiple times, remember the answer
   - **Example:** First time takes 2 seconds, next time takes 0.1 seconds (from cache)

3. **Update models** without breaking the website
   - **Why?** When we train better models, we need to replace old ones without stopping the service
   - **Example:** Like changing a car tire while driving - keep the service running

4. **Handle both single items and batches**
   - **Why?** Sometimes users check 1 barcode, sometimes 1000 barcodes at once
   - **Example:** Single: "Check this one EPC", Batch: "Check this CSV file with 10,000 EPCs"

5. **Manage computer memory** efficiently
   - **Why?** ML models are big and use lots of memory - we need to be smart about loading them
   - **Example:** Only load LSTM when needed, unload it when not used for a while