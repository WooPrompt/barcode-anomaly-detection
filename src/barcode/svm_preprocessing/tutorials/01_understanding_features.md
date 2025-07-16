# Understanding Features - A Beginner's Guide

This tutorial explains what "features" are, why they matter for SVM, and how our system creates them from barcode scan data.

## 🤔 What Are Features?

**Features** are numerical values that describe important characteristics of your data. Think of them as "measurements" that help a computer understand patterns.

### Real-World Analogy 🏠
Imagine you're trying to predict house prices. You might use these features:
- **Size**: 2,500 square feet → `2500.0`
- **Bedrooms**: 3 bedrooms → `3.0`
- **Age**: Built in 2010 → `14.0` (years old)
- **Pool**: Has a pool → `1.0` (yes) or `0.0` (no)

The computer doesn't understand "3 bedrooms" but it understands `3.0`.

### For Barcode Anomalies 📦
Similarly, for barcode anomaly detection, we need to convert scan data into numbers:
- **EPC Format**: "001.8804823.1234567" → `1.0` (valid) or `0.0` (invalid)
- **Event Count**: Product scanned 5 times → `5.0`
- **Time Span**: Events over 3 days → `72.0` (hours)
- **Location Jumps**: Impossible travel → `2.0` (violation count)

## 🎯 Why Does SVM Need Fixed-Length Features?

**SVM (Support Vector Machine)** is like a very precise measuring tool. It needs every sample to have **exactly the same measurements**.

### The Problem 📏
Real barcode data has variable complexity:
- **Simple Product**: 2 scans → 2 time differences
- **Complex Product**: 8 scans → 7 time differences

But SVM needs consistent input:
- **Every Product**: Must have exactly 10 feature values

### Our Solution ✨
We use **intelligent processing** to convert variable data into fixed features:

```python
# Variable input (different products, different complexity)
simple_product = ["scan1", "scan2"]                    # 2 events
complex_product = ["scan1", "scan2", ..., "scan8"]     # 8 events

# Fixed output (every product gets exactly 10 features)
simple_features = [1.0, 2.0, 4.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.8]   # 10 values
complex_features = [8.0, 7.2, 3.1, 2.4, 1.8, 0.9, 0.3, 0.1, 4.2, 0.6]  # 10 values
```

## 🏗️ Our 5 Feature Types

We extract features for 5 different types of anomalies:

### 1. 🏷️ EPC Fake Features (10 dimensions)
**Purpose**: Detect invalid EPC format
**Input**: Single EPC code string
**Output**: 10 numerical features

```python
# Example EPC: "001.8804823.1234567.123456.20240101.123456789"
features = [
    1.0,    # structure_valid (6 parts separated by dots)
    1.0,    # header_valid (starts with "001")
    1.0,    # company_valid (known company code)
    1.0,    # product_valid (7-digit product code)
    1.0,    # lot_valid (6-digit lot number)
    1.0,    # date_valid (valid manufacturing date)
    0.0,    # date_future_error (not in future)
    0.0,    # date_old_error (not too old)
    1.0,    # serial_valid (9-digit serial number)
    0.68    # epc_length_normalized (length relative to standard)
]
```

### 2. 🔄 EPC Duplicate Features (8 dimensions)
**Purpose**: Detect impossible duplicate scans
**Input**: All scans for one EPC
**Output**: 8 numerical features

```python
# Example: Product scanned 4 times
features = [
    4.0,    # event_count (total scans)
    3.0,    # unique_locations_count (3 different places)
    2.0,    # unique_event_types_count (2 different event types)
    72.5,   # time_span_hours (scanned over 3+ days)
    18.2,   # avg_time_between_events (average gap)
    48.0,   # max_time_gap (longest gap between scans)
    0.25,   # location_repetition_ratio (some repeated locations)
    0.50    # event_type_repetition_ratio (some repeated events)
]
```

### 3. 📅 Event Order Features (12 dimensions)
**Purpose**: Detect wrong sequence of supply chain events
**Input**: Sequence of events for one EPC
**Output**: 12 numerical features

```python
# Example: Events in wrong order (retail before logistics)
features = [
    5.0,    # sequence_length (5 events total)
    3.0,    # unique_event_types (3 different types)
    4.0,    # transition_count (4 transitions between events)
    2.0,    # backward_transitions (2 violations of expected order)
    0.0,    # temporal_disorder_count (time is chronological)
    0.71,   # max_time_gap_hours (normalized largest gap)
    0.20,   # production_ratio (20% production events)
    0.60,   # logistics_ratio (60% logistics events)
    0.20,   # retail_ratio (20% retail events)
    1.85,   # event_type_entropy (diversity of event types)
    0.75,   # sequence_regularity (how predictable the pattern is)
    0.40    # violation_density (violations per event)
]
```

### 4. 🗺️ Location Error Features (15 dimensions)
**Purpose**: Detect impossible location transitions
**Input**: Sequence of locations for one EPC
**Output**: 15 numerical features

```python
# Example: Product jumps from factory to retail (skipping logistics)
features = [
    6.0,    # sequence_length (6 location changes)
    4.0,    # unique_locations (4 different places)
    3.0,    # hierarchy_violations (3 impossible jumps)
    0.0,    # unknown_locations (all locations are known)
    1.0,    # reverse_transitions (1 backward movement)
    0.67,   # max_hierarchy_jump (normalized biggest jump)
    0.17,   # level0_ratio (17% factory-level locations)
    0.33,   # level1_ratio (33% logistics-level locations)
    0.17,   # level2_ratio (17% distribution-level locations)
    0.33,   # level3_ratio (33% retail-level locations)
    1.92,   # location_entropy (diversity of locations)
    0.60,   # hierarchy_consistency (how well it follows expected flow)
    0.67,   # geographic_scatter (how spread out the locations are)
    0.45,   # transition_regularity (how predictable transitions are)
    0.50    # location_anomaly_density (anomalies per location change)
]
```

### 5. ⏰ Time Jump Features (10 dimensions)
**Purpose**: Detect impossible travel times between locations
**Input**: Timestamps and locations for one EPC
**Output**: 10 numerical features

```python
# Example: Product travels impossible distances in short time
features = [
    4.0,    # sequence_length (4 time intervals)
    168.5,  # total_time_span_hours (total journey time)
    42.1,   # avg_time_between_events (average interval)
    120.0,  # max_time_gap_hours (longest interval)
    0.25,   # min_time_gap_seconds (shortest interval in hours)
    1842.3, # time_gap_variance (how irregular the timing is)
    2.0,    # impossible_jumps_count (2 physically impossible travels)
    0.0,    # negative_time_count (no time going backwards)
    0.0,    # zero_time_gaps (no simultaneous events)
    0.23    # time_regularity_score (low = irregular timing)
]
```

## 🧮 How Fixed Dimensions Work

### The Challenge
Each product has a different journey:
- **Product A**: Factory → Store (2 events)
- **Product B**: Factory → Warehouse → Distribution → Store (4 events)

### The Solution
We use **statistical summaries** instead of raw sequences:

```python
# Instead of storing all events (variable length):
product_a_events = ["factory", "store"]                    # Length: 2
product_b_events = ["factory", "warehouse", "dist", "store"]  # Length: 4

# We calculate fixed statistics (same length):
product_a_features = [2.0, 1.0, 0.5, ...]  # Length: 10
product_b_features = [4.0, 3.0, 0.8, ...]  # Length: 10
```

### Types of Statistical Features
1. **Counts**: How many events, locations, transitions
2. **Ratios**: Percentage of each type of event
3. **Time Statistics**: Average, maximum, variance of time gaps
4. **Pattern Measures**: Entropy, regularity, violation density
5. **Aggregations**: Sum, mean, min, max of various measurements

## 🔍 Feature Engineering Examples

### Example 1: Detecting Fake EPCs
```python
# Real EPC: "001.8804823.1234567.123456.20240101.123456789"
real_epc_features = [1, 1, 1, 1, 1, 1, 0, 0, 1, 0.68]  # Mostly 1s (valid)

# Fake EPC: "INVALID.FORMAT"
fake_epc_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15]  # Mostly 0s (invalid)

# SVM learns: "If most values are 0, it's probably fake"
```

### Example 2: Detecting Duplicates
```python
# Normal product (scanned at different times/places)
normal_features = [3, 3, 2, 48.0, 16.0, 24.0, 0.0, 0.33]  # Spread out

# Duplicate problem (same time, multiple places)
duplicate_features = [5, 3, 1, 0.0, 0.0, 0.0, 0.6, 0.8]  # Clustered

# SVM learns: "If time_span=0 but locations>1, it's impossible"
```

## 🎯 Why This Approach Works

### 1. **Consistency** 📏
Every sample has exactly the same feature structure, so SVM can compare them meaningfully.

### 2. **Information Preservation** 📊
Statistical summaries capture the essential patterns while standardizing the format.

### 3. **Scalability** 🚀
Works with any complexity of product journey - from simple to complex supply chains.

### 4. **Interpretability** 🔍
Each feature has a clear meaning, so you can understand why the model made a decision.

## 🚀 Next Steps

Now that you understand features, try these:

1. **Run the basic example**: `python examples/basic_usage.py`
2. **Explore feature extractors**: Look at `02_features/epc_fake_extractor.py`
3. **See features in action**: `python 01_core/base_preprocessor.py`
4. **Learn about labels**: Read `02_creating_labels.md`

## 💡 Key Takeaways

- **Features** = numerical measurements that describe data patterns
- **Fixed dimensions** = every sample must have the same number of features
- **Statistical summaries** = how we convert variable data to fixed features
- **5 anomaly types** = each gets its own specialized feature extraction
- **Interpretable features** = you can understand what each number means

Ready to dive deeper? Try the next tutorial! 🎓