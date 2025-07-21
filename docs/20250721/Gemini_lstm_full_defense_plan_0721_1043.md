## LSTM Anomaly Detection Pipeline: Full Defense and Implementation Plan

**Author:** Gemini, Data Scientist and Vector Space Researcher
**Date:** 2025-07-21
**Version:** 4.0

This document provides a comprehensive, academically-defensible, and production-ready plan for implementing an LSTM-based anomaly detection system. It is designed to be understood by non-technical stakeholders while providing the rigorous detail required by a technical and academic audience.

---

### **Part 1: Dataset and Labeling Strategy**

This section addresses the fundamental questions about the data used to train and evaluate our model.

#### **1.1. What dataset was used?**

*   **For Non-Technical Stakeholders:** We are using a realistic, simulated dataset that mirrors the real-world logistics data we expect to see. The data, located in `data/raw/*.csv`, represents the complete lifecycle of products as they move through the supply chain, from the factory to the final destination.
*   **For Technical Audience:** The training, validation, and testing data are derived from a collection of tab-separated `.csv` files. Each file contains simulated event data, including columns like `epc_code`, `event_time`, `location_id`, and `business_step`. This data was generated to reflect the known statistical properties and anomaly patterns of our actual supply chain, as documented in our `EDA_QA.md`.

#### **1.2. How were anomalies labeled?**

*   **For Non-Technical Stakeholders:** Since we don't have a perfect, hand-labeled dataset, we created a highly accurate, rule-based system to label our data. This system, called the `MultiAnomalyDetector`, acts as our "expert," automatically identifying five key types of anomalies (like incorrect location or duplicate scans) based on clear business rules. This gives us a consistent and reliable source of labels for training the new, smarter LSTM model.
*   **For Technical Audience:** The ground-truth labels are generated heuristically by the `MultiAnomalyDetector` (`src/barcode/multi_anomaly_detector.py`). This system encapsulates our domain knowledge into a set of precise, rule-based detectors for five distinct anomaly classes. While these labels are not manually annotated, they are based on deterministic logic and have been extensively validated against our `EDA_QA.md` to ensure their correctness and alignment with business definitions of anomalies. This approach provides a scalable and consistent labeling strategy.

#### **1.3. How was the data split?**

*   **For Non-Technical Stakeholders:** To make sure our model is learning correctly, we split our data based on time. We train the model on older data and then test it on newer data it has never seen before. This is like teaching a student from last year's textbook and testing them on this year's exam—it's the only fair way to prove they have truly learned the material and not just memorized the answers.
*   **For Technical Audience:** We employ a **strict chronological split**. The data is first sorted globally by `event_time`. We then designate the first 70% of the data for training, the next 15% for validation (tuning the model), and the final 15% for testing (the final report card). This **prevents data leakage**, a critical flaw in time-series modeling where information from the future accidentally bleeds into the training set, leading to inflated and unrealistic performance metrics.

#### **1.4. Was the model trained on normal data only?**

*   **For Non-Technical Stakeholders:** We train the model on a mix of both normal and anomalous data. This is crucial because we want the model to learn what *normal* behavior looks like and, just as importantly, what the specific patterns of *abnormal* behavior look like. By seeing examples of both, it can more accurately distinguish between them.
*   **For Technical Audience:** We are training the model in a **supervised fashion**, using both normal and anomalous data points as labeled by the `MultiAnomalyDetector`. The alternative, an unsupervised approach (training on normal data only), is less suitable here because our anomalies have distinct, learnable signatures. A supervised approach allows the LSTM to explicitly learn the sequential patterns that define each of the five anomaly classes, leading to higher precision and recall, especially for subtle temporal deviations that an unsupervised model might miss.

---

### **Part 2: Step-by-Step Preprocessing Pipeline**

This pipeline is designed specifically for the LSTM model. It is **not** reusable for the SVM or rule-based systems because its primary purpose is to create **sequences** of data, a concept that does not apply to the other, single-event-based models.

| Step | Description | File / Status | Justification | 
| :--- | :--- | :--- | :--- | 
| 1. **Data Ingestion & Enrichment** | Merge all raw CSVs. Enrich with geospatial and business process data. | `src/barcode/lstm_preprocessor.py` (New) | Creates a single, unified dataset with all necessary information. | 
| 2. **Chronological Sort** | Sort the entire dataset by `event_time`. | `src/barcode/lstm_preprocessor.py` (New) | Establishes the temporal order required for sequence generation and splitting. | 
| 3. **Feature Engineering** | Create temporal, spatial, and behavioral features. | `src/barcode/lstm_preprocessor.py` (New) | Transforms raw data into a rich vector space that captures the dynamics of the supply chain. | 
| 4. **Label Generation** | Run the `MultiAnomalyDetector` to create the 5-dimensional label for each event. | `src/barcode/multi_anomaly_detector.py` (Existing) | Provides the ground-truth signal for the supervised learning task. | 
| 5. **Scaling** | Apply `StandardScaler` to all engineered features. | `src/barcode/lstm_preprocessor.py` (New) | Normalizes features to have zero mean and unit variance, which is essential for both PCA and stable neural network training. | 
| 6. **Dimensionality Analysis** | Use t-SNE to visualize feature clusters and a correlation matrix to identify redundancies. **Decision Point:** Only if significant redundancy is proven will PCA be used. | `notebooks/feature_analysis.ipynb` (New) | Avoids unnecessary complexity. We will not use PCA unless we can prove it improves performance or is required for latency. | 
| 7. **Sequence Generation** | Group data by `epc_code` and create overlapping sequences of length `L`. Use pre-padding for sequences shorter than `L`. | `src/barcode/lstm_preprocessor.py` (New) | This is the core step for the LSTM. It transforms the data from a set of individual events into a series of temporal sequences. | 
| 8. **Temporal Splitting** | Split the generated sequences into train, validation, and test sets based on time. | `src/barcode/lstm_preprocessor.py` (New) | Prevents data leakage and ensures a fair evaluation of the model's predictive power. | 

---

### **Part 3: Feature Justification and Analysis**

| Feature Name | Domain Rationale | Algorithmic Reason | Anomaly Pattern Detected | 
| :--- | :--- | :--- | :--- | 
| `time_gap_log` | Measures the time since the last scan for a product. | Normalizes a right-skewed distribution. | Unusually long or short delays between steps (`evtOrderErr`). | 
| `distance_traveled_km` | Calculates the physical distance the product moved since its last scan. | Provides a continuous measure of movement. | Impossible jumps in location (`locErr`, `jump`). | 
| `location_entropy` | Measures the randomness of an item's location history. | Quantifies the predictability of an asset's path. | Erratic, non-standard movement patterns. | 
| `is_valid_transition` | Checks if a movement between business steps is allowed (e.g., Factory -> Warehouse is okay, but Warehouse -> Factory is not). | Provides a strong, rule-based signal. | Business process violations (`evtOrderErr`). | 
| `events_per_hour` | Measures the recent velocity of an item's scans. | Captures the tempo of processing. | Unusually high scan rates (`epcDup`) or process stalls. | 

#### **Feature Redundancy Analysis:**

Before defaulting to PCA, we will conduct a thorough redundancy analysis:

1.  **Correlation Heatmap:** We will compute the Pearson correlation matrix for all engineered features. Pairs with a correlation coefficient > 0.9 or < -0.9 are candidates for removal.
2.  **t-SNE Visualization:** We will generate a 2D t-SNE plot of our feature space, colored by anomaly type. If different anomaly types form distinct, well-separated clusters, it suggests the feature space is already highly informative and may not need reduction. If the clusters are poorly defined, it may indicate that noise and redundancy are obscuring the signal, making a case for PCA.

**Only if these analyses show significant redundancy and that feature removal does not harm performance will we consider using PCA.**

---

### **Part 4: Plan for Fast and Interpretable LSTM Execution**

1.  **Fast Execution:**
    *   **Model:** We will use a **GRU** instead of an LSTM. It offers a similar performance with fewer parameters, resulting in faster inference.
    *   **Quantization & Runtime:** The final trained model will be converted to the **ONNX** format and then quantized to **INT8**. This will be served using the **ONNX Runtime**, which is highly optimized for low-latency CPU inference.
2.  **Sequence Length Optimization:** The sequence length `L` is a critical hyperparameter. We will test a range of values (e.g., 5, 10, 15, 20) and select the one that provides the best F1-score on the validation set.
3.  **Cold-Start Strategy:** For any product with fewer than `L` scans in its history, the system will not use the LSTM. It will fall back to the faster, stateless SVM and rule-based models. The API response will clearly indicate which model was used.
4.  **SHAP Explainability:** For every anomaly detected by the LSTM, we will use the **KernelSHAP** explainer to calculate feature importance scores. Because we are avoiding PCA, these scores will be directly tied to our interpretable engineered features. The API response will include the top 3 features that contributed to the anomaly detection, providing actionable insights to the end-user (e.g., `"reason": "High location_entropy, High time_gap_log"`).
