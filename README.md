# Barcode Anomaly Detection - Prompt Engineering Learning Project

## Project Purpose

This project serves as a **learning laboratory for prompt engineering and AI collaboration** in the context of supply chain anomaly detection. As a data analyst, I'm using this real-world use case to experiment with and develop systematic AI interaction methodologies.

### Learning Objectives
- **Prompt Engineering**: Developing reusable, systematic AI interaction patterns
- **AI Collaboration**: Building frameworks for efficient human-AI workflows  
- **Domain Application**: Applying prompt engineering to data analysis and ML tasks
- **Process Documentation**: Creating reproducible methodologies for AI-assisted development

### My Role
**Data Analyst** focusing on:
- Anomaly detection algorithm design and implementation
- Statistical analysis and model validation
- AI-assisted code development and optimization
- Systematic prompt engineering experimentation

## Technical Implementation

### Anomaly Detection System
Real-time API for supply chain barcode anomaly detection featuring:
- **Rule-based Detection**: 5 anomaly types (epcFake, epcDup, locErr, evtOrderErr, jump)
- **Machine Learning**: One-Class SVM for statistical outlier detection
- **Future Scope**: Graph Neural Networks (GNN) for relationship-based anomaly detection

### Prompt Engineering Framework
Advanced AI collaboration system with metadata lineage tracking:
- **Structured Protocols**: Analysis logs, decision documentation, context management
- **Automation Templates**: JSON-based command system for reproducible interactions  
- **Knowledge Accumulation**: Persistent decision history and pattern recognition
- **Metadata Lineage**: Complete tracking of prompt evolution and derivation
- **Separation of Concerns**: context (WHAT), protocol (HOW), meta (prompt-for-prompt), templates (REUSABLE), task (EXECUTE)

## Quick Start

### Prerequisites
- Python 3.8+ with conda environment named `ds`
- Git for version control and collaboration tracking

### Environment Setup
```bash
# Activate conda environment
conda activate ds

# Navigate to project directory
cd path/to/barcode-anomaly-detection

# Install dependencies  
pip install -r requirements.txt
```

### Running the System
```bash
# 1. Rule-based anomaly detection (modularized)
python src/barcode/anomaly_detection_decomposed.py

# 2. Train SVM model
python src/barcode/svm_anomaly_detection_v2.py

# 3. Start API server
python src/barcode/api.py
```

## Project Structure

### Core Components
```
├── automation/                     # Session lifecycle management
│   ├── README.md                   # How to use automation system  
│   ├── command.json               # Entry point (session setup)
│   ├── ai_handoff.json            # Exit point (task handoff)
│   ├── update_index.json          # Directory structure maintenance
│   └── directory_scan.json        # Pure scanning utility
├── src/barcode/                    # Main application code
├── prompts/                        # AI interaction framework (restructured)
│   ├── context/                    # WHAT the project is
│   │   ├── ai_handoff.txt          # Complete project context
│   │   ├── principle.llm.txt       # Project specifications
│   │   └── metadata.json           # Lineage tracking
│   ├── protocol/                   # HOW AI should behave
│   │   ├── learning_v1.llm.txt     # Educational interaction
│   │   ├── analysis_log_behavior.llm.txt # Decision tracking
│   │   ├── question_loop.llm.txt   # Systematic questioning protocol
│   │   └── metadata.json           # Lineage tracking
│   ├── meta/                       # HOW to design prompts (prompt-for-prompt)
│   │   ├── automation_guide.txt    # Meta-automation guidance
│   │   ├── file_registry.json      # Path tracking design patterns
│   │   └── metadata.json           # Lineage tracking
│   ├── templates/                  # REUSABLE blank forms (copy to customize)
│   │   ├── function_generation_template.json # Blank form for code generation
│   │   ├── analysis_template.json  # Blank form for analysis tasks  
│   │   └── metadata.json           # Lineage tracking
│   ├── task/                       # FILLED forms ready to execute (domain-specific)
│   │   ├── anomaly_detection/      # Current domain: barcode anomalies
│   │   │   ├── function_generation.json # Filled template for this project
│   │   │   ├── edge.txt            # Domain-specific edge cases
│   │   │   └── refactoring_workflow.json # Domain-specific workflow
│   │   └── metadata.json           # Lineage tracking
│   └── log/                        # WHAT happened (conversation history)
│       └── metadata.json           # Lineage tracking
├── index.llm.txt                   # Project summary for AI consultation
├── data/                           # Raw and processed datasets
└── docs/                           # Project documentation
```

### Key Learning Artifacts
- **Analysis Logs**: `prompts/log/` - Documented decision-making process
- **Automation Guide**: `prompts/meta/` - Meta-prompts for creating new prompts
- **Protocol Files**: `prompts/protocol/` - Systematic AI interaction rules
- **Metadata Lineage**: `*/metadata.json` - Complete prompt evolution tracking
- **Command System**: `command.json` - Automated task execution with logging
- **Git History**: Detailed commit messages for AI training data

## Learning Outcomes & Applications

### Prompt Engineering Techniques Developed
1. **Systematic Inquiry Protocol**: 4-step analysis process (motivation → ambiguities → understanding → strategies)
2. **Context Management**: Persistent knowledge files with automatic loading
3. **Decision Documentation**: Analysis logs for building cumulative expertise
4. **Automation Templates**: JSON-based commands for reproducible AI interactions
5. **Metadata Lineage Tracking**: Complete prompt genealogy and evolution tracking
6. **Separation of Concerns**: Clear distinction between context, protocol, meta, templates, and tasks
7. **Command Automation**: Single-command execution of complex AI workflows

### Transferable Frameworks
- **Multi-modal AI Collaboration**: Structured approach applicable to any technical domain
- **Knowledge Accumulation**: Methods for building persistent AI collaboration expertise
- **Process Documentation**: Templates for systematic AI-assisted development

### Future Applications
- Training other developers in effective AI collaboration
- Building AI-assisted data analysis workflows
- Developing domain-specific prompt engineering patterns
- Creating enterprise AI collaboration standards

## Anomaly Detection Results

To run the anomaly detection system, follow these steps:

1.  **Open Command Prompt (CMD) or Terminal:**
    *   **Windows:** Search for "cmd" in the Start Menu and open Command Prompt.
    *   **macOS/Linux:** Open the "Terminal" application.

2.  **Navigate to the Project Directory:**
    In the command prompt/terminal, use the following command to navigate to the root directory of this project. (Replace the example path with your actual project path.)
    ```bash
        # Replace <path/to/your/downloaded/barcode-anomaly-detection-folder> with the actual path to the directory where you downloaded this project.
    cd <path/to/your/downloaded/barcode-anomaly-detection-folder>
    ```

3.  **Install Dependencies:**
    Ensure you have Python installed. (If not, you can download it from [python.org](https://www.python.org/downloads/)). Then, install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Data Preprocessing (Rule-Based Anomaly Detection):**
    Run the script for initial data processing and rule-based anomaly detection. This step generates cleaned data and identifies rule-based anomalies.
    ```bash
    python src/barcode-anomaly-detection/anomaly_detection_v5.py
    ```

5.  **Train Machine Learning Model (SVM Anomaly Detection):**
    Execute the script to train the One-Class SVM model. This model is used for statistical anomaly detection.
    ```bash
    python src/barcode-anomaly-detection/svm_anomaly_detection_v2.py
    ```

6.  **Start the Anomaly Detection API:**
    Launch the FastAPI application. This API will serve as the interface for real-time anomaly detection.
    ```bash
    python src/barcode-anomaly-detection/api.py
    ```
    The API will typically run on `http://127.0.0.1:8000`.

    Example output after starting the API:
    ```
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    ... (other log messages) ...
    ```

```
$ python anomaly_detection_v5.py 

Starting to load raw data from: /Users/sd/Works/samples/AnomalyDetection/raw
- Successfully loaded and added hws.csv
- Successfully loaded and added icn.csv
- Successfully loaded and added kum.csv
- Successfully loaded and added ygs.csv
Successfully merged 4 files with a total of 920000 records.
--- Rule-Based Anomaly Detection Results (Local Test) ---
Total Rule-Based Anomaly Counts:
anomaly_type
evtOrderErr    600
Name: count, dtype: int64
Rule-based clean data saved to: /Users/sd/Works/samples/AnomalyDetection/csv/local_rule_clean_test.csv

$ python svm_anomaly_detection_v2.py

--- Full Anomaly Detection and Reporting Workflow ---
--- Running SVM Anomaly Detection Logic ---
Training new One-Class SVM model...
SVM 모델이 저장되었습니다: /Users/sd/Works/samples/AnomalyDetection/model/svm_20250709_122328.pkl
Predicting anomalies...
Predicting anomalies...
Found 610854 statistical anomalies using One-Class SVM.
SVM anomaly report saved to: /Users/sd/Works/samples/AnomalyDetection/csv/svm_anomalies_report_v1.csv

$ python model_serving_api.py
/Users/sd/Works/samples/AnomalyDetection/model_serving_api.py:155: DeprecationWarning: 
        on_event is deprecated, use lifespan event handlers instead.

        Read more about it in the
        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).
        
  @app.on_event("startup")
INFO:     Started server process [24211]
INFO:     Waiting for application startup.
2025-07-09 12:25:42,528 - __main__ - INFO - Loading static data and models for anomaly detection...
2025-07-09 12:25:42,530 - __main__ - INFO - Loaded transition stats: 82 records
2025-07-09 12:25:42,531 - __main__ - INFO - Loaded geospatial data: 58 unique locations
2025-07-09 12:25:42,531 - __main__ - INFO - Loaded SVM model: svm_20250709_122328.pkl
2025-07-09 12:25:42,531 - __main__ - INFO - Static data loaded successfully
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
2025-07-09 12:26:01,915 - __main__ - INFO - --- Received Request for Anomaly Detection ---
2025-07-09 12:26:01,917 - __main__ - INFO - Starting rule-based anomaly detection...
2025-07-09 12:26:01,923 - __main__ - INFO - Rule-based detection found 0 anomalies
2025-07-09 12:26:01,923 - __main__ - INFO - Starting SVM anomaly detection...
--- Running SVM Anomaly Detection Logic ---
Using pre-trained SVM model
Predicting anomalies...
Found 9 statistical anomalies using One-Class SVM.
2025-07-09 12:26:01,927 - __main__ - INFO - SVM detection found 9 anomalies
2025-07-09 12:26:01,929 - __main__ - INFO - Detection completed in 0.01 seconds
INFO:     127.0.0.1:55659 - "POST /detect_anomalies HTTP/1.1" 200 OK

{"message":"Anomaly detection completed successfully","processing_time_seconds":0.012598,"model_version":"svm_20250709_122328.pkl","statistics":{"total_input_records":13,"rule_based_anomalies":0,"svm_anomalies":9,"total_anomalies":9,"anomaly_rate_percent":69.23},"warnings":[],"anomalies":[{"epc_code":"001.8804823.1293291.010004.20250701.000004002","event_time":"2025-07-01 10:34:17","scan_location":"인천공장","event_type":"Aggregation","anomaly_type":"svm_anomaly","factory":"icn"},{"epc_code":"001.8804823.1293291.010004.20250701.000004002","event_time":"2025-07-01 11:02:38","scan_location":"인천공장창고","event_type":"WMS_Inbound","anomaly_type":"svm_anomaly","factory":"icn"},{"epc_code":"001.8804823.1293291.010004.20250701.000004002","event_time":"2025-07-30 02:09:38","scan_location":"수도권물류센터","event_type":"HUB_Inbound","anomaly_type":"svm_anomaly","factory":"icn"},{"epc_code":"001.8804823.1293291.010004.20250701.000004002","event_time":"2025-08-14 13:20:38","scan_location":"수도권물류센터","event_type":"HUB_Outbound","anomaly_type":"svm_anomaly","factory":"icn"},{"epc_code":"001.8804823.1293291.010004.20250701.000004002","event_time":"2025-09-18 00:14:38","scan_location":"수도권_도매상3_권역_소매상2","event_type":"R_Stock_Inbound","anomaly_type":"svm_anomaly","factory":"icn"},{"epc_code":"001.8804823.1293291.010004.20250701.000004002","event_time":"2025-09-24 16:23:38","scan_location":"수도권_도매상3_권역_소매상2","event_type":"R_Stock_Outbound","anomaly_type":"svm_anomaly","factory":"icn"},{"epc_code":"001.8804823.1293291.010004.20250701.000004002","event_time":"2025-10-27 08:43:38","scan_location":"수도권_도매상3_권역_소매상2","event_type":"POS_Sell","anomaly_type":"svm_anomaly","factory":"icn"},{"epc_code":"001.8804823.1293291.010004.20250701.000004003","event_time":"2025-07-01 10:34:17","scan_location":"인천공장","event_type":"Aggregation","anomaly_type":"svm_anomaly","factory":"icn"},{"epc_code":"001.8804823.1293291.010004.20250701.000004003","event_time":"2025-07-01 11:02:38","scan_location":"인천공장창고","event_type":"WMS_Inbound","anomaly_type":"svm_anomaly","factory":"icn"}]}%                                                           
```
"# barcode-anomaly-detection" 
