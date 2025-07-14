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
# 1. Start FastAPI server (recommended)
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload

# 2. Alternative server start
python fastapi_server.py

# 3. Test multi-anomaly detection system
python test_anomaly_api.py

# 4. Test built-in examples
python src/barcode/multi_anomaly_detector.py
```

## Project Structure

### Core Components
```
├── src/barcode/                    # Main application code
├── prompts/                        # AI interaction framework (restructured)
│   ├── automation/                 # Session lifecycle management
│   │   ├── README.md              # How to use automation system  
│   │   ├── init.json              # Entry point (session initialization)
│   │   ├── ai_handoff.json        # Exit point (task handoff)
│   │   ├── update_index.json      # Directory structure maintenance
│   │   └── directory_scan.json    # Pure scanning utility
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

## How to Test the Anomaly Detection System

### Quick Testing (Recommended)

1. **Start the FastAPI server:**
   ```bash
   uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Test with browser:**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Quick Test: http://localhost:8000/ 

3. **Test with sample data:**
   ```bash
   python test_anomaly_api.py
   ```
   This uses `test_data_sample.json` and tests all 5 anomaly types.

4. **Test built-in examples:**
   ```bash
   python src/barcode/multi_anomaly_detector.py
   ```

### Manual Testing

1. **Navigate to project directory:**
   ```bash
   cd path/to/barcode-anomaly-detection
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run comprehensive tests:**
   ```bash
   # Start FastAPI server
   uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
   
   # Test multi-anomaly detection (in another terminal)
   python test_anomaly_api.py
   
   # Test individual components  
   python src/barcode/anomaly_detection_combined.py
   ```

### Understanding Test Results

The system detects 5 types of anomalies:
- **epcFake**: Invalid EPC format (structure, company code, dates)
- **epcDup**: Impossible duplicate scans (same EPC, different locations, same time)
- **jump**: Impossible travel times between locations
- **evtOrderErr**: Invalid event sequences (consecutive inbound/outbound)
- **locErr**: Location hierarchy violations (retail → wholesale)

### Sample Output
```
DETECTION RESULTS:
Total anomalies found: 1
Multi-anomaly EPCs: 1
Summary: {'epcFake': 0, 'epcDup': 1, 'locErr': 0, 'evtOrderErr': 1, 'jump': 1}

1. EPC: 001.8809437.1203199.150002.20250701.000000001
   Anomaly Types: ['epcDup', 'jump', 'evtOrderErr']
   Scores: {'epcDup': 90, 'jump': 95, 'evtOrderErr': 25}
   Primary Issue: jump
   Severity: HIGH
```

## API Integration

### Using the Detection API

1. **Start the FastAPI server:**
   ```bash
   uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Test with sample data (Postman or curl):**
   ```bash
   # Using curl
   curl -X POST "http://localhost:8000/api/v1/barcode-anomaly-detect" \
        -H "Content-Type: application/json" \
        -d @real_sample_data.json
   
   # Or simple test
   curl -X POST "http://localhost:8000/api/v1/test-with-sample"
   ```

3. **Expected API response:**
   ```json
   {
     "EventHistory": [
       {
         "epcCode": "001.8809437.1203199.150002.20250701.000000001",
         "anomalyTypes": ["epcDup", "jump", "evtOrderErr"],
         "anomalyScores": {"epcDup": 90, "jump": 95, "evtOrderErr": 25},
         "severity": "HIGH",
         "primaryAnomaly": "jump"
       }
     ],
     "summaryStats": {"epcFake": 0, "epcDup": 1, "locErr": 0, "evtOrderErr": 1, "jump": 1},
     "totalAnomalyCount": 1
   }
   ```

## Troubleshooting

### Common Issues

1. **Unicode/Encoding Errors (Windows):**
   - Test files have been updated to remove emoji characters
   - Use `python test_anomaly_api.py` for cross-platform compatibility

2. **Missing Dependencies:**
   ```bash
   pip install -r requirements.txt
   # Or manually:
   pip install fastapi uvicorn pandas numpy scikit-learn pydantic
   ```

3. **Module Import Errors:**
   - Ensure you're in the project root directory
   - Check Python path includes `src/` directory 
