# Project Structure & Organization

## Root Level Files
- **fastapi_server.py** - Main FastAPI application entry point
- **requirements.txt** - Python dependencies
- **README.md** - Comprehensive project documentation
- **restart_server.bat** - Windows server restart script
- **test_*.py** - Various API and integration test files
- **postman_test_data.json** - Sample data for API testing

## Core Application Structure

### `/src/barcode/` - Main Application Code
- **multi_anomaly_detector.py** - Primary detection engine with all 5 anomaly types
- **anomaly_detection_combined.py** - Combined detection logic
- **api.py** - API endpoint definitions
- **benchmark.py** - Performance testing utilities
- **transition_time_analyzer_v2.py** - Travel time analysis for jump detection

### `/data/` - Data Storage
- **`/raw/`** - Original CSV files and datasets
- **`/processed/`** - Cleaned data including `location_id_withGeospatial.csv`
- **`/external/`** - Third-party data sources
- **`/detection_logs/`** - Anomaly detection results and logs

### `/model/` - Machine Learning Models
- **svm_*.pkl** - Trained SVM models with timestamps
- **.keep** - Ensures directory exists in git

### `/docs/` - Documentation
- **project_analysis.md** - Comprehensive project analysis
- **COMPLETE_TEAM_REQUIREMENTS.md** - Team integration requirements
- **DETAILED_LOGIC_EXPLANATION.md** - Anomaly detection logic
- **folder_structure.txt** - Directory organization

### `/prompts/` - AI Collaboration Framework
- **`/automation/`** - Session lifecycle management
- **`/context/`** - Project specifications and principles
- **`/protocol/`** - AI behavior guidelines
- **`/meta/`** - Prompt engineering patterns
- **`/templates/`** - Reusable prompt templates
- **`/task/`** - Domain-specific filled templates
- **`/log/`** - Conversation history and decisions

### `/tests/` - Testing Framework
- **test_json_output.py** - JSON format validation
- **README.md** - Testing guidelines

### `/playground/` - Development Sandbox
- **test_data_generate.ipynb** - Jupyter notebook for data generation
- **README.md** - Playground usage instructions

### `/.kiro/` - Kiro IDE Configuration
- **`/specs/`** - Feature specifications and requirements
- **`/steering/`** - AI assistant guidance rules

## Key Architectural Patterns

### API-First Design
- FastAPI with automatic OpenAPI documentation
- Pydantic models for request/response validation
- RESTful endpoints with consistent JSON format

### Multi-Anomaly Detection Pipeline
- **Rule-based detection** for clear violations
- **Statistical analysis** for travel time anomalies
- **Format validation** for EPC structure
- **Sequence analysis** for event order errors
- **Geospatial validation** for location hierarchy

### Data Flow Architecture
1. **Input**: JSON with eventId, location_id, business_step, event_type
2. **Processing**: CSV location mapping + anomaly detection functions
3. **Output**: Clean JSON with null value removal and confidence scores

### File Naming Conventions
- **test_*.py** - Test files with descriptive suffixes
- **svm_YYYYMMDD_HHMMSS.pkl** - Timestamped model files
- **location_id_withGeospatial.csv** - Descriptive data file names
- **multi_anomaly_detector.py** - Primary module names reflect functionality

### Configuration Management
- **Environment-based** - Uses conda `ds` environment
- **CSV-driven** - Location mapping via external CSV files
- **Model versioning** - Timestamped pickle files for ML models
- **API documentation** - Auto-generated from code annotations