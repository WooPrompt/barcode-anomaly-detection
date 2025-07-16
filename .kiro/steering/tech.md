# Technology Stack & Build System

## Core Technologies

### Backend Stack
- **Python 3.8+** - Primary development language
- **FastAPI** - REST API framework with automatic documentation
- **Uvicorn** - ASGI server for production deployment
- **Pydantic** - Data validation and serialization
- **Pandas** - Data processing and CSV handling
- **NumPy** - Numerical computations
- **Scikit-learn** - Machine learning models (SVM)

### Development Environment
- **Conda environment**: `ds` (data science environment)
- **Git** - Version control with detailed commit history for AI training
- **JSON** - Primary data exchange format

### Data Processing
- **CSV files** - Location mapping and geospatial data
- **JSON** - API request/response format
- **Pickle** - Model serialization (SVM models in `/model` directory)

## Common Commands

### Environment Setup
```bash
# Activate conda environment
conda activate ds

# Install dependencies
pip install -r requirements.txt
```

### Development Server
```bash
# Start FastAPI server (recommended)
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000 --reload

# Alternative start method
python fastapi_server.py

# Windows batch restart
restart_server.bat
```

### Testing
```bash
# Test multi-anomaly detection
python test_anomaly_api.py

# Test individual detection modules
python src/barcode/multi_anomaly_detector.py

# Test current output format
python test_current_output.py

# Test backend integration
python test_backend_fix.py
```

### API Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Performance Requirements
- **Response Time**: <100ms for real-time detection
- **Data Volume**: Handle 920,000+ records efficiently
- **Concurrent Processing**: Multi-anomaly detection in parallel
- **Memory Management**: Efficient model loading/unloading

## Key Dependencies
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pandas==2.1.4
numpy==1.25.2
scikit-learn==1.3.2
python-multipart==0.0.6
```

## Production Considerations
- **CORS enabled** for frontend integration
- **Automatic model reloading** without service interruption
- **Error handling** with proper HTTP status codes
- **Logging** for debugging and monitoring