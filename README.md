# Supply Chain Anomaly Detection API

## 설치 및 실행

이상 감지 시스템을 실행하려면 다음 단계를 따르십시오:

1.  **명령 프롬프트(CMD) 또는 터미널 열기:**
    *   **Windows:** 시작 메뉴에서 "cmd"를 검색하여 명령 프롬프트를 엽니다.
    *   **macOS/Linux:** "터미널" 앱을 엽니다.

2.  **프로젝트 디렉토리로 이동:**
    명령 프롬프트/터미널에서 다음 명령을 사용하여 이 프로젝트의 루트 디렉토리로 이동합니다. (예시 경로는 실제 프로젝트 경로로 대체해야 합니다.)
    ```bash
        # Replace <path/to/your/downloaded/barcode-anomaly-detection-folder> with the actual path to the directory where you downloaded this project.
    cd <path/to/your/downloaded/barcode-anomaly-detection-folder>
    ```

3.  **종속성 설치:**
    Python이 설치되어 있는지 확인하십시오. (Python이 없다면 [python.org](https://www.python.org/downloads/)에서 설치할 수 있습니다.) 그런 다음 pip를 사용하여 필요한 라이브러리를 설치하십시오:
    ```bash
    pip install -r requirements.txt
    ```

4.  **데이터 전처리 (규칙 기반 이상 감지):**
    초기 데이터 처리 및 규칙 기반 이상 감지를 위한 스크립트를 실행하십시오. 이 단계는 정제된 데이터를 생성하고 규칙 기반 이상을 식별합니다.
    ```bash
    python src/barcode-anomaly-detection/anomaly_detection_v5.py
    ```

5.  **머신러닝 모델 학습 (SVM 이상 감지):**
    One-Class SVM 모델을 학습시키기 위한 스크립트를 실행하십시오. 이 모델은 통계적 이상 감지에 사용됩니다.
    ```bash
    python src/barcode-anomaly-detection/svm_anomaly_detection_v2.py
    ```

6.  **이상 감지 API 시작:**
    FastAPI 애플리케이션을 시작하십시오. 이 API는 실시간 이상 감지를 위한 인터페이스 역할을 합니다.
    ```bash
    python src/barcode-anomaly-detection/api.py
    ```
    API는 일반적으로 `http://127.0.0.1:8000`에서 실행됩니다.

    API 시작 후 예시 출력:
    ```
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    ... (기타 로그 메시지) ...
    ```

## Installation and Execution

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
