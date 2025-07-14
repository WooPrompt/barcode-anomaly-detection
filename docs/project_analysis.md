# Project Analysis: 2D Barcode-Based Logistics Supply Chain Illegal Distribution Analysis Web Service

## 1. Project Overview & Team Structure

**Team Leader & Data Analyst**: [Your Role Here]
**Frontend Team**: [Frontend Developer 1], [Frontend Developer 2]
**Backend Team**: [Backend Developer]

This project aims to develop a web service for analyzing illegal distribution in a logistics supply chain using 2D barcode data. It involves:
*   **Anomaly Detection:** Identifying various types of anomalies (e.g., fake EPCs, duplicate scans, location errors, event order errors, impossible travel times) using both rule-based logic and machine learning (SVM, GNN).
*   **Data Management:** Handling CSV file uploads, processing, and storage of barcode scan data.
*   **User Management:** Admin functionalities for user approval and status management.
*   **Visualization & Reporting:** Providing a web-based dashboard for visualizing logistics flow, anomaly occurrences, and generating detailed reports (PDF/Excel).
*   **API-driven:** The system is built with a Python FastAPI backend and a React frontend, communicating via REST APIs.

## 2. Anomaly Types & Definitions

The project focuses on detecting the following anomalies:

*   **`jump` (시공간 점프):** Non-logical spatiotemporal movement (e.g., impossible travel time).
*   **`evtOrderErr` (이벤트 순서 오류):** Non-logical event type sequence (e.g., inbound after outbound at the same location).
*   **`epcFake` (위조):** EPC code generation rule violation (malformed EPC).
*   **`epcDup` (복제):** Same EPC scanned at different locations at the exact same time.
*   **`locErr` (경로 위조):** Product at a wrong or illogical location (e.g., retail back to factory).
*   **`model`:** Anomalies detected by ML models (e.g., SVM, GNN) that require expert review.

## 3. Data Schema & Key Entities

The system deals with several key data entities:

*   **Event History:** Core barcode scan data including `scan_location`, `location_id`, `hub_type`, `business_step`, `event_type`, `epc_code` (decomposed into `epc_header`, `epc_company`, `epc_product`, `epc_lot`, `epc_manufacture`, `epc_serial`), `product_name`, `event_time`, `manufacture_date`, `expiry_date`.
*   **User:** `userId`, `password`, `userName`, `email`, `phone`, `locationId`, `role`, `status`.
*   **Product:** `epcProduct`, `productName`.
*   **Location:** `locationId`, `scanLocation`.
*   **CSV File Metadata:** `fileId`, `fileName`, `filePath`, `fileSize`, `createdAt`, `memberId`.
*   **Anomaly Output:** `epcCode`, `productName`, `eventType`, `businessStep`, `scanLocation`, `eventTime`, `anomaly` (boolean), `anomalyType` (Korean description), `anomalyCode` (e.g., "jump"), `description`.

## 4. API Data Flow & Your Role as Data Analyst

### Input from Backend Team
**Backend will send you JSON data in this format:**
```json
{
  "data": [
    {
      "scan_location": "인천공장",
      "location_id": 1,
      "hub_type": "ICN_Factory",
      "business_step": "Factory",
      "event_type": "Aggregation",
      "operator_id": 1,
      "device_id": 1,
      "epc_code": "001.8805843.2932031.010001.20250701.000000001",
      "epc_header": "001",
      "epc_company": "8805843",
      "epc_product": "2932031",
      "epc_lot": "010001",
      "epc_manufacture": "20250701",
      "epc_serial": "000000001",
      "product_name": "Product 1",
      "event_time": "2025-07-01 10:23:38",
      "manufacture_date": "2025-07-01 10:23:38",
      "expiry_date": "20251231"
    }
    // ... more events
  ]
}
```

### Your API Endpoint
**Backend will call your API:**
```
POST /api/v1/barcode-anomaly-detect
```

### Output to Backend Team
**You need to return JSON in this format:**
```json
{
  "EventHistory": [
    {
      "epcCode": "001.8809437.1203199.150002.20250701.000000002",
      "productName": "Product 2",
      "eventType": "jump",
      "businessStep": "Factory",
      "scanLocation": "구미공장",
      "eventTime": "2025-07-01 10:23:39",
      "anomaly": true,
      "anomalyType": "비논리적인 시공간 이동",
      "anomalyCode": "jump",
      "description": "공장에서 리테일로 점프 이동 발생"
    },
    {
      "epcCode": "001.8805843.2932031.150001.20250701.000000001",
      "productName": "Product 1",
      "eventType": "evtOrderErr",
      "businessStep": "Factory",
      "scanLocation": "구미공장",
      "eventTime": "2025-07-01 10:23:39",
      "anomaly": true,
      "anomalyType": "이벤트 순서 오류",
      "anomalyCode": "evtOrderErr",
      "description": "공장 입고 후 공장 출고 순서 오류"
    }
  ]
}
```

## 5. Backend Team's Needs (from Data Analyst)

The Backend Developer is responsible for the FastAPI implementation and integrating your anomaly detection logic. They need the following from you:

*   **Anomaly Detection Logic:** The core Python functions for each anomaly type (`epcFake`, `epcDup`, `locErr`, `evtOrderErr`, `jump`). These should be robust, efficient, and return results in the specified JSON format.
*   **ML Model Integration:** Trained SVM and GNN models for anomaly detection. This includes:
    *   Clean, labeled datasets for training (especially normal data for SVM).
    *   Understanding how to handle `n` values for SVM (e.g., `n=0.01` issue).
    *   Clarification on how to handle "model" detected anomalies (those not caught by rules).
*   **Data Input/Output Formats:** Ensure your Python anomaly detection functions can receive data in the specified JSON format (list of event objects) and output anomalies in the required JSON format (list of anomaly objects with `epcCode`, `anomalyType`, `summaryStats`, etc.).
*   **Location Data:** Provide accurate geospatial data (coordinates) for `scan_location` and `location_id` to enable `jump` anomaly detection and map visualization.
*   **KPI Calculation Logic:** Define the exact logic and data sources for calculating KPIs like `totalTripCount`, `uniqueProductCount`, `codeCount`, `anomalyCount`, `anomalyRate`, `salesRate`, `dispatchRate`, `inventoryRate`, and `avgLeadTime`.
*   **Data Schema Confirmation:** Work with the backend to finalize the database schema based on the data models discussed.
*   **Performance Metrics:** Provide runtime analysis for each anomaly detection function to help the backend optimize API timeout settings.
*   **Anomaly Probability/Confidence:** The Backend Developer is interested in showing a "percentage of anomaly" for an EPC. This is likely related to the output of ML models. You need to explain how this can be achieved (e.g., using confidence scores from SVM or GNN) and how to present this concept to the team.

## 6. Frontend Team's Needs (from Data Analyst)

The Frontend Developers are building the React UI, dashboards, and reports. They need the following from you:

### **Frontend Developer 1 (Report Generation)**
- **Product/Lot-based Reports**: `GET /api/reports?product=제품A&lot=로트1-001&menu=이상탐지리포트&type=시공간점프`
- **Report Details**: `GET /api/report/detail?reportId=report_002`
- **Statistical Summaries**: Chart.js integration with summaryStats format

### **Frontend Developer 2 (Dashboard & Role Management)**  
- **Role-based Access**: 총괄 관리자 vs 공장별 매니저 권한
- **Map Visualization**: Using location_id_withGeospatial.csv coordinates
- **Factory-specific Dashboards**: Each factory manager sees only their data
- **Visualizing Anomalies in Sequence**: For a given EPC, if multiple anomalies are detected, the Frontend Developer wants to visualize *where* in the sequence each anomaly occurred (e.g., red dots on the map at the problematic scan locations). This requires your anomaly detection output to pinpoint the exact event(s) that triggered the anomaly.

The frontend team needs the following from you:

*   **API Endpoints & Data:** Consistent and reliable data from the backend via the defined API endpoints for:
    *   CSV file preview (`GET /api/{role}/file/{fileName}`)
    *   Report lists (`GET /api/report`)
    *   Report details (`GET /api/report/detail`)
    *   Node coordinates (`GET /api/manager/nodes`)
    *   Anomalous trip data (`GET /api/manager/anomalies`)
    *   All trip data (`GET /api/manager/trips`)
    *   KPI summary (`GET /api/manager/kpi`)
    *   Inventory distribution (`GET /api/manager/inventory`)
*   **Data for Visualization:** Specifically, `summaryStats` in the report detail API for Chart.js integration (bar and doughnut charts).
*   **Geospatial Data for Maps:** Accurate coordinates for locations to enable map-based visualizations of logistics flow and anomalies. The frontend has a strategy for mapping `avg_hours` to physical distances.
*   **Clear Anomaly Descriptions:** The `anomalyType` and `description` fields in the anomaly output are crucial for user understanding in reports.
*   **CSV File Handling:** The frontend handles CSV file upload (`POST /api/manager/upload`) and download (`GET /download/{fileLogId}`). Your backend logic needs to support this.
*   **Filtering & Search:** The frontend is implementing robust filtering and search capabilities. Ensure the backend APIs can support these queries efficiently.

## 7. Anomaly Detection Principles & Logic (for Presentation)

As the Data Analyst, your role is to not only implement the anomaly detection but also to clearly articulate its principles and logic to the team, especially for presentation purposes.

### General Approach: Rule-Based First, Then Machine Learning

Our strategy is to first apply a set of well-defined, rule-based anomaly detection functions. These rules are designed to catch common, clear-cut violations of expected logistics flow. For more subtle or complex anomalies that might not be caught by rules, we will leverage Machine Learning (ML) models.

### Detailed Logic for Each Rule-Based Anomaly

Each anomaly detection function operates on the event history of an EPC code, analyzing specific aspects of its journey.

#### 7.1. `epcFake` (Fake EPC)

*   **Principle:** Validates the structural integrity of the EPC code itself. EPCs follow a strict format (e.g., `001.CompanyCode.ProductCode.LotCode.ManufactureDate.SerialNumber`). Any deviation from this format indicates a potentially fake or malformed EPC.
*   **Logic:**
    1.  **Parsing:** The EPC string is parsed into its constituent parts (header, company, product, lot, manufacture date, serial).
    2.  **Format Validation:** Each part is checked against predefined rules (e.g., `epc_header` must be "001", `epc_company` must be 7 digits, `epc_manufacture` must be a valid date in YYYYMMDD format).
    3.  **Edge Cases:** Handles cases where parts are missing, have incorrect lengths, or contain invalid characters. If any part fails validation, the EPC is flagged as fake.
*   **Problem Solved:** Prevents invalid or non-standard EPCs from entering the system, ensuring data quality at the source.

#### 7.2. `epcDup` (Duplicate EPC)

*   **Principle:** Detects instances where the *same* EPC code is scanned at *different physical locations* at the *exact same timestamp*. This is physically impossible for a single item and indicates a duplicate or cloned EPC.
*   **Logic:**
    1.  **Grouping:** Group events by `epc_code` and `event_time`.
    2.  **Location Check:** For each group, if there are multiple events with the same `epc_code` and `event_time` but different `scan_location` values, it's flagged as a duplicate.
*   **Edge Cases:**
    *   **Near-simultaneous scans:** The current rule is strict on "exact same timestamp." If scans are milliseconds apart but at different locations, they might be considered legitimate. This can be refined with a small time window if needed.
    *   **Same location, multiple scans:** Not an `epcDup` anomaly, as it's physically possible.
*   **Problem Solved:** Identifies cloned or duplicated products in the supply chain, which can indicate counterfeiting or unauthorized parallel imports.

#### 7.3. `locErr` (Location Error / Path Forgery)

*   **Principle:** Identifies illogical or impossible movements of a product within the supply chain based on predefined business steps and geographical constraints. This includes backward movements or jumps to unrelated locations.
*   **Logic:**
    1.  **Sequence Analysis:** For each EPC, events are ordered by `event_time`.
    2.  **Business Step Hierarchy:** A predefined hierarchy of `business_step` (e.g., Factory -> WMS -> Logistics_HUB -> W_Stock -> R_Stock -> POS_Sell) is used.
    3.  **Geospatial Validation:** Utilizes `location_id_withGeospatial.csv` to check if the movement between two `scan_location` points is geographically plausible or if it violates expected regional flows (e.g., a product from a specific factory should not appear in a distant region without passing through intermediate hubs).
    4.  **Reverse Flow:** Explicitly flags any movement that goes against the general downstream flow (e.g., from Retail back to Factory).
*   **Edge Cases:**
    *   **Legitimate Returns/Repairs:** The current logic flags reverse flows. If legitimate returns or repairs are part of the business process, these would be flagged as anomalies and require manual review.
    *   **Missing Location Data:** If `scan_location` or `location_id` is missing, the anomaly cannot be accurately determined for that event.
*   **Problem Solved:** Detects products that deviate from their expected logistics path, indicating potential diversion, theft, or misrouting.

#### 7.4. `evtOrderErr` (Event Order Error)

*   **Principle:** Checks if the sequence of `event_type` within a specific `scan_location` is logically consistent. For example, an "Outbound" event should generally follow an "Inbound" event at the same location.
*   **Logic:**
    1.  **Location-Specific Grouping:** Events are grouped by `epc_code` and `scan_location`.
    2.  **Event Type Sequence:** Within each group, the sequence of `event_type` (e.g., Inbound, Outbound, Aggregation) is validated against predefined logical transitions.
    3.  **Example:** An "Inbound" followed immediately by another "Inbound" at the same location might be an error.
*   **Edge Cases:**
    *   **Complex Event Flows:** Some business processes might have more complex event sequences. The rules need to be flexible enough to accommodate these or flag them for review.
    *   **Missing Event Types:** Events with null or unrecognized `event_type` cannot be validated.
*   **Problem Solved:** Ensures the integrity of internal logistics processes within a single facility, identifying errors in handling or recording.

#### 7.5. `jump` (Spatiotemporal Jump)

*   **Principle:** Identifies instances where a product appears to have traveled an impossible distance in an impossibly short amount of time, or an unusually long time for a short distance. This is based on statistical analysis of typical travel times between business steps.
*   **Logic:**
    1.  **Travel Time Calculation:** Calculates the time difference between consecutive events for an EPC.
    2.  **Statistical Baseline:** Compares the calculated travel time against a statistical baseline of average travel times between `business_step` transitions (e.g., from Factory to WMS) using `data/processed/business_step_transition_avg_v2.csv`.
    3.  **Outlier Detection:** Flags events where the travel time significantly deviates (e.g., too fast or too slow) from the established average, often using statistical methods like IQR (Interquartile Range) or Z-scores.
*   **Edge Cases:**
    *   **Data Gaps:** Large time gaps between scans might not indicate a jump but rather unrecorded events.
    *   **Legitimate Delays:** Unusually long travel times might be due to legitimate delays (e.g., customs, weather) rather than anomalies.
*   **Problem Solved:** Detects highly improbable movements, which can indicate data entry errors, unrecorded events, or even illicit transportation.

### Handling Multiple Anomalies per EPC & Sequence Visualization

A single EPC code can trigger multiple anomaly types. Our system is designed to detect *all* applicable anomalies for a given EPC. For example, an EPC might be both `epcFake` and also exhibit a `jump` anomaly.

Furthermore, for visualization, it's crucial to identify *where* in the EPC's event sequence an anomaly occurred. For instance, a `jump` anomaly is associated with a specific "from" and "to" event pair, while an `evtOrderErr` is tied to a sequence of events at a particular location. Our output will pinpoint the exact event(s) or sequence of events that triggered each anomaly, allowing the frontend to highlight these problematic points on the map or in the event log.

### Addressing Edge Cases & Problem Solving Philosophy

Our approach to edge cases is to:

1.  **Define Clear Rules:** For rule-based anomalies, we establish precise criteria.
2.  **Flag for Review:** If a situation is ambiguous or falls outside the clear rules (e.g., a legitimate return flagged as `locErr`), it is still flagged as an anomaly. The `description` field provides context, and the "model" anomaly type is specifically for cases requiring expert review. This prevents false negatives and ensures all potential issues are brought to attention.
3.  **Iterative Refinement:** As we encounter more real-world data and receive feedback, we can refine our rules and models to better distinguish between true anomalies and legitimate but unusual events.

### Explaining Anomaly Percentage (ML Models)

For anomalies detected by Machine Learning models (e.g., SVM, GNN), we can provide a "percentage of anomaly" or a confidence score.

*   **SVM:** One-Class SVM models can output a "decision function" value. This value indicates how far an instance is from the decision boundary of the "normal" class. A more negative value indicates a higher likelihood of being an outlier. This value can be normalized or scaled to represent a "percentage of anomaly" or a confidence score.
*   **GNN:** For Graph Neural Networks, anomaly scores can be derived from various methods, such as reconstruction error (for autoencoder-based GNNs) or deviation from expected node/edge properties. These scores can also be normalized to provide a percentage.

The key is to explain that this percentage represents the model's *confidence* that an event is anomalous, based on the patterns it learned from the training data. It's a statistical measure, not a definitive "true/false" like rule-based anomalies, and often requires human interpretation.

## 8. Your Immediate Action Plan (as Team Leader & Data Analyst)

**🔥 TOP PRIORITY - Your team is waiting for this API:**

### Phase 1: Complete API Integration (Today)
1. **Create Unified API Endpoint:**
   ```python
   # POST /api/v1/barcode-anomaly-detect
   # Integrate your 5 existing detection functions:
   # - epcFake_structure_check.py
   # - epcDup_duplicate_scan.py
   # - locErr_hierarchy_check.py
   # - evtOrderErr_sequence_check.py
   # - jump_travel_time.py
   ```

2. **Implement Exact JSON Response Format:**
   ```python
   # Transform your function outputs to match EventHistory schema
   # Each anomaly must include: epcCode, productName, eventType, businessStep,
   # scanLocation, eventTime, anomaly, anomalyType, anomalyCode, description
   # Ensure that for each anomaly, the specific event(s) in the sequence that triggered it are identifiable.
   ```

3. **Add Error Handling & Performance Optimization:**
   ```python
   # Target: <100ms response time for backend integration
   # Handle edge cases: null data, invalid EPC formats, missing locations
   ```

### Phase 2: Support Backend Integration (Tomorrow)
1. **Test API with Backend Developer:**
   - Verify JSON input/output formats match exactly
   - Performance test with real CSV data (920,000 records)
   - Debug any integration issues

2. **Provide Documentation:**
   - API specification with examples
   - Error codes and handling
   - Performance benchmarks
   - Explanation of anomaly percentage/confidence for ML models.

### Phase 3: Support Frontend Features (This Week)
1. **Enable Report Generation:**
   - Support product/lot filtering
   - Generate `summaryStats` for Chart.js integration
   - Provide detailed anomaly descriptions
   - Ensure the output allows for highlighting specific anomalous events in the sequence.

2. **Support Dashboard KPIs:**
   - Calculate anomaly rates and statistics
   - Support business step filtering (Factory, WMS, Logistics_HUB, W_Stock, R_Stock, POS_Sell)

**Your API is the critical blocker for the entire team. Once completed, both backend and frontend can immediately integrate and test their completed features.**

