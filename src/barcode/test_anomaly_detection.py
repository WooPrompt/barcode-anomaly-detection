import unittest
import json
import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import random

# Add the parent directory to the Python path to find the module to test
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from barcode.anomaly_detection_combined import detect_anomalies_from_json

class TestAnomalyDetection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Prepare the comprehensive test data once for all tests."""
        print("Generating large-scale test data...")
        cls.test_input_json_str = cls.generate_large_test_data(1000)
        print("Test data generation complete.")

    @classmethod
    def generate_large_test_data(cls, num_rows):
        product_id = "0000001"
        lot_id = "000001"
        base_time = datetime(2024, 7, 10, 9, 0, 0)
        locations = ["서울 공장", "부산 물류센터", "대전 물류센터", "광주 도매상", "대구 소매상"]
        data = []

        # --- Anomaly Counts ---
        cls.expected_epcFake = 5
        cls.expected_evtOrderErr = 5
        cls.expected_locErr = 5
        cls.expected_jump = 5
        cls.expected_model_anomalies = 5 # Anomalies to be caught by SVM

        # --- Generate Normal Data ---
        normal_rows = num_rows - (cls.expected_epcFake + cls.expected_evtOrderErr + cls.expected_locErr + cls.expected_jump + cls.expected_model_anomalies)
        for i in range(normal_rows):
            serial = f"{i:09d}"
            epc = f"001.8804823.{product_id}.{lot_id}.20240701.{serial}"
            data.append({
                "epc_code": epc,
                "event_time": (base_time + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "scan_location": random.choice(locations),
                "event_type": "Inbound",
                "worker_id": f"W{i%100:03d}",
                "factory": "hws"
            })

        # --- Inject Anomalies ---
        # 1. epcFake
        for i in range(cls.expected_epcFake):
            data.append({
                "epc_code": f"fake-epc-{i}",
                "event_time": (base_time + timedelta(minutes=10*i)).strftime("%Y-%m-%d %H:%M:%S"),
                "scan_location": "광주 도매상", "event_type": "Inbound", "worker_id": "WFAKE", "factory": "ygs"
            })

        # 2. evtOrderErr
        for i in range(cls.expected_evtOrderErr):
            serial = f"{10000+i:09d}"
            epc = f"001.8804823.{product_id}.{lot_id}.20240701.{serial}"
            t = base_time + timedelta(days=1, hours=i)
            data.append({"epc_code": epc, "event_time": t.strftime("%Y-%m-%d %H:%M:%S"), "scan_location": "대전 물류센터", "event_type": "Outbound", "worker_id": "WEVT", "factory": "icn"})
            data.append({"epc_code": epc, "event_time": (t + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"), "scan_location": "대전 물류센터", "event_type": "Inbound", "worker_id": "WEVT", "factory": "icn"})

        # 3. locErr
        for i in range(cls.expected_locErr):
            serial = f"{20000+i:09d}"
            epc = f"001.8804823.{product_id}.{lot_id}.20240701.{serial}"
            t = base_time + timedelta(days=2, hours=i)
            data.append({"epc_code": epc, "event_time": t.strftime("%Y-%m-%d %H:%M:%S"), "scan_location": "광주 도매상", "event_type": "Inbound", "worker_id": "WLOC", "factory": "kum"})
            data.append({"epc_code": epc, "event_time": (t + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S"), "scan_location": "서울 공장", "event_type": "Inbound", "worker_id": "WLOC", "factory": "kum"})

        # 4. jump
        for i in range(cls.expected_jump):
            serial = f"{30000+i:09d}"
            epc = f"001.8804823.{product_id}.{lot_id}.20240701.{serial}"
            t = base_time + timedelta(days=3, hours=i)
            data.append({"epc_code": epc, "event_time": t.strftime("%Y-%m-%d %H:%M:%S"), "scan_location": "서울 공장", "event_type": "Outbound", "worker_id": "WJUMP", "factory": "hws"})
            data.append({"epc_code": epc, "event_time": (t + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S"), "scan_location": "부산 물류센터", "event_type": "Inbound", "worker_id": "WJUMP", "factory": "hws"})

        # 5. Data for SVM model anomalies (unusual time and manufacture gap)
        for i in range(cls.expected_model_anomalies):
            serial = f"{40000+i:09d}"
            epc = f"001.8804823.{product_id}.{lot_id}.20240701.{serial}"
            # Make the event time very far from the manufacture date
            event_time_svm = base_time + timedelta(days=365 * 2 + i) # 2 years after manufacture
            data.append({
                "epc_code": epc,
                "event_time": event_time_svm.strftime("%Y-%m-%d %H:%M:%S"),
                "scan_location": "서울 공장", "event_type": "Inbound", "worker_id": "WSVM", "factory": "hws"
            })

        input_data = {
            "product_id": product_id,
            "lot_id": lot_id,
            "data": data,
            "transition_stats": [
                {"from_scan_location": "서울 공장", "to_scan_location": "부산 물류센터", "time_taken_hours_mean": 5.0, "time_taken_hours_std": 1.0}
            ],
            "geo_data": [
                {"scan_location": "서울 공장", "Latitude": 37.5665, "Longitude": 126.9780},
                {"scan_location": "부산 물류센터", "Latitude": 35.1796, "Longitude": 129.0756},
                {"scan_location": "대전 물류센터", "Latitude": 36.3504, "Longitude": 127.3845},
                {"scan_location": "광주 도매상", "Latitude": 35.1595, "Longitude": 126.8526},
                {"scan_location": "대구 소매상", "Latitude": 35.8714, "Longitude": 128.6014}
            ]
        }
        return json.dumps(input_data)

    def test_anomaly_detection_with_large_data(self):
        """
        Tests the main anomaly detection function with a large, generated dataset.
        It checks if the summary counts match the number of injected anomalies.
        """
        # Execute the function with the generated test data
        result_json_str = detect_anomalies_from_json(self.test_input_json_str)
        self.assertIsNotNone(result_json_str, "Function should return a JSON string.")
        result_data = json.loads(result_json_str)

        # Assert the summary statistics
        summary = result_data.get("summaryStats", {})
        self.assertEqual(summary.get("epcFake", 0), self.expected_epcFake, f"Should detect {self.expected_epcFake} fake EPCs")
        self.assertEqual(summary.get("evtOrderErr", 0), self.expected_evtOrderErr, f"Should detect {self.expected_evtOrderErr} event order errors")
        self.assertEqual(summary.get("jump", 0), self.expected_jump, f"Should detect {self.expected_jump} time/space jumps")
        self.assertEqual(summary.get("locErr", 0), self.expected_locErr, f"Should detect {self.expected_locErr} location errors")
        self.assertGreaterEqual(summary.get("model", 0), 1, "Should detect at least one model-based anomaly")

        # Assert that specific anomalies are present in the details
        details = result_data.get("details", [])
        self.assertTrue(any("fake-epc" in d and "epcFake" in d for d in details), "epcFake detail missing")
        self.assertTrue(any("WEVT" in d and "evtOrderErr" in d for d in details), "evtOrderErr detail missing")
        self.assertTrue(any("WJUMP" in d and "jump" in d for d in details), "jump detail missing")
        self.assertTrue(any("WLOC" in d and "locErr" in d for d in details), "locErr detail missing")
        self.assertTrue(any("WSVM" in d and "model" in d for d in details), "Model anomaly detail missing")

if __name__ == '__main__':
    print("Running anomaly detection tests with large dataset...")
    unittest.main(verbosity=2)