import requests
import sys
import json
from datetime import datetime

class LoanAPITester:
    def __init__(self, base_url="https://loan-analyzer-3.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details
        })

    def test_api_root(self):
        """Test API root endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Message: {data.get('message', 'N/A')}"
            self.log_test("API Root", success, details)
            return success
        except Exception as e:
            self.log_test("API Root", False, str(e))
            return False

    def test_dataset_stats(self):
        """Test dataset statistics endpoint"""
        try:
            response = requests.get(f"{self.api_url}/dataset/stats", timeout=15)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                required_fields = ['total_records', 'default_rate', 'avg_income', 'avg_loan_amount', 'feature_distributions']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    success = False
                    details = f"Missing fields: {missing_fields}"
                else:
                    details = f"Total records: {data['total_records']}, Default rate: {data['default_rate']:.2%}"
                    # Validate data types and ranges
                    if data['total_records'] <= 0:
                        success = False
                        details += " - Invalid total_records"
                    elif not (0 <= data['default_rate'] <= 1):
                        success = False
                        details += " - Invalid default_rate range"
            else:
                details = f"Status: {response.status_code}"
                
            self.log_test("Dataset Stats", success, details)
            return success, data if success else None
        except Exception as e:
            self.log_test("Dataset Stats", False, str(e))
            return False, None

    def test_model_performance(self):
        """Test model performance endpoint"""
        try:
            response = requests.get(f"{self.api_url}/model/performance", timeout=15)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                required_fields = ['accuracy', 'total_samples', 'default_rate', 'feature_importance', 'model_type']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    success = False
                    details = f"Missing fields: {missing_fields}"
                else:
                    details = f"Accuracy: {data['accuracy']:.2%}, Model: {data['model_type']}"
                    # Validate accuracy range
                    if not (0 <= data['accuracy'] <= 1):
                        success = False
                        details += " - Invalid accuracy range"
            else:
                details = f"Status: {response.status_code}"
                
            self.log_test("Model Performance", success, details)
            return success, data if success else None
        except Exception as e:
            self.log_test("Model Performance", False, str(e))
            return False, None

    def test_loan_prediction_high_risk(self):
        """Test loan prediction with high-risk profile"""
        high_risk_application = {
            "applicant_income": 2500,  # Low income
            "coapplicant_income": 0,
            "loan_amount": 250,  # High loan amount
            "loan_amount_term": 360,
            "credit_history": 0,  # Poor credit
            "gender": "Male",
            "married": "No",
            "dependents": "3+",
            "education": "Not Graduate",
            "self_employed": "Yes",  # Higher risk
            "property_area": "Rural"
        }
        
        try:
            response = requests.post(f"{self.api_url}/predict", json=high_risk_application, timeout=15)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                required_fields = ['predicted_default_probability', 'predicted_class', 'risk_level', 'application']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    success = False
                    details = f"Missing fields: {missing_fields}"
                else:
                    prob = data['predicted_default_probability']
                    details = f"Class: {data['predicted_class']}, Risk: {data['risk_level']}, Probability: {prob:.2%}"
                    
                    # Validate prediction logic for high-risk profile
                    if prob < 0.3:  # Should be higher risk
                        details += " - WARNING: Low probability for high-risk profile"
            else:
                details = f"Status: {response.status_code}"
                if response.status_code == 422:
                    details += f" - Validation error: {response.text}"
                
            self.log_test("High-Risk Prediction", success, details)
            return success, data if success else None
        except Exception as e:
            self.log_test("High-Risk Prediction", False, str(e))
            return False, None

    def test_loan_prediction_low_risk(self):
        """Test loan prediction with low-risk profile"""
        low_risk_application = {
            "applicant_income": 8000,  # High income
            "coapplicant_income": 3000,
            "loan_amount": 120,  # Reasonable loan amount
            "loan_amount_term": 360,
            "credit_history": 1,  # Good credit
            "gender": "Female",
            "married": "Yes",
            "dependents": "1",
            "education": "Graduate",
            "self_employed": "No",
            "property_area": "Urban"
        }
        
        try:
            response = requests.post(f"{self.api_url}/predict", json=low_risk_application, timeout=15)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                required_fields = ['predicted_default_probability', 'predicted_class', 'risk_level', 'application']
                missing_fields = [field for field in required_fields if field not in data]
                
                if missing_fields:
                    success = False
                    details = f"Missing fields: {missing_fields}"
                else:
                    prob = data['predicted_default_probability']
                    details = f"Class: {data['predicted_class']}, Risk: {data['risk_level']}, Probability: {prob:.2%}"
                    
                    # Validate prediction logic for low-risk profile
                    if prob > 0.7:  # Should be lower risk
                        details += " - WARNING: High probability for low-risk profile"
            else:
                details = f"Status: {response.status_code}"
                if response.status_code == 422:
                    details += f" - Validation error: {response.text}"
                
            self.log_test("Low-Risk Prediction", success, details)
            return success, data if success else None
        except Exception as e:
            self.log_test("Low-Risk Prediction", False, str(e))
            return False, None

    def test_prediction_history(self):
        """Test prediction history endpoint"""
        try:
            response = requests.get(f"{self.api_url}/predictions/history", timeout=10)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                details = f"Retrieved {len(data)} predictions"
                
                # Validate structure if predictions exist
                if data and len(data) > 0:
                    first_pred = data[0]
                    required_fields = ['id', 'predicted_default_probability', 'predicted_class', 'risk_level']
                    missing_fields = [field for field in required_fields if field not in first_pred]
                    if missing_fields:
                        details += f" - Missing fields in predictions: {missing_fields}"
            else:
                details = f"Status: {response.status_code}"
                
            self.log_test("Prediction History", success, details)
            return success, data if success else None
        except Exception as e:
            self.log_test("Prediction History", False, str(e))
            return False, None

    def test_invalid_prediction(self):
        """Test prediction with invalid data"""
        invalid_application = {
            "applicant_income": -1000,  # Invalid negative income
            "loan_amount": "invalid",  # Invalid data type
            "credit_history": 2,  # Invalid value (should be 0 or 1)
        }
        
        try:
            response = requests.post(f"{self.api_url}/predict", json=invalid_application, timeout=10)
            # Should return 422 for validation error
            success = response.status_code == 422
            details = f"Status: {response.status_code} (Expected 422 for invalid data)"
            
            self.log_test("Invalid Data Handling", success, details)
            return success
        except Exception as e:
            self.log_test("Invalid Data Handling", False, str(e))
            return False

    def run_all_tests(self):
        """Run all backend API tests"""
        print("🚀 Starting Loan Default Prediction API Tests")
        print(f"📍 Testing API at: {self.api_url}")
        print("=" * 60)
        
        # Test API availability first
        if not self.test_api_root():
            print("\n❌ API is not accessible. Stopping tests.")
            return False
        
        # Test core endpoints
        stats_success, stats_data = self.test_dataset_stats()
        perf_success, perf_data = self.test_model_performance()
        
        # Test prediction functionality
        high_risk_success, high_risk_data = self.test_loan_prediction_high_risk()
        low_risk_success, low_risk_data = self.test_loan_prediction_low_risk()
        
        # Test history and error handling
        history_success, history_data = self.test_prediction_history()
        invalid_success = self.test_invalid_prediction()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed! API is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the details above.")
            
        # Print key insights if data is available
        if stats_data and perf_data:
            print(f"\n📈 Key Metrics:")
            print(f"   • Dataset size: {stats_data['total_records']} records")
            print(f"   • Default rate: {stats_data['default_rate']:.2%}")
            print(f"   • Model accuracy: {perf_data['accuracy']:.2%}")
            print(f"   • Average income: ${stats_data['avg_income']:,.0f}")
            
        return self.tests_passed == self.tests_run

def main():
    tester = LoanAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())