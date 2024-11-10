import requests

# Base URL for the FastAPI server
BASE_URL = "http://localhost:8000"

def test_health_check():
    # Testing health endpoint
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    print("Health check passed!")

def test_predict_veracity():
    # Corrected input with "claim" key as expected by the API
    input_data = {"claim": "COVID-19 can be cured by garlic"}
    
    # Sending POST request to prediction endpoint
    response = requests.post(f"{BASE_URL}/claim/v1/predict", json=input_data)
    
    # Verifying response status
    assert response.status_code == 200, f"Expected status code 200 but got {response.status_code}"
    response_json = response.json()
    
    # Verifying the structure of the response
    assert "claim" in response_json, "Missing 'claim' in response"
    assert "veracity" in response_json, "Missing 'veracity' in response"
    
    # Printing response
    print(f"Input: {input_data['claim']}")
    print(f"Prediction: {response_json['veracity']}")
    print("Prediction test passed!")

if __name__ == "__main__":
    print("Testing API Endpoints...\n")
    test_health_check()
    test_predict_veracity()
    print("\nAll tests completed successfully!")

