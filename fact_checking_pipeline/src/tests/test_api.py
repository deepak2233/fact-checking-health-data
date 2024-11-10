# src/tests/test_api.py

import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from fastapi.testclient import TestClient
from src.serve import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_predict_veracity():
    # Use "claim" instead of "text" as per the API's expected input format
    response = client.post("/claim/v1/predict", json={"claim": "COVID-19 can be cured by garlic"})
    assert response.status_code == 200
    assert "veracity" in response.json()
    assert isinstance(response.json()["veracity"], str)  # Expected output should be a string label

