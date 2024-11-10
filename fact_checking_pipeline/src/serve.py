# src/serve.py

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from prometheus_client import Summary, Counter, generate_latest

app = FastAPI()

# Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_count', 'Number of requests processed')

# Load model and tokenizer
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models")
  
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH,use_safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

class ClaimRequest(BaseModel):
    claim: str

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/metrics")
def metrics():
    return generate_latest()

@app.post("/claim/v1/predict")
@REQUEST_TIME.time()
def predict_veracity(request: ClaimRequest):
    REQUEST_COUNT.inc()
    try:
        # Tokenize the input claim text
        inputs = tokenizer(request.claim, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
        
        # Predict with the model
        with torch.no_grad():
            logits = model(**inputs).logits
        prediction = torch.argmax(logits, dim=1).item()

        # Define label mapping
        label_map = {0: "true", 1: "false", 2: "mixture", 3: "unproven"}
        
        # Return the response
        return {"claim": request.claim, "veracity": label_map.get(prediction, "unknown")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

