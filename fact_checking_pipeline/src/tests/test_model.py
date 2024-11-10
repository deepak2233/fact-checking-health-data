# Tests for model inference and performance
# src/tests/test_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def test_model_load():
    model = AutoModelForSequenceClassification.from_pretrained("models")
    assert model is not None, "Failed to load model"

def test_model_inference():
    tokenizer = AutoTokenizer.from_pretrained("nbroad/longformer-base-health-fact")
    model = AutoModelForSequenceClassification.from_pretrained("models")
    inputs = tokenizer("Example claim for testing.", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    assert logits is not None, "Inference failed"
    assert logits.shape[1] == 4, "Unexpected number of output classes"

