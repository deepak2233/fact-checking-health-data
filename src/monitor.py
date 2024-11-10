# Monitoring and data drift detection script

# src/monitor.py
import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import pairwise_distances

def calculate_embeddings(dataset, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []

    for example in dataset:
        inputs = tokenizer(example["claim"], return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embeddings.append(embedding)
    
    return np.array(embeddings)

def monitor_data_drift(new_data_path, reference_data_path, model_name, threshold=0.5):
    reference_data = load_from_disk(reference_data_path)['train']
    new_data = load_from_disk(new_data_path)['validation']
    reference_embeddings = calculate_embeddings(reference_data, model_name)
    new_embeddings = calculate_embeddings(new_data, model_name)
    drift_distance = pairwise_distances(reference_embeddings, new_embeddings, metric="cosine").mean()
    if drift_distance > threshold:
        print("Data drift detected! Retraining needed.")
    else:
        print("No significant drift detected.")

if __name__ == "__main__":
    monitor_data_drift("data/processed", "data/raw", "nbroad/longformer-base-health-fact")

