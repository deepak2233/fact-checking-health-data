# Script to download and save the dataset
# src/ingest.py
import argparse
import os
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)

def download_data(save_path):
    logging.info("Starting download of PUBHEALTH dataset.")
    dataset = load_dataset("ImperialCollegeLondon/health_fact")
    os.makedirs(save_path, exist_ok=True)
    dataset.save_to_disk(save_path)
    logging.info(f"Dataset downloaded and saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the PUBHEALTH dataset.")
    parser.add_argument("--save_path", type=str, default="data/raw", help="Path to save the dataset")
    args = parser.parse_args()
    download_data(args.save_path)

