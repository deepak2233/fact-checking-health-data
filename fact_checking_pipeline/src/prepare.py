# src/prepare.py
import argparse
import yaml
import logging
from datasets import load_from_disk
from transformers import AutoTokenizer
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def prepare_data(config_file):
    """Prepares data by tokenizing and adding sentiment and length features."""
    config = load_config(config_file)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    dataset = load_from_disk(config['data_path'])

    def process_example(example):
        # Process each entry in the batch
        example['cleaned_claim'] = [claim.lower().replace(r"[^a-zA-Z0-9 ]", "") for claim in example['claim']]
        example['input_ids'] = tokenizer(example['cleaned_claim'], truncation=True, padding='max_length', max_length=config['max_length'])['input_ids']
        example['sentiment'] = [TextBlob(claim).sentiment.polarity for claim in example['claim']]
        example['explanation_length'] = [len(explanation.split()) for explanation in example['explanation']]
        return example

    # Filter dataset to ensure labels are valid
    def filter_invalid_labels(example):
        return example['label'] in [0, 1, 2, 3]  # Assuming 4 labels: 0, 1, 2, 3

    # Apply filtering and processing
    dataset = dataset.filter(filter_invalid_labels)
    processed_dataset = dataset.map(process_example, batched=True)
    processed_dataset.save_to_disk(config['processed_path'])
    logging.info(f"Data processed and saved to {config['processed_path']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dataset for training.")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()
    prepare_data(args.config_file)
