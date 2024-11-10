# src/train.py
import argparse
import yaml
import logging
import mlflow
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score

logging.basicConfig(level=logging.INFO)

def load_config(config_file):
    """Loads the configuration from a YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_metrics(pred):
    """Computes accuracy and F1 score for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

def validate_labels(dataset):
    """Ensure all labels are within the expected range."""
    valid_labels = [0, 1, 2, 3]
    for split in ["train", "validation"]:
        split_labels = dataset[split]["label"]
        if any(label not in valid_labels for label in split_labels):
            raise ValueError(f"Invalid labels found in {split} split")

def train_model(config_file):
    """Trains the model using Hugging Face Trainer API with memory optimization options."""
    config = load_config(config_file)
    
    # Ensure learning rate is a float
    learning_rate = float(config['lr'])
    
    # Load dataset
    logging.info("Loading processed dataset...")
    dataset = load_from_disk(config['processed_path'])

    # Validate labels
    validate_labels(dataset)
    
    # Load model and tokenizer
    logging.info(f"Loading model: {config['model_name']} for fine-tuning...")
    model = AutoModelForSequenceClassification.from_pretrained(config['model_name'], num_labels=4)
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Define training arguments with memory optimizations
    logging.info("Setting up training arguments with memory optimizations...")
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        save_total_limit=1,
        logging_steps=10,
        report_to=["none"],       # Disable WandB
        fp16=True,                # Enable mixed precision training
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batches
        max_steps=100,            # Optional: limit the number of training steps
        no_cuda=False             # Set to True if you want to run on CPU only
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_metrics,
    )

    # Start MLFlow logging
    logging.info("Starting MLFlow run...")
    mlflow.start_run()

    # Log model parameters
    params_to_log = {
        "model_name": config['model_name'],
        "learning_rate": learning_rate,
        "batch_size": config['batch_size'],
        "epochs": config['epochs'],
        "max_length": config['max_length'],
    }
    for param_name, param_value in params_to_log.items():
        try:
            mlflow.log_param(param_name, param_value)
        except Exception as e:
            logging.warning(f"Parameter '{param_name}' could not be logged: {e}")

    # Begin training
    logging.info("Training the model...")
    trainer.train()

    # Evaluate and log metrics
    logging.info("Evaluating the model...")
    metrics = trainer.evaluate()
    mlflow.log_metrics(metrics)

    # Save model with configuration and tokenizer
    logging.info(f"Saving the model and tokenizer to {config['output_dir']}...")
    model.save_pretrained(config['output_dir'])      # Save model with configuration
    tokenizer.save_pretrained(config['output_dir'])  # Save tokenizer

    # End MLFlow run
    mlflow.end_run()
    logging.info("Training completed and MLFlow run ended.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()
    
    # Start the training process
    train_model(args.config_file)
