# Exploratory Data Analysis script
# src/eda.py
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk
import pandas as pd

def plot_distribution(data, column, title, filename):
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], bins=20, kde=True)
    plt.title(title)
    plt.savefig(filename)

def eda(data_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    dataset = load_from_disk(data_path)
    df = pd.DataFrame(dataset['train'])

    plot_distribution(df, 'label', 'Label Distribution', os.path.join(output_dir, 'label_distribution.png'))
    df['claim_length'] = df['claim'].apply(lambda x: len(x.split()))
    plot_distribution(df, 'claim_length', 'Claim Length Distribution', os.path.join(output_dir, 'claim_length_distribution.png'))
    plot_distribution(df, 'sentiment', 'Sentiment Distribution', os.path.join(output_dir, 'sentiment_distribution.png'))

if __name__ == "__main__":
    eda("data/processed", "eda_output")

