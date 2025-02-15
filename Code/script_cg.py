# =====================================
# Setup and Dependencies
# =====================================
'''
# Install necessary libraries
!pip install pandas transformers datasets wandb evaluate rouge-score sentencepiece
'''

# Import necessary libraries
import gc
#import os
import json
#import evaluate
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration, 
    AdamW, Seq2SeqTrainer, Seq2SeqTrainingArguments
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
#from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import re
import argparse
import time
from datetime import datetime, timedelta

# =====================================
# Configuration and Hyperparameters
# =====================================
class Config:
    # Paths
    DATA_DIR = './data'
    MODEL_DIR = './dgen_models_train'
    OUTPUT_DIR = './dgen_candidates'
    TRAIN_DATA_PATH = f'{DATA_DIR}/dgen_train_converted.json'
    TEST_DATA_PATH = f'{DATA_DIR}/dgen_test_converted.json'
    
    # Model Configuration
    MODEL_NAME = 't5-base'
    MODEL_VERSION = '003'  # Version number for saving models
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Training Hyperparameters
    TRAIN_BATCH_SIZE = 4
    EVAL_BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    WEIGHT_DECAY = 0.01
    
    # Tokenizer Settings
    MAX_INPUT_LENGTH = 512
    MAX_TARGET_LENGTH = 128
    
    # Generation Settings
    GEN_MAX_LENGTH = 50
    NO_REPEAT_NGRAM_SIZE = 2
    
    # Evaluation Settings
    EVAL_METRICS = ["P@1", "R@1", "F1@1", "P@3", "R@3", "F1@3",
                   "P@5", "R@5", "F1@5", "P@10", "R@10", "F1@10",
                   "MRR@3", "MAP@5", "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10"]

    @classmethod
    def get_model_save_path(cls, model_type="ft"):
        """Get the path for saving the model"""
        return f"{cls.MODEL_DIR}/t5_base_ft_cg_{cls.MODEL_VERSION}"
    
    @classmethod
    def get_output_path(cls, model_type="ft", split=False):
        """Get the path for saving generation outputs"""
        base_path = f"{cls.OUTPUT_DIR}/t5_base_{model_type}_cg_{cls.MODEL_VERSION}"
        return f"{base_path}.json"
    
# =====================================
# Device Configuration
# =====================================    
print(f"Using device: {Config.DEVICE}")
torch.cuda.empty_cache()
gc.collect()

# =====================================
# Data Loading and Preprocessing
# =====================================
def load_data():
    """Load and preprocess the training and testing data"""
    # Load datasets
    with open(Config.TRAIN_DATA_PATH, 'r') as file:
        train_data = json.load(file)
    with open(Config.TEST_DATA_PATH, 'r') as file:
        test_data = json.load(file)

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Add prefixes
    train_df['input'] = 'Question: ' + train_df['sentence'] + ', Answer: ' + train_df['answer']
    test_df['input'] = 'Question: ' + test_df['sentence'] + ', Answer: ' + test_df['answer']
    
    # Split into train/val
    train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Print dataset sizes
    print("\nDataset sizes:")
    print(f"Training Set:      {len(train_data):,} examples")
    print(f"Validation Set:    {len(val_data):,} examples")
    print(f"Test Set:          {len(test_df):,} examples")
    print(f"Total Dataset:     {len(train_data) + len(val_data) + len(test_df):,} examples\n")
    
    return train_data, val_data, test_df

# =====================================
# Model Training
# =====================================

def prepare_data_for_training(df):
    """
    Prepare data for the model by pairing each input with individual distractors
    Returns a dataset with input-distractor pairs
    """
    input_texts = []
    target_texts = []
    
    for idx, row in df.iterrows():
        input_text = row['input']
        distractors = row['distractors']
        
        # Add each distractor separately with the same input
        for distractor in distractors:
            input_texts.append(input_text)
            target_texts.append(distractor)
    
    return Dataset.from_dict({
        "input_text": input_texts,
        "target_text": target_texts
    })

def tokenize_data(examples, tokenizer):
    """Tokenize the data"""
    model_inputs = tokenizer(
        examples["input_text"], 
        max_length=Config.MAX_INPUT_LENGTH, 
        truncation=True, 
        padding="max_length"
    )
    labels = tokenizer(
        examples["target_text"], 
        max_length=Config.MAX_TARGET_LENGTH, 
        truncation=True, 
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def format_time_duration(seconds):
    """
    Format time duration into days, hours, minutes, and seconds
    Args:
        seconds (float): Time in seconds
    Returns:
        str: Formatted time string
    """
    duration = timedelta(seconds=int(seconds))
    days = duration.days
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60
    
    time_parts = []
    if days > 0:
        time_parts.append(f"{days} days")
    if hours > 0:
        time_parts.append(f"{hours} hours")
    if minutes > 0:
        time_parts.append(f"{minutes} minutes")
    if seconds > 0 or not time_parts:  # include seconds if it's the only non-zero value
        time_parts.append(f"{seconds} seconds")
    
    return ", ".join(time_parts)


def print_dataset_samples(dataset, name, num_samples=3):
    """
    Print the first num_samples examples from a dataset
    Args:
        dataset: The dataset to inspect
        name: Name of the dataset for display
        num_samples: Number of samples to print (default=3)
    """
    print(f"\n=== First {num_samples} samples from {name} ===")
    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i+1}:")
        print(f"Input text: {dataset[i]['input_text']}")
        print(f"Target text: {dataset[i]['target_text']}")
        print("-" * 80)

def train_model(train_data, val_data, model_type="ft"):
    """Train the model using either fine-tuning or contrastive learning"""
    
    print("\nStarting model training...")
    training_start_time = time.time()
    
    # Initialize model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(Config.MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(Config.MODEL_NAME).to(Config.DEVICE)
    
    # Prepare datasets
    train_dataset = prepare_data_for_training(train_data)
    val_dataset = prepare_data_for_training(val_data)
    
    # Print sample data
    #print_dataset_samples(train_dataset, "Training Dataset")
    #print_dataset_samples(val_dataset, "Validation Dataset")
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda x: tokenize_data(x, tokenizer), 
        batched=True
    )
    tokenized_val = val_dataset.map(
        lambda x: tokenize_data(x, tokenizer), 
        batched=True
    )
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./t5-distractor-generator",
        evaluation_strategy="epoch",
        learning_rate=Config.LEARNING_RATE,
        per_device_train_batch_size=Config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=Config.EVAL_BATCH_SIZE,
        num_train_epochs=Config.NUM_EPOCHS,
        weight_decay=Config.WEIGHT_DECAY,
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        load_best_model_at_end=True,
        logging_dir="./logs",
        logging_steps=10,
        push_to_hub=False
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
       model=model,
       args=training_args,
       train_dataset=tokenized_train,
       eval_dataset=tokenized_val,
       tokenizer=tokenizer
    )
    
    # Train
    print("Training base model...")
    trainer.train()

    # Save fine-tuned model
    ft_model_path = Config.get_model_save_path("ft")
    model.save_pretrained(ft_model_path)
    tokenizer.save_pretrained(ft_model_path)
    print(f"\nFine-tuned model saved to: {ft_model_path}")
    

    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"\nTotal training time: {format_time_duration(total_training_time)}")
    
    return model, tokenizer

# =====================================
# Generation
# =====================================
def generate_distractors(model, tokenizer, sentence, answer):
    """Generate distractors for a given question and answer"""
    input_text = f"Question: {sentence}, Answer: {answer}"
    input_ids = tokenizer(
        input_text, 
        return_tensors='pt'
    ).input_ids.to(Config.DEVICE)
    
    outputs = model.generate(
        input_ids,
        max_length=Config.GEN_MAX_LENGTH,
        num_return_sequences = 10,
        num_beams = 10,
        no_repeat_ngram_size=Config.NO_REPEAT_NGRAM_SIZE,
        early_stopping=True
    )
    
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

def run_generation(model_type="ft"):
    """Run generation on test dataset"""
    # Load model
    model = T5ForConditionalGeneration.from_pretrained(
        Config.get_model_save_path(model_type),
        config=None,            # Suppress config output
        local_files_only=True,  # Avoid downloading warnings
    ).to(Config.DEVICE)
    tokenizer = T5Tokenizer.from_pretrained(
        Config.get_model_save_path(model_type),
        local_files_only=True  # Avoid downloading warnings
        )
    
    # Load test data
    _, _, test_data = load_data()
    
    # Generate distractors
    results = []
    for item in tqdm(test_data.to_dict('records'), desc="Generating distractors"):
        generated_distractors = generate_distractors(
            model, tokenizer, 
            item['sentence'], item['answer']
        )
        results.append({
            "sentence": item['sentence'],
            "answer": item['answer'],
            "distractors": item['distractors'],
            "generated_distractors": generated_distractors
        })
    
    # Save results
    with open(Config.get_output_path(model_type), 'w') as f:
        json.dump(results, f, indent=2)

# =====================================
# Evaluation
# =====================================
def dcg_at_k(r, k):
    """Calculate DCG@k"""
    r = np.asfarray(r)[:k]
    if r.size:
        return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
    return 0.

def ndcg_at_k(r, k):
    """Calculate NDCG@k"""
    idcg = dcg_at_k(sorted(r, reverse=True), k)
    if not idcg:
        return 0.
    return dcg_at_k(r, k) / idcg

def evaluate_result(result):
    """Evaluate a single result"""
    eval_metrics = {metric: 0.0 for metric in Config.EVAL_METRICS}
    
    # Normalize text for comparison
    distractors = [d.lower() for d in result["distractors"]]
    generations = [d.lower() for d in result["generated_distractors"]]
    relevants = [int(generation in distractors) for generation in generations]
    
    # Calculate metrics for different k values (1, 3, 5, 10)
    for k in [1, 3, 5, 10]:
        # Precision@k
        eval_metrics[f"P@{k}"] = relevants[:k].count(1) / k if len(relevants) >= k else 0
        
        # Recall@k
        eval_metrics[f"R@{k}"] = relevants[:k].count(1) / len(distractors)
        
        # F1@k
        try:
            p_k = eval_metrics[f"P@{k}"]
            r_k = eval_metrics[f"R@{k}"]
            eval_metrics[f"F1@{k}"] = (2 * p_k * r_k) / (p_k + r_k) if (p_k + r_k) > 0 else 0
        except ZeroDivisionError:
            eval_metrics[f"F1@{k}"] = 0
        
        # NDCG@k
        eval_metrics[f"NDCG@{k}"] = ndcg_at_k(relevants, k)
    
    # MRR@5
    eval_metrics["MRR@3"] = next(
        (1 / (i+1) for i in range(3) if i < len(relevants) and relevants[i] == 1),
        0
    )
    
    # MAP@5
    rel_num = 0
    for i in range(min(5, len(relevants))):
        if relevants[i] == 1:
            rel_num += 1
            eval_metrics["MAP@5"] += rel_num / (i+1)
    eval_metrics["MAP@5"] = eval_metrics["MAP@5"] / len(distractors) if distractors else 0
    
    return eval_metrics

def evaluate_generations(model_type="ft"):
    """Evaluate the generated distractors"""
    input_file = Config.get_output_path(model_type)
    
    # Load and evaluate results
    with open(input_file, 'r') as file:
        results = json.load(file)
    
    # Initialize average metrics
    avg_metrics = {metric: 0.0 for metric in Config.EVAL_METRICS}
    
    # Calculate metrics for all results
    print("Evaluating...")
    for result in results:
        metrics = evaluate_result(result)
        for k in avg_metrics.keys():
            avg_metrics[k] += metrics[k]
    
    # Calculate averages
    for k in avg_metrics.keys():
        avg_metrics[k] = (avg_metrics[k] / len(results)) * 100
        print(f"{k}: {avg_metrics[k]:.2f}%")
    
    print("Evaluation complete!")
    return avg_metrics

# =====================================
# Main Pipeline
# =====================================
def run_pipeline(model_type="ft"):
    """Run the complete pipeline"""
    print(f"Starting pipeline for model type: {model_type}")
    
    # Load data
    print("Loading and preprocessing data...")
    train_data, val_data, test_data = load_data()
    
    # Train model
    print(f"Training {model_type} model...")
    model, tokenizer = train_model(train_data, val_data, model_type)
    
    # Generate distractors
    print("Generating distractors...")
    run_generation(model_type)
    
    # Evaluate results
    print("Evaluating results...")
    metrics = evaluate_generations(model_type)
    
    return metrics

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Run distractor generation pipeline')
   parser.add_argument('--train_only', action='store_true',
                     help='Only train the model without generation and evaluation')
   parser.add_argument('--generate_only', action='store_true',
                     help='Only generate distractors using existing model')
   parser.add_argument('--evaluate_only', action='store_true',
                     help='Only evaluate existing generations')
   
   args = parser.parse_args()
   
   if args.train_only:
       print("Training model only...")
       train_data, val_data, _ = load_data()
       train_model(train_data, val_data)
   elif args.generate_only:
       print("Generating distractors only...")
       run_generation()
   elif args.evaluate_only:
       print("Evaluating existing generations...")
       evaluate_generations()
   else:
       print("Running complete pipeline...")
       metrics = run_pipeline()
       
   print("Done!")
