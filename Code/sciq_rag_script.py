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
    MODEL_DIR = './sciq_models_train_rag'
    OUTPUT_DIR = './sciq_rag_outputs'
    TRAIN_DATA_PATH = f'{DATA_DIR}/sciq_train_rag.json'
    VALID_DATA_PATH = f'{DATA_DIR}/sciq_valid_rag.json'
    TEST_DATA_PATH = f'{DATA_DIR}/sciq_test_1.json'
    
    # Model Configuration
    MODEL_NAME = 't5-base'
    MODEL_VERSION = 'sciq_rag_correct_006'  # Version number for saving models  
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
    
    # Contrastive Learning Parameters
    CL_BATCH_SIZE = 4
    CL_LEARNING_RATE = 1e-4
    CL_EPOCHS = 10
    CL_MARGIN = 0.01     #0.01
    CL_TEMPERATURE = 0.1
    CL_LOSS_TYPE = "info_nce"  # Options: "info_nce" or "triplet"
    CL_GEN_LOSS_WEIGHT = 0.5   # Weight for generation loss
    CL_CONT_LOSS_WEIGHT = 0.5 # Weight for contrastive loss
    
    # Evaluation Settings
    EVAL_METRICS = ["P@1", "R@1", "F1@1", "P@3", "R@3", "F1@3",
                   "P@5", "R@5", "F1@5", "P@10", "R@10", "F1@10",
                   "MRR@3", "MAP@5", "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@10"]

    @classmethod
    def get_model_save_path(cls, model_type="ft"):
        """Get the path for saving the model"""
        if model_type == "cl":
            return f"{cls.MODEL_DIR}/t5_base_cl_{cls.CL_LOSS_TYPE}_{cls.MODEL_VERSION}"
        return f"{cls.MODEL_DIR}/t5_base_ft_{cls.MODEL_VERSION}"
    
    @classmethod
    def get_output_path(cls, model_type="ft", split=False):
        """Get the path for saving generation outputs"""
        base_path = f"{cls.OUTPUT_DIR}/t5_base_{model_type}_{cls.MODEL_VERSION}"
        return f"{base_path}_split.json" if split else f"{base_path}.json"
    
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
    with open(Config.VALID_DATA_PATH, 'r') as file:
        val_data = json.load(file)

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    val_df = pd.DataFrame(val_data)
    
    
    # Add prefixes
    train_df['input'] = 'Question: ' + train_df['sentence'] + ', Answer: ' + train_df['answer']
    test_df['input'] = 'Question: ' + test_df['sentence'] + ', Answer: ' + test_df['answer']
    val_df['input'] = 'Question: ' + val_df['sentence'] + ', Answer: ' + val_df['answer']
    
    # Split into train/val
    #train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)
    
    # Print dataset sizes
    print("\nDataset sizes:")
    print(f"Training Set:      {len(train_data):,} examples")
    print(f"Validation Set:    {len(val_data):,} examples")
    print(f"Test Set:          {len(test_df):,} examples")
    print(f"Total Dataset:     {len(train_data) + len(val_data) + len(test_df):,} examples\n")
    
    example = train_df.iloc[0]  # You can change the index (e.g., .iloc[5]) to get another
    
    # Step 3: Display the input and output
    print("Input:", example['input'])
    print("Target:", example['distractors'])  # Assuming 'distractors' is the target
        
    return train_df, val_df, test_df

# =====================================
# Model Training
# =====================================
def prepare_data_for_training(df):
    """Prepare data for the model"""
    inputs = df['input'].tolist()
    outputs = [', '.join((d)) for d in df['distractors'].tolist()]
    return Dataset.from_dict({"input_text": inputs, "target_text": outputs})

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
    
    if model_type == "cl":
        print("\nStarting contrastive learning...")
        train_with_contrastive_loss(model, tokenizer, tokenized_train, tokenized_val)
    
    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    print(f"\nTotal training time: {format_time_duration(total_training_time)}")
    
    # Write to text file
    with open("sciq_rag_correct_006.txt", "w") as file:
      file.write(f"Total training time: {format_time_duration(total_training_time)}\n")
    
    return model, tokenizer

# =====================================
# Contrastive Learning
# =====================================
def contrastive_loss(anchor, positive, negatives, temperature= 0.1):
    """Calculate InfoNCE contrastive loss"""
    positive_score = F.cosine_similarity(anchor, positive, dim=-1) / temperature
    negative_scores = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1) / temperature
    logits = torch.cat([positive_score.unsqueeze(1), negative_scores], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(anchor.device)
    return F.cross_entropy(logits, labels)

def triplet_loss(anchor, positive, negative, margin=0.01, similarity='euclidean'):
    """Calculate triplet loss"""
    if similarity == 'cosine':
        sim_positive = F.cosine_similarity(anchor, positive, dim=-1)
        sim_negative = F.cosine_similarity(anchor, negative, dim=-1)
    elif similarity == 'euclidean':
        sim_positive = torch.norm(anchor - positive, dim=-1)
        sim_negative = torch.norm(anchor - negative, dim=-1)
    else:
        raise ValueError("Invalid similarity metric")
    
    loss = F.relu(sim_negative - sim_positive + margin)
    return loss.mean()

def train_with_contrastive_loss(model, tokenizer, train_dataset, val_dataset):
    """
    Train using contrastive learning with either InfoNCE or Triplet loss
    """
    
    cl_start_time = time.time()
    
    #shuffle=True
    train_loader = DataLoader(train_dataset, batch_size=Config.CL_BATCH_SIZE, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=Config.CL_LEARNING_RATE)
    
    model.train()
    for epoch in range(Config.CL_EPOCHS):
        epoch_start_time = time.time()
        total_loss = 0
        for batch in train_loader:
            # Prepare inputs
            inputs = tokenizer(
                batch['input_text'], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=Config.MAX_INPUT_LENGTH
            ).to(model.device)
            outputs = tokenizer(
                batch['target_text'], 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=Config.MAX_TARGET_LENGTH
            ).to(model.device)
            
            # Forward pass
            encoder_outputs = model.encoder(**inputs)
            decoder_outputs = model(**inputs, labels=outputs['input_ids'])
            
            # Get embeddings
            anchor = encoder_outputs.last_hidden_state.mean(dim=1)
            positive = decoder_outputs.encoder_last_hidden_state.mean(dim=1)
            batch_size = anchor.size(0)
            
            # Calculate generation loss
            gen_loss = decoder_outputs.loss
            
            # Select negatives and calculate contrastive loss based on loss type
            if Config.CL_LOSS_TYPE == "info_nce":
                # Random negatives for InfoNCE
                negatives = positive[torch.randperm(batch_size)]
                cont_loss = contrastive_loss(
                    anchor, positive, negatives, 
                    temperature=Config.CL_TEMPERATURE
                )
            elif Config.CL_LOSS_TYPE == "triplet":
                # Shifted negatives for Triplet Loss
                negatives = positive[torch.roll(torch.arange(batch_size), shifts=1)]
                cont_loss = triplet_loss(
                    anchor, positive, negatives, 
                    margin=Config.CL_MARGIN
                )
            else:
                raise ValueError("Invalid loss_type. Choose 'info_nce' or 'triplet'.")
            
            # Combine losses with configurable weights
            final_loss = (Config.CL_GEN_LOSS_WEIGHT * gen_loss + Config.CL_CONT_LOSS_WEIGHT * cont_loss)
            
            # Backward pass
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            
            total_loss += final_loss.item()
            
        epoch_end_time = time.time()
        epoch_duration = format_time_duration(epoch_end_time - epoch_start_time)
        print(f"Epoch {epoch + 1}/{Config.CL_EPOCHS}, Loss: {total_loss / len(train_loader):.4f}, Duration: {epoch_duration}")
        
    # Calculate total CL training time
    cl_end_time = time.time()
    total_cl_time = cl_end_time - cl_start_time
    print(f"\nContrastive learning training time: {format_time_duration(total_cl_time)}")
    
    # Save contrastively trained model
    cl_model_path = Config.get_model_save_path("cl")
    model.save_pretrained(cl_model_path)
    tokenizer.save_pretrained(cl_model_path)
    print(f"\nContrastively trained model ({Config.CL_LOSS_TYPE}) saved to: {cl_model_path}")
    
    

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
# Post-Processing
# =====================================
def process_generated_distractors(input_path, output_path):
    """Process and clean up generated distractors"""
    with open(input_path, 'r') as input_file:
        data = json.load(input_file)
    
    for item in data:
        # Split on commas and clean up
        item["generated_distractors"] = re.split(r', ', item["generated_distractors"][0])
        item["generated_distractors"] = [d.rstrip('.') for d in item["generated_distractors"]]
    
    with open(output_path, 'w') as output_file:
        json.dump(data, output_file, indent=4)

def post_process_results(model_type="ft"):
    """Run post-processing on generated results"""
    input_path = Config.get_output_path(model_type)
    output_path = Config.get_output_path(model_type, split=True)
    process_generated_distractors(input_path, output_path)

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
    input_file = Config.get_output_path(model_type, split=True)
    
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
    
    # Post-process results
    print("Post-processing results...")
    post_process_results(model_type)
    
    # Evaluate results
    print("Evaluating results...")
    metrics = evaluate_generations(model_type)
    
    return metrics

if __name__ == "__main__":
   
   parser = argparse.ArgumentParser(description='Run distractor generation pipeline')
   parser.add_argument('--model_type', type=str, default='ft', choices=['ft', 'cl'],
                     help='Model type: fine-tuned (ft) or contrastive learning (cl)')
   parser.add_argument('--cl_loss_type', type=str, default='info_nce', choices=['info_nce', 'triplet'],
                     help='Contrastive learning loss type: InfoNCE or Triplet loss')
   parser.add_argument('--gen_loss_weight', type=float, default=0.5,
                     help='Weight for generation loss (default: 0.5)')
   parser.add_argument('--cont_loss_weight', type=float, default=0.5,
                     help='Weight for contrastive loss (default: 0.5)')
   parser.add_argument('--train_only', action='store_true',
                     help='Only train the model without generation and evaluation')
   parser.add_argument('--generate_only', action='store_true',
                     help='Only generate distractors using existing model')
   parser.add_argument('--evaluate_only', action='store_true',
                     help='Only evaluate existing generations')
   
   args = parser.parse_args()
   
   # Update configurations if specified
   if args.cl_loss_type:
       Config.CL_LOSS_TYPE = args.cl_loss_type
   if args.gen_loss_weight is not None:
       Config.CL_GEN_LOSS_WEIGHT = args.gen_loss_weight
   if args.cont_loss_weight is not None:
       Config.CL_CONT_LOSS_WEIGHT = args.cont_loss_weight
   
   if args.train_only:
       print(f"Training model only... (Type: {args.model_type}, "
             f"CL Loss: {Config.CL_LOSS_TYPE}, "
             f"Gen Weight: {Config.CL_GEN_LOSS_WEIGHT}, "
             f"Cont Weight: {Config.CL_CONT_LOSS_WEIGHT})")
       train_data, val_data, _ = load_data()
       train_model(train_data, val_data, args.model_type)
   elif args.generate_only:
       print("Generating distractors only...")
       run_generation(args.model_type)
       post_process_results(args.model_type)
   elif args.evaluate_only:
       print("Evaluating existing generations...")
       evaluate_generations(args.model_type)
   else:
       print("Running complete pipeline...")
       metrics = run_pipeline(args.model_type)
       
   print("Done!")

# # Run with default weights (0.5 each)
# python script.py --model_type cl
# # Run with custom weights
# python script.py --model_type cl --gen_loss_weight 0.7 --cont_loss_weight 0.3
# # Run with custom weights and specific loss type
# python script.py --model_type cl --gen_loss_weight 0.1 --cont_loss_weight 0.1 --cl_loss_type info_nce

#python script.py --model_type cl --cl_loss_type info_nce
