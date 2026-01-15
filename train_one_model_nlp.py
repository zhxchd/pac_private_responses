import numpy as np
np.random.seed(42)

import torch
torch.manual_seed(42)
torch.cuda.manual_seed(42)

import json
import os
import argparse
import transformers
from utils import load_text_dataset
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('split_id', type=str)
parser.add_argument('model', type=str, default='bert_small')
args = parser.parse_args()

if args.model != 'bert_small':
    raise ValueError("Only 'bert_small' model is supported in this script.")

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
data, test_split, num_classes = load_text_dataset(args.dataset, tokenizer)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

curr_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(curr_dir, 'saved_models', args.dataset, args.split_id)

with open(os.path.join(output_dir, "train_data_indices.json"), 'r') as f:
    train_data_indices = json.load(f)

subsampled_train_data = data['train'].select(train_data_indices)

# we are training a BERT-small model from scratch
model = AutoModelForSequenceClassification.from_config(
    BertConfig(
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
        num_labels=num_classes,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
)

training_args = TrainingArguments(
    output_dir=output_dir,
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=64,
    num_train_epochs=50, # this is large, but we use early stopping
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    weight_decay=0.01,
    warmup_steps=500,
    eval_strategy="epoch", 
    logging_strategy="epoch",
    fp16=True,
    seed=42,
    report_to="none",
    save_total_limit=1
)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

split_datasets = subsampled_train_data.train_test_split(test_size=0.1)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split_datasets['train'],
    eval_dataset=split_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[
        transformers.EarlyStoppingCallback(early_stopping_patience=5)
    ]
)

trainer.train()

trainer.save_model(output_dir)

train_predictions = trainer.predict(data['train']).predictions
train_scores = softmax(train_predictions, axis=1)
np.save(os.path.join(output_dir, f"{args.model}_train_scores.npy"), train_scores)

test_predictions = trainer.predict(data[test_split]).predictions
test_scores = softmax(test_predictions, axis=1)
test_acc = (np.argmax(test_scores, axis=1) == data[test_split]['labels']).mean()
print(f"Test accuracy: {test_acc}")
np.save(os.path.join(output_dir, f"{args.model}_test_scores.npy"), test_scores)

# delete the checkpoint folders to save space
for folder in os.listdir(output_dir):
    if folder.startswith('checkpoint-'):
        folder_path = os.path.join(output_dir, folder)
        if os.path.isdir(folder_path):
            import shutil
            shutil.rmtree(folder_path)
