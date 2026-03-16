import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# ==========================
# 📥 Load CSV Datasets
# ==========================
dataset = load_dataset(
    "csv",
    data_files={
        "train": "multi_train.csv",       # Bulk training set
        "validation": "single_train.csv", # Validation set
        "test": "expert_clean.csv"        # Gold test set
    }
)

print("✅ Dataset loaded successfully")
print(dataset)

# ==========================
# 🧠 Encode Labels
# ==========================
# Collect all labels to fit encoder on complete label space
all_labels = list(dataset['train']['label']) + \
             list(dataset['validation']['label']) + \
             list(dataset['test']['label'])

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

num_labels = len(label_encoder.classes_)
print(f"✅ Number of unique labels: {num_labels}")

# Encode labels for each split
def encode_labels(batch):
    batch["labels"] = label_encoder.transform(batch["label"])
    return batch

dataset = dataset.map(encode_labels)

# ==========================
# ✍️ Tokenization
# ==========================
MODEL_NAME = "bert-base-uncased"  # you can change to legalBERT, etc.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(batch):
    return tokenizer(
        batch["text"],  # column name in CSV
        truncation=True,
        padding="max_length",
        max_length=256,
    )

dataset = dataset.map(preprocess_function, batched=True)
dataset = dataset.remove_columns(["text", "label"])  # remove raw cols
dataset.set_format("torch")

# ==========================
# ⚡ Load Model
# ==========================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

# ==========================
# 📊 Metrics
# ==========================
accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# ==========================
# 🏋️ Training
# ==========================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==========================
# 💾 Save Model & Tokenizer
# ==========================
os.makedirs("fine_tuned_model", exist_ok=True)
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

# ==========================
# 💾 Save Label Encoder
# ==========================
with open("fine_tuned_model/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Model, tokenizer, and label encoder saved successfully to fine_tuned_model/")
