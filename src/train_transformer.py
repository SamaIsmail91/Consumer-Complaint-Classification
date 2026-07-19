"""
train_transformer.py
---------------------
Fine-tunes a pretrained DistilBERT model (HuggingFace) for complaint classification.
Saves the fine-tuned model + tokenizer to models/transformer_model/

الاستخدام:
    python src/train_transformer.py

يفضل تشغيله على GPU (Google Colab / Kaggle) لأنه أبطأ بكتير من RNN/LSTM/GRU على CPU.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset

DATA_DIR = "data"
ARTIFACTS_DIR = "models"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = os.path.join(ARTIFACTS_DIR, "transformer_model")
MAX_LEN = 128
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_raw_splits():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_raw.csv")).dropna()
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_raw.csv")).dropna()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_raw.csv")).dropna()
    return train_df, val_df, test_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    with open(os.path.join(ARTIFACTS_DIR, "config.json")) as f:
        config = json.load(f)
    num_classes = config["num_classes"]

    train_df, val_df, test_df = load_raw_splits()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = Dataset.from_pandas(train_df.rename(columns={"label_id": "label"}))
    val_ds = Dataset.from_pandas(val_df.rename(columns={"label_id": "label"}))
    test_ds = Dataset.from_pandas(test_df.rename(columns={"label_id": "label"}))

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)
    test_ds = test_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("🚀 Fine-tuning DistilBERT ...")
    trainer.train()

    print("\n📊 Evaluating on test set ...")
    test_results = trainer.evaluate(test_ds)
    print(test_results)

    # Save final model + tokenizer for the Gradio app
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(ARTIFACTS_DIR, "transformer_test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    print(f"\n✅ Transformer fine-tuned and saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
