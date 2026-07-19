"""
evaluate.py
-----------
يقيّم الأربع موديلات (SimpleRNN, LSTM, GRU, Transformer) على test set:
    Accuracy, Precision, Recall, F1-score, Confusion Matrix
ويحفظ:
    outputs/model_comparison.csv
    outputs/comparison_chart.png
    outputs/confusion_matrix_<model>.png  (لكل موديل)

الاستخدام:
    python src/evaluate.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

import tensorflow as tf

DATA_DIR = "data"
ARTIFACTS_DIR = "models"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_test_data():
    data = np.load(os.path.join(DATA_DIR, "processed_data.npz"))
    return data["X_test"], data["y_test"]


def load_config_and_encoder():
    with open(os.path.join(ARTIFACTS_DIR, "config.json")) as f:
        config = json.load(f)
    le = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))
    return config, le


def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{model_name.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   Saved: {path}")


def evaluate_keras_model(model_path, model_name, X_test, y_test, classes):
    print(f"\n📈 Evaluating {model_name} ...")
    model = tf.keras.models.load_model(model_path)
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    plot_confusion_matrix(y_test, y_pred, classes, model_name)

    return {"Model": model_name, "Accuracy": acc, "Precision": precision, "Recall": recall, "F1-score": f1}


def evaluate_transformer(X_test_raw, y_test, classes):
    model_path = os.path.join(ARTIFACTS_DIR, "transformer_model")
    if not os.path.exists(model_path):
        print("⚠️  Transformer model not found, skipping. Run src/train_transformer.py first.")
        return None

    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except ImportError:
        print("⚠️  'transformers'/'torch' not installed, skipping transformer evaluation.")
        return None

    print("\n📈 Evaluating Transformer (DistilBERT) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    preds = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_test_raw), batch_size):
            batch = X_test_raw[i:i + batch_size]
            enc = tokenizer(list(batch), truncation=True, padding=True, max_length=128, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

    y_pred = np.array(preds)
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
    plot_confusion_matrix(y_test, y_pred, classes, "Transformer")

    return {"Model": "Transformer (DistilBERT)", "Accuracy": acc, "Precision": precision, "Recall": recall, "F1-score": f1}


def main():
    X_test, y_test = load_test_data()
    config, le = load_config_and_encoder()
    classes = config["classes"]

    results = []
    for name in ["simplernn", "lstm", "gru"]:
        model_path = os.path.join(ARTIFACTS_DIR, f"{name}_model.keras")
        if os.path.exists(model_path):
            results.append(evaluate_keras_model(model_path, name.upper() if name != "simplernn" else "SimpleRNN",
                                                 X_test, y_test, classes))
        else:
            print(f"⚠️  {model_path} not found, skipping.")

    test_raw_df = pd.read_csv(os.path.join(DATA_DIR, "test_raw.csv")).dropna()
    transformer_result = evaluate_transformer(test_raw_df["text"].values, test_raw_df["label_id"].values, classes)
    if transformer_result:
        results.append(transformer_result)

    comparison_df = pd.DataFrame(results).sort_values("F1-score", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    print("\n" + "=" * 70)
    print("🏆 MODEL COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # Comparison bar chart
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(comparison_df))
    width = 0.2
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, comparison_df[metric], width, label=metric)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(comparison_df["Model"])
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparison_chart.png"), dpi=150)
    plt.close()

    best_model = comparison_df.iloc[0]["Model"]
    print(f"\n🥇 Best model: {best_model}")
    with open(os.path.join(OUTPUT_DIR, "best_model.txt"), "w") as f:
        f.write(best_model)


if __name__ == "__main__":
    main()
