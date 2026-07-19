"""
predict.py
----------
دوال تنبؤ موحدة تُستخدم من app.py (Gradio) لأي من الموديلات الأربعة.
تحمّل الموديلات مرة واحدة (lazy loading) وتُرجع: الفئة المتوقعة + confidence + top-k احتمالات.
"""

import os
import json
import joblib
import numpy as np
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

ARTIFACTS_DIR = "models"

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"x{2,}", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


class ComplaintClassifier:
    """Lazily loads whichever models are available on disk and exposes a unified predict() API."""

    def __init__(self):
        self._keras_models = {}
        self._tokenizer = None
        self._label_encoder = None
        self._config = None
        self._transformer_model = None
        self._transformer_tokenizer = None
        self.available_models = []
        self._discover()

    # ------------------------------------------------------------------
    def _discover(self):
        cfg_path = os.path.join(ARTIFACTS_DIR, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                self._config = json.load(f)

        le_path = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
        if os.path.exists(le_path):
            self._label_encoder = joblib.load(le_path)

        tok_path = os.path.join(ARTIFACTS_DIR, "tokenizer.pkl")
        if os.path.exists(tok_path):
            self._tokenizer = joblib.load(tok_path)

        for name in ["simplernn", "lstm", "gru"]:
            if os.path.exists(os.path.join(ARTIFACTS_DIR, f"{name}_model.keras")):
                self.available_models.append(
                    {"simplernn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU"}[name]
                )

        if os.path.exists(os.path.join(ARTIFACTS_DIR, "transformer_model")):
            self.available_models.append("Transformer (DistilBERT)")

        if not self.available_models:
            self.available_models = ["SimpleRNN", "LSTM", "GRU", "Transformer (DistilBERT)"]

    # ------------------------------------------------------------------
    def _load_keras(self, model_key):
        import tensorflow as tf
        if model_key not in self._keras_models:
            path = os.path.join(ARTIFACTS_DIR, f"{model_key}_model.keras")
            self._keras_models[model_key] = tf.keras.models.load_model(path)
        return self._keras_models[model_key]

    def _load_transformer(self):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        if self._transformer_model is None:
            path = os.path.join(ARTIFACTS_DIR, "transformer_model")
            self._transformer_tokenizer = AutoTokenizer.from_pretrained(path)
            self._transformer_model = AutoModelForSequenceClassification.from_pretrained(path)
            self._transformer_model.eval()
        return self._transformer_model, self._transformer_tokenizer

    # ------------------------------------------------------------------
    def predict(self, text: str, model_name: str, top_k: int = 3):
        if not text or not text.strip():
            return None, None, []

        classes = self._config["classes"] if self._config else self._label_encoder.classes_.tolist()

        if model_name == "Transformer (DistilBERT)":
            import torch
            model, tokenizer = self._load_transformer()
            enc = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=1).squeeze().numpy()
        else:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            key = model_name.lower().replace(" ", "")
            model = self._load_keras(key)
            cleaned = clean_text(text)
            seq = self._tokenizer.texts_to_sequences([cleaned])
            max_len = self._config["max_seq_len"]
            padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
            probs = model.predict(padded, verbose=0)[0]

        top_idx = np.argsort(probs)[::-1][:top_k]
        top_results = [(classes[i], float(probs[i])) for i in top_idx]

        pred_label = top_results[0][0]
        confidence = top_results[0][1]
        return pred_label, confidence, top_results


classifier = ComplaintClassifier()
