"""
preprocessing.py
-----------------
- تحميل الداتا
- تنظيف النصوص (lowercase, remove punctuation/numbers, stopwords, lemmatization)
- تحليل توازن الفئات
- Tokenization + Padding
- تقسيم train / val / test
- حفظ الـ tokenizer و label encoder و الداتا الجاهزة (.npz / .pkl)

الاستخدام:
    python src/preprocessing.py
"""

import os
import re
import string
import json
import joblib
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
DATA_PATH = "data/complaints.csv"
TEXT_COL = "Consumer complaint narrative"
LABEL_COL = "Product"

TOP_N_CLASSES = 10          # خذ أهم N فئة فقط (None = كل الفئات)
SAMPLE_SIZE = None          # مثال: 50000 لو الداتا كبيرة جدًا، None = استخدم الكل
MIN_WORDS = 3                # احذف الشكاوى القصيرة جدًا (نص غير مفيد)

MAX_VOCAB_SIZE = 20000
MAX_SEQ_LEN = 150
OOV_TOKEN = "<OOV>"

TEST_SIZE = 0.15
VAL_SIZE = 0.15              # نسبة من الـ train المتبقي
RANDOM_STATE = 42

ARTIFACTS_DIR = "models"
DATA_OUT_DIR = "data"

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


# ------------------------------------------------------------------
# NLTK setup
# ------------------------------------------------------------------
def ensure_nltk():
    for pkg in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.data.find(f"corpora/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)


ensure_nltk()
STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ------------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase -> remove punctuation/numbers/special chars -> remove stopwords -> lemmatize"""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"x{2,}", " ", text)              # redact placeholders like "XXXX" used in CFPB data
    text = re.sub(r"http\S+|www\S+", " ", text)      # urls
    text = re.sub(r"[^a-z\s]", " ", text)             # remove numbers, punctuation, special chars
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


# ------------------------------------------------------------------
# Load & explore
# ------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    print(f"📂 Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df = df.rename(columns={TEXT_COL: "text", LABEL_COL: "label"})
    print(f"Initial rows: {len(df)}")

    if SAMPLE_SIZE is not None and len(df) > SAMPLE_SIZE:
        df = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    if TOP_N_CLASSES is not None:
        top_classes = df["label"].value_counts().nlargest(TOP_N_CLASSES).index
        df = df[df["label"].isin(top_classes)]

    print("\n📊 Class distribution (before cleaning):")
    print(df["label"].value_counts())

    return df.reset_index(drop=True)


def handle_class_imbalance(df: pd.DataFrame, max_ratio: float = 3.0) -> pd.DataFrame:
    """
    Down-sample dominant classes so the majority class is at most `max_ratio`
    times the size of the minority class (keeps things balanced without
    throwing away too much data). Combined with class_weight during training.
    """
    counts = df["label"].value_counts()
    min_count = counts.min()
    cap = int(min_count * max_ratio)

    parts = []
    for label, group in df.groupby("label"):
        if len(group) > cap:
            group = group.sample(cap, random_state=RANDOM_STATE)
        parts.append(group)

    balanced = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"\n⚖️  Balanced dataset: {len(df)} -> {len(balanced)} rows (cap={cap} per class)")
    return balanced


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def run():
    df = load_data()

    print("\n🧹 Cleaning text ...")
    df["clean_text"] = df["text"].apply(clean_text)
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df = df[df["word_count"] >= MIN_WORDS].reset_index(drop=True)
    print(f"Rows after cleaning & filtering short texts: {len(df)}")

    df = handle_class_imbalance(df)

    # Label encoding
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    num_classes = len(le.classes_)
    print(f"\n🏷️  Classes ({num_classes}): {list(le.classes_)}")

    # Train / val / test split (stratified)
    X_temp, X_test, y_temp, y_test = train_test_split(
        df["clean_text"].values, df["label_id"].values,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label_id"].values
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"\n✂️  Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Tokenizer
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token=OOV_TOKEN)
    tokenizer.fit_on_texts(X_train)

    def to_padded(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=MAX_SEQ_LEN, padding="post", truncating="post")

    X_train_pad = to_padded(X_train)
    X_val_pad = to_padded(X_val)
    X_test_pad = to_padded(X_test)

    # Class weights (for imbalance handling during training)
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    # Save artifacts
    np.savez(
        os.path.join(DATA_OUT_DIR, "processed_data.npz"),
        X_train=X_train_pad, y_train=y_train,
        X_val=X_val_pad, y_val=y_val,
        X_test=X_test_pad, y_test=y_test,
    )
    # Raw cleaned text (needed for transformer, which uses its own tokenizer)
    pd.DataFrame({"text": X_train, "label_id": y_train}).to_csv(f"{DATA_OUT_DIR}/train_raw.csv", index=False)
    pd.DataFrame({"text": X_val, "label_id": y_val}).to_csv(f"{DATA_OUT_DIR}/val_raw.csv", index=False)
    pd.DataFrame({"text": X_test, "label_id": y_test}).to_csv(f"{DATA_OUT_DIR}/test_raw.csv", index=False)

    joblib.dump(tokenizer, f"{ARTIFACTS_DIR}/tokenizer.pkl")
    joblib.dump(le, f"{ARTIFACTS_DIR}/label_encoder.pkl")
    joblib.dump(class_weight_dict, f"{ARTIFACTS_DIR}/class_weights.pkl")

    config = {
        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "num_classes": num_classes,
        "classes": list(le.classes_),
    }
    with open(f"{ARTIFACTS_DIR}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("\n✅ Preprocessing complete. Artifacts saved to 'models/' and 'data/'.")
    print(f"   Vocabulary size (actual): {len(tokenizer.word_index)}")
    print(f"   Sequence length: {MAX_SEQ_LEN}")


if __name__ == "__main__":
    run()
