"""
train_deep_models.py
---------------------
يدرب SimpleRNN و LSTM و GRU على الداتا المعالجة، ويحفظ:
    - أوزان كل موديل (.keras)
    - تاريخ التدريب (history) لكل موديل (لعمل رسوم بيانية)

الاستخدام:
    python src/train_deep_models.py
"""

import os
import json
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from models_rnn import MODEL_BUILDERS

DATA_DIR = "data"
ARTIFACTS_DIR = "models"
EPOCHS = 20
BATCH_SIZE = 64

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_processed_data():
    data = np.load(os.path.join(DATA_DIR, "processed_data.npz"))
    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
    )


def main():
    tf.random.set_seed(42)

    X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()
    with open(os.path.join(ARTIFACTS_DIR, "config.json")) as f:
        config = json.load(f)
    class_weight_dict = joblib.load(os.path.join(ARTIFACTS_DIR, "class_weights.pkl"))

    vocab_size = min(config["max_vocab_size"], 20000) + 1
    max_len = config["max_seq_len"]
    num_classes = config["num_classes"]

    all_histories = {}

    for name, builder in MODEL_BUILDERS.items():
        print("\n" + "=" * 70)
        print(f"🚀 Training {name} ...")
        print("=" * 70)

        model = builder(vocab_size, max_len, num_classes)
        model.summary()

        ckpt_path = os.path.join(ARTIFACTS_DIR, f"{name.lower()}_model.keras")
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks,
            verbose=2,
        )

        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ {name} — Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

        all_histories[name] = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        all_histories[name]["test_accuracy"] = float(test_acc)
        all_histories[name]["test_loss"] = float(test_loss)

    with open(os.path.join(ARTIFACTS_DIR, "training_history.json"), "w") as f:
        json.dump(all_histories, f, indent=2)

    print("\n🎉 All three models trained and saved to 'models/'.")


if __name__ == "__main__":
    main()
