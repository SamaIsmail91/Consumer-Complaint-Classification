"""
models_rnn.py
-------------
بناء ثلاثة موديلات من الصفر باستخدام Keras:
    - SimpleRNN
    - LSTM
    - GRU
كل موديل يستخدم نفس الـ Embedding layer configuration عشان المقارنة تكون عادلة.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout,
    Bidirectional, GlobalMaxPooling1D, SpatialDropout1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


EMBEDDING_DIM = 128


def build_simple_rnn(vocab_size: int, max_len: int, num_classes: int) -> Sequential:
    model = Sequential(name="SimpleRNN_Model")
    # mask_zero=True is critical: it tells the recurrent layers to ignore the
    # padding timesteps, otherwise dozens of trailing zero-steps wash out
    # the signal from the (usually much shorter) real text.
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(SimpleRNN(64, return_sequences=True)))
    model.add(SimpleRNN(32))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm(vocab_size: int, max_len: int, num_classes: int) -> Sequential:
    model = Sequential(name="LSTM_Model")
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(LSTM(32))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_gru(vocab_size: int, max_len: int, num_classes: int) -> Sequential:
    model = Sequential(name="GRU_Model")
    model.add(Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, mask_zero=True))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(GRU(64, return_sequences=True)))
    model.add(GRU(32))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu", kernel_regularizer=l2(1e-4)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


MODEL_BUILDERS = {
    "SimpleRNN": build_simple_rnn,
    "LSTM": build_lstm,
    "GRU": build_gru,
}
