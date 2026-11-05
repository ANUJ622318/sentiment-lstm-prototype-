"""
Text Sentiment Analysis using LSTM
Project ID: DL-014

train_model.py
--------------
This script trains a Long Short-Term Memory (LSTM) deep learning model
on the IMDb movie-review dataset to classify sentiment as Positive or Negative.

Features:
- Uses TensorFlow/Keras built-in IMDb dataset (no manual downloads needed).
- Manageable dataset subset configured by default for fast, smooth training on standard laptops.
- Fixed random seed for reproducibility.
- Saves the trained model, word-to-index vocabulary dictionary, and training metrics locally.
"""

import os
import sys

# Configure environment before imports to silence verbose logs and set writable cache paths
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["KERAS_HOME"] = "/tmp/keras"

import re
import json
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
MAX_VOCAB = 10000          # Top 10,000 words in vocabulary
MAX_LEN = 200              # Sequence length (cut/pad reviews to 200 tokens)
EMBEDDING_DIM = 64         # Word vector dimension
LSTM_UNITS = 64            # Number of LSTM memory cells
BATCH_SIZE = 64            # Training batch size
DEFAULT_EPOCHS = 3         # Quick and effective for student laptops
DEFAULT_TRAIN_SIZE = 10000 # Student-friendly subset (out of 25,000)
DEFAULT_TEST_SIZE = 2500   # Student-friendly evaluation subset

# Directory paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_lstm_model.keras")
WORD_INDEX_FILE = os.path.join(MODEL_DIR, "word_index.json")
METRICS_FILE = os.path.join(MODEL_DIR, "model_info.json")
HISTORY_PLOT_FILE = os.path.join(MODEL_DIR, "training_history.png")


def build_and_save_tokenizer():
    """
    Extracts the IMDb word index and creates a clean word-to-id mapping
    accounting for Keras special offset tokens:
      0: <PAD>
      1: <START>
      2: <UNK>
      3: <UNUSED>
    """
    print("[1/5] Preparing word vocabulary mapping from IMDb dataset...")
    raw_word_index = imdb.get_word_index()

    # Keras offset: actual tokens in sequences are offset by 3
    word_to_id = {
        "<PAD>": 0,
        "<START>": 1,
        "<UNK>": 2,
    }

    for word, rank in raw_word_index.items():
        token_id = rank + 3
        if token_id < MAX_VOCAB:
            word_to_id[word] = token_id

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(WORD_INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(word_to_id, f, indent=2)

    print(f"      Saved vocabulary ({len(word_to_id):,} tokens) to: {WORD_INDEX_FILE}")
    return word_to_id


def load_dataset(train_size=DEFAULT_TRAIN_SIZE, test_size=DEFAULT_TEST_SIZE):
    """
    Loads IMDb dataset with fixed vocabulary size and cuts/pads sequences to MAX_LEN.
    Uses a representative subset for student laptop friendly training.
    """
    print(f"[2/5] Loading IMDb dataset (subset: {train_size:,} train, {test_size:,} test)...")
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = imdb.load_data(
        num_words=MAX_VOCAB,
        index_from=3
    )

    # Slice manageable subsets
    x_train_sub = x_train_raw[:train_size]
    y_train_sub = y_train_raw[:train_size]
    x_test_sub = x_test_raw[:test_size]
    y_test_sub = y_test_raw[:test_size]

    # Pad sequences to fixed length (pre-padding is standard in NLP)
    x_train = pad_sequences(x_train_sub, maxlen=MAX_LEN, padding="pre", truncating="pre")
    x_test = pad_sequences(x_test_sub, maxlen=MAX_LEN, padding="pre", truncating="pre")
    y_train = np.array(y_train_sub, dtype=np.float32)
    y_test = np.array(y_test_sub, dtype=np.float32)

    print(f"      Training samples shape: {x_train.shape}")
    print(f"      Testing samples shape:  {x_test.shape}")
    return x_train, y_train, x_test, y_test


def build_lstm_model(vocab_size=MAX_VOCAB, embedding_dim=EMBEDDING_DIM, lstm_units=LSTM_UNITS):
    """
    Builds the Sequential LSTM neural network architecture:
      1. Embedding Layer: Maps word indices into dense continuous vectors.
      2. SpatialDropout1D: Prevents feature co-adaptation across embeddings.
      3. LSTM Layer: Recurrent cells with forget/input/output gates for context memory.
      4. Dense Hidden Layer + ReLU: High-level non-linear feature combination.
      5. Dropout Layer: Regularization to avoid overfitting.
      6. Output Layer + Sigmoid: Probability of positive sentiment (0 to 1).
    """
    print("[3/5] Constructing LSTM neural network architecture...")
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="word_embedding"
        ),
        SpatialDropout1D(0.2, name="spatial_dropout"),
        LSTM(
            lstm_units,
            dropout=0.2,
            recurrent_dropout=0.0,
            name="lstm_layer"
        ),
        Dense(32, activation="relu", name="dense_hidden"),
        Dropout(0.3, name="dropout_regularization"),
        Dense(1, activation="sigmoid", name="output_sigmoid")
    ], name="IMDb_Sentiment_LSTM")

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def clean_and_tokenize_text(text, word_to_id, max_len=MAX_LEN):
    """
    Helper function to preprocess raw user review text consistently with training data:
    1. Lowercase text.
    2. Extract word tokens using regular expression.
    3. Look up token IDs in vocabulary (fallback to <UNK> = 2).
    4. Prepend <START> token (1).
    5. Pad or truncate sequence to max_len.
    """
    text_clean = text.lower()
    tokens = re.findall(r"[a-z0-9']+", text_clean)
    if not tokens:
        return np.zeros((1, max_len), dtype=np.int32)

    token_ids = [1]  # <START> token
    for t in tokens:
        token_ids.append(word_to_id.get(t, 2))  # 2 is <UNK>

    padded = pad_sequences([token_ids], maxlen=max_len, padding="pre", truncating="pre")
    return padded


def plot_and_save_history(history):
    """
    Plots training and validation accuracy and loss curves and saves to disk.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy Plot
    ax1.plot(history.history["accuracy"], label="Train Accuracy", color="#1e3a8a", linewidth=2)
    if "val_accuracy" in history.history:
        ax1.plot(history.history["val_accuracy"], label="Val Accuracy", color="#2563eb", linestyle="--", linewidth=2)
    ax1.set_title("Model Accuracy Over Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0.5, 1.0])
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Loss Plot
    ax2.plot(history.history["loss"], label="Train Loss", color="#dc2626", linewidth=2)
    if "val_loss" in history.history:
        ax2.plot(history.history["val_loss"], label="Val Loss", color="#f97316", linestyle="--", linewidth=2)
    ax2.set_title("Model Loss Over Epochs", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_FILE, dpi=150)
    plt.close()
    print(f"      Saved training history plot to: {HISTORY_PLOT_FILE}")


def train(epochs=DEFAULT_EPOCHS, train_size=DEFAULT_TRAIN_SIZE, test_size=DEFAULT_TEST_SIZE, force=False):
    """
    Main training routine.
    """
    if os.path.exists(MODEL_FILE) and os.path.exists(WORD_INDEX_FILE) and os.path.exists(METRICS_FILE) and not force:
        print(f"✓ Model already exists at: {MODEL_FILE}")
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  Existing Model Test Accuracy: {meta.get('test_accuracy', 0)*100:.2f}%")
        print("  Pass --force to retrain.")
        return

    word_to_id = build_and_save_tokenizer()
    x_train, y_train, x_test, y_test = load_dataset(train_size, test_size)
    model = build_lstm_model()

    print(f"\n[4/5] Training LSTM model for {epochs} epochs...")
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        validation_split=0.15,
        verbose=1
    )

    print("\n[5/5] Evaluating model on unseen test set...")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"      → Test Loss:     {test_loss:.4f}")
    print(f"      → Test Accuracy: {test_acc * 100:.2f}%")

    # Save trained model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_FILE)
    print(f"      ✓ Trained model saved to: {MODEL_FILE}")

    # Plot history
    plot_and_save_history(history)

    # Save real metadata
    metadata = {
        "project_id": "DL-014",
        "project_title": "Text Sentiment Analysis using LSTM",
        "dataset_name": "IMDb Large Movie Review Dataset (Keras built-in)",
        "train_samples": int(train_size),
        "test_samples": int(test_size),
        "vocabulary_size": int(len(word_to_id)),
        "max_sequence_length": MAX_LEN,
        "embedding_dimensions": EMBEDDING_DIM,
        "lstm_units": LSTM_UNITS,
        "epochs_trained": int(epochs),
        "batch_size": BATCH_SIZE,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
        "final_val_accuracy": float(history.history["val_accuracy"][-1]) if "val_accuracy" in history.history else None,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"      ✓ Metadata & metrics saved to: {METRICS_FILE}")

    # Quick sanity check with sample reviews
    print("\n--- Model Verification Sanity Check ---")
    samples = [
        "This movie was an absolute masterpiece! The plot was gripping and the performances were sensational.",
        "A truly awful film. The pacing dragged endlessly and the dialogue was completely cringe-worthy."
    ]
    for sample in samples:
        padded = clean_and_tokenize_text(sample, word_to_id, MAX_LEN)
        prob = float(model.predict(padded, verbose=0)[0][0])
        sentiment = "Positive" if prob >= 0.5 else "Negative"
        confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100
        print(f"Review: \"{sample[:60]}...\"")
        print(f"Prediction: {sentiment} ({confidence:.2f}% confidence)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Sentiment Analysis Model on IMDb Dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs to train")
    parser.add_argument("--train_size", type=int, default=DEFAULT_TRAIN_SIZE, help="Number of training samples")
    parser.add_argument("--test_size", type=int, default=DEFAULT_TEST_SIZE, help="Number of testing samples")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model files exist")
    args = parser.parse_args()

    train(epochs=args.epochs, train_size=args.train_size, test_size=args.test_size, force=args.force)
