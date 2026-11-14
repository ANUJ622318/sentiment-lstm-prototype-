"""
Text Sentiment Analysis with Bidirectional LSTM (Project DL-014)

train_model.py
--------------
Trains a deep Bidirectional LSTM model for 3-class sentiment analysis:
  0: Negative
  1: Neutral
  2: Positive

Key Improvements:
1. Multi-Domain Balanced Dataset: Combines Rotten Tomatoes, SST-2, Twitter Sentiment,
   decoded IMDb reviews, and curated everyday conversational emotional statements.
2. Robust Preprocessing: Contraction expansion ("don't" -> "do not"), negation preservation,
   repeated character normalization, and clean tokenization.
3. Bidirectional LSTM Architecture: Dual-direction recurrent memory cells capture preceding
   and succeeding context (crucial for negations like "I am not happy").
4. Masking (mask_zero=True): Dynamically masks padding tokens so short conversational sentences
   ("I am demotivated with her.") are not overwhelmed by padding.
5. Baseline Comparison: Trains a TF-IDF + Logistic Regression classifier on identical data
   for academic rigor and comparative benchmarking.
6. Exhaustive Evaluation: Produces Accuracy, Weighted/Macro Precision, Recall, F1-scores,
   Confusion Matrix plot, and Loss/Accuracy curves.
"""

import os
import sys

# Silence verbose TF warnings and configure writable cache
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["KERAS_HOME"] = "/tmp/keras"

import json
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense,
    Dropout, SpatialDropout1D, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from dataset import get_train_val_test_splits, LABEL_MAP
from preprocessing import (
    create_tokenizer, save_tokenizer,
    texts_to_padded_sequences, MAX_SEQUENCE_LENGTH, MAX_VOCAB_SIZE
)

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
EMBEDDING_DIM = 100
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
DEFAULT_EPOCHS = 6

# Paths
SAVED_MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_FILE = os.path.join(SAVED_MODEL_DIR, "sentiment_bilstm_model.keras")
TOKENIZER_FILE = os.path.join(SAVED_MODEL_DIR, "tokenizer.json")
METRICS_FILE = os.path.join(SAVED_MODEL_DIR, "model_info.json")
HISTORY_PLOT_FILE = os.path.join(SAVED_MODEL_DIR, "training_history.png")
CONFUSION_MATRIX_FILE = os.path.join(SAVED_MODEL_DIR, "confusion_matrix.png")


def build_bilstm_model(vocab_size=MAX_VOCAB_SIZE, embedding_dim=EMBEDDING_DIM, max_len=MAX_SEQUENCE_LENGTH):
    """
    Constructs the Bidirectional LSTM neural network architecture with masking:
      1. Embedding (mask_zero=True): Ignores padding tokens across variable lengths.
      2. SpatialDropout1D: Drops 1D word feature maps for regularization.
      3. Bidirectional LSTM: Reads sequence in forward and backward directions.
      4. Dense + BatchNormalization + ReLU: High-level contextual synthesis.
      5. Dropout (0.3): Regularization against overfitting.
      6. Dense (3, softmax): 3-class probability distribution.
    """
    print("[3/6] Constructing Bidirectional LSTM Architecture with Masking...")
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            mask_zero=True,
            name="word_embedding_with_mask"
        ),
        SpatialDropout1D(0.2, name="spatial_dropout"),
        Bidirectional(
            LSTM(
                LSTM_UNITS,
                dropout=0.2,
                recurrent_dropout=0.0,
                return_sequences=False
            ),
            name="bidirectional_lstm"
        ),
        Dense(DENSE_UNITS, activation="relu", name="dense_representation"),
        BatchNormalization(name="batch_norm"),
        Dropout(DROPOUT_RATE, name="dropout_regularization"),
        Dense(3, activation="softmax", name="sentiment_output_3class")
    ], name="BiLSTM_Sentiment_Classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Trains a TF-IDF + Logistic Regression model as an academic baseline for comparison.
    """
    print("\n[Baseline] Training TF-IDF + Logistic Regression comparison model...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")

    print(f"  → Baseline (TF-IDF + LogReg) Test Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
    return {
        "model_name": "TF-IDF + Logistic Regression (Baseline)",
        "test_accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1)
    }


def plot_and_save_confusion_matrix(y_true, y_pred, class_names=["Negative", "Neutral", "Positive"]):
    """Plots and saves confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="BiLSTM Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold"
            )

    fig.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    plt.close()
    print(f"      Saved confusion matrix plot to: {CONFUSION_MATRIX_FILE}")
    return cm.tolist()


def plot_and_save_training_curves(history):
    """Plots training and validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy Plot
    ax1.plot(history.history["accuracy"], label="Train Accuracy", color="#1e3a8a", linewidth=2)
    if "val_accuracy" in history.history:
        ax1.plot(history.history["val_accuracy"], label="Val Accuracy", color="#2563eb", linestyle="--", linewidth=2)
    ax1.set_title("Model Accuracy Over Epochs", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
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
    print(f"      Saved training curves to: {HISTORY_PLOT_FILE}")


def train(epochs=DEFAULT_EPOCHS, force=False):
    """Main training routine."""
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE) and os.path.exists(METRICS_FILE) and not force:
        print(f"✓ Trained model already exists at: {MODEL_FILE}")
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"  BiLSTM Test Accuracy: {meta.get('test_accuracy', 0)*100:.2f}% | F1: {meta.get('f1_score_weighted', 0):.4f}")
        print("  Pass --force to retrain.")
        return

    # 1. Load dataset splits
    print("[1/6] Loading multi-domain balanced dataset splits...")
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test) = get_train_val_test_splits()

    # 2. Fit and save tokenizer
    print("[2/6] Building and saving vocabulary tokenizer...")
    tokenizer = create_tokenizer(X_train_raw, vocab_size=MAX_VOCAB_SIZE)
    save_tokenizer(tokenizer, TOKENIZER_FILE)
    vocab_size_actual = len(tokenizer.word_index) + 1
    print(f"      Fitted tokenizer with {len(tokenizer.word_index):,} unique tokens.")

    # 3. Transform texts to padded sequence tensors
    X_train_pad = texts_to_padded_sequences(X_train_raw, tokenizer, max_len=MAX_SEQUENCE_LENGTH)
    X_val_pad = texts_to_padded_sequences(X_val_raw, tokenizer, max_len=MAX_SEQUENCE_LENGTH)
    X_test_pad = texts_to_padded_sequences(X_test_raw, tokenizer, max_len=MAX_SEQUENCE_LENGTH)

    # Compute class weights for balanced gradients
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_dict = {c: float(w) for c, w in zip(classes, weights)}
    print(f"      Computed class weights: {class_weights_dict}")

    # 4. Build and train baseline model
    baseline_metrics = train_baseline_model(X_train_raw, y_train, X_test_raw, y_test)

    # 5. Build and train Bidirectional LSTM
    model = build_bilstm_model(vocab_size=min(vocab_size_actual, MAX_VOCAB_SIZE))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=MODEL_FILE, monitor="val_accuracy", save_best_only=True, verbose=1)
    ]

    print(f"\n[4/6] Training Bidirectional LSTM for {epochs} epochs...")
    history = model.fit(
        X_train_pad,
        y_train,
        validation_data=(X_val_pad, y_val),
        batch_size=BATCH_SIZE,
        epochs=epochs,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Comprehensive Test Evaluation
    print("\n[5/6] Evaluating on unseen test set...")
    test_loss, test_acc = model.evaluate(X_test_pad, y_test, verbose=0)
    y_pred_probs = model.predict(X_test_pad, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro")
    class_report_str = classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"])

    print("\n" + "=" * 60)
    print("BI-DIRECTIONAL LSTM TEST EVALUATION REPORT:")
    print("=" * 60)
    print(f"Test Loss:             {test_loss:.4f}")
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Weighted Precision:    {precision_w:.4f}")
    print(f"Weighted Recall:       {recall_w:.4f}")
    print(f"Weighted F1-Score:     {f1_w:.4f}")
    print(f"Macro F1-Score:        {f1_m:.4f}")
    print("-" * 60)
    print(class_report_str)
    print("=" * 60)

    # Save plots
    print("\n[6/6] Generating evaluation plots & saving metadata...")
    plot_and_save_training_curves(history)
    cm_matrix = plot_and_save_confusion_matrix(y_test, y_pred)

    # Save comprehensive metrics metadata
    metadata = {
        "project_id": "DL-014",
        "project_title": "Text Sentiment Analysis with Bidirectional LSTM",
        "architecture": "Embedding (mask_zero=True) -> SpatialDropout1D -> BiLSTM(64) -> Dense(64) -> BatchNorm -> Dropout(0.3) -> Dense(3, Softmax)",
        "dataset_description": "Multi-domain balanced dataset (Rotten Tomatoes, SST-2, Twitter Sentiment, IMDb decoded reviews, and curated everyday emotional/negation sentences)",
        "classes": ["Negative", "Neutral", "Positive"],
        "num_classes": 3,
        "train_samples": int(len(X_train_pad)),
        "val_samples": int(len(X_val_pad)),
        "test_samples": int(len(X_test_pad)),
        "vocab_size": int(min(vocab_size_actual, MAX_VOCAB_SIZE)),
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "epochs_trained": int(len(history.history["loss"])),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_score_weighted": float(f1_w),
        "f1_score_macro": float(f1_m),
        "confusion_matrix": cm_matrix,
        "baseline_comparison": baseline_metrics
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Model saved to:    {MODEL_FILE}")
    print(f"✓ Tokenizer to:      {TOKENIZER_FILE}")
    print(f"✓ Metadata to:       {METRICS_FILE}")

    # Immediate verification on user's problematic sentence
    print("\n--- Diagnostic Check: User Problematic Example ---")
    problem_sentence = "I am demotivated with her."
    pad_prob = texts_to_padded_sequences([problem_sentence], tokenizer, max_len=MAX_SEQUENCE_LENGTH)
    prob_vec = model.predict(pad_prob, verbose=0)[0]
    pred_class = LABEL_MAP[int(np.argmax(prob_vec))]
    conf = float(np.max(prob_vec)) * 100.0
    print(f"Input: \"{problem_sentence}\"")
    print(f"Prediction: {pred_class} ({conf:.2f}% confidence)")
    print(f"Class Probabilities: [Neg: {prob_vec[0]:.3f}, Neu: {prob_vec[1]:.3f}, Pos: {prob_vec[2]:.3f}]")
    if pred_class == "Negative":
        print(">>> SUCCESS: Fixed underlying prediction failure! <<<")
    else:
        print(">>> WARNING: Still not predicting Negative. Further tuning needed. <<<")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM Sentiment Analysis Model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs to train")
    parser.add_argument("--force", action="store_true", help="Force retrain even if model files exist")
    args = parser.parse_args()

    train(epochs=args.epochs, force=args.force)
