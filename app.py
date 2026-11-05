"""
Text Sentiment Analysis using LSTM
Project ID: DL-014

app.py
------
Streamlit Web Application for real-time text sentiment analysis using
a trained Long Short-Term Memory (LSTM) deep learning model.
"""

import os
import sys

# Silence verbose TF warnings and configure writable cache
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["KERAS_HOME"] = "/tmp/keras"

import re
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Text Sentiment Analysis using LSTM | DL-014",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
KERAS_MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_lstm_model.keras")
H5_MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_lstm_model.h5")
MODEL_FILE = KERAS_MODEL_FILE if os.path.exists(KERAS_MODEL_FILE) else H5_MODEL_FILE

WORD_INDEX_FILE = os.path.join(MODEL_DIR, "word_index.json")
METRICS_FILE = os.path.join(MODEL_DIR, "model_info.json")
HISTORY_PLOT_FILE = os.path.join(MODEL_DIR, "training_history.png")

MAX_LEN = 200

# -----------------------------------------------------------------------------
# Navy / Blue Modern UI Theme (Custom CSS)
# -----------------------------------------------------------------------------
NAVY_CSS = """
<style>
    /* Global styles */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Header banner */
    .hero-banner {
        background: linear-gradient(135deg, #0a192f 0%, #1e3a8a 60%, #3b82f6 100%);
        color: #ffffff;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.15);
        color: #93c5fd;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-bottom: 0.6rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        color: #ffffff;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        margin-top: 0.6rem;
        line-height: 1.5;
        max-width: 800px;
    }

    /* Cards */
    .card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* Metric pill */
    .metric-pill {
        background: #f1f5f9;
        border-left: 4px solid #1e3a8a;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .metric-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
    }

    /* Sentiment outcome containers */
    .sentiment-positive {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(16, 185, 129, 0.2);
    }
    .sentiment-negative {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 2px solid #ef4444;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(239, 68, 68, 0.2);
    }
    .verdict-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .verdict-positive {
        color: #047857;
    }
    .verdict-negative {
        color: #b91c1c;
    }

    /* How it works pipeline steps */
    .pipeline-step {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.75rem;
    }
    .pipeline-num {
        display: inline-block;
        background: #1e3a8a;
        color: white;
        border-radius: 50%;
        width: 26px;
        height: 26px;
        text-align: center;
        line-height: 26px;
        font-weight: bold;
        font-size: 0.85rem;
        margin-right: 8px;
    }

    /* Primary button custom styling */
    div.stButton > button:first-child {
        background-color: #1e3a8a;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.55rem 1.4rem;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #2563eb;
        color: white;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }
</style>
"""
st.markdown(NAVY_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Helper Functions: Loading Model & Vocabulary
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sentiment_model(file_path):
    """Loads the compiled Keras model from disk."""
    if not os.path.exists(file_path):
        return None
    try:
        model = load_model(file_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_word_index():
    """Loads the word-to-id mapping dictionary."""
    if not os.path.exists(WORD_INDEX_FILE):
        return None
    with open(WORD_INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_metrics():
    """Loads saved training metrics."""
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_text(text: str, word_to_id: dict, max_len: int = MAX_LEN):
    """
    Cleans and tokenizes input text consistently with model training:
    - Lowercase
    - Regex tokenization
    - Vocabulary mapping (fallback to <UNK> = 2)
    - Prepends <START> token (1)
    - Pre-pads to max_len
    """
    text_clean = text.lower()
    tokens = re.findall(r"[a-z0-9']+", text_clean)
    if not tokens:
        return np.zeros((1, max_len), dtype=np.int32), []

    token_ids = [1]  # <START> token
    mapped_tokens = [("<START>", 1)]
    for t in tokens:
        idx = word_to_id.get(t, 2)  # 2 is <UNK>
        token_ids.append(idx)
        mapped_tokens.append((t, idx))

    padded = pad_sequences([token_ids], maxlen=max_len, padding="pre", truncating="pre")
    return padded, mapped_tokens


def train_model_automatically():
    """Runs train_model.py in-process if model does not exist."""
    import train_model
    train_model.train(epochs=3, train_size=10000, test_size=2500, force=False)


# -----------------------------------------------------------------------------
# Check & Ensure Model Exists
# -----------------------------------------------------------------------------
active_model_file = KERAS_MODEL_FILE if os.path.exists(KERAS_MODEL_FILE) else H5_MODEL_FILE

if not os.path.exists(active_model_file) or not os.path.exists(WORD_INDEX_FILE):
    st.info("📦 **Initial Setup Detected:** Trained model files not found locally. Initiating model training now on the IMDb dataset (subset configured for quick student laptop execution)...")
    with st.spinner("Training LSTM model on IMDb dataset... This usually takes ~30-60 seconds on standard laptops..."):
        train_model_automatically()
        st.cache_resource.clear()
        st.cache_data.clear()
    active_model_file = KERAS_MODEL_FILE if os.path.exists(KERAS_MODEL_FILE) else H5_MODEL_FILE
    st.success("✅ Model training completed and files saved successfully!")
    st.rerun()

model = load_sentiment_model(active_model_file)
word_to_id = load_word_index()
metrics = load_metrics()

if model is None or word_to_id is None:
    st.error("⚠️ Failed to load model or vocabulary. Please check the `saved_model/` folder.")
    st.stop()


# -----------------------------------------------------------------------------
# UI Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <span class="hero-badge">PROJECT ID: DL-014 • DEEP LEARNING</span>
    <h1 class="hero-title">🎬 Text Sentiment Analysis using LSTM</h1>
    <p class="hero-subtitle">
        An end-to-end Natural Language Processing (NLP) system trained on real IMDb movie reviews.
        Enter any review or choose a sample to analyze whether the underlying sentiment is <b>Positive</b> or <b>Negative</b>.
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State for User Review Text
# -----------------------------------------------------------------------------
if "review_input" not in st.session_state:
    st.session_state.review_input = ""

if "last_analyzed" not in st.session_state:
    st.session_state.last_analyzed = None

# Sample reviews
SAMPLES = {
    "🌟 Masterpiece (Positive)": (
        "This movie was an absolute masterpiece! The directing was visionary, the acting was powerful and heartfelt, "
        "and the emotional soundtrack stayed with me long after the credits rolled. Highly recommended to everyone!"
    ),
    "🍿 Fun & Heartwarming (Positive)": (
        "A delightful and heartwarming story with great comedic timing and charming chemistry between the lead actors. "
        "Truly enjoyable from start to finish."
    ),
    "💔 Boring & Disappointing (Negative)": (
        "A complete waste of time. The script was hollow, characters made no sense, and the pacing dragged endlessly. "
        "I was so bored I almost walked out halfway through. Very disappointing."
    ),
    "👎 Terrible Acting (Negative)": (
        "Horrible dialogue and atrocious acting throughout. The plot twists were absurd and predictable. "
        "Save your money and skip this disaster."
    )
}


# -----------------------------------------------------------------------------
# Sidebar: Model Architecture & Evaluation Stats
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Model Specifications")
    st.markdown(
        """
        <div style="font-size:0.9rem; color:#475569; margin-bottom:1rem;">
        Real metrics evaluated on unseen IMDb test reviews:
        </div>
        """,
        unsafe_allow_html=True
    )

    if metrics:
        test_acc = metrics.get("test_accuracy", 0.0) * 100
        test_loss = metrics.get("test_loss", 0.0)
        vocab_size = metrics.get("vocabulary_size", 10000)
        seq_len = metrics.get("max_sequence_length", 200)
        epochs = metrics.get("epochs_trained", 3)
        train_samples = metrics.get("train_samples", 10000)

        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-title">Test Accuracy (Real)</div>
            <div class="metric-value">{test_acc:.2f}%</div>
        </div>
        <div class="metric-pill">
            <div class="metric-title">Test Loss</div>
            <div class="metric-value">{test_loss:.4f}</div>
        </div>
        <div class="metric-pill">
            <div class="metric-title">Dataset Subset</div>
            <div class="metric-value">{train_samples:,} reviews</div>
        </div>
        <div class="metric-pill">
            <div class="metric-title">Vocabulary Size</div>
            <div class="metric-value">{vocab_size:,} words</div>
        </div>
        <div class="metric-pill">
            <div class="metric-title">Sequence Length</div>
            <div class="metric-value">{seq_len} tokens</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Metrics not found in saved_model/model_info.json")

    st.markdown("---")
    st.markdown("### 🧠 Network Layers")
    st.markdown(
        """
        - **Embedding Layer**: 64 dimensions
        - **SpatialDropout1D**: 0.2 rate
        - **LSTM Layer**: 64 recurrent units
        - **Dense Hidden**: 32 units (ReLU)
        - **Dropout**: 0.3 rate
        - **Dense Output**: 1 unit (Sigmoid)
        """
    )

    if os.path.exists(HISTORY_PLOT_FILE):
        st.markdown("---")
        st.markdown("### 📈 Training Curves")
        st.image(HISTORY_PLOT_FILE, caption="Accuracy & Loss progression over training epochs", use_container_width=True)


# -----------------------------------------------------------------------------
# Main Application Layout: Input & Analysis
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown("### ✍️ Enter Movie or Product Review")
    st.caption("Type your review below or click one of the quick test samples to load realistic text:")

    # Sample review chips
    st.markdown("**Click a sample to test:**")
    scol1, scol2 = st.columns(2)
    with scol1:
        if st.button("🌟 Positive Sample 1", key="btn_pos1", use_container_width=True):
            st.session_state.review_input = SAMPLES["🌟 Masterpiece (Positive)"]
        if st.button("🍿 Positive Sample 2", key="btn_pos2", use_container_width=True):
            st.session_state.review_input = SAMPLES["🍿 Fun & Heartwarming (Positive)"]
    with scol2:
        if st.button("💔 Negative Sample 1", key="btn_neg1", use_container_width=True):
            st.session_state.review_input = SAMPLES["💔 Boring & Disappointing (Negative)"]
        if st.button("👎 Negative Sample 2", key="btn_neg2", use_container_width=True):
            st.session_state.review_input = SAMPLES["👎 Terrible Acting (Negative)"]

    # Text Area
    user_review = st.text_area(
        label="Review Text",
        value=st.session_state.review_input,
        placeholder="e.g. This movie was surprisingly touching and well crafted...",
        height=160,
        key="review_text_area",
        help="Input a movie, book, or product review to classify."
    )

    # Word and Character counter
    word_count = len(user_review.split()) if user_review.strip() else 0
    char_count = len(user_review)
    st.markdown(f"<div style='font-size:0.85rem; color:#64748b; margin-top:-10px; margin-bottom:15px;'>Words: <b>{word_count}</b> | Characters: <b>{char_count}</b></div>", unsafe_allow_html=True)

    # Action buttons: Analyze and Clear
    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        analyze_clicked = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    with btn_col2:
        clear_clicked = st.button("🔄 Clear / Reset", use_container_width=True)

    if clear_clicked:
        st.session_state.review_input = ""
        st.session_state.last_analyzed = None
        st.rerun()


# -----------------------------------------------------------------------------
# Prediction & Display Logic
# -----------------------------------------------------------------------------
with col_right:
    st.markdown("### 🎯 Prediction Results")

    # If Analyze button is clicked or we have an active input to analyze
    if analyze_clicked:
        clean_text = user_review.strip()
        if not clean_text:
            st.warning("⚠️ **Friendly reminder:** Please enter or select a review before clicking Analyze.")
        else:
            # Preprocess text
            padded_input, mapped_tokens = preprocess_text(clean_text, word_to_id, max_len=MAX_LEN)

            # Predict probability
            prob_raw = float(model.predict(padded_input, verbose=0)[0][0])
            prob_positive = prob_raw
            prob_negative = 1.0 - prob_raw

            is_positive = prob_positive >= 0.5
            sentiment_label = "Positive" if is_positive else "Negative"
            confidence_pct = (prob_positive if is_positive else prob_negative) * 100.0

            st.session_state.last_analyzed = {
                "text": clean_text,
                "sentiment": sentiment_label,
                "confidence": confidence_pct,
                "prob_positive": prob_positive,
                "prob_negative": prob_negative,
                "tokens": mapped_tokens
            }

    # Render results if available
    if st.session_state.last_analyzed:
        res = st.session_state.last_analyzed
        is_pos = res["sentiment"] == "Positive"

        if is_pos:
            st.markdown(f"""
            <div class="sentiment-positive">
                <div style="font-size: 2.8rem;">🎉</div>
                <div class="verdict-title verdict-positive">Positive Sentiment</div>
                <div style="font-size: 1.15rem; color: #065f46; font-weight: 600;">
                    Model Confidence: <b>{res['confidence']:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="sentiment-negative">
                <div style="font-size: 2.8rem;">⚠️</div>
                <div class="verdict-title verdict-negative">Negative Sentiment</div>
                <div style="font-size: 1.15rem; color: #991b1b; font-weight: 600;">
                    Model Confidence: <b>{res['confidence']:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence Bar / Probability Distribution Chart
        st.markdown("##### Probability Breakdown")
        col_p1, col_p2 = st.columns(2)
        col_p1.metric("Positive Probability", f"{res['prob_positive']*100:.2f}%")
        col_p2.metric("Negative Probability", f"{res['prob_negative']*100:.2f}%")

        # Custom Horizontal Probability Bar
        pos_width = max(1, min(99, int(res['prob_positive'] * 100)))
        neg_width = 100 - pos_width
        st.markdown(f"""
        <div style="background-color: #e2e8f0; border-radius: 9999px; height: 26px; overflow: hidden; display: flex; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); margin-top: 5px;">
            <div style="background: linear-gradient(90deg, #10b981, #059669); width: {pos_width}%; color: white; font-size: 0.75rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                {pos_width}% Pos
            </div>
            <div style="background: linear-gradient(90deg, #dc2626, #ef4444); width: {neg_width}%; color: white; font-size: 0.75rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                {neg_width}% Neg
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Token breakdown preview in an expander
        with st.expander("🔍 View Preprocessing & Token Sequence"):
            st.caption(f"First 15 tokens mapped to IMDb vocabulary IDs (Total review words: {len(res['tokens'])-1}):")
            sample_tokens = res["tokens"][:15]
            token_display = ", ".join([f"`{t}` → **{i}**" for t, i in sample_tokens])
            st.markdown(token_display)
            st.markdown(f"*Input tensor was padded to fixed shape: `(1, {MAX_LEN})`*")

    elif not analyze_clicked:
        st.markdown("""
        <div style="background:#ffffff; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 2.5rem 1.5rem; text-align: center; color: #64748b;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">⏱️</div>
            <div style="font-weight: 600; font-size: 1.1rem; color: #1e293b;">Awaiting Review Input</div>
            <p style="font-size: 0.9rem; margin-top: 0.4rem;">
                Select a sample from the left or enter your own review text and click <b>Analyze Sentiment</b> to see prediction.
            </p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# "How It Works" Educational Deep Dive Section
# -----------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("### 📘 How It Works: The LSTM Deep Learning Pipeline")
st.markdown("A step-by-step beginner-friendly explanation of how this BTech DL prototype turns raw English text into sentiment predictions:")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-num">1</span><b>Tokenization</b>
        <p style="font-size:0.88rem; color:#475569; margin-top:6px;">
            Raw text is cleaned, converted to lowercase, and split into individual word tokens.
            Each word is translated into an integer index based on frequency rank in the IMDb vocabulary.
        </p>
    </div>
    <div class="pipeline-step">
        <span class="pipeline-num">2</span><b>Sequence Padding</b>
        <p style="font-size:0.88rem; color:#475569; margin-top:6px;">
            Neural networks require fixed-dimension tensors. Using <code>pad_sequences(maxlen=200)</code>,
            short reviews are pre-padded with 0 (<PAD>) and longer reviews are truncated safely.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-num">3</span><b>Word Embedding</b>
        <p style="font-size:0.88rem; color:#475569; margin-top:6px;">
            Instead of sparse one-hot vectors, an <code>Embedding(10000, 64)</code> layer projects each integer into a 64-dimensional continuous vector space where semantically similar words cluster together.
        </p>
    </div>
    <div class="pipeline-step">
        <span class="pipeline-num">4</span><b>LSTM Memory Cells</b>
        <p style="font-size:0.88rem; color:#475569; margin-top:6px;">
            Standard RNNs suffer from vanishing gradients. An <b>LSTM</b> uses <em>Forget, Input, and Output gates</em> to preserve long-range dependencies and understand negations like <i>"not great"</i>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-num">5</span><b>Dense & Dropout</b>
        <p style="font-size:0.88rem; color:#475569; margin-top:6px;">
            The LSTM output sequence feeds into a 32-unit Dense layer with ReLU activation. Dropout (0.3) randomly deactivates neurons during training to prevent overfitting.
        </p>
    </div>
    <div class="pipeline-step">
        <span class="pipeline-num">6</span><b>Sigmoid Classification</b>
        <p style="font-size:0.88rem; color:#475569; margin-top:6px;">
            The final neuron uses the <code>Sigmoid</code> function: <br>
            <code>S(z) = 1 / (1 + e<sup>-z</sup>)</code> <br>
            It squashes the value between 0.0 (Negative) and 1.0 (Positive) with a 0.5 threshold.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.82rem; margin-top: 1.5rem;">
    Project DL-014 • Text Sentiment Analysis using LSTM • Built with TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
