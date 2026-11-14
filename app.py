"""
Text Sentiment Analysis with Bidirectional LSTM
Project ID: DL-014

app.py
------
Streamlit Web Application for real-time sentiment analysis using
a trained Bidirectional Long Short-Term Memory (BiLSTM) deep learning model.
Supports 3-Class Sentiment: Positive, Negative, Neutral, and Uncertain.
"""

import os
import sys

# Silence verbose TF warnings and configure writable cache
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["KERAS_HOME"] = "/tmp/keras"

import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

from preprocessing import (
    clean_text, load_tokenizer_from_file,
    texts_to_padded_sequences, MAX_SEQUENCE_LENGTH
)

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Text Sentiment Analysis with BiLSTM | DL-014",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(__file__)
SAVED_MODEL_DIR = os.path.join(BASE_DIR, "saved_model")
MODEL_FILE = os.path.join(SAVED_MODEL_DIR, "sentiment_bilstm_model.keras")
TOKENIZER_FILE = os.path.join(SAVED_MODEL_DIR, "tokenizer.json")
METRICS_FILE = os.path.join(SAVED_MODEL_DIR, "model_info.json")
HISTORY_PLOT_FILE = os.path.join(SAVED_MODEL_DIR, "training_history.png")
CONFUSION_MATRIX_FILE = os.path.join(SAVED_MODEL_DIR, "confusion_matrix.png")

LABEL_NAMES = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# -----------------------------------------------------------------------------
# Modern Navy / Blue Theme (Custom CSS)
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
        max-width: 850px;
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
        font-size: 0.78rem;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 1.2rem;
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
    .sentiment-neutral {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 2px solid #0284c7;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(2, 132, 199, 0.2);
    }
    .sentiment-uncertain {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 2px solid #f59e0b;
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 8px 20px -4px rgba(245, 158, 11, 0.2);
    }

    .verdict-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.2rem 0;
    }
    .verdict-positive { color: #047857; }
    .verdict-negative { color: #b91c1c; }
    .verdict-neutral  { color: #0369a1; }
    .verdict-uncertain{ color: #b45309; }

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
# Caching Model & Tokenizer
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    """Loads compiled Keras Bidirectional LSTM model."""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        return load_model(MODEL_FILE)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Loads fitted tokenizer from JSON."""
    if not os.path.exists(TOKENIZER_FILE):
        return None
    try:
        return load_tokenizer_from_file(TOKENIZER_FILE)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None


@st.cache_data(show_spinner=False)
def get_metrics():
    """Loads model evaluation metrics metadata."""
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def trigger_training():
    """Trains the model automatically if missing."""
    import train_model
    train_model.train(epochs=6, force=False)


# Check & ensure model exists
if not os.path.exists(MODEL_FILE) or not os.path.exists(TOKENIZER_FILE):
    st.info("📦 **Initial Setup Detected:** Trained BiLSTM model files not found locally. Initiating training pipeline...")
    with st.spinner("Training Bidirectional LSTM on balanced multi-domain dataset... (~60-90s on laptop)..."):
        trigger_training()
        st.cache_resource.clear()
        st.cache_data.clear()
    st.success("✅ Model training completed and files saved successfully!")
    st.rerun()

model = get_model()
tokenizer = get_tokenizer()
metrics = get_metrics()

if model is None or tokenizer is None:
    st.error("⚠️ Failed to load model or tokenizer. Please run `python3 train_model.py --force` to retrain.")
    st.stop()


# -----------------------------------------------------------------------------
# Header Banner
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <span class="hero-badge">PROJECT ID: DL-014 • DEEP LEARNING (IMPROVED)</span>
    <h1 class="hero-title">🎬 Text Sentiment Analysis with Bidirectional LSTM</h1>
    <p class="hero-subtitle">
        An advanced Natural Language Processing system powered by a <b>Bidirectional LSTM</b> with dynamic padding masking.
        Trained on balanced multi-domain data (conversational emotions, negations, social media, movie reviews, and factual statements).
    </p>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Session State Management
# -----------------------------------------------------------------------------
if "review_input" not in st.session_state:
    st.session_state.review_input = ""

if "last_analyzed" not in st.session_state:
    st.session_state.last_analyzed = None

# Comprehensive Samples
SAMPLES = {
    "💔 Negative: Demotivation": "I am demotivated with her.",
    "👎 Negative: Complaint": "I hate this product and regret my decision to buy it. Total waste of money.",
    "🚫 Negative: Negation": "This is not good. Nobody enjoyed the presentation.",
    "🌟 Positive: Masterpiece": "This movie was an absolute masterpiece! Directing was visionary and acting was superb.",
    "🎉 Positive: Motivation": "I am so happy and motivated with the fantastic results our team achieved.",
    "⚖️ Neutral: Daily Fact": "I went to college today.",
    "📚 Neutral: Informational": "The library opens at 9 AM and the train arrives at platform four at 4 PM.",
    "🤔 Subtle: Double Negation": "I don't think this is bad at all. Actually pretty decent."
}


# -----------------------------------------------------------------------------
# Sidebar: Model Architecture & Evaluation Comparison
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Academic Specifications")

    if metrics:
        test_acc = metrics.get("test_accuracy", 0.0) * 100
        f1_score = metrics.get("f1_score_weighted", 0.0)
        vocab_size = metrics.get("vocab_size", 15000)
        train_samples = metrics.get("train_samples", 13650)
        baseline = metrics.get("baseline_comparison", {})

        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-title">BiLSTM Test Accuracy (Real)</div>
            <div class="metric-value">{test_acc:.2f}%</div>
        </div>
        <div class="metric-pill">
            <div class="metric-title">BiLSTM F1-Score (Weighted)</div>
            <div class="metric-value">{f1_score:.4f}</div>
        </div>
        <div class="metric-pill">
            <div class="metric-title">Training Samples</div>
            <div class="metric-value">{train_samples:,} sentences</div>
        </div>
        <div class="metric-pill">
            <div class="metric-title">Vocabulary Size</div>
            <div class="metric-value">{vocab_size:,} words</div>
        </div>
        """, unsafe_allow_html=True)

        if baseline:
            st.markdown("---")
            st.markdown("### ⚖️ Baseline Comparison")
            base_acc = baseline.get("test_accuracy", 0.0) * 100
            base_f1 = baseline.get("f1_score", 0.0)
            st.caption("TF-IDF + Logistic Regression Baseline:")
            st.markdown(f"""
            - **Baseline Accuracy**: `{base_acc:.2f}%`
            - **Baseline F1-Score**: `{base_f1:.4f}`
            - **BiLSTM Gain**: `+{test_acc - base_acc:.2f}%`
            """)

    st.markdown("---")
    st.markdown("### 🧠 Neural Architecture")
    st.markdown(
        """
        - **Embedding Layer**: 100-dim (`mask_zero=True`)
        - **SpatialDropout1D**: 0.2 rate
        - **Bidirectional LSTM**: 64 units forward + 64 backward
        - **Dense Representation**: 64 units (ReLU)
        - **Batch Normalization**: Stabilizes hidden activations
        - **Dropout**: 0.3 regularization
        - **Softmax Output**: 3 classes (Neg / Neu / Pos)
        """
    )

    if os.path.exists(CONFUSION_MATRIX_FILE):
        st.markdown("---")
        st.markdown("### 🔍 Confusion Matrix")
        st.image(CONFUSION_MATRIX_FILE, caption="Evaluation Confusion Matrix on Test Split", use_container_width=True)

    if os.path.exists(HISTORY_PLOT_FILE):
        st.markdown("---")
        st.markdown("### 📈 Training Curves")
        st.image(HISTORY_PLOT_FILE, caption="Accuracy & Loss progression over epochs", use_container_width=True)


# -----------------------------------------------------------------------------
# Main Application Layout: Input & Analysis
# -----------------------------------------------------------------------------
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    st.markdown("### ✍️ Input Sentence or Review")
    st.caption("Type any sentence below or click a verified test sample to evaluate:")

    # Sample Buttons organized by sentiment
    st.markdown("**Click a sample to test:**")

    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        st.markdown("<small><b>Negative Samples</b></small>", unsafe_allow_html=True)
        if st.button("💔 Demotivated", key="s_neg1", use_container_width=True):
            st.session_state.review_input = SAMPLES["💔 Negative: Demotivation"]
        if st.button("👎 Regret & Waste", key="s_neg2", use_container_width=True):
            st.session_state.review_input = SAMPLES["👎 Negative: Complaint"]
        if st.button("🚫 Negation ('not good')", key="s_neg3", use_container_width=True):
            st.session_state.review_input = SAMPLES["🚫 Negative: Negation"]

    with scol2:
        st.markdown("<small><b>Positive Samples</b></small>", unsafe_allow_html=True)
        if st.button("🌟 Masterpiece", key="s_pos1", use_container_width=True):
            st.session_state.review_input = SAMPLES["🌟 Positive: Masterpiece"]
        if st.button("🎉 Motivated & Happy", key="s_pos2", use_container_width=True):
            st.session_state.review_input = SAMPLES["🎉 Positive: Motivation"]
        if st.button("🤔 Not Bad (Double Neg)", key="s_pos3", use_container_width=True):
            st.session_state.review_input = SAMPLES["🤔 Subtle: Double Negation"]

    with scol3:
        st.markdown("<small><b>Neutral Samples</b></small>", unsafe_allow_html=True)
        if st.button("⚖️ Went to college", key="s_neu1", use_container_width=True):
            st.session_state.review_input = SAMPLES["⚖️ Neutral: Daily Fact"]
        if st.button("📚 Library & Train", key="s_neu2", use_container_width=True):
            st.session_state.review_input = SAMPLES["📚 Neutral: Informational"]

    # Text Area
    user_review = st.text_area(
        label="Input Text",
        value=st.session_state.review_input,
        placeholder="e.g., I am demotivated with her. OR The meeting begins at 10 AM.",
        height=140,
        key="review_text_area",
        help="Input any conversational statement, review, or factual sentence."
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

    if analyze_clicked:
        raw_text = user_review.strip()
        if not raw_text:
            st.warning("⚠️ **Friendly reminder:** Please enter or select a sentence before clicking Analyze.")
        else:
            # Preprocess using identical training pipeline
            cleaned = clean_text(raw_text)
            padded = texts_to_padded_sequences([raw_text], tokenizer, max_len=MAX_SEQUENCE_LENGTH)

            # Predict probabilities
            probs = model.predict(padded, verbose=0)[0]
            neg_p = float(probs[0])
            neu_p = float(probs[1])
            pos_p = float(probs[2])

            pred_idx = int(np.argmax(probs))
            top_prob = float(probs[pred_idx])

            # Uncertainty calibration:
            # If top probability is < 0.45 or difference between top two is < 0.08, flag as Uncertain
            sorted_probs = sorted([neg_p, neu_p, pos_p], reverse=True)
            diff_top_two = sorted_probs[0] - sorted_probs[1]
            is_uncertain = (top_prob < 0.45) or (diff_top_two < 0.08)

            if is_uncertain:
                verdict = "Uncertain"
            else:
                verdict = LABEL_NAMES[pred_idx]

            # Extract token IDs for inspection
            seq_tokens = tokenizer.texts_to_sequences([cleaned])[0]
            token_pairs = []
            words_in_clean = cleaned.split()
            for w in words_in_clean:
                tid = tokenizer.word_index.get(w, tokenizer.word_index.get("<UNK>", 1))
                token_pairs.append((w, tid))

            st.session_state.last_analyzed = {
                "raw_text": raw_text,
                "cleaned_text": cleaned,
                "verdict": verdict,
                "top_prob": top_prob * 100.0,
                "prob_neg": neg_p,
                "prob_neu": neu_p,
                "prob_pos": pos_p,
                "tokens": token_pairs,
                "is_uncertain": is_uncertain
            }

    # Render results
    if st.session_state.last_analyzed:
        res = st.session_state.last_analyzed
        v = res["verdict"]

        if v == "Positive":
            st.markdown(f"""
            <div class="sentiment-positive">
                <div style="font-size: 2.6rem;">🎉</div>
                <div class="verdict-title verdict-positive">Positive Sentiment</div>
                <div style="font-size: 1.1rem; color: #065f46; font-weight: 600;">
                    Model Confidence: <b>{res['top_prob']:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif v == "Negative":
            st.markdown(f"""
            <div class="sentiment-negative">
                <div style="font-size: 2.6rem;">⚠️</div>
                <div class="verdict-title verdict-negative">Negative Sentiment</div>
                <div style="font-size: 1.1rem; color: #991b1b; font-weight: 600;">
                    Model Confidence: <b>{res['top_prob']:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif v == "Neutral":
            st.markdown(f"""
            <div class="sentiment-neutral">
                <div style="font-size: 2.6rem;">⚖️</div>
                <div class="verdict-title verdict-neutral">Neutral Statement</div>
                <div style="font-size: 1.1rem; color: #075985; font-weight: 600;">
                    Model Confidence: <b>{res['top_prob']:.2f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="sentiment-uncertain">
                <div style="font-size: 2.6rem;">🤔</div>
                <div class="verdict-title verdict-uncertain">Uncertain / Ambiguous</div>
                <div style="font-size: 0.95rem; color: #92400e; font-weight: 600; margin-top: 5px;">
                    Model indicates mixed signals or balanced sentiment probabilities.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3-Class Probability Breakdown
        st.markdown("##### 3-Class Probability Distribution")
        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("Negative", f"{res['prob_neg']*100:.2f}%")
        pcol2.metric("Neutral", f"{res['prob_neu']*100:.2f}%")
        pcol3.metric("Positive", f"{res['prob_pos']*100:.2f}%")

        # Custom 3-segment probability meter
        neg_w = max(1, min(98, int(res['prob_neg'] * 100)))
        neu_w = max(1, min(98 - neg_w, int(res['prob_neu'] * 100)))
        pos_w = max(1, 100 - neg_w - neu_w)

        st.markdown(f"""
        <div style="background-color: #e2e8f0; border-radius: 9999px; height: 26px; overflow: hidden; display: flex; box-shadow: inset 0 2px 4px rgba(0,0,0,0.1); margin-top: 5px;">
            <div style="background: linear-gradient(90deg, #dc2626, #ef4444); width: {neg_w}%; color: white; font-size: 0.72rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                {neg_w}% Neg
            </div>
            <div style="background: linear-gradient(90deg, #0284c7, #38bdf8); width: {neu_w}%; color: white; font-size: 0.72rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                {neu_w}% Neu
            </div>
            <div style="background: linear-gradient(90deg, #10b981, #059669); width: {pos_w}%; color: white; font-size: 0.72rem; font-weight: bold; display: flex; align-items: center; justify-content: center;">
                {pos_w}% Pos
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessing & Token Breakdown
        with st.expander("🔍 View Preprocessing Pipeline & Tokens"):
            st.markdown(f"**Cleaned & Contraction Expanded**: `{res['cleaned_text']}`")
            sample_tokens = res["tokens"][:18]
            token_display = ", ".join([f"`{w}` → **{tid}**" for w, tid in sample_tokens])
            st.markdown(f"**Vocabulary Mappings**: {token_display}")
            st.caption(f"Input is zero-padded with dynamic mask (`mask_zero=True`) to fixed tensor shape `(1, {MAX_SEQUENCE_LENGTH})`.")

    elif not analyze_clicked:
        st.markdown("""
        <div style="background:#ffffff; border: 2px dashed #cbd5e1; border-radius: 12px; padding: 2.5rem 1.5rem; text-align: center; color: #64748b;">
            <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">⏱️</div>
            <div style="font-weight: 600; font-size: 1.1rem; color: #1e293b;">Awaiting Input Text</div>
            <p style="font-size: 0.9rem; margin-top: 0.4rem;">
                Enter your own sentence or choose one of the verified sample buttons on the left and click <b>Analyze Sentiment</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# "How It Works" Educational Architecture Deep Dive
# -----------------------------------------------------------------------------
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("### 📘 How It Works: The Improved BiLSTM Deep Learning Pipeline")
st.markdown("An end-to-end breakdown of how this academic prototype resolves negations, short emotional statements, and factual neutrality:")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-num">1</span><b>Contraction & Negation Normalization</b>
        <p style="font-size:0.86rem; color:#475569; margin-top:6px;">
            Contractions like <i>"don't"</i>, <i>"can't"</i>, and <i>"isn't"</i> are expanded to <code>"do not"</code>, <code>"cannot"</code>, and <code>"is not"</code>. Negation words are strictly protected from stop-word filters.
        </p>
    </div>
    <div class="pipeline-step">
        <span class="pipeline-num">2</span><b>Multi-Domain Vocabulary Tokenizer</b>
        <p style="font-size:0.86rem; color:#475569; margin-top:6px;">
            Unlike movie-only vocabularies, our 15,000-word vocabulary spans social media, conversational statements, and emotional expressions like <i>"demotivated"</i>, <i>"disappointed"</i>, and <i>"regret"</i>.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-num">3</span><b>Masked Embedding Layer</b>
        <p style="font-size:0.86rem; color:#475569; margin-top:6px;">
            Using <code>Embedding(mask_zero=True)</code>, the network instructs recurrent layers to ignore padding zeros completely. A 5-word sentence is not drowned out by 115 padding steps.
        </p>
    </div>
    <div class="pipeline-step">
        <span class="pipeline-num">4</span><b>Bidirectional LSTM</b>
        <p style="font-size:0.86rem; color:#475569; margin-top:6px;">
            Standard LSTMs only read forward. A <b>BiLSTM</b> processes text in both forward and backward directions, allowing <i>"happy"</i> to be immediately conditioned by preceding <i>"not"</i> tokens.
        </p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="pipeline-step">
        <span class="pipeline-num">5</span><b>Batch Normalization & Dropout</b>
        <p style="font-size:0.86rem; color:#475569; margin-top:6px;">
            Dense layer features are normalized across training batches, and 30% dropout is applied to prevent the model from memorizing specific training phrases.
        </p>
    </div>
    <div class="pipeline-step">
        <span class="pipeline-num">6</span><b>3-Class Softmax & Calibration</b>
        <p style="font-size:0.86rem; color:#475569; margin-top:6px;">
            Outputs a calibrated probability vector over <b>[Negative, Neutral, Positive]</b>. If probabilities are balanced or confidence is low, it safely flags <b>Uncertain</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.82rem; margin-top: 1.5rem;">
    Project DL-014 • Text Sentiment Analysis with Bidirectional LSTM • Built with TensorFlow & Streamlit
</div>
""", unsafe_allow_html=True)
