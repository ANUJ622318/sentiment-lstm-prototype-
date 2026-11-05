# Text Sentiment Analysis using LSTM (Project DL-014)

A complete, beginner-friendly Deep Learning prototype built for BTech projects. This web application predicts whether a movie or product review conveys a **Positive** or **Negative** sentiment using a **Long Short-Term Memory (LSTM)** neural network trained on the real **IMDb movie reviews dataset**.

![Project DL-014 Banner](https://img.shields.io/badge/Project%20ID-DL--014-blue.svg)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00.svg?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B.svg?logo=streamlit&logoColor=white)

---

## 📌 Project Overview

- **Title**: Text Sentiment Analysis using LSTM
- **Project ID**: DL-014
- **Domain**: Natural Language Processing (NLP) / Deep Learning
- **Dataset**: Built-in IMDb Movie Reviews Dataset (`tensorflow.keras.datasets.imdb`)
- **Web Interface**: Interactive Streamlit Web App styled with a modern Navy/Blue theme
- **Model Architecture**: Embedding Layer $\rightarrow$ Spatial Dropout $\rightarrow$ LSTM $\rightarrow$ Dense (ReLU) $\rightarrow$ Dropout $\rightarrow$ Dense (Sigmoid)

---

## 📁 Repository Structure

```text
sentiment-lstm-prototype/
├── app.py                     # Interactive Streamlit Web Application
├── train_model.py             # Script to train and evaluate the LSTM model on IMDb
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation & execution instructions
├── .gitignore                 # Standard Python & virtual environment ignore list
└── saved_model/               # Saved model artifacts (generated upon training)
    ├── sentiment_lstm_model.h5  # Trained Keras LSTM model weights & architecture
    ├── word_index.json          # Vocabulary mapping dictionary (word -> ID)
    ├── model_info.json          # Real evaluation metrics (test accuracy, loss, params)
    └── training_history.png     # Plot of training & validation loss/accuracy curves
```

---

## 🚀 Step-by-Step macOS Execution Guide

Open **Terminal** on your Mac (`Applications` > `Utilities` > `Terminal`) and navigate to the project directory:

```bash
cd /Users/apple/Downloads/sentiment-lstm-prototype
```

### 1. Create and Activate a Python Virtual Environment

Use Python 3 to initialize an isolated virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

> **Note**: Whenever your terminal prompt shows `(venv)`, the virtual environment is active.

### 2. Install Project Dependencies

Upgrade `pip` and install all necessary libraries:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train the Model

Train the LSTM network on a student-laptop-friendly subset of the IMDb dataset (10,000 training reviews, 2,500 test reviews, 3 epochs):

```bash
python3 train_model.py
```

*This step automatically creates the `saved_model/` folder containing the trained model, word vocabulary dictionary, actual test accuracy metrics, and loss/accuracy plots.*

### 4. Launch the Streamlit Web Application

Start the web application:

```bash
streamlit run app.py
```

Streamlit will launch local server at:
👉 **`http://localhost:8501`**

Open your web browser and navigate to `http://localhost:8501`.

---

## 🧠 Deep Learning Architecture & Pipeline

```text
Raw User Review ("This movie was wonderful...")
       │
       ▼
1. Preprocessing & Tokenization
   • Lowercasing & regex word extraction
   • Word-to-index mapping using IMDb vocabulary (Top 10,000 words)
   • Out-of-vocabulary tokens safely assigned to <UNK> (ID: 2)
       │
       ▼
2. Sequence Padding (`pad_sequences`)
   • Pre-padded / truncated to fixed length: 200 tokens
       │
       ▼
3. Word Embedding Layer (`Embedding(10000, 64)`)
   • Transforms sparse integer IDs into dense 64-dimensional semantic vectors
       │
       ▼
4. Spatial Dropout (`SpatialDropout1D(0.2)`)
   • Drops 1D feature maps to prevent co-adaptation across embedding dimensions
       │
       ▼
5. LSTM Layer (`LSTM(64)`)
   • 64 recurrent memory units with Forget, Input, and Output gates
   • Captures sequential dependencies, contextual semantics, and negations ("not good")
       │
       ▼
6. Dense Hidden Layer (`Dense(32, activation='relu')`)
   • High-level non-linear feature representation
       │
       ▼
7. Regularization (`Dropout(0.3)`)
   • Reduces overfitting on training reviews
       │
       ▼
8. Output Classification (`Dense(1, activation='sigmoid')`)
   • Produces probability $P \in [0.0, 1.0]$:
     - $P \ge 0.5 \implies$ **Positive Sentiment**
     - $P < 0.5 \implies$ **Negative Sentiment**
```

---

## 🎯 Application Features

1. **Modern Navy/Blue Dashboard**: Sleek aesthetic with clean cards, badges, and responsive layout.
2. **Real-Time Sentiment Prediction**: Instant feedback on custom user-typed reviews or selected samples.
3. **Confidence Meter & Distribution**: Visual probability breakdown between Positive and Negative classes.
4. **Clickable Sample Reviews**: Four pre-loaded positive and negative samples for instant 1-click testing.
5. **Preprocessing Inspector**: View how raw English words were transformed into vocabulary integer IDs.
6. **Live Model Specifications**: Shows authentic test accuracy and loss evaluated on unseen test sets (no fabricated numbers).
7. **Input Validation**: Gracefully alerts users if empty text is submitted.
8. **Reset / Clear**: 1-click reset to clear text and predictions.
9. **Educational Guide**: Beginner-friendly breakdown of NLP concepts (Tokenization, Padding, Embeddings, LSTM, Sigmoid).

---

## 🛠️ Frequently Asked Questions (FAQ)

### What if I want to retrain the model from scratch?
Run:
```bash
python3 train_model.py --force
```

### How do I deactivate the virtual environment when done?
Run:
```bash
deactivate
```

---

**Developed for BTech Project DL-014**
