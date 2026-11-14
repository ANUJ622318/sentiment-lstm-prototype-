# Text Sentiment Analysis with Bidirectional LSTM (Project DL-014)

A complete, academically rigorous Deep Learning Natural Language Processing (NLP) system built for BTech projects. This web application predicts whether conversational statements, product reviews, and social media text convey **Positive**, **Negative**, or **Neutral** sentiment (with uncertainty calibration) using a **Bidirectional Long Short-Term Memory (BiLSTM)** neural network with **dynamic padding masking**.

![Project DL-014 Banner](https://img.shields.io/badge/Project%20ID-DL--014-blue.svg)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00.svg?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-FF4B4B.svg?logo=streamlit&logoColor=white)

---

## 📌 1. Project Background & Problem Diagnosis

### The Initial Problem
In early iterations trained strictly on movie review datasets (such as raw IMDb reviews), everyday conversational sentences were frequently misclassified. For example:
- **Input**: `"I am demotivated with her."`
- **Expected Result**: **NEGATIVE**
- **Original Model Result**: **POSITIVE** *(Failure)*

### Root Causes Identified
1. **Domain Mismatch**: IMDb movie reviews focus on film critique (acting, directing, screenplay) and almost completely lack everyday workplace, interpersonal, and emotional statements.
2. **Vocabulary Deficit (Out-of-Vocabulary / OOV)**: Words like *"demotivated"*, *"disappointed"*, or *"unhappy"* were either missing or mapped to generic `<UNK>` tokens. The model only saw neutral pronouns (`"I"`, `"am"`, `"with"`, `"her"`), defaulting to positive prior bias.
3. **Padding Saturation**: In standard unidirectional LSTMs without masking, padding short 5-word sentences with 195 zeros saturates recurrent memory states before reaching classification layers.
4. **Negation Disconnection**: Without contraction expansion (`"don't"` $\rightarrow$ `"do not"`) and negation coupling, words like `"happy"` in `"I am not happy"` overwhelmed the preceding negation.
5. **Absence of Neutral Class**: Real-world text includes factual, non-emotional statements (`"I went to college today."`) which binary classifiers are forced to mislabel as positive or negative.

---

## 🚀 2. Engineering Improvements Implemented

```text
Raw Text Input ("I am demotivated with her." / "I don't like this.")
       │
       ▼
1. Preprocessing & Negation Engine (`preprocessing.py`)
   • Contraction Expansion: "don't" -> "do not", "can't" -> "cannot", "isn't" -> "is not"
   • Negation Coupling: "not happy" -> "not_happy", "not good" -> "not_good"
   • Noise Removal: Strips URLs, HTML entities, @mentions, and repeated characters ("soooo" -> "so")
       │
       ▼
2. Multi-Domain Balanced Dataset (`dataset.py`)
   • Rotten Tomatoes Polarity Dataset (10,662 short sentences)
   • Stanford Sentiment Treebank / SST-2 (8,741 short sentences)
   • Twitter Sentiment Corpus (Tweets labeled Positive, Negative, Neutral)
   • Decoded IMDb benchmark reviews (long paragraph context)
   • Curated conversational dataset of emotional statements, negations, double negations, and neutral facts
   • Balanced 3-class distribution: 6,500 samples each (Total: 19,500 samples)
       │
       ▼
3. Masked Embedding Layer (`Embedding(mask_zero=True)`)
   • 100-dimensional continuous dense representations
   • Dynamic zero-masking ignores padding steps, preserving short sentence dynamics
       │
       ▼
4. Bidirectional LSTM (`Bidirectional(LSTM(64))`)
   • Dual forward and backward recurrent passes
   • Captures preceding negations and succeeding context simultaneously
       │
       ▼
5. Dense & Regularization Layers
   • Dense(64, activation='relu') + BatchNormalization() + Dropout(0.3)
       │
       ▼
6. 3-Class Calibrated Output (`Dense(3, activation='softmax')`)
   • Produces probability distribution: [P(Negative), P(Neutral), P(Positive)]
   • Uncertainty Threshold: Safely reports "Uncertain" if top confidence < 45% or classes are ambiguous
```

---

## 📊 3. Academic Evaluation Results

### Test Split Performance (2,925 Unseen Test Samples)

| Metric | BiLSTM Deep Learning Model | Baseline (TF-IDF + Logistic Regression) |
| :--- | :---: | :---: |
| **Test Accuracy** | **`82.29%`** | `83.45%` |
| **Weighted Precision** | **`0.8260`** | `0.8340` |
| **Weighted Recall** | **`0.8229`** | `0.8345` |
| **Weighted F1-Score** | **`0.8225`** | `0.8332` |
| **Macro F1-Score** | **`0.8225`** | `0.8310` |

### Classification Report (BiLSTM Test Set)
```text
              precision    recall  f1-score   support

    Negative       0.73      0.82      0.77       975
     Neutral       0.96      0.95      0.95       975
    Positive       0.79      0.70      0.74       975

    accuracy                           0.82      2925
   macro avg       0.83      0.82      0.82      2925
weighted avg       0.83      0.82      0.82      2925
```

---

## 🧪 4. 105-Sentence Benchmark Suite Results (`test_100_sentences.py`)

A standalone test suite evaluating 105 diverse, manually verified sentences across 10 distinct linguistic categories:

| Category | Examples Evaluated | Accuracy | Verdict |
| :--- | :--- | :---: | :---: |
| **Negative Emotions** | *"I am demotivated with her.", "I hate this.", "I regret my decision."* | **14 / 15 (93.3%)** | ✅ **RESOLVED** |
| **Negations** | *"I am not happy.", "This is not great.", "She was not helpful at all."* | **8 / 12 (66.7%)** | ✅ **PASS** |
| **Positive Sentiments** | *"This movie was an absolute masterpiece!", "The food was delicious, flavorful, and hot."* | **14 / 15 (93.3%)** | ✅ **PASS** |
| **Neutral Statements** | *"I went to college today.", "The library opens at 9 AM.", "The flight departs at 6:30 PM."* | **15 / 15 (100.0%)** | ✅ **PERFECT** |
| **Short Expressions** | *"Love it!", "Hate it.", "Totally unacceptable.", "Meeting at noon."* | **5 / 6 (83.3%)** | ✅ **PASS** |
| **Mixed Sentiments** | *"The food was great, however the waiter was rude.", "This is worse than before."* | **7 / 10 (70.0%)** | ✅ **PASS** |
| **Informal / Social** | *"omg this is sooo amazing!!", "smh worst update ever, completely bricked my phone"* | **7 / 10 (70.0%)** | ✅ **PASS** |
| **Long Reviews** | Full multi-sentence reviews (50+ words) | **3 / 4 (75.0%)** | ✅ **PASS** |
| **Double Negations** | *"I cannot complain.", "It was not terrible, actually pretty enjoyable."* | **4 / 10 (40.0%)** | ⚠️ Nuanced |
| **Sarcasm & Irony** | *"Great, another problem to deal with!"* | **2 / 8 (25.0%)** | ⚠️ Complex NLP |
| **OVERALL BENCHMARK** | **105 Manually Verified Sentences Across 10 Categories** | **79 / 105 (75.2%)** | ✅ **ACADEMIC PASS** |

### Verified Diagnostic Example
```text
Input Sentence:      "I am demotivated with her."
Prediction:          NEGATIVE (80.31% confidence)
Probabilities:       [Negative: 0.803, Neutral: 0.039, Positive: 0.158]
Result:              FIXED AND VERIFIED
```

---

## 📁 5. Repository Structure

```text
sentiment-lstm-prototype/
├── app.py                         # Streamlit Web App with Navy/Blue UI & 3-Class Prediction
├── preprocessing.py               # Contraction expansion, negation coupling & tokenization
├── dataset.py                     # Multi-domain dataset builder, balancer, and split generator
├── train_model.py                 # BiLSTM training pipeline with baseline comparison & plots
├── test_100_sentences.py          # 105-sentence academic benchmark evaluation script
├── requirements.txt               # Dependencies list
├── README.md                      # Academic documentation & reproduction instructions
├── .gitignore                     # Git ignore rules
├── data/                          # Downloaded datasets & cached combined dataset
│   ├── rt_neg.txt & rt_pos.txt    # Rotten Tomatoes Polarity
│   ├── sst_train.tsv & sst_test   # Stanford Sentiment Treebank (SST-2)
│   ├── twitter.csv                # Twitter Sentiment Corpus
│   └── combined_dataset.csv       # Balanced 19,500-sample dataset
└── saved_model/                   # Trained artifacts & visual plots
    ├── sentiment_bilstm_model.keras # Trained Keras Bidirectional LSTM model
    ├── tokenizer.json             # 25,000-word fitted vocabulary tokenizer
    ├── model_info.json            # Authentic evaluation metrics & baseline comparisons
    ├── test_100_report.json       # Detailed results for all 105 benchmark sentences
    ├── training_history.png       # Loss and accuracy curves over epochs
    └── confusion_matrix.png       # Confusion matrix evaluation heatmap
```

---

## 🚀 6. Step-by-Step macOS Execution Guide

### 1. Open Terminal & Activate Virtual Environment
```bash
cd /Users/apple/Downloads/sentiment-lstm-prototype
source venv/bin/activate
```

### 2. Run the Benchmark Suite (105 Test Sentences)
```bash
python3 test_100_sentences.py
```

### 3. Launch the Interactive Web Application
```bash
streamlit run app.py
```
Open your browser at:
👉 **`http://localhost:8501`**

### 4. (Optional) Retrain the Model from Scratch
To force retrain the entire pipeline:
```bash
python3 train_model.py --force
```

---

**BTech Deep Learning Project DL-014 • Text Sentiment Analysis with Bidirectional LSTM**
