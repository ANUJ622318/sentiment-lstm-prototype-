"""
Text Sentiment Analysis with Bidirectional LSTM (Project DL-014)

preprocessing.py
----------------
Robust text preprocessing module designed to preserve critical sentiment cues:
- Expands contractions (don't -> do not, can't -> cannot, etc.)
- Couples negation words with sentiment words (not happy -> not_happy, not good -> not_good)
- Normalizes repeated characters (soooo -> so, goooood -> good)
- Preserves negation tokens and custom compounds
- Cleans informal social media noise (URLs, mentions, HTML entities)
- Consistent tokenization and padding for both training and inference.
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
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Comprehensive contractions mapping
CONTRACTIONS = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "might've": "might have",
    "must've": "must have",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "would've": "would have",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

# Regex pattern for contractions
CONTRACTIONS_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in CONTRACTIONS.keys()) + r")\b", flags=re.IGNORECASE)

# Negation words to couple with following adjective or verb
NEGATION_WORDS = {"not", "no", "never", "cannot", "neither", "nor", "hardly", "barely", "scarcely"}

# Standard sequence length & vocabulary
MAX_SEQUENCE_LENGTH = 120
MAX_VOCAB_SIZE = 25000

# Custom filters that do NOT remove underscores (so compound tokens like not_happy remain intact)
CUSTOM_TOKENIZER_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n'


def expand_contractions(text: str) -> str:
    """Replaces contractions with their expanded forms while preserving case style."""
    def replace(match):
        match_str = match.group(0).lower()
        expanded = CONTRACTIONS.get(match_str, match_str)
        return expanded
    return CONTRACTIONS_RE.sub(replace, text)


def normalize_repeated_characters(text: str) -> str:
    """Normalizes characters repeated 3 or more times (e.g., 'sooooo' -> 'so', 'goooood' -> 'good')."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def couple_negations(text: str) -> str:
    """
    Couples negation words with their immediate successor word:
    'not happy' -> 'not_happy'
    'not good' -> 'not_good'
    'not bad' -> 'not_bad'
    'never again' -> 'never_again'
    'cannot recommend' -> 'cannot_recommend'
    This creates dedicated vocabulary tokens for negated contexts,
    preventing positive words (like 'happy') from overriding preceding negations.
    """
    words = text.split()
    new_words = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in NEGATION_WORDS and i + 1 < len(words):
            next_w = words[i + 1]
            new_words.append(f"{w}_{next_w}")
            i += 2
        else:
            new_words.append(w)
            i += 1
    return " ".join(new_words)


def clean_text(text: str) -> str:
    """
    Cleans raw review/sentence text:
    1. Expands contractions to make negations explicit (e.g., "don't" -> "do not").
    2. Strips URLs and HTML entities.
    3. Removes social media handles (@username).
    4. Normalizes repeated characters.
    5. Strips non-alphanumeric noise while keeping spaces and underscores.
    6. Couples negation words to their targets (e.g. "not happy" -> "not_happy").
    7. Converts to lowercase and normalizes whitespace.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if not text:
        return ""

    # Replace common HTML entities
    text = text.replace("&amp;", " and ").replace("&lt;", " < ").replace("&gt;", " > ")
    text = text.replace("<br />", " ").replace("<br>", " ")

    # Expand contractions (e.g., "don't" -> "do not")
    text = expand_contractions(text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove @mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize character repetitions (e.g. "sooooo" -> "soo")
    text = normalize_repeated_characters(text)

    # Remove punctuation except spaces, alphanumerics, and apostrophes
    text = re.sub(r"[^a-zA-Z0-9\s'_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    # Couple negation tokens
    text = couple_negations(text)

    return text


def create_tokenizer(texts, vocab_size=MAX_VOCAB_SIZE, oov_token="<UNK>"):
    """
    Fits and returns a Keras Tokenizer on clean texts with custom filters
    that preserve compound negation tokens (e.g., 'not_happy', 'not_bad').
    Index 0 is reserved for padding (enables mask_zero=True in Embedding layer).
    """
    tokenizer = Tokenizer(
        num_words=vocab_size,
        filters=CUSTOM_TOKENIZER_FILTERS,
        oov_token=oov_token,
        lower=True
    )
    tokenizer.fit_on_texts(texts)
    return tokenizer


def save_tokenizer(tokenizer, file_path):
    """Saves tokenizer configuration and word index to JSON file."""
    tokenizer_json = tokenizer.to_json()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(tokenizer_json)


def load_tokenizer_from_file(file_path):
    """Loads tokenizer from saved JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    return tokenizer_from_json(tokenizer_json)


def texts_to_padded_sequences(texts, tokenizer, max_len=MAX_SEQUENCE_LENGTH):
    """
    Transforms text list into padded integer sequence tensors.
    Uses 'post' padding with mask_zero=True in Embedding for optimal LSTM dynamics.
    """
    cleaned = [clean_text(t) for t in texts]
    sequences = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return padded
