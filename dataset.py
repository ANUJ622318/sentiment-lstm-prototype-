"""
Text Sentiment Analysis with Bidirectional LSTM (Project DL-014)

dataset.py
----------
Multi-domain dataset loader and aggregator:
- Combines Rotten Tomatoes (short sentences), SST-2 (Stanford Sentiment Treebank),
  Twitter Sentiment Corpus, decoded IMDb reviews, and curated everyday conversational sentences.
- Supports 3 sentiment classes:
    0: Negative
    1: Neutral
    2: Positive
- Prepares balanced train / validation / test splits.
"""

import os
import sys

# Configure environment before imports to ensure sandbox compatibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["KERAS_HOME"] = "/tmp/keras"

import csv
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import imdb
from preprocessing import clean_text

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RT_NEG_FILE = os.path.join(DATA_DIR, "rt_neg.txt")
RT_POS_FILE = os.path.join(DATA_DIR, "rt_pos.txt")
SST_TRAIN_FILE = os.path.join(DATA_DIR, "sst_train.tsv")
SST_TEST_FILE = os.path.join(DATA_DIR, "sst_test.tsv")
TWITTER_FILE = os.path.join(DATA_DIR, "twitter.csv")
CACHE_DATASET_FILE = os.path.join(DATA_DIR, "combined_dataset.csv")

# Label mappings
LABEL_NEGATIVE = 0
LABEL_NEUTRAL = 1
LABEL_POSITIVE = 2

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}


def load_curated_conversational_data():
    """
    Returns curated everyday conversational, workplace, emotional,
    negation, double-negation, and neutral factual sentences.
    Specifically guarantees rich representation of expressions like:
    'I am demotivated with her', 'I don't like this', 'I regret my decision', etc.
    """
    texts = []
    labels = []

    # 1. Direct Negative Emotional Statements (Class 0)
    neg_statements = [
        "I am demotivated with her.",
        "I feel so demotivated and exhausted today.",
        "This completely demotivates our whole team.",
        "I am unmotivated to continue working on this project.",
        "I am deeply disappointed with her behavior.",
        "I am disappointed with the final exam results.",
        "I hate this product so much.",
        "I hate the way she treated me yesterday.",
        "I do not like her at all.",
        "I don't like this movie or its characters.",
        "This is not good news.",
        "This is not acceptable under any circumstance.",
        "I regret my decision to buy this laptop.",
        "I regret trusting him with the task.",
        "I feel terrible about the entire situation.",
        "I am unhappy with the customer service.",
        "I expected much better quality than this garbage.",
        "Nothing is going right in my life right now.",
        "I am not happy with how things turned out.",
        "I cannot stand his rude behavior.",
        "This is the worst experience I have ever had.",
        "A total disaster from beginning to end.",
        "The food was horrible and completely cold.",
        "Everything went completely wrong today.",
        "I am frustrated and tired of these constant bugs.",
        "I feel hopeless about the future.",
        "The software crashed and deleted all my work.",
        "Do not waste your money on this scam.",
        "I was completely let down by the service.",
        "This is utterly useless and horribly designed.",
        "The battery life is pathetic and drains in an hour.",
        "I am heartbroken and devastated by what happened.",
        "The staff was unhelpful, dismissive, and obnoxious.",
        "I felt humiliated and depressed after the meeting.",
        "Horrible acting, clunky dialogue, and terrible pacing.",
        "This ruined my entire weekend.",
        "I cannot recommend this to anyone in good conscience.",
        "Terrible quality, broke on the very first day.",
        "I am sick and tired of these false promises.",
        "A complete waste of time, effort, and money.",
        "I am dissatisfied with the performance.",
        "It made me extremely uncomfortable and upset.",
        "The delivery was two weeks late and damaged.",
        "I despise this kind of deceptive marketing.",
        "Nothing worked as advertised.",
        "I feel so lonely, sad, and discouraged.",
        "This is painfully boring and slow.",
        "I hate waiting in this endless line.",
        "The manager was extremely rude and uncooperative.",
        "Never buying from this company again.",
        "I am not happy.",
        "I am not happy with this.",
        "I am not happy at all.",
        "This is not great.",
        "It is not great.",
        "It is not good.",
        "She was not helpful at all.",
        "The staff was not helpful.",
        "Customer support was not helpful.",
        "I will never come back here again.",
        "Never coming back.",
        "Nobody enjoyed the presentation.",
        "Nobody liked it.",
        "No one enjoyed the movie.",
        "I regret my decision.",
        "I regret my choice.",
        "I deeply regret buying this.",
        "Totally unacceptable.",
        "Completely unacceptable behavior."
    ]

    # Expand negative statements with contextual templates
    negative_templates = [
        "I am very unhappy with {target}.",
        "I am extremely demotivated with {target}.",
        "I feel terrible about {target}.",
        "I completely hate {target}.",
        "I do not like {target}.",
        "I expected better from {target}.",
        "I am deeply disappointed in {target}.",
        "I regret dealing with {target}.",
        "{target} is truly awful and disappointing.",
        "{target} made me feel depressed and angry.",
        "{target} was an absolute nightmare to handle.",
        "There is nothing good to say about {target}."
    ]
    targets_neg = [
        "her attitude", "his work", "the service", "the system", "the results",
        "the management", "this device", "the presentation", "the team lead",
        "their support", "the application", "the hotel room", "the professor",
        "the assignment", "the food", "this update", "the performance"
    ]
    for tmpl in negative_templates:
        for t in targets_neg:
            neg_statements.append(tmpl.format(target=t))

    for s in neg_statements:
        texts.append(s)
        labels.append(LABEL_NEGATIVE)

    # 2. Direct Positive Emotional & Evaluative Statements (Class 2)
    pos_statements = [
        "I am so happy and motivated today.",
        "I feel deeply inspired by her dedication.",
        "This motivates me to work even harder.",
        "I am delighted with the outstanding results.",
        "I love this product and use it every single day.",
        "I really like her warm and helpful attitude.",
        "This is exceptionally good work.",
        "I am proud of my decision to join this team.",
        "I feel wonderful and energetic this morning.",
        "The customer support was quick, courteous, and polite.",
        "This exceeded all my highest expectations.",
        "Everything went smoothly and perfectly.",
        "I am extremely happy with how this project turned out.",
        "The food was delicious, fresh, and hot.",
        "Best purchase I have made in a very long time.",
        "A truly heartwarming and beautiful story.",
        "Brilliant acting and masterclass directing throughout.",
        "Highly recommended to all my friends and family.",
        "The staff was incredibly welcoming and attentive.",
        "Wonderful experience, will definitely return again soon.",
        "The battery life is phenomenal and lasts two whole days.",
        "Outstanding performance, lightning fast and responsive.",
        "I feel so optimistic, confident, and grateful.",
        "This saved me hours of tedious work.",
        "Remarkable craftsmanship and top notch quality.",
        "Delivered super fast and packed with great care.",
        "I love the clean interface and sleek design.",
        "Such a joyful and uplifting film to watch.",
        "Every single detail was executed with perfection.",
        "I had a fantastic time with my family here.",
        "The food was delicious, flavorful, and hot.",
        "The food was delicious, fresh, and hot.",
        "I feel deeply grateful for all your support.",
        "Fast shipping and exceptional customer service.",
        "Highly recommended to anyone looking for quality.",
        "The battery life exceeded all my expectations.",
        "I love this product and use it daily.",
        "This app is super lit and dope.",
        "Huge shoutout to the devs for fixing this fast!!",
        "Yay, got my dream internship offer today!!",
        "Love it!",
        "Absolutely fantastic.",
        "From the opening scene to the emotional final credits, this film captured my heart."
    ]

    positive_templates = [
        "I am very happy with {target}.",
        "I feel truly inspired by {target}.",
        "I really love {target}.",
        "I thoroughly enjoyed {target}.",
        "I am so impressed by {target}.",
        "{target} is fantastic and exceeded my expectations.",
        "{target} was wonderfully executed and clean.",
        "{target} made me smile and feel energized.",
        "You can always rely on {target}.",
        "Full marks to {target} for an incredible job.",
        "Fast and reliable {target}.",
        "I am deeply grateful for {target}.",
        "Highly recommended {target}."
    ]
    targets_pos = [
        "her dedication", "his creativity", "the service", "the update",
        "the results", "the new interface", "this phone", "the team",
        "their response", "the meal", "the presentation", "the teacher",
        "the customer service", "the fast shipping", "the delicious food",
        "their helpful support", "the exceptional quality"
    ]
    for tmpl in positive_templates:
        for t in targets_pos:
            pos_statements.append(tmpl.format(target=t))

    for s in pos_statements:
        texts.append(s)
        labels.append(LABEL_POSITIVE)

    # 3. Double Negations & Subtle Sentiments (Class 2 / Positive or Soft Positive)
    double_negations = [
        ("I do not think this is bad.", LABEL_POSITIVE),
        ("This is not bad at all.", LABEL_POSITIVE),
        ("It is not bad.", LABEL_POSITIVE),
        ("Not bad.", LABEL_POSITIVE),
        ("It is hard to dislike this movie.", LABEL_POSITIVE),
        ("I cannot complain about the service.", LABEL_POSITIVE),
        ("I cannot complain.", LABEL_POSITIVE),
        ("Nothing short of wonderful.", LABEL_POSITIVE),
        ("Not disappointing in any way.", LABEL_POSITIVE),
        ("I definitely do not regret my choice.", LABEL_POSITIVE),
        ("It never fails to impress me.", LABEL_POSITIVE),
        ("You will not be disappointed.", LABEL_POSITIVE),
        ("Not a bad option for students.", LABEL_POSITIVE),
        ("It was not horrible, in fact it was pretty decent.", LABEL_POSITIVE),
        ("It was not terrible, actually pretty enjoyable.", LABEL_POSITIVE),
        ("She is not unpleasant to be around.", LABEL_POSITIVE),
        ("I do not hate her, she is actually quite pleasant.", LABEL_POSITIVE)
    ]
    # Replicate subtle double-negation patterns to reinforce learned representation
    for s, l in double_negations * 15:
        texts.append(s)
        labels.append(l)

    # 4. Neutral Factual & Informational Sentences (Class 1)
    neutral_statements = [
        "I went to college today.",
        "The lecture begins promptly at 10 AM.",
        "The library opens at 9 AM and closes at 8 PM.",
        "The train arrives at platform four at 4 PM.",
        "The flight departs at 6:30 PM this evening.",
        "Today is Tuesday and tomorrow is Wednesday.",
        "It is 24 degrees Celsius in the city right now.",
        "The package was delivered by courier yesterday.",
        "The textbook has three hundred and fifty pages.",
        "He submitted the laboratory assignment on time.",
        "She is reading a technical document in the hall.",
        "The retail store is located on the second floor.",
        "We discussed the project outline during the meeting.",
        "There are four wooden chairs around the conference table.",
        "The smartphone comes packaged with a USB charger.",
        "The annual financial report contains twelve distinct sections.",
        "I drank a glass of water before leaving the house.",
        "They walked two blocks to reach the subway station.",
        "The operating system installed on this machine is macOS.",
        "He wrote the contact address down on a piece of paper.",
        "The local bus stop is located five minutes away from here.",
        "The baking recipe calls for two eggs and one cup of flour.",
        "She works in the accounting department from nine to five.",
        "The documentation was exported in standard PDF format.",
        "The digital camera has a twelve megapixel image sensor.",
        "We attended the scheduled physics lecture on Monday morning.",
        "The smartphone battery capacity is four thousand milliampere hours.",
        "The city historical museum is closed on national holidays.",
        "He resides in a residential apartment in Bangalore.",
        "The suspension bridge connects the two northern islands.",
        "I received an official email confirmation from the university.",
        "The family sedan has four doors and five seats.",
        "The annual academic conference is held every November.",
        "The campus cafeteria serves vegetarian meals every afternoon.",
        "The software prototype was developed using Python and Keras.",
        "The feature film has a runtime of two hours and ten minutes.",
        "The live music concert will commence at seven in the evening.",
        "They traveled to the neighboring town by passenger car.",
        "The office laser printer is connected via local Wi-Fi.",
        "The cardboard package includes an instruction manual."
    ]

    # Programmatic neutral templates to ensure robust neutral representation
    neutral_templates = [
        "The {item} was purchased on {day}.",
        "The {item} is scheduled for {day}.",
        "The meeting regarding {topic} took place yesterday.",
        "He placed the {item} on the table.",
        "She opened the {item} at {time}.",
        "The train from {city} arrives at {time}.",
        "We discussed {topic} during the lecture.",
        "The document about {topic} has {pages} pages.",
        "There are {count} items stored in the {place}.",
        "The distance between {city} and the station is {dist} kilometers."
    ]
    items = ["notebook", "laptop", "textbook", "package", "ticket", "report", "folder", "charger", "projector"]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    topics = ["neural networks", "linear algebra", "database schema", "operating systems", "microprocessors"]
    cities = ["Delhi", "Mumbai", "Chennai", "Kolkata", "Hyderabad", "Pune"]
    times = ["9 AM", "11:30 AM", "2 PM", "4:15 PM", "6 PM"]

    for _ in range(120):
        t_choice = random.choice(neutral_templates)
        s = t_choice.format(
            item=random.choice(items),
            day=random.choice(days),
            topic=random.choice(topics),
            city=random.choice(cities),
            time=random.choice(times),
            pages=random.randint(10, 500),
            count=random.randint(2, 20),
            place="shelf",
            dist=random.randint(5, 50)
        )
        neutral_statements.append(s)

    for s in neutral_statements:
        texts.append(s)
        labels.append(LABEL_NEUTRAL)

    return texts, labels


def load_rotten_tomatoes_data():
    """Loads Rotten Tomatoes Polarity dataset (short sentences)."""
    texts = []
    labels = []

    if os.path.exists(RT_NEG_FILE):
        with open(RT_NEG_FILE, "r", encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(LABEL_NEGATIVE)

    if os.path.exists(RT_POS_FILE):
        with open(RT_POS_FILE, "r", encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(LABEL_POSITIVE)

    print(f"  [Rotten Tomatoes] Loaded {len(texts):,} short sentence samples.")
    return texts, labels


def load_sst_data():
    """Loads Stanford Sentiment Treebank (SST-2) short sentences."""
    texts = []
    labels = []

    for path in [SST_TRAIN_FILE, SST_TEST_FILE]:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        sentence = parts[0].strip()
                        lbl_str = parts[1].strip()
                        if lbl_str in ["0", "1"] and sentence:
                            # 0: negative, 1: positive
                            mapped = LABEL_NEGATIVE if lbl_str == "0" else LABEL_POSITIVE
                            texts.append(sentence)
                            labels.append(mapped)

    print(f"  [SST-2 Dataset] Loaded {len(texts):,} short sentence samples.")
    return texts, labels


def load_twitter_data():
    """Loads Twitter Sentiment Corpus (includes Positive, Negative, Neutral)."""
    texts = []
    labels = []

    if os.path.exists(TWITTER_FILE):
        with open(TWITTER_FILE, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 5:
                    sent = row[1].strip().lower()
                    tweet = row[4].strip()
                    if not tweet:
                        continue
                    if sent == "negative":
                        texts.append(tweet)
                        labels.append(LABEL_NEGATIVE)
                    elif sent == "positive":
                        texts.append(tweet)
                        labels.append(LABEL_POSITIVE)
                    elif sent == "neutral":
                        texts.append(tweet)
                        labels.append(LABEL_NEUTRAL)

    print(f"  [Twitter Corpus] Loaded {len(texts):,} tweet samples.")
    return texts, labels


def load_imdb_decoded_subset(num_samples=4000):
    """
    Decodes a balanced subset of IMDb reviews back to raw English text
    to provide long review context.
    """
    print(f"  [IMDb Dataset] Decoding {num_samples:,} movie reviews...")
    (x_train, y_train), _ = imdb.load_data(num_words=15000, index_from=3)
    raw_word_index = imdb.get_word_index()
    index_word = {i + 3: w for w, i in raw_word_index.items()}
    index_word[0] = ""
    index_word[1] = ""
    index_word[2] = ""

    texts = []
    labels = []
    pos_count = 0
    neg_count = 0
    target_per_class = num_samples // 2

    for seq, y in zip(x_train, y_train):
        if y == 1 and pos_count < target_per_class:
            words = [index_word.get(idx, "") for idx in seq if idx > 2]
            text = " ".join(words).strip()
            if len(text.split()) >= 10:
                texts.append(text)
                labels.append(LABEL_POSITIVE)
                pos_count += 1
        elif y == 0 and neg_count < target_per_class:
            words = [index_word.get(idx, "") for idx in seq if idx > 2]
            text = " ".join(words).strip()
            if len(text.split()) >= 10:
                texts.append(text)
                labels.append(LABEL_NEGATIVE)
                neg_count += 1

        if pos_count >= target_per_class and neg_count >= target_per_class:
            break

    print(f"  [IMDb Dataset] Decoded {len(texts):,} reviews.")
    return texts, labels


def build_unified_dataset(cache=True, samples_per_class=6500):
    """
    Builds a balanced, multi-domain dataset covering:
    - Short sentences
    - Everyday emotional expressions
    - Negations and double negations
    - Tweets and social media language
    - Movie reviews
    - Neutral factual statements

    Balances dataset across:
    0: Negative
    1: Neutral
    2: Positive
    """
    if cache and os.path.exists(CACHE_DATASET_FILE):
        print(f"Loading cached dataset from: {CACHE_DATASET_FILE}")
        df = pd.read_csv(CACHE_DATASET_FILE)
        return df["text"].tolist(), df["label"].tolist()

    print("Building unified multi-domain sentiment dataset...")
    all_texts = []
    all_labels = []

    # 1. Curated conversational dataset
    c_texts, c_labels = load_curated_conversational_data()
    all_texts.extend(c_texts)
    all_labels.extend(c_labels)

    # 2. Rotten Tomatoes (short sentences)
    rt_texts, rt_labels = load_rotten_tomatoes_data()
    all_texts.extend(rt_texts)
    all_labels.extend(rt_labels)

    # 3. SST-2 (short sentences)
    sst_texts, sst_labels = load_sst_data()
    all_texts.extend(sst_texts)
    all_labels.extend(sst_labels)

    # 4. Twitter Corpus (positive, negative, neutral)
    tw_texts, tw_labels = load_twitter_data()
    all_texts.extend(tw_texts)
    all_labels.extend(tw_labels)

    # 5. IMDb decoded reviews
    imdb_texts, imdb_labels = load_imdb_decoded_subset(num_samples=4000)
    all_texts.extend(imdb_texts)
    all_labels.extend(imdb_labels)

    # Clean and organize into classes
    class_buckets = {0: [], 1: [], 2: []}
    seen_texts = set()

    for text, lbl in zip(all_texts, all_labels):
        cleaned = clean_text(text)
        if len(cleaned.split()) >= 2 and cleaned not in seen_texts:
            seen_texts.add(cleaned)
            class_buckets[lbl].append(cleaned)

    # To ensure balance, balance classes
    # If neutral bucket is smaller, oversample or replicate curated neutrals
    min_count = min(len(class_buckets[0]), len(class_buckets[2]))
    target_count = min(samples_per_class, min_count)

    print(f"Raw class counts before balancing:")
    print(f"  Negative (0): {len(class_buckets[0]):,}")
    print(f"  Neutral  (1): {len(class_buckets[1]):,}")
    print(f"  Positive (2): {len(class_buckets[2]):,}")

    # Balance positive and negative
    balanced_texts = []
    balanced_labels = []

    random.seed(42)
    random.shuffle(class_buckets[0])
    random.shuffle(class_buckets[1])
    random.shuffle(class_buckets[2])

    neg_selected = class_buckets[0][:target_count]
    pos_selected = class_buckets[2][:target_count]

    # Neutral upsampling to maintain representative 3-class distribution
    neutrals = class_buckets[1]
    if len(neutrals) < target_count:
        # Repeat with slight perturbations or cyclical sampling to reach target_count
        repeat_factor = (target_count // len(neutrals)) + 1
        expanded_neutrals = (neutrals * repeat_factor)[:target_count]
        neu_selected = expanded_neutrals
    else:
        neu_selected = neutrals[:target_count]

    for t in neg_selected:
        balanced_texts.append(t)
        balanced_labels.append(LABEL_NEGATIVE)

    for t in neu_selected:
        balanced_texts.append(t)
        balanced_labels.append(LABEL_NEUTRAL)

    for t in pos_selected:
        balanced_texts.append(t)
        balanced_labels.append(LABEL_POSITIVE)

    # Shuffle combined dataset
    combined = list(zip(balanced_texts, balanced_labels))
    random.shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)
    balanced_texts = list(balanced_texts)
    balanced_labels = list(balanced_labels)

    print(f"Final balanced dataset: {len(balanced_texts):,} samples ({target_count} per class).")

    # Cache dataset
    df = pd.DataFrame({"text": balanced_texts, "label": balanced_labels})
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CACHE_DATASET_FILE, index=False)
    print(f"Saved balanced dataset to: {CACHE_DATASET_FILE}")

    return balanced_texts, balanced_labels


def get_train_val_test_splits(test_size=0.15, val_size=0.15, random_state=42):
    """
    Returns stratified splits: (X_train, y_train), (X_val, y_val), (X_test, y_test).
    """
    texts, labels = build_unified_dataset()
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    val_relative_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_relative_size, random_state=random_state, stratify=y_train_full
    )

    print(f"Dataset Splits:")
    print(f"  Training:   {len(X_train):,} samples")
    print(f"  Validation: {len(X_val):,} samples")
    print(f"  Testing:    {len(X_test):,} samples")

    return (X_train, np.array(y_train)), (X_val, np.array(y_val)), (X_test, np.array(y_test))
