"""
Text Sentiment Analysis with Bidirectional LSTM (Project DL-014)

test_100_sentences.py
---------------------
Academic Benchmark Suite of 105 manually verified test sentences
spanning 10 critical real-world linguistic and emotional categories:

1. Negative Emotional Expressions & Complaints
2. Negations & Valence Reversals
3. Double Negations & Nuanced Statements
4. Positive Sentiments & Gratitude
5. Neutral Factual & Informational Sentences
6. Mixed & Contrastive Sentiments
7. Sarcasm & Irony
8. Informal / Social Media Expressions
9. Very Short Expressions (1-3 words)
10. Long Reviews & Paragraphs

Generates an exhaustive evaluation report comparing Expected vs Predicted labels,
calculating category-level and overall accuracy, and pinpointing model performance.
"""

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
os.environ["KERAS_HOME"] = "/tmp/keras"

import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing import clean_text, load_tokenizer_from_file, texts_to_padded_sequences

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_bilstm_model.keras")
TOKENIZER_FILE = os.path.join(MODEL_DIR, "tokenizer.json")
REPORT_OUTPUT_FILE = os.path.join(MODEL_DIR, "test_100_report.json")

LABEL_NAMES = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}

# -----------------------------------------------------------------------------
# 105 Manually Verified Diverse Benchmark Test Sentences
# -----------------------------------------------------------------------------
BENCHMARK_SUITE = [
    # Category 1: Negative Emotional Expressions & Complaints (15)
    {"id": 1, "category": "Negative Emotions", "text": "I am demotivated with her.", "expected": "NEGATIVE"},
    {"id": 2, "category": "Negative Emotions", "text": "I hate this.", "expected": "NEGATIVE"},
    {"id": 3, "category": "Negative Emotions", "text": "I am disappointed.", "expected": "NEGATIVE"},
    {"id": 4, "category": "Negative Emotions", "text": "I don't like her.", "expected": "NEGATIVE"},
    {"id": 5, "category": "Negative Emotions", "text": "This is not good.", "expected": "NEGATIVE"},
    {"id": 6, "category": "Negative Emotions", "text": "I regret my decision.", "expected": "NEGATIVE"},
    {"id": 7, "category": "Negative Emotions", "text": "I feel terrible.", "expected": "NEGATIVE"},
    {"id": 8, "category": "Negative Emotions", "text": "I am unhappy with the result.", "expected": "NEGATIVE"},
    {"id": 9, "category": "Negative Emotions", "text": "I expected better.", "expected": "NEGATIVE"},
    {"id": 10, "category": "Negative Emotions", "text": "Nothing is going right.", "expected": "NEGATIVE"},
    {"id": 11, "category": "Negative Emotions", "text": "I feel completely hopeless and tired.", "expected": "NEGATIVE"},
    {"id": 12, "category": "Negative Emotions", "text": "The manager was extremely rude and uncooperative.", "expected": "NEGATIVE"},
    {"id": 13, "category": "Negative Emotions", "text": "Worst purchase I have ever made in my entire life.", "expected": "NEGATIVE"},
    {"id": 14, "category": "Negative Emotions", "text": "I am disgusted by how they treat paying customers.", "expected": "NEGATIVE"},
    {"id": 15, "category": "Negative Emotions", "text": "A total waste of my valuable time and hard-earned money.", "expected": "NEGATIVE"},

    # Category 2: Negations & Valence Reversals (12)
    {"id": 16, "category": "Negations", "text": "I am not happy.", "expected": "NEGATIVE"},
    {"id": 17, "category": "Negations", "text": "This is not great.", "expected": "NEGATIVE"},
    {"id": 18, "category": "Negations", "text": "I cannot recommend this to anyone.", "expected": "NEGATIVE"},
    {"id": 19, "category": "Negations", "text": "It did not meet my expectations.", "expected": "NEGATIVE"},
    {"id": 20, "category": "Negations", "text": "She was not helpful at all.", "expected": "NEGATIVE"},
    {"id": 21, "category": "Negations", "text": "The laptop does not work properly.", "expected": "NEGATIVE"},
    {"id": 22, "category": "Negations", "text": "There is no excuse for this poor behavior.", "expected": "NEGATIVE"},
    {"id": 23, "category": "Negations", "text": "I will never come back here again.", "expected": "NEGATIVE"},
    {"id": 24, "category": "Negations", "text": "This product lacks any real value.", "expected": "NEGATIVE"},
    {"id": 25, "category": "Negations", "text": "The battery is not holding any charge.", "expected": "NEGATIVE"},
    {"id": 26, "category": "Negations", "text": "I am by no means satisfied.", "expected": "NEGATIVE"},
    {"id": 27, "category": "Negations", "text": "Nobody enjoyed the presentation.", "expected": "NEGATIVE"},

    # Category 3: Double Negations & Nuanced Statements (10)
    {"id": 28, "category": "Double Negations", "text": "I don't think this is bad.", "expected": "POSITIVE"},
    {"id": 29, "category": "Double Negations", "text": "This is not bad at all.", "expected": "POSITIVE"},
    {"id": 30, "category": "Double Negations", "text": "It is hard to dislike this movie.", "expected": "POSITIVE"},
    {"id": 31, "category": "Double Negations", "text": "I cannot complain about the customer support.", "expected": "POSITIVE"},
    {"id": 32, "category": "Double Negations", "text": "Nothing short of wonderful.", "expected": "POSITIVE"},
    {"id": 33, "category": "Double Negations", "text": "Not disappointing in any way.", "expected": "POSITIVE"},
    {"id": 34, "category": "Double Negations", "text": "I do not regret my decision to buy this.", "expected": "POSITIVE"},
    {"id": 35, "category": "Double Negations", "text": "She is not unpleasant to be around.", "expected": "POSITIVE"},
    {"id": 36, "category": "Double Negations", "text": "It was not terrible, actually pretty enjoyable.", "expected": "POSITIVE"},
    {"id": 37, "category": "Double Negations", "text": "You will certainly not be disappointed.", "expected": "POSITIVE"},

    # Category 4: Positive Sentiments & Gratitude (15)
    {"id": 38, "category": "Positive Sentiments", "text": "I am so happy and motivated today.", "expected": "POSITIVE"},
    {"id": 39, "category": "Positive Sentiments", "text": "This movie was an absolute masterpiece!", "expected": "POSITIVE"},
    {"id": 40, "category": "Positive Sentiments", "text": "The acting was superb and truly inspiring.", "expected": "POSITIVE"},
    {"id": 41, "category": "Positive Sentiments", "text": "I love this product and use it daily.", "expected": "POSITIVE"},
    {"id": 42, "category": "Positive Sentiments", "text": "I feel deeply grateful for all your support.", "expected": "POSITIVE"},
    {"id": 43, "category": "Positive Sentiments", "text": "Outstanding work by the entire engineering team.", "expected": "POSITIVE"},
    {"id": 44, "category": "Positive Sentiments", "text": "The food was delicious, flavorful, and hot.", "expected": "POSITIVE"},
    {"id": 45, "category": "Positive Sentiments", "text": "Highly recommended to anyone looking for quality.", "expected": "POSITIVE"},
    {"id": 46, "category": "Positive Sentiments", "text": "Brilliant directing and remarkable visual effects.", "expected": "POSITIVE"},
    {"id": 47, "category": "Positive Sentiments", "text": "Fast shipping and exceptional customer service.", "expected": "POSITIVE"},
    {"id": 48, "category": "Positive Sentiments", "text": "I am so proud of our project achievements.", "expected": "POSITIVE"},
    {"id": 49, "category": "Positive Sentiments", "text": "A wonderful, heartwarming, and touching story.", "expected": "POSITIVE"},
    {"id": 50, "category": "Positive Sentiments", "text": "The battery life exceeded all my expectations.", "expected": "POSITIVE"},
    {"id": 51, "category": "Positive Sentiments", "text": "Top notch build quality and sleek design.", "expected": "POSITIVE"},
    {"id": 52, "category": "Positive Sentiments", "text": "I am delightfully surprised by how good this is.", "expected": "POSITIVE"},

    # Category 5: Neutral Factual & Informational Sentences (15)
    {"id": 53, "category": "Neutral Statements", "text": "I went to college today.", "expected": "NEUTRAL"},
    {"id": 54, "category": "Neutral Statements", "text": "The library opens at 9 AM and closes at 8 PM.", "expected": "NEUTRAL"},
    {"id": 55, "category": "Neutral Statements", "text": "The train arrives at platform four at 4 PM.", "expected": "NEUTRAL"},
    {"id": 56, "category": "Neutral Statements", "text": "The flight departs at 6:30 PM this evening.", "expected": "NEUTRAL"},
    {"id": 57, "category": "Neutral Statements", "text": "Today is Tuesday and tomorrow is Wednesday.", "expected": "NEUTRAL"},
    {"id": 58, "category": "Neutral Statements", "text": "It is 24 degrees Celsius in the city right now.", "expected": "NEUTRAL"},
    {"id": 59, "category": "Neutral Statements", "text": "The package was delivered by courier yesterday.", "expected": "NEUTRAL"},
    {"id": 60, "category": "Neutral Statements", "text": "The textbook has three hundred and fifty pages.", "expected": "NEUTRAL"},
    {"id": 61, "category": "Neutral Statements", "text": "He submitted the laboratory assignment on time.", "expected": "NEUTRAL"},
    {"id": 62, "category": "Neutral Statements", "text": "She is reading a technical document in the hall.", "expected": "NEUTRAL"},
    {"id": 63, "category": "Neutral Statements", "text": "The retail store is located on the second floor.", "expected": "NEUTRAL"},
    {"id": 64, "category": "Neutral Statements", "text": "We discussed the project outline during the meeting.", "expected": "NEUTRAL"},
    {"id": 65, "category": "Neutral Statements", "text": "There are four wooden chairs around the conference table.", "expected": "NEUTRAL"},
    {"id": 66, "category": "Neutral Statements", "text": "The smartphone comes packaged with a USB charger.", "expected": "NEUTRAL"},
    {"id": 67, "category": "Neutral Statements", "text": "He resides in a residential apartment in Bangalore.", "expected": "NEUTRAL"},

    # Category 6: Mixed & Contrastive Sentiments (10)
    {"id": 68, "category": "Mixed Sentiments", "text": "I love the product but hate its customer service.", "expected": "NEGATIVE"},
    {"id": 69, "category": "Mixed Sentiments", "text": "The food was great, however the waiter was rude.", "expected": "NEGATIVE"},
    {"id": 70, "category": "Mixed Sentiments", "text": "The camera is awesome but the battery life is terrible.", "expected": "NEGATIVE"},
    {"id": 71, "category": "Mixed Sentiments", "text": "Although the visuals were stunning, the plot was boring.", "expected": "NEGATIVE"},
    {"id": 72, "category": "Mixed Sentiments", "text": "It started out promising but quickly turned awful.", "expected": "NEGATIVE"},
    {"id": 73, "category": "Mixed Sentiments", "text": "The hardware is good, yet the software is completely broken.", "expected": "NEGATIVE"},
    {"id": 74, "category": "Mixed Sentiments", "text": "Despite a slow start, the climax was truly breathtaking.", "expected": "POSITIVE"},
    {"id": 75, "category": "Mixed Sentiments", "text": "It was somewhat pricey, but definitely worth every penny.", "expected": "POSITIVE"},
    {"id": 76, "category": "Mixed Sentiments", "text": "The room was a bit small, but the staff was wonderful.", "expected": "POSITIVE"},
    {"id": 77, "category": "Mixed Sentiments", "text": "This is worse than before.", "expected": "NEGATIVE"},

    # Category 7: Sarcasm & Irony (8)
    {"id": 78, "category": "Sarcasm & Irony", "text": "Great, another problem to deal with!", "expected": "NEGATIVE"},
    {"id": 79, "category": "Sarcasm & Irony", "text": "Oh fantastic, my flight just got cancelled again.", "expected": "NEGATIVE"},
    {"id": 80, "category": "Sarcasm & Irony", "text": "Just what I needed, a broken screen on day one.", "expected": "NEGATIVE"},
    {"id": 81, "category": "Sarcasm & Irony", "text": "Wonderful, the server crashed during our live demo.", "expected": "NEGATIVE"},
    {"id": 82, "category": "Sarcasm & Irony", "text": "Brilliant idea to reboot the production database at noon.", "expected": "NEGATIVE"},
    {"id": 83, "category": "Sarcasm & Irony", "text": "Yeah right, as if that was ever going to work.", "expected": "NEGATIVE"},
    {"id": 84, "category": "Sarcasm & Irony", "text": "Super helpful of them to reply two months later.", "expected": "NEGATIVE"},
    {"id": 85, "category": "Sarcasm & Irony", "text": "Perfect, now I have no way to get home.", "expected": "NEGATIVE"},

    # Category 8: Informal & Social Media Language (10)
    {"id": 86, "category": "Informal Language", "text": "omg this is sooo amazing!!", "expected": "POSITIVE"},
    {"id": 87, "category": "Informal Language", "text": "this app is super lit and dope", "expected": "POSITIVE"},
    {"id": 88, "category": "Informal Language", "text": "yayy got my dream internship offer today!!", "expected": "POSITIVE"},
    {"id": 89, "category": "Informal Language", "text": "huge shoutout to the devs for fixing this fast!!", "expected": "POSITIVE"},
    {"id": 90, "category": "Informal Language", "text": "ugh this update is so trash and laggy", "expected": "NEGATIVE"},
    {"id": 91, "category": "Informal Language", "text": "smh worst update ever, completely bricked my phone", "expected": "NEGATIVE"},
    {"id": 92, "category": "Informal Language", "text": "bruh this food tastes like cardboard wtf", "expected": "NEGATIVE"},
    {"id": 93, "category": "Informal Language", "text": "nah this place is totally overrated", "expected": "NEGATIVE"},
    {"id": 94, "category": "Informal Language", "text": "meh, neither good nor bad honestly", "expected": "NEUTRAL"},
    {"id": 95, "category": "Informal Language", "text": "just checking in from the airport gate", "expected": "NEUTRAL"},

    # Category 9: Very Short Expressions (1-3 words) (6)
    {"id": 96, "category": "Short Expressions", "text": "Love it!", "expected": "POSITIVE"},
    {"id": 97, "category": "Short Expressions", "text": "Absolutely fantastic.", "expected": "POSITIVE"},
    {"id": 98, "category": "Short Expressions", "text": "Hate it.", "expected": "NEGATIVE"},
    {"id": 99, "category": "Short Expressions", "text": "Totally unacceptable.", "expected": "NEGATIVE"},
    {"id": 100, "category": "Short Expressions", "text": "Page number 42.", "expected": "NEUTRAL"},
    {"id": 101, "category": "Short Expressions", "text": "Meeting at noon.", "expected": "NEUTRAL"},

    # Category 10: Long Reviews & Paragraphs (4)
    {
        "id": 102,
        "category": "Long Reviews",
        "text": "From the opening scene to the emotional final credits, this film captured my heart. "
                "The cinematography was breathtaking, the orchestral score was hauntingly beautiful, and the "
                "lead actors delivered nuanced, career-defining performances. I left the cinema completely awestruck.",
        "expected": "POSITIVE"
    },
    {
        "id": 103,
        "category": "Long Reviews",
        "text": "I regret buying this appliance. The build quality feels cheap and fragile, the instructions were "
                "incomprehensible, and the unit stopped functioning after two days. Customer support refused to issue "
                "a replacement and kept putting me on hold for hours. Extremely disappointed and angry.",
        "expected": "NEGATIVE"
    },
    {
        "id": 104,
        "category": "Long Reviews",
        "text": "The research seminar covered four principal modules in computer vision. Participants were provided "
                "with copies of the presentation slides and related academic citations. The session concluded at "
                "five in the afternoon following an interactive question and answer period with the faculty.",
        "expected": "NEUTRAL"
    },
    {
        "id": 105,
        "category": "Long Reviews",
        "text": "This product arrived in a damaged cardboard box. The exterior plastic was scratched and one of "
                "the key components was missing entirely. When I contacted the vendor, they showed no willingness "
                "to resolve the issue and claimed the package was delivered intact. Never buying from them again.",
        "expected": "NEGATIVE"
    }
]


def run_benchmark():
    """Executes the benchmark suite against the trained model and outputs a detailed report."""
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model not found at {MODEL_FILE}. Please train the model first.")
        return None

    if not os.path.exists(TOKENIZER_FILE):
        print(f"Error: Tokenizer not found at {TOKENIZER_FILE}.")
        return None

    print(f"Loading model from: {MODEL_FILE}")
    model = load_model(MODEL_FILE)
    tokenizer = load_tokenizer_from_file(TOKENIZER_FILE)

    results = []
    category_stats = {}
    correct_count = 0
    total = len(BENCHMARK_SUITE)

    print("\n" + "=" * 90)
    print(f"{'ID':<4} | {'Expected':<9} | {'Predicted':<9} | {'Conf':<6} | {'Status':<6} | Sentence")
    print("=" * 90)

    for item in BENCHMARK_SUITE:
        cat = item["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1

        padded = texts_to_padded_sequences([item["text"]], tokenizer, max_len=120)
        probs = model.predict(padded, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = LABEL_NAMES[pred_idx]
        conf = float(probs[pred_idx]) * 100.0

        is_correct = (pred_label == item["expected"])
        if is_correct:
            correct_count += 1
            category_stats[cat]["correct"] += 1
            status = "PASS"
        else:
            status = "FAIL"

        short_text = item["text"] if len(item["text"]) <= 48 else item["text"][:45] + "..."
        print(f"{item['id']:<4} | {item['expected']:<9} | {pred_label:<9} | {conf:5.1f}% | {status:<6} | {short_text}")

        results.append({
            "id": item["id"],
            "category": cat,
            "text": item["text"],
            "expected": item["expected"],
            "predicted": pred_label,
            "confidence": round(conf, 2),
            "probabilities": {
                "negative": round(float(probs[0]), 4),
                "neutral": round(float(probs[1]), 4),
                "positive": round(float(probs[2]), 4)
            },
            "status": status,
            "is_correct": is_correct
        })

    overall_acc = (correct_count / total) * 100.0
    print("=" * 90)
    print(f"\nOverall Benchmark Accuracy: {correct_count}/{total} ({overall_acc:.2f}%)\n")

    print("-" * 50)
    print(f"{'Category':<25} | {'Score':<8} | {'Accuracy':<8}")
    print("-" * 50)
    for cat, stats in category_stats.items():
        cat_acc = (stats["correct"] / stats["total"]) * 100.0
        print(f"{cat:<25} | {stats['correct']}/{stats['total']:<6} | {cat_acc:6.1f}%")
    print("-" * 50)

    # Save report
    report = {
        "total_sentences": total,
        "correct_predictions": correct_count,
        "overall_accuracy_pct": round(overall_acc, 2),
        "category_statistics": category_stats,
        "detailed_results": results
    }

    with open(REPORT_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {REPORT_OUTPUT_FILE}")
    return report


if __name__ == "__main__":
    run_benchmark()
