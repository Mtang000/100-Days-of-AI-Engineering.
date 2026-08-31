
from transformers import pipeline

print("Downloading AI Brain (This might take a few seconds on the first run)...")
sentiment_analyzer = pipeline("sentiment-analysis")
print("AI is ready!\n")

reviews = [
    "I absolutely love this new keyboard, the keys feel amazing!",
    "It arrived on time. Standard packaging.",
    "This is the worst phone I have ever used. The battery dies in 2 hours.",
    "I'm not exactly thrilled about the price, but the quality is okay."
]

print("--- 📊 Analyzing Customer Feedback ---\n")

results = sentiment_analyzer(reviews)
for i in range(len(reviews)):
    review_text = reviews[i]
    ai_label = results[i]['label']
    ai_confidence = results[i]['score'] * 100

    if ai_label == "POSITIVE":
        icon = "🟢"
    elif ai_label == "NEGATIVE":
        icon = "🔴"
    else:
        icon = "⚪"

    print(f"{icon} {ai_label} ({ai_confidence:.1f}% Match)")
    print(f"   Review: '{review_text}'\n")
