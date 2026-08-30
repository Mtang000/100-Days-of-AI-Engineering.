import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

emails = [
    "Hey team, just a reminder about the meeting tomorrow at 10 AM.",
    "URGENT! You have been selected as the winner of a $1000 gift card! Click here now!",
    "Can you send over the Q3 financial report when you have a second?",
    "CONGRATULATIONS! Claim your FREE crypto coins immediately before time runs out.",
    "Lunch is in the breakroom, let me know if you want a slice of pizza."
]
labels = [0, 1, 0, 1, 0]
vectorizer = TfidfVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(emails)

ai_classifier = MultinomialNB()
ai_classifier.fit(X_train, labels)

new_inbox = [
    "Don't forget to review the project proposal before the weekend.",
    "URGENT ACTION REQUIRED: Claim your free luxury vacation package now!!!"
]


X_new = vectorizer.transform(new_inbox)

predictions = ai_classifier.predict(X_new)
probabilities = ai_classifier.predict_proba(X_new)

for i in range(len(new_inbox)):
    email_text = new_inbox[i]
    spam_chance = probabilities[i][1] * 100

    if predictions[i] == 1:
        print(f"🚨 SPAM DETECTED ({spam_chance:.1f}% confidence):")
        print(f"   '{email_text}'\n")
    else:
        print(f"✅ SAFE INBOX ({100 - spam_chance:.1f}% confidence):")
        print(f"   '{email_text}'\n")
