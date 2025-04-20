# 📦 Import libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import movie_reviews
import random

# 📥 Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('movie_reviews')

# 🎯 Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# 🗂️ Load and shuffle movie review data
documents = [(movie_reviews.raw(fileid), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)

# 🔁 Analyze sentiments
predicted = []
actual = []

for review, label in documents:
    score = analyzer.polarity_scores(review)
    sentiment = 1 if score['compound'] >= 0 else 0  # positive = 1, negative = 0
    predicted.append(sentiment)
    actual.append(1 if label == 'pos' else 0)

# 📊 Evaluation
from sklearn.metrics import confusion_matrix, classification_report
print("\nConfusion Matrix:")
print(confusion_matrix(actual, predicted))

print("\nClassification Report:")
print(classification_report(actual, predicted))
