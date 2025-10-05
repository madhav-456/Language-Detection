import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv('language_dataset.csv')
X = df['text']  # Column with text data
y = df['language']  # Column with language labels

# Initialize and fit TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Initialize and train the classifier
classifier = LogisticRegression(max_iter=500)
classifier.fit(X_tfidf, y)

# Save the fitted vectorizer and classifier together
with open('lrmodel.pckl', 'wb') as model_file:
    pickle.dump((vectorizer, classifier), model_file)

print("Model and vectorizer trained and saved successfully!")

