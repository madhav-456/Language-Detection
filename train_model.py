import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv('language_dataset.csv')
X = df['text']  # Assuming 'text' column contains the input text
y = df['language']  # Assuming 'language' column contains the language labels

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

classifier = LogisticRegression()
classifier.fit(X_tfidf, y)

# Save the fitted model and vectorizer
with open('lrmodel.pckl', 'wb') as model_file:
    pickle.dump((vectorizer, classifier), model_file)
