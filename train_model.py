import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("language_dataset.csv")

X = df["text"]
y = df["label"]

# Convert text to numeric features
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train classifier
model = LogisticRegression(max_iter=500)
model.fit(X_vec, y)

# Save vectorizer + model together
with open("lrmodel.pckl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("Model trained and saved as lrmodel.pckl")
