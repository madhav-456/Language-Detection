from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import os

# Load the fitted vectorizer and classifier
try:
    model_path = os.path.join(os.path.dirname(__file__), "lrmodel.pckl")
    with open(model_path, "rb") as f:
        vectorizer, classifier = pickle.load(f)
except FileNotFoundError:
    vectorizer = None
    classifier = None
    print("Warning: lrmodel.pckl not found. Predictions will fail.")

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if vectorizer is None or classifier is None:
        return jsonify({"error": "Model not loaded"}), 500

    # Get text input from form
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Transform input text
        X_vec = vectorizer.transform([text])

        # Predict language
        prediction = classifier.predict(X_vec)[0]

        # Get probabilities for all classes
        probabilities = classifier.predict_proba(X_vec)[0]
        prob_dict = {lang: float(prob) for lang, prob in zip(classifier.classes_, probabilities)}

        return jsonify({
            "input_text": text,
            "predicted_language": prediction,
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use Render's PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port,debug=false)


