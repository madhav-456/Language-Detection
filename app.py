from flask import Flask, request, jsonify, render_template
import pickle

# Load the fitted vectorizer and classifier
with open("lrmodel.pckl", "rb") as f:
    vectorizer, classifier = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
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
    app.run(debug=True)
