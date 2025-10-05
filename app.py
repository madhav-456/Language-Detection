from flask import Flask, request, jsonify, render_template
import pickle, os

# Load trained model + vectorizer
with open("lrmodel.pckl", "rb") as f:
    vectorizer, model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    try:
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        prob_dict = {lang: float(prob) for lang, prob in zip(model.classes_, probabilities)}

        return jsonify({
            "input_text": text,
            "predicted_language": prediction,
            "probabilities": prob_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
