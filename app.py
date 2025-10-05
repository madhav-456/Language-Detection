from flask import Flask, request, jsonify, render_template
import pickle

# Load vectorizer and model
with open("lrmodel.pckl", "rb") as f:
    vectorizer, model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        X_vec = vectorizer.transform([text])
        prediction = model.predict(X_vec)[0]
        probabilities = model.predict_proba(X_vec)[0]
        prob_dict = {lang: float(prob) for lang, prob in zip(model.classes_, probabilities)}

        return jsonify({
            "input_text": text,
            "predicted_language": prediction,
            "probabilities": prob_dict
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
