from flask import Flask, request, jsonify, render_template
import pickle
import os

# Load the trained model
with open("lrmodel.pckl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form.get("text", "")
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        prob_dict = {
            lang: float(prob) for lang, prob in zip(model.classes_, probabilities)
        }

        return jsonify({
            "input_text": text,
            "predicted_language": prediction,
            "probabilities": prob_dict
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ✅ This ensures the app runs properly on Render (which provides $PORT)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
