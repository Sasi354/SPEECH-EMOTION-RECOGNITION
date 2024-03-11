# 
import pickle
import collections
from collections.abc import Iterable
import numpy as np
from tensorflow import keras
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Importing collections and Iterable before Keras
model = None
try:
    model = pickle.load(open('F:\SPEECH-EMOTION-RECOGNITION\speechEmotionRecognition.pkl', 'rb'))
except Exception as e:
    print("Error loading model:", e)

feature_extraction = None
try:
    feature_extraction = pickle.load(open('feature_extraction.pkl', 'rb'))
except Exception as e:
    print("Error loading feature extraction:", e)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    if model is None or feature_extraction is None:
        return jsonify({"error": "Model or feature extraction is not loaded properly."})

    labels = ["angry", "happy", "fear", "disgust", "neutral", "ps", "sad"]
    audio_input = request.files['audio']
    feature = feature_extraction(audio_input)
    feature = np.expand_dims(feature, -1)
    print(feature.shape)
    predicted = model.predict(feature)
    predicted_labels_indices = np.argmax(predicted, axis=1)
    predicted_labels = [labels[idx] for idx in predicted_labels_indices]
    most_common_label = collections.Counter(predicted_labels).most_common(1)[0][0]

    return jsonify({"emotion": most_common_label})

if __name__ == "__main__":
    app.run(debug=True)
