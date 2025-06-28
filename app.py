from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model/Model_BodyParts.h5')

# Preprocessing function (adjust as needed based on your training)
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))  # Adjust to your model's expected input
    image_array = np.array(image) / 255.0  # Normalize if trained that way
    return np.expand_dims(image_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    img_bytes = image_file.read()

    try:
        input_tensor = preprocess_image(img_bytes)
        prediction = model.predict(input_tensor)[0]
        predicted_class = int(np.argmax(prediction))

        return jsonify({
            'prediction': predicted_class,
            'confidence': float(np.max(prediction))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "Model API is running"

