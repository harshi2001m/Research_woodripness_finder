from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = load_model('tree_bark_model.keras')
img_height, img_width = 150, 150

# Home route
@app.route('/')
def home():
    return render_template('index.html')  # Create an index.html file for the UI

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded!'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected!'}), 400

    # Save the uploaded file temporarily
    filepath = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)

    # Preprocess the image
    image = load_img(filepath, target_size=(img_height, img_width))
    image_array = img_to_array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    prediction = model.predict(image_array)
    label = 'ripe' if prediction < 0.00015 else 'not ripe'
    probability = float(prediction[0])

    # Cleanup
    os.remove(filepath)

    return jsonify({'label': label, 'probability': probability})

if __name__ == '__main__':
    app.run(debug=True)
