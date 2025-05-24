from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO  # ✅ Import BytesIO

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ Load the Pretrained Model
MODEL_PATH = "soybean_fungal_model.h5"  # Ensure this file exists

model = load_model(MODEL_PATH)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])  # ✅ Fix

# ✅ Correct Class Labels List (Index-based)
class_labels = ['Broken soybeans', 'Immature soybeans', 'Intact soybeans', 'Skin-damaged soybeans', 'Spotted soybeans']

# ✅ Route for Home Page
@app.route('/')
def home():
    return render_template('index.html')

# ✅ Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # ✅ Load Image with Correct Preprocessing
        img = image.load_img(BytesIO(file.read()), target_size=(300, 300))  # ✅ Fix Here
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize if needed

        # ✅ Model Prediction
        prediction = model.predict(img_array)
        print(f"Model Raw Output: {prediction}")  # Debugging

        predicted_class = np.argmax(prediction)  # Get index of max value
        result = class_labels[predicted_class]   # ✅ Correct Mapping

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
