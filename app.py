from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model once
model = load_model('BrainTumor4ClassModel.h5')
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
INPUT_SIZE = 64

def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = img.resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "No file part", 400
        file = request.files['image']
        if file.filename == '':
            return "No selected file", 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict
        img_batch = preprocess_image(filepath)
        preds = model.predict(img_batch)
        idx = np.argmax(preds)
        prediction = classes[idx]

        os.remove(filepath)
        return render_template('result.html',
                               prediction=prediction,
                               confidence=f"{preds[0][idx]*100:.2f}%")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
