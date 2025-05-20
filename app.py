from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import os

app = Flask(__name__)
model = load_model('BrainTumor4ClassModel.h5')

classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict_image(image_path):
    image = cv2.imread(image_path)  # BGR format
    if image is None:
        return "Error reading image"
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize using PIL
    img = Image.fromarray(image).resize((64, 64))
    
    img = np.array(img).astype('float32') / 255.0
    
    # If model expects 4D input: (batch, height, width, channels)
    img = np.expand_dims(img, axis=0)
    
    pred = model.predict(img)
    
    class_idx = np.argmax(pred, axis=1)[0]
    return classes[class_idx]



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template('result.html', prediction=prediction, image_path=filepath)
    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)  


