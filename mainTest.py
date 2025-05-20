import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

model = load_model('BrainTumor4ClassModel.h5')

classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

img = cv2.imread("C:\\Users\\Siddharth\\Desktop\\Major Project (ML model)\\Testing\\notumor\\Te-no_0010.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img).resize((64, 64))
img = np.expand_dims(np.array(img) / 255.0, axis=0)

pred = model.predict(img)
class_idx = np.argmax(pred)
print("Predicted class:", classes[class_idx])




