import cv2
import numpy as np
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = load_model('BrainTumor4ClassModel.h5')

classes = ['glioma', 'meningioma', 'pituitary', 'notumor']

# Read and preprocess image
img_path = "C:\\Users\\Siddharth\\Desktop\\Major Project (ML model)\\Testing\\notumor\\Te-no_0010.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img).resize((64, 64))
img_array = np.array(img_pil) / 255.0
img_input = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_input)
class_idx = np.argmax(pred)
predicted_class = classes[class_idx]

# Print prediction
print("Predicted class:", predicted_class)

# Plot the MRI image
plt.imshow(img_pil)
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()

# Plot prediction probabilities
plt.bar(classes, pred[0])
plt.ylabel("Probability")
plt.title("Prediction Probabilities")
plt.show()




