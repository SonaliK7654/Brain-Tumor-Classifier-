import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import normalize

# Paths
data_dir = 'Training/'
classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
INPUT_SIZE = 64

# Prepare data
dataset = []
labels = []

for idx, cls in enumerate(classes):
    cls_folder = os.path.join(data_dir, cls)
    for img_name in os.listdir(cls_folder):
        if img_name.endswith('.jpg'):
            img_path = os.path.join(cls_folder, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            image = Image.fromarray(image).resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            labels.append(idx)

dataset = np.array(dataset)
labels = np.array(labels)

# Normalize images
dataset = dataset / 255.0

# Split
x_train, x_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=4)
y_test = to_categorical(y_test, num_classes=4)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

#-------------- Model Architecture--------------
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))  # 4 classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

# Training
model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=50,
    callbacks=[early_stop]
)

# Save model
model.save('BrainTumor4ClassModel.h5')
