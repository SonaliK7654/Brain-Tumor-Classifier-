# Brain Tumor Classification Web App

This project is a deep learning-based web application to classify brain tumor types from MRI images into four categories:
- Glioma
- Meningioma
- Pituitary tumor
- No tumor

## Features
- CNN model trained on MRI brain scans
- Flask backend serving the model
- Simple frontend with image upload and prediction display
- Data augmentation and early stopping during training
- Confusion matrix and classification report for model evaluation

## Project Structure
Brain-Tumor-Classifier-/
├── app.py # Flask backend application
├── BrainTumor4ClassModel.h5 # Trained CNN model for brain tumor classification
├── requirements.txt # Python dependencies
├── static/ # Static files (JavaScript, CSS)
│ └── uploads/ # Folder for storing uploaded MRI images
├── templates/ # HTML templates for web pages
│ ├── index.html # Home page with image upload form
│ └── result.html # Page displaying prediction results
├── training/ # Folder containing training MRI images
├── testing/ # Folder containing testing MRI images
├── maintrain.py # Script to train the CNN model
├── maintest.py # Script to test the CNN model
└── README.md # Project documentation


---

## Features

- Upload brain MRI images through the web interface
- Real-time tumor classification into 4 classes:
  - Glioma
  - Meningioma
  - Pituitary tumor
  - No tumor
- View prediction probabilities and visualize the MRI image
- Scripts available for model training and testing on MRI datasets

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/SonaliK7654/Brain-Tumor-Classifier-.git
   cd Brain-Tumor-Classifier-

2. Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

3. Install required Python packages:

pip install -r requirements.txt
 
4. Run the Flask app:

python app.py

5. Open your browser and go to:

http://localhost:5000


# Deployment / Hosting
Deployment can be done on any hosting site.

# Features
Upload MRI brain scan images through the web interface.

Real-time tumor classification using a pre-trained CNN model.

Displays prediction probabilities and class labels.

Clean and responsive frontend using HTML, CSS, and JavaScript.

# File Descriptions
app.py: Main Flask application to handle routing and prediction.

BrainTumor4ClassModel.h5: Pre-trained CNN model for classification.

maintrain.py: Script to train the CNN model on MRI dataset.

maintest.py: Script to test the trained model on new images.

static/: Contains JavaScript, CSS, and upload folders.

templates/: HTML templates for web pages.

training/ and testing/: Image datasets for model training and testing.

# Contributing
Feel free to fork this repository and submit pull requests for improvements or fixes.

# License
Specify your license here (e.g., MIT License).

