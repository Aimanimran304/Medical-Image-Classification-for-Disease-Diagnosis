# app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model('pneumonia_classification_model.h5')

# Define the class names
class_names = ['Normal', 'Pneumonia']

# Set the title and description
st.title('Pneumonia Detection from Chest X-ray Images')
st.write("""
This app uses a Convolutional Neural Network (CNN) to classify chest X-ray images as **Normal** or **Pneumonia**.
""")

# File uploader allows user to upload image
uploaded_file = st.file_uploader("Please upload a chest X-ray image (JPEG/PNG format)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image_data.resize((150, 150))
    img = img.convert('RGB')  # Ensure image is in RGB format
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0,1]

    # Make prediction
    prediction = model.predict(img_array)
    score = prediction[0][0]


    # Display the prediction
    if score < 0.5:
        st.write("### Prediction: **Normal**")
        st.write(f"Confidence: {(1 - score) * 100:.2f}%")
    else:
        st.write("### Prediction: **Pneumonia**")
        st.write(f"Confidence: {score * 100:.2f}%")
else:
    st.write("Please upload an image to get a prediction.")
