import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("cat_dog_model.keras")
class_names = ["cat", "dog"]  # check this matches your training label order

st.title("Cat vs Dog Classifier")
file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB").resize((64, 64))  # match your training image size
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    pred = model.predict(arr)
    st.image(img)
    st.write(f"Prediction: {class_names[np.argmax(pred)]}")