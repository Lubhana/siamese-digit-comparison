import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

model1 = tf.keras.models.load_model("models/model1.keras")
model2 = tf.keras.models.load_model("models/model2.keras")

st.title("Siamese Digit Similarity Detector")

img1 = st.file_uploader("Upload Image 1")
img2 = st.file_uploader("Upload Image 2")

def preprocess(img):
    img = np.array(img.convert("L"))
    img = cv2.resize(img,(28,28))
    img = img/255.0
    return img.reshape(1,28,28)

if img1 and img2:

    image1 = preprocess(Image.open(img1))
    image2 = preprocess(Image.open(img2))

    st.image([img1,img2], width=150)

    pred1 = model1.predict([image1,image2])
    pred2 = model2.predict([image1,image2])

    st.subheader("Results")

    st.write("Model 1 (Binary Crossentropy):",pred1[0][0])
    st.write("Model 2 (Contrastive Loss):",pred2[0][0])