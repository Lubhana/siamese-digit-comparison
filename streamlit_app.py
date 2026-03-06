import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image


def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - y_pred, 0))
    return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)

model1 = tf.keras.models.load_model("models/model1.keras", compile=False)
model2 = tf.keras.models.load_model("models/model2.keras", compile=False)

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
    pred1 = model1.predict([image1, image2])[0][0]
    pred2 = model2.predict([image1, image2])[0][0]

    st.subheader("Results")

# Model 1 interpretation
    if pred1 > 0.5:
        result1 = "Same Digit"
    else:
        result1 = "Different Digit"

# Model 2 interpretation
    if pred2 < 0.5:
        result2 = "Same Digit"
    else:
        result2 = "Different Digit"

    st.write(f"Model 1 (Binary Crossentropy) Score: {pred1:.4f}")
    st.success(f"Model 1 Prediction: {result1}")

st.write(f"Model 2 (Contrastive Loss) Distance: {pred2:.4f}")
st.success(f"Model 2 Prediction: {result2}")
