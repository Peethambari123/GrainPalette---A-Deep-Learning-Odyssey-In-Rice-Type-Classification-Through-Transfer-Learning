import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the model
model = tf.keras.models.load_model("rice_type_model.h5")
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

st.title("🍚 Rice Type Classifier")

# Upload image
uploaded_file = st.file_uploader("Upload an image of rice grain", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button("Predict"):
        # Preprocess image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model.predict(img_array)
        predicted_class = class_names[np.argmax(pred)]
        confidence = np.max(pred) * 100

        st.success(f"✅ Predicted Rice Type: **{predicted_class}** ({confidence:.2f}% confidence)")
