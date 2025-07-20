import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load model
model = tf.keras.models.load_model("rice_type_model.h5")

# Class labels (should match your folder names in the same order used by ImageDataGenerator)
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Load and preprocess image
img_path ="C:\\Users\\Peethambari\\Pictures\\Screenshots\\Screenshot 2025-06-26 113703.png"
 # Replace with the image you want to test
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
pred = model.predict(img_array)
predicted_class = class_names[np.argmax(pred)]

print(f"✅ Predicted Rice Type: {predicted_class}")