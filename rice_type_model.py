import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Paths
data_dir = "Rice_Image_Dataset"
image_size = 224
batch_size = 32

# Data Generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important: keep this False for correct label alignment
)

# Base Model
base_model = MobileNetV2(input_shape=(image_size, image_size, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

# Custom Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Summary of Model Structure
model.summary()

# Train
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data
)


train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]

print(f"\n✅ Final Training Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Final Validation Accuracy: {val_acc * 100:.2f}%\n")

# Save model
model.save("rice_type_model.h5")

# ✅✅✅ CLASSIFICATION METRICS FOR PROJECT REPORT ✅✅✅

# Reset validation generator
val_data.reset()

# True Labels
y_true = val_data.classes
class_labels = list(val_data.class_indices.keys())

# Predictions
y_pred_probs = model.predict(val_data)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n✅ Confusion Matrix:")
print(cm)

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy Score: {acc * 100:.2f}%")

# Classification Report
print("\n✅ Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ✅✅✅ END OF EVALUATION SECTION ✅✅✅

# Plot Accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy")
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")
plt.show()
