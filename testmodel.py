import tensorflow as tf

model = tf.keras.models.load_model("rice_type_model.h5")
model.summary()