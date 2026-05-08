import tensorflow as tf

model = tf.keras.models.load_model("cnn_model.keras")

model.save("cnn_model.h5")

print("Model converted successfully!")