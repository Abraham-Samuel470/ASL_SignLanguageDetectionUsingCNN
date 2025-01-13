import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

# Simple test
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
print("Model created successfully")
