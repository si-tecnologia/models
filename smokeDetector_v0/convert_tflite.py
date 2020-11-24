import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("fine_tuned_model/saved_model")
tflite_model = converter.convert()

# Save the model.
with open('tflite/model.tflite', 'wb') as f:
    f.write(tflite_model)
