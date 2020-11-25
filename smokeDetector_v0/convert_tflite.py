import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("fine_tuned_model/saved_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()

open('tflite/model.tflite', 'wb').write(tflite_model)
