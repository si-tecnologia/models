import tensorflow as tf
import numpy as np
from pathlib import Path


class TfLiteModel:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, input_data):
        self.interpreter.set_tensor(self.input_index, input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)
