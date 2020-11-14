import tensorflow as tf
import numpy as np
from pathlib import Path

class ServingSmokeDetector:
    '''Wrapper class of the Smoke Detection Model used for serving as an API'''
    
    versions_path = {
        0:'fine_tuned_model/saved_model',
        1:'fine_tuned_model/saved_model_v1',
    }

    def __init__(self, v=0):
        if v not in self.versions_path.keys():
            raise ValueError(f'Only versions {self.versions_path.keys()} are available')
        
        self.detect_fn = tf.saved_model.load(str(Path(__file__).parent.resolve().joinpath(self.versions_path[v])))

    def predict(self, image_np):
        input_tensor = np.expand_dims(image_np, 0)
        detections = self.detect_fn(input_tensor)
        return detections