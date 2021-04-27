import tensorflow as tf
import numpy as np

from io import BytesIO
from PIL import Image
from object_detection.utils import visualization_utils as viz_utils

class SmokeDetector:
    '''Wrapper class of the Smoke Detection Model'''
    
    category_index = {
        1: {'id': 1, 'name': 'Fumaça'}
    }

    versions_path = {
        0:'smokeDetector_v0/fine_tuned_model/saved_model',
        1:'smokeDetector_v0/fine_tuned_model/saved_model_v1',
        2:'smokeDetector_v0/fine_tuned_model/saved_model_v2',
        3:'smokeDetector_v0/fine_tuned_model/saved_model_v3',
        4:'smokeDetector_v0/fine_tuned_model/saved_model_v4',
    }

    def __init__(self, min_score_thresh=0.3, v=0):
        if v not in self.versions_path.keys():
            raise ValueError(f'Only versions {self.versions_path.keys()} are available')
        
        self.detect_fn = tf.saved_model.load(self.versions_path[v])
        self.min_score_thresh = min_score_thresh


    def load_img_from_path(self, path):
        image = Image.open(path)
        return np.array(image, dtype='uint8')


    def load_img_from_bytes(self, img_bytes):
        image = Image.open(BytesIO(img_bytes))
        return np.array(image, dtype='uint8')


    def draw_detections(self, image_np, boxes, classes, scores):
        """Wrapper function to visualize detections. 
        Args:
        image_np: uint8 numpy array with shape (img_height, img_width, 3)
        boxes: a numpy array of shape [N, 4]
        classes: a numpy array of shape [N]. Note that class indices are 1-based,
            and match the keys in the label map.
        scores: a numpy array of shape [N] or None.  If scores=None, then
            this function assumes that the boxes to be plotted are groundtruth
            boxes and plot all boxes as black with no classes or scores.
        category_index: a dict containing category dictionaries (each holding
            category index `id` and category name `name`) keyed by category indices.
        """
        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np,
            boxes,
            classes,
            scores,
            self.category_index,
            use_normalized_coordinates=True,
            min_score_thresh=self.min_score_thresh)
        return image_np


    def predict(self, img, from_path=True):
        '''Predicts the presence of smoke on the image
        Args:
        - img: the path to the image, OR the encoded img in bytes
        - from_path: if set to False, the function reads the image in bytes
        Return, a dict with the keys
        - has_smoke::bool
        - smoke_prob::float[0,1]
        - detection_img::np.array -> image with the bounding box drawn (if there is smoke)
        - rel_bbox::list -> the bounding box coordinates
        '''

        if from_path:
            image_np = self.load_img_from_path(img)
        else:
            image_np = self.load_img_from_bytes(img)
        
        input_tensor = np.expand_dims(image_np, 0)
        detections = self.detect_fn(input_tensor)
        
        smoke_prob = max(detections['detection_scores'][0].numpy())

        if smoke_prob > self.min_score_thresh:

            image_np_with_detections = image_np.copy()
            draw_img = self.draw_detections(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.uint32),
                detections['detection_scores'][0].numpy())

            return {
                'has_smoke':True,
                'smoke_prob':smoke_prob,
                'detection_img': draw_img,
                'rel_bbox':[0,0,0,0],
                'bbox': detections['detection_boxes'][0].numpy()
            }
        
        else:
            return {
                'has_smoke':False,
                'smoke_prob':smoke_prob,
                'bbox': detections['detection_boxes'][0].numpy()
                }