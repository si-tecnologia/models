import tensorflow as tf
import numpy as np

from io import BytesIO
from PIL import Image, ImageDraw

from matplotlib import cm
from object_detection_my.utils import visualization_utils as viz_utils

class SmokeDetector_v2:
    '''Wrapper class of the Smoke Detection Model'''
    
    category_index = {
        1: {'id': 1, 'name': 'Fumaça'}
    }

    versions_path = {
        0:'smokeDetector_v0/fine_tuned_model/d0',
        6:'smokeDetector_v0/fine_tuned_model/d6',
        7:'smokeDetector_v0/fine_tuned_model/d6',
    }
    
    def draw_prediction_on_image(self, array_image, bbox):
        im = Image.fromarray(np.uint8(array_image))
        draw = ImageDraw.Draw(im)
        width, height = im.size
        x1, y1, x2, y2 = bbox

        bbox_pix = (x1*width, y1*height, x2*width, y2*height)
        draw.rectangle(bbox_pix, outline ="red",width=2) 
        return im
    
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
            y1, x1, y2, x2 = tuple(detections['detection_boxes'][0][0].numpy())
            bbox = [x1, y1, x2, y2]
            draw_img = self.draw_prediction_on_image(image_np_with_detections,bbox)
            draw_img2 = self.draw_detections(
                image_np_with_detections,
                detections['detection_boxes'][0].numpy(),
                detections['detection_classes'][0].numpy().astype(np.uint32),
                detections['detection_scores'][0].numpy())
            return {
                'has_smoke':True,
                'smoke_prob':detections['detection_scores'][0][:5].numpy(),
                'detection_img': draw_img,
                'detection_img_tflib': draw_img2,
                'rel_bbox':[0,0,0,0],
                'bbox': detections['detection_boxes'][0][:5].numpy()
            }
        
        else:
            return {
                'has_smoke':False,
                'smoke_prob':detections['detection_scores'][0][:5].numpy(),
                'bbox': detections['detection_boxes'][0][:5].numpy()
                }