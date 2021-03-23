
import requests
import json
import base64

from PIL import Image, ImageDraw
from io import BytesIO
from time import sleep
from timeit import default_timer as timer
import numpy as np


def get_model_server_predictions(pil_img, model_url):
    start = timer()
    data = json.dumps({"signature_name": "serving_default", "instances": [np.array(pil_img, dtype='uint8').tolist()]})
    end = timer()
    print(f'- Convert img to Json {end - start:.2f}s')

    start = timer()
    response = requests.post(model_url, data=data, headers={"content-type": "application/json"})
    end = timer()
    print(f'- Get Model prediction {end - start:.2f}s')
    print(response)
    return response.json()['predictions']
    

image = Image.open('tests/test_img.png')
model_url = "http://localhost:8501/v1/models/smoke_detector:predict"
prediction = get_model_server_predictions(image, model_url)

print(prediction[0])

print(prediction[0]['detection_scores'][0])

print(prediction[0]['detection_boxes'][0])