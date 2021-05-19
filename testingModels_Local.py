from smokeDetector_v0.model_v2 import SmokeDetector_v2

from PIL import Image
import numpy as np

sd = SmokeDetector_v2(min_score_thresh=0.3, v=6)

print("image 1...........")

#image = Image.open('tests/test_img.png')
image = 'tests/test_img.png'

prediction = sd.predict(image)

print(prediction)


print(prediction.keys())