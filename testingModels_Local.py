from smokeDetector_v0.model import SmokeDetector
from PIL import Image
import numpy as np

sd = SmokeDetector()

print("image 1...........")

#image = Image.open('tests/test_img.png')
image = 'tests/test_img.png'

prediction = sd.predict(image)

print(prediction)


print(prediction.keys())