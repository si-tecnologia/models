from PIL import Image

import sys
sys.path.insert(0, '..')
from smokeDetector_v0.serving_model import * 

sd = ServingSmokeDetector()

image = Image.open('test_img.png')

prediction = sd.predict(np.array(image))

print(prediction)