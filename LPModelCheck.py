import cv2
import itertools
import keras
import os

import matplotlib.pyplot as plt
import numpy as np
from airplanes.SelectiveSearch import SelectiveSearch

model_final = keras.models.load_model('ieeercnn_vgg16_1.h5')
model_final.summary()

selective_search = SelectiveSearch()
images_path = "Images"
for e, file in enumerate(f for f in os.listdir(images_path) if f.startswith("car")):
    print(f"{e}\t{file}")
    ss_results, img_out = selective_search.process_image(os.path.join(images_path, file))
    plt.imshow(img_out)
    for result in itertools.islice(ss_results, 2000):
        x, y, w, h = result
        resized = cv2.resize(img_out, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(resized, axis=0)
        out = model_final.predict(img)
        if out[0][0] >= 0.65:
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
    plt.figure()
    plt.imshow(img_out)
    plt.savefig(f'out/{file}')
model_final.save('model')
