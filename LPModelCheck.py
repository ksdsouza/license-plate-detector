import cv2
import itertools
import keras
import os

import matplotlib.pyplot as plt
import numpy as np
from airplanes.SelectiveSearch import SelectiveSearch

model_final = keras.models.load_model('model')
model_final.summary()

selective_search = SelectiveSearch()
images_path = "Images"
for e, file in enumerate(f for f in os.listdir(images_path) if f.startswith("car")):
    print(f"{e}\t{file}")
    ss_results, img_out = selective_search.process_image(os.path.join(images_path, file))
    plt.imshow(img_out)
    for result in itertools.islice(ss_results, 2000):
        x, y, w, h = result
        selection = img_out[y:y+h, x:x+w]
        resized_selection = cv2.resize(selection, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(resized_selection, axis=0)
        out = model_final.predict(img)
        if out[0][0] >= 0.85:
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
    plt.figure()
    plt.imshow(img_out)
    plt.savefig(f'out/{file}')
model_final.save('model')
