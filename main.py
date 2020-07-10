import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

model_final = keras.models.load_model('model')
path = "Images"
annot = "Airplanes_Annotations"
cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

z = 0
for e, i in enumerate(os.listdir(path)):
    if i.startswith("4"):
        z += 1
        img = cv2.imread(os.path.join(path, i))
        ss.setBaseImage(img)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = img.copy()
        for e, result in enumerate(ssresults):
            if e < 2000:
                x, y, w, h = result
                timage = imout[y:y + h, x:x + w]
                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                out = model_final.predict(img)
                if out[0][0] > 0.90:
                    cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        print("AAAAAAA")
        plt.figure()
        plt.imsave(f'out/{i}', imout)
        plt.imshow(imout)