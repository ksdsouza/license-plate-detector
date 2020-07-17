import itertools
import os
import sys

import cv2
import numpy as np

from SelectiveSearch import SelectiveSearch

images_path = "Images"
annotations = "Annotations"

cv2.setUseOptimized(True)
selective_search = SelectiveSearch()

SS_IMG_SIZE = (224, 224)

filenames = enumerate(
    f for f in os.listdir(annotations)
    if f.startswith('wts')
)

for index, file in filenames:
    try:
        filename = file.split('.')[0]
        img_filename = os.path.join(images_path, filename)
        print(f"{index}\t{img_filename}")
        ss_results, img_out = selective_search.process_image(f'{img_filename}.jpg')

        ss_results = list(itertools.islice(ss_results, 2000))
        np.save(f'SS/{filename}', ss_results)
    except Exception as e:
        print(e)
        print(f"Error occurred in {file}")
