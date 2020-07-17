import cv2
import itertools
import keras
import os

import matplotlib.pyplot as plt
import numpy as np
from airplanes.SelectiveSearch import SelectiveSearch
from LPSelectiveSearch import get_iou, get_annotation, annotations

model_final = keras.models.load_model('model')

iou_threshold = 0.65
selective_search = SelectiveSearch()
images_path = "Images"
average_ious = []
f = open("results.csv", "w")
f.write("filename,hit_avg_iou,hit_num,false_pos_avg_iou,false_pos_num,false_neg_avg_iou,false_neg_count,miss_avg_iou,miss_count\n")

for e, file in enumerate(f for f in os.listdir(images_path) if f.startswith("car")):
    print(f"{e}\t{file}")

    filename = file[:file.index('.')]
    ss_results = np.load(f'{os.path.join("SS", filename)}.npy')
    img_out = cv2.imread(f'{os.path.join(images_path, filename)}.jpg')
    # ss_results, img_out = selective_search.process_image(os.path.join(images_path, file))
    plt.imshow(img_out)

    hit_ious = [] #LP correctly identified
    fneg_ious = [] #LP false negative
    fpos_ious = [] #non-LP false positive
    miss_ious = [] #non-LP correctly identified

    boundary_box, img_filename = get_annotation(f'{os.path.join(annotations, filename)}.txt')

    for result in itertools.islice(ss_results, 2000):
        x, y, w, h = result

        selection = img_out[y:y+h, x:x+w]
        resized_selection = cv2.resize(selection, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        img = np.expand_dims(resized_selection, axis=0)
        out = model_final.predict(img)

        iou = get_iou(boundary_box, {
            'x1': x,
            'y1': y,
            'x2': x + w,
            'y2': y + h
        })

        # print("IoU: {:.4f}".format(iou))

        if out[0][0] >= 0.85:
            cv2.rectangle(img_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            if iou >= iou_threshold:
                hit_ious.append(iou)
            else:
                fpos_ious.append(iou)
        else:
            if iou >= iou_threshold:
                fneg_ious.append(iou)
            else:
                miss_ious.append(iou)

    avg_hit = sum(hit_ious) / len(hit_ious) if len(hit_ious) > 0 else 0
    avg_fpos = sum(fpos_ious) / len(fpos_ious) if len(fpos_ious) > 0 else 0
    avg_fneg = sum(fneg_ious) / len(fneg_ious) if len(fneg_ious) > 0 else 0
    avg_miss = sum(miss_ious) / len(miss_ious) if len(miss_ious) > 0 else 0

    result = f"{file},{avg_hit:.4f},{len(hit_ious)},{avg_fpos:.4f},{len(fpos_ious)},{avg_fneg:.4f},{len(fneg_ious)},{avg_miss:.4f},{len(miss_ious)}\n"
    print(result)
    f.write(result)
    f.flush()

    plt.figure()
    plt.imshow(img_out)
    plt.imsave(f'out/{filename}.jpg', img_out)

f.close()
model_final.save('model')