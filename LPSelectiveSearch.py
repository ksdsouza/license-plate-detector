import itertools
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

from SelectiveSearch import SelectiveSearch

images_path = "Images"
annotations = "Annotations"

cv2.setUseOptimized(True)
selective_search = SelectiveSearch()
train_images = []
train_labels = []

SS_IMG_SIZE = (224, 224)

chunk = int(sys.argv[1])


class Annotation:
    def __init__(self, file: str):
        f = open(file)
        [filename, x1, y1, dx, dy] = f.readline().split('\t')[0:5]
        f.close()

        self.img_filename = os.path.join(images_path, filename)
        self.boundary_box = {
            'x1': int(x1),
            'y1': int(y1),
            'x2': int(x1) + int(dx),
            'y2': int(y1) + int(dy)
        }


def show_annotated_boundaries():
    for e, file in enumerate(itertools.islice(os.listdir(annotations), 10)):
        annotation = Annotation(os.path.join(annotations, file))
        img = cv2.imread(annotation.img_filename)
        plt.imshow(img)

        boundary_box = annotation.boundary_box
        cv2.rectangle(
            img=img,
            pt1=(boundary_box['x1'], boundary_box['y1']),
            pt2=(boundary_box['x2'], boundary_box['y2']),
            color=(255, 0, 0),
            thickness=2
        )
        plt.figure()
        plt.imshow(img)
        plt.savefig(f'test_img_{annotation}.jpg')
    plt.show()

def get_iou(bb1, bb2):
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


filenames = list(enumerate(
    f for f in os.listdir(annotations)
    if f.startswith('wts')
))[chunk * 20:(chunk + 1) * 20]

for index, file in filenames:
    try:
        print(f"{index}\t{file}")
        annotation = Annotation(os.path.join(annotations, file))
        ss_results, img_out = selective_search.process_image(annotation.img_filename)
        boundary_box = annotation.boundary_box
        counter = 0
        falsecounter = 0
        fflag = 0
        bflag = 0

        images = []
        labels = []
        for result in itertools.islice(ss_results, 2000):
            x1, y1, dx, dy = result

            iou = get_iou(boundary_box, {
                'x1': x1,
                'y1': y1,
                'x2': x1 + dx,
                'y2': y1 + dy
            })

            if counter < 30:
                if iou > 0.85:
                    test_image = img_out[y1: y1 + dy, x1:x1 + dx]
                    resized = cv2.resize(test_image, SS_IMG_SIZE, interpolation=cv2.INTER_AREA)
                    images.append(resized)
                    labels.append(1)
                    counter += 1
            else:
                fflag = 1
            if falsecounter < 30:
                if iou < 0.3:
                    test_image = img_out[y1: y1 + dy, x1:x1 + dx]
                    resized = cv2.resize(test_image, SS_IMG_SIZE, interpolation=cv2.INTER_AREA)
                    images.append(resized)
                    labels.append(0)
                    falsecounter += 1
            else:
                bflag = 1
            if fflag == 1 and bflag == 1:
                break
        if counter != 0:
            train_images.extend(images)
            train_labels.extend(labels)
        else:
            print("Skipping")
    except Exception as e:
        print(e)
        print(f"Error occurred in {file}")


np.save(f'train_images_chunk{chunk}', train_images)
np.save(f'train_labels_chunk{chunk}', train_labels)
