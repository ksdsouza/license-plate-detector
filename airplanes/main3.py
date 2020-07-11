import itertools
import os, cv2, keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from Binarizer import Binarizer
from SelectiveSearch import SelectiveSearch
from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping

path = "Images"
annot = "Airplanes_Annotations"

cv2.setUseOptimized(True)
ss = SelectiveSearch()

train_images = []
train_labels = []

def get_points_from_row(row):
    row_points = row[1][0].split(" ")
    x1 = int(row_points[0])
    y1 = int(row_points[1])
    x2 = int(row_points[2])
    y2 = int(row_points[3])
    return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}

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


for index, file in itertools.islice(enumerate(f for f in os.listdir(annot) if f.startswith("airplane")), 20):
    try:
        filename = file.split(".")[0] + ".jpg"
        print(index, filename)
        df = pd.read_csv(os.path.join(annot, file))
        ground_truth_bounding_boxes = []
        [ground_truth_bounding_boxes.append(get_points_from_row(row)) for row in df.iterrows()]
        ssresults, imout = ss.process_image(os.path.join(path, filename))
        counter = 0
        falsecounter = 0
        flag = 0
        fflag = 0
        bflag = 0
        for result in itertools.islice(ssresults, 2000):
            for gt_bounding_box in ground_truth_bounding_boxes:
                x, y, w, h = result
                iou = get_iou(gt_bounding_box, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                if counter < 30:
                    if iou > 0.70:
                        test_image = imout[y:y + h, x:x + w]
                        resized = cv2.resize(test_image, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(1)
                        counter += 1
                else:
                    fflag = 1
                if falsecounter < 30:
                    if iou < 0.3:
                        test_image = imout[y:y + h, x:x + w]
                        resized = cv2.resize(test_image, (224, 224), interpolation=cv2.INTER_AREA)
                        train_images.append(resized)
                        train_labels.append(0)
                        falsecounter += 1
                else:
                    bflag = 1
            if fflag == 1 and bflag == 1:
                break
        print(counter, falsecounter)
    except Exception as e:
        print(e)
        print("error in " + filename)
        continue

X_new = np.array(train_images)
y_new = np.array(train_labels)
print(f"Successes: {len([i for i in train_labels if i == 1])} Failures: {len([i for i in train_labels if i != 1])}")


vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()

for layers in (vggmodel.layers)[:15]:
    print(layers)
    layers.trainable = False

X = vggmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model_final = Model(vggmodel.input, predictions)


opt = Adam(lr=0.0001)

model_final.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])

model_final.summary()




lenc = Binarizer()
Y = lenc.fit_transform(y_new)

X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

traindata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90).flow(x=X_train, y=y_train)
testdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90).flow(x=X_test, y=y_test)


checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model_final.fit_generator(generator=traindata, steps_per_epoch=40, epochs=10, validation_data=testdata,
                                 validation_steps=2, callbacks=[checkpoint, early])

z = 0
for e, i in enumerate(os.listdir(path)):
    if i.startswith("4"):
        z += 1
        ssresults, imout = ss.process_image(os.path.join(path, i))
        for e, result in enumerate(ssresults):
            if e < 2000:
                x, y, w, h = result
                test_image = imout[y:y + h, x:x + w]
                resized = cv2.resize(test_image, (224, 224), interpolation=cv2.INTER_AREA)
                img = np.expand_dims(resized, axis=0)
                out = model_final.predict(img)
                if out[0][0] > 0.65:
                    cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        plt.figure()
        plt.imsave(f'out/{i}', imout)
