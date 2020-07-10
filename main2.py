import os, cv2, keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

path = "Images"
annot = "Airplanes_Annotations"

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


def get_points_from_row(row):
    row_points = row[1][0].split(" ")
    x1 = int(row_points[0])
    y1 = int(row_points[1])
    x2 = int(row_points[2])
    y2 = int(row_points[3])
    return {"x1": x1, "x2": x2, "y1": y1, "y2": y2}


for e, i in enumerate(os.listdir(annot)):
    if e < 10:
        filename = i.split(".")[0] + ".jpg"
        print(filename)
        img = cv2.imread(os.path.join(path, filename))
        df = pd.read_csv(os.path.join(annot, i))
        plt.imshow(img)

        for row in df.iterrows():
            row_points = get_points_from_row(row)
            cv2.rectangle(
                img=img,
                pt1=(row_points['x1'], row_points['y1']),
                pt2=(row_points['x2'], row_points['y2']),
                color=(255, 0, 0),
                thickness=2
            )
        plt.figure()
        plt.imshow(img)
        break

cv2.setUseOptimized(True)
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

train_images = []
train_labels = []


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

    return intersection_area / float(bb1_area + bb2_area - intersection_area)


for e, i in enumerate(os.listdir(annot)):
    try:
        if i.startswith("airplane"):
            filename = i.split(".")[0] + ".jpg"
            print(e, filename)
            image = cv2.imread(os.path.join(path, filename))
            df = pd.read_csv(os.path.join(annot, i))
            gtvalues = [get_points_from_row(row) for row in df.iterrows()]
            ss.setBaseImage(image)
            ss.switchToSelectiveSearchFast()
            ssresults = ss.process()
            imout = image.copy()
            counter = 0
            falsecounter = 0
            flag = 0
            fflag = 0
            bflag = 0
            for e, result in enumerate(ssresults):
                if e < 2000 and flag == 0:
                    for gtval in gtvalues:
                        x, y, w, h = result
                        iou = get_iou(gtval, {"x1": x, "x2": x + w, "y1": y, "y2": y + h})
                        if counter < 30:
                            if iou > 0.70:
                                timage = imout[y:y + h, x:x + w]
                                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(1)
                                counter += 1
                        else:
                            fflag = 1
                        if falsecounter < 30:
                            if iou < 0.3:
                                timage = imout[y:y + h, x:x + w]
                                resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                train_images.append(resized)
                                train_labels.append(0)
                                falsecounter += 1
                        else:
                            bflag = 1
                    if fflag == 1 and bflag == 1:
                        print("inside")
                        flag = 1
    except Exception as e:
        print(e)
        print("error in " + filename)
        continue

X_new = np.array(train_images)
y_new = np.array(train_labels)

from keras.layers import Dense
from keras import Model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

vggmodel = VGG16(weights='imagenet', include_top=True)
vggmodel.summary()

for layers in (vggmodel.layers)[:15]:
    print(layers)
    layers.trainable = False

X = vggmodel.layers[-2].output
predictions = Dense(2, activation="softmax")(X)
model_final = Model(vggmodel.input, predictions)

from keras.optimizers import Adam

opt = Adam(lr=0.0001)

model_final.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"])

model_final.summary()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1 - Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)


lenc = MyLabelBinarizer()
Y = lenc.fit_transform(y_new)
X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
traindata = trdata.flow(x=X_train, y=y_train)
tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
testdata = tsdata.flow(x=X_test, y=y_test)

from keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("ieeercnn_vgg16_1.h5", monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

hist = model_final.fit(traindata,
                       steps_per_epoch=40,
                       epochs=76,
                       validation_data=testdata,
                       validation_steps=2,
                       callbacks=[checkpoint, early])

import matplotlib.pyplot as plt

# plt.plot(hist.history["acc"])
# plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Loss", "Validation Loss"])
plt.show()
plt.savefig('chart loss.png')

im = X_test[-1]
plt.imshow(im)
img = np.expand_dims(im, axis=0)
out = model_final.predict(img)
if out[0][0] > out[0][1]:
    print("plane")
else:
    print("not plane")

z = 0
for e, i in enumerate(os.listdir(path)):
    if i.startswith("40"):
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
                if out[0][0] > 0.25:
                    cv2.rectangle(imout, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        print("AAAAAAA")
        plt.figure()
        plt.imsave(f'out/image_new{i}.jpg', imout)
        plt.imshow(imout)
model_final.save('model')
