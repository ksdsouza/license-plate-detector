import keras
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from Binarizer import Binarizer
import os
import sys

chunk = int(sys.argv[1])

X_new = np.load(f'train_images_chunk{chunk}.npy')
y_new = np.load(f'train_labels_chunk{chunk}.npy')

if os.path.exists('ieeercnn_vgg16_1.h5'):
    model_final = keras.models.load_model('ieeercnn_vgg16_1.h5')
else:
    vggmodel = VGG16(weights='imagenet', include_top=True)
    vggmodel.summary()

    for layers in vggmodel.layers[:15]:
        print(layers)
        layers.trainable = False

    X = vggmodel.layers[-2].output
    predictions = Dense(2, activation="softmax")(X)
    model_final = Model(vggmodel.input, predictions)
    opt = Adam(lr=0.0001)
    model_final.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy']
    )
    model_final.summary()

lenc = Binarizer()
Y = lenc.fit_transform(y_new)

X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.10)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

traindata = ImageDataGenerator().flow(x=X_train, y=y_train)

testdata = ImageDataGenerator().flow(x=X_test, y=y_test)

checkpoint = ModelCheckpoint(
    "ieeercnn_vgg16_1.h5",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto', period=1)

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')

hist = model_final.fit(traindata,
                       batch_size=5,
                       steps_per_epoch=100,
                       epochs=5,
                       validation_data=testdata,
                       validation_steps=25,
                       callbacks=[checkpoint, early])
