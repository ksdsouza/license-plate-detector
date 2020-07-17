import os

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

X_new = np.concatenate(
    (
        np.load(f'train_images_chunk0.npy'),
        np.load(f'train_images_chunk1.npy'),
        np.load(f'train_images_chunk2.npy'),
        np.load(f'train_images_chunk3.npy'),
        np.load(f'train_images_chunk4.npy'),
        np.load(f'train_images_chunk5.npy'),
        np.load(f'train_images_chunk6.npy'),
        np.load(f'train_images_chunk7.npy'),
        np.load(f'train_images_chunk8.npy'),
    ),
    axis=0
)

y_new = np.concatenate(
    (
        np.load(f'train_labels_chunk0.npy'),
        np.load(f'train_labels_chunk1.npy'),
        np.load(f'train_labels_chunk2.npy'),
        np.load(f'train_labels_chunk3.npy'),
        np.load(f'train_labels_chunk4.npy'),
        np.load(f'train_labels_chunk5.npy'),
        np.load(f'train_labels_chunk6.npy'),
        np.load(f'train_labels_chunk7.npy'),
        np.load(f'train_labels_chunk8.npy'),
    ),
    axis=0
)

if os.path.exists('ieeercnn_vgg16_1.h5'):
    model_final = keras.models.load_model('ieeercnn_vgg16_1.h5')
else:
    vggmodel = VGG16(weights='imagenet', include_top=True)
    vggmodel.summary()

    for layers in vggmodel.layers[:5]:
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

Y = Binarizer().fit_transform(y_new)

X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.10)

traindata = ImageDataGenerator().flow(x=X_train, y=y_train)

testdata = ImageDataGenerator().flow(x=X_test, y=y_test)

checkpoint = ModelCheckpoint(
    "model",
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto', period=1)

early = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=20, verbose=1, mode='auto')

hist = model_final.fit(traindata,
                       steps_per_epoch=40,
                       epochs=100,
                       validation_data=testdata,
                       validation_steps=30,
                       callbacks=[checkpoint, early])
