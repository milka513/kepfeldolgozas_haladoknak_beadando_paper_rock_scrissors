import numpy as np
from glob import glob
#from tqdm import tqdm
import tensorflow as tf
import scipy.io as io
#from scipy import misc
#from scipy import ndimage
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from sklearn.datasets import load_files
from keras.applications import xception
from keras.utils import np_utils
import os
import PIL
from mpl_toolkits.axes_grid1 import ImageGrid
from IPython.display import display, Image
#from IPython.core.display import HTML
#from sklearn.preprocessing import LabelEncoder
import cv2
from ProcessImages import ProcessImages as PI
from keras.callbacks import ModelCheckpoint
from ImageProcessor import image as IM


class Network(object):

    def __init__(self):
        pass

    #melyhalo elkeszitese
    def make_network(self):
        self.model2=Sequential()
        self.model2.add(Conv2D(16, (3,3), padding='same', use_bias=False, input_shape=(1, 300, 200)))
        self.model2.add(BatchNormalization(axis=3, scale=False))
        self.model2.add(Activation("relu"))
        self.model2.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
        self.model2.add(Dropout(0.2))

        self.model2.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
        self.model2.add(BatchNormalization(axis=3, scale=False))
        self.model2.add(Activation("relu"))
        self.model2.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
        self.model2.add(Dropout(0.2))

        self.model2.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
        self.model2.add(BatchNormalization(axis=3, scale=False))
        self.model2.add(Activation("relu"))
        self.model2.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same'))
        self.model2.add(Dropout(0.2))

        self.model2.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
        self.model2.add(BatchNormalization(axis=3, scale=False))
        self.model2.add(Activation("relu"))
        self.model2.add(Flatten())
        self.model2.add(Dropout(0.2))

        # model2.add(GlobalAveragePooling2D())
        self.model2.add(Dense(256, activation='relu'))
        self.model2.add(Dense(3, activation='softmax'))
        self.model2.summary()

    #melyhalo tanittatasa
    def run(self):
        self.model2.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        epochs = 100
        checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch_to_beadando.hdf5',
                                       monitor='loss' ,verbose=1, save_best_only=True, mode='min')

        pi=PI('train')
        train_tensors, y_train=pi.make_all()
        print(np.shape(train_tensors),' ', np.shape(y_train))
        pi2=PI('valid')
        valid_tensors, y_valid=pi2.make_all()
        self.model2.fit(train_tensors, y_train,
                   validation_data=(valid_tensors, y_valid),
                   epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)

    #melyhalo betoltese
    def load(self):
        self.model2.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        self.model2.load_weights('saved_models/weights.best.from_scratch_to_beadando.hdf5')
        pi=PI('test')
        test_tensors, y_test=pi.make_all()


        valami=self.model2.evaluate(x=test_tensors, y=y_test, batch_size=50, verbose=1)

        print(valami)
    #kep kiertekelese
    def get_image(self, img_path):
        im=IM(img_path)
        img=im.get_image()
        img = img.reshape( 1, 300, 200)
        arr = self.model2.predict(np.expand_dims(img, axis=0))
        # pr=[np.argmax(arr[i]) for i in range(len(arr))]
        return np.argmax(arr)