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
import NeuralNetwork as network
from ImageProcessor import image as im





def main():

    #cv2.imshow('Contours', image)


    #cv2.imshow('valami', image)
    #pi=PI('C:\\Users\\Heinc Emília\\Documents\\university\\kepfeldolgozas_haladoknak\\beadando\\rock-paper-scissors-dataset\\Rock-Paper-Scissors\\test')

    #gray = cv2.cvtColor(list[0], cv2.COLOR_BGR2GRAY)


    #cv2.imshow('paper2', list[10])


    n=network.Network()
    n.make_network()
    #n.run()
    n.load()

    out=n.get_image('rocktest1.png')
    out5=n.get_image('rocktest1.png')
    out2=n.get_image('test1.png')
    out3=n.get_image('test2.png')
    out4=n.get_image('test3.png')
    print('test1: ',out2 )
    print('test2: ',out3 )
    print('test3: ',out4 )
    print('test4: ',out5 )

    print(out)
    #i = im('test3.png')
    #cv2.imshow('test', i.get_image())
    #cv2.waitKey(0)

if __name__ == '__main__':
    main()

