from ImageProcessor import image
from PIL import Image
from keras.applications import xception
import numpy as np
import glob


#Visszater az adott mappaban a labelezett kepekkel
class ProcessImages(object):

    def __init__(self, eleresi_utvonal, INPUT_SIZE_X=300, INPUT_SIZE_Y=200):
        self.eleresi_utvonal=eleresi_utvonal
        self.INPUT_SIZE_X=INPUT_SIZE_X
        self.INPUT_SIZE_Y=INPUT_SIZE_Y

    #kepek feldolgozasa
    def make_images(self, name):
        list=np.zeros((len(glob.glob(self.eleresi_utvonal+"\\"+name+"\\*.png")), self.INPUT_SIZE_X, self.INPUT_SIZE_Y), dtype='float32')

        index=0
        for filename in glob.glob(self.eleresi_utvonal+"\\"+name+"\\*.png"):
            #print(filename)
            i=image(filename)
            im=i.get_image()
            x=np.expand_dims(im.copy(), axis=0)
            list[index]=x
            index=index+1
        return list


    def make_scissors(self):
        return self.make_images('scissors')

    def make_rock(self):
        return self.make_images('rock')

    def make_paper(self):
        return self.make_images('paper')

    #ko, papir ollo kiertekele
    def make_all(self):
        scissors=self.make_scissors()
        rock=self.make_rock()
        paper=self.make_paper()
        s1,_,_=scissors.shape #0: ollo
        s2,_,_=rock.shape #1: ko
        s3,_,_=paper.shape # 2: papir
        x=scissors.copy()
        x=np.append(x, rock.copy(), axis=0)
        x=np.append(x, paper.copy(), axis=0)
        x=x.reshape(x.shape[0], 1, self.INPUT_SIZE_X, self.INPUT_SIZE_Y)
        y=np.zeros(s1+s2+s3)
        for i in range(s1, s2+s1):
            y[i]=1
        for i in range(s1+s2, s1+s2+s3):
            y[i]=2
        y=y.flatten()
        return (x,y)

