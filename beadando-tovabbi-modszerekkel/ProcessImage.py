
from PIL import Image
import numpy as np
import glob
import measure
import cv2


#Visszater az adott mappaban a labelezett kepekkel
class ProcessImages(object):

    def __init__(self, eleresi_utvonal, INPUT_SIZE_X=7, two_value=False, list_=[]):
        self.eleresi_utvonal=eleresi_utvonal
        self.INPUT_SIZE_X=INPUT_SIZE_X
        self.two_value=two_value
        self.list_=list_

    def make_images(self, name):
        if self.two_value:
            self.INPUT_SIZE_X=len(self.list_)
        list=np.zeros((len(glob.glob(self.eleresi_utvonal+"\\"+name+"\\*.png")), self.INPUT_SIZE_X), dtype='float32')

        index=0
        for filename in glob.glob(self.eleresi_utvonal+"\\"+name+"\\*.png"):
            #print(filename)
            im=measure.process(filename, two_values=self.two_value, list=self.list_)
            list[index]=im
            index=index+1
        #print(name,': ', list)
        return list

    def __find_picture(self, name, par):
        if self.two_value:
            self.INPUT_SIZE_X=len(self.list_)

        for filename in glob.glob(self.eleresi_utvonal + "\\" + name + "\\*.png"):
            measure.find_picture(filename, par, two_values=self.two_value, parameters=self.list_)

    def find_pictures(self, par):
        self.__find_picture('scissors', par)
        self.__find_picture('rock', par)
        self.__find_picture('paper', par)

    def make_scissors(self):
        return self.make_images('scissors')

    def make_rock(self):
        return self.make_images('rock')

    def make_paper(self):
        return self.make_images('paper')

    def make_all(self):
        scissors=self.make_scissors()
        rock=self.make_rock()
        paper=self.make_paper()
        s1,_=scissors.shape #0: ollo
        s2,_=rock.shape #1: ko
        s3,_=paper.shape # 2: papir
        x=scissors.copy()
        x=np.append(x, rock.copy(), axis=0)
        x=np.append(x, paper.copy(), axis=0)
        #x=x.reshape(x.shape[0], 1, self.INPUT_SIZE_X, self.INPUT_SIZE_Y)
        #x=np.reshape(x, (60000, s1+s2+s3))
        #np.concatenate((x, rock))
        #np.concatenate((x, paper))
        y=np.zeros(s1+s2+s3)
        for i in range(s1, s2+s1):
            y[i]=1
        for i in range(s1+s2, s1+s2+s3):
            y[i]=2
        y=y.flatten()
        return (x,y)

