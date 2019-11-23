
from PIL import Image
import numpy as np
import glob
import measure

#Visszater az adott mappaban a labelezett kepekkel
class ProcessImages(object):

    def __init__(self, eleresi_utvonal, INPUT_SIZE_X=6):
        self.eleresi_utvonal=eleresi_utvonal
        self.INPUT_SIZE_X=INPUT_SIZE_X

    def make_images(self, name):
        list=np.zeros((len(glob.glob(self.eleresi_utvonal+"\\"+name+"\\*.png")), self.INPUT_SIZE_X), dtype='float32')

        index=0
        for filename in glob.glob(self.eleresi_utvonal+"\\"+name+"\\*.png"):
            im=measure.process(filename)
            list[index]=im
            index=index+1

        return list

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
        y=np.zeros(s1+s2+s3)
        for i in range(s1, s2+s1):
            y[i]=1
        for i in range(s1+s2, s1+s2+s3):
            y[i]=2
        y=y.flatten()
        return (x,y)

