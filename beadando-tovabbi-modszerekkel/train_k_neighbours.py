
from sklearn import  neighbors
import pickle
import ProcessImage as image
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
#k-legkozelebbi szomszed modszere
class train(object):

    def __init__(self, neigh_numbers, first_two=False, features=[]):
        self.neigh_numbers=neigh_numbers
        self.first_two=first_two
        self.features=features


    def train(self, X, y):
        if self.first_two:
            X=X[:, :2]
        for weight in ['uniform', 'distance']:
            self.neigh=neighbors.KNeighborsClassifier(self.neigh_numbers, weights=weight)
            self.neigh.fit(X, y)
            name='out/kneigh_'+str(weight)+'_'+str(self.neigh_numbers)+'.pkl'
            #with open(name, 'wb') as f:
            #    f.flush()
            #    pickle.dump(self.neigh, f)

    def predict(self, X):
        if self.first_two:
            X=X[:, :2]
        #list=[X]
        return self.neigh.predict(X)

    def test(self):
        im=image.ProcessImages('test', two_value=self.first_two, list_=self.features)
        X,Y=im.make_all()
        if self.first_two:
            X=X[:, :2]
        return self.neigh.score(X, Y)
    def valid(self):
        im=image.ProcessImages('validation', two_value=self.first_two, list_=self.features)
        X, Y = im.make_all()
        if self.first_two:
            X=X[:, :2]
        return self.neigh.score(X, Y)

    def load(self, neigh_numbers,  type='distance'):
        filename= name='out/kneigh_'+type+'_'+str(self.neigh_numbers)+'.pkl'
        with open(filename, 'rb') as f:
            self.neigh=pickle.load(f)

    #resource:
    def train_first_two_features(self, X, y, weight='uniform'):
        X=X[:, :2]
        self.neigh=neighbors.KNeighborsClassifier(self.neigh_numbers, weights=weight)
        self.neigh.fit(X, y)
        x_min, x_max= X[:, 0].min()-1, X[:, 0].max()+1
        y_min, y_max= X[:, 1].min()-1, X[:, 1].max()+1
        x_, y_ = np.meshgrid(np.arange(x_min, x_max, 0.005),
                             np.arange(y_min, y_max, 0.005))
        Z = self.neigh.predict(np.c_[x_.ravel(), y_.ravel()])
        Z=Z.reshape(x_.shape)
        plt.figure()
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        plt.pcolormesh(x_, y_, Z, cmap=cmap_light)
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(x_.min(), x_.max())
        plt.ylim(y_.min(), y_.max())
        plt.show()

    def test_train_two_features(self, testX, testY, trainX, trainY, weight='uniform'):
        trainX=trainX[:, :2]
        testX=testX[:, :2]
        self.neigh = neighbors.KNeighborsClassifier(self.neigh_numbers, weights=weight)
        self.neigh.fit(trainX, trainY)
        x_min, x_max = trainX[:, 0].min() - 1, trainX[:, 0].max() + 1
        y_min, y_max = trainX[:, 1].min() - 1, trainX[:, 1].max() + 1
        x_, y_ = np.meshgrid(np.arange(x_min, x_max, 0.004),
                             np.arange(y_min, y_max, 0.004))
        Z = self.neigh.predict(np.c_[x_.ravel(), y_.ravel()])
        Z = Z.reshape(x_.shape)
        plt.figure()
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        plt.pcolormesh(x_, y_, Z, cmap=cmap_light)
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        plt.scatter(testX[:, 0], testX[:, 1], c=testY, cmap=cmap_bold,
                    edgecolor='k', s=20)
        plt.xlim(x_.min(), x_.max())
        plt.ylim(y_.min(), y_.max())
        plt.show()
