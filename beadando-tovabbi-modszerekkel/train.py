import cv2
import pickle
from sklearn.linear_model import SGDClassifier as SGD
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
class train(object):

    def __init__(self):
        pass

    def train(self, X, y):
        steps = [('pca', PCA()), ('clf', SVC(kernel='rbf'))]
        pipe = Pipeline(steps)
        pca__n_components=[6]
        n_splits = 6
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
        clf__gamma = np.logspace(-4, -2, 3)  # [.0001, .001, .01]
        clf__C = np.logspace(0, 2, 3)  # [1, 10, 100]
        grid_params = dict(pca__n_components=pca__n_components,
                           clf__gamma=clf__gamma,
                           clf__C=clf__C)
        self.grid = GridSearchCV(pipe, grid_params, cv=cv, refit=True, n_jobs=-1, scoring='f1_micro')
        self.grid.fit(X, y)
        with open('out/clf.pkl', 'wb') as f:
            f.flush()
            pickle.dump(self.grid, f)

    def load(self):
        filename='out/clf.pkl'
        with open(filename, 'rb') as f:
            self.grid=pickle.load(f)


    def predict(self, X):
        list=[X]
        return self.grid.predict(list)