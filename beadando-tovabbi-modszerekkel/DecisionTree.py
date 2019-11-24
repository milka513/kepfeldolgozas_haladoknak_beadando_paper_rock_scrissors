from sklearn import tree
import ProcessImage as image
import matplotlib.pyplot as plt

class train(object):
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()

    def train(self, X, Y):

        self.clf.fit(X, Y)

    def predict(self, X):
        list=[X]

        return self.clf.predict(list)

    def show(self, X, Y):
        plt.figure()
        tree.plot_tree(self.clf.fit(X, Y))
        plt.show()

    def test(self):
        im = image.ProcessImages('test')
        X, Y = im.make_all()
        return self.clf.score(X, Y)

    def valid(self):
        im = image.ProcessImages('validation')
        X, Y = im.make_all()
        return self.clf.score(X, Y)
