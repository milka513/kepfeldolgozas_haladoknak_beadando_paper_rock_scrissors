import  measure
import ProcessImage as image
from train import train
import cv2
from train_k_neighbours import train as train_neigh
import matplotlib.pyplot as plt
from DecisionTree import train as DT

#legjobb K ertek megtalasa a k-legkozelebbi szomszednal
def getBestKValue(maxK, two_feature=False, features=['ecc', 'extent']):
    im=image.ProcessImages('train', two_value=two_feature, list_=features)
    X, Y=im.make_all()
    max = 0.0
    index = 3
    x_list = []
    y_list = []
    #iteracio
    for x in range(3, maxK, 1):
        # for weight in ['uniform', 'distance']:
        print(' negighbours: ', 'distance', ' ', x)
        t2 = train_neigh(x, first_two=True, features=features)
        t2.train(X, Y)
        # t2.load(x, weight)
        v = t2.valid()
        #validacios halmazon elert legjobb eredmeny
        if (v > max):
            max = v
            index = x
            print(v)
        x_list.append(x)
        y_list.append(v)

    #megjelenitese
    plt.plot(x_list, y_list)
    plt.show()

    print('maximum: ', max, ' neighbours: ', index)
    return (max, index)

#2 parameteres feature megjelenitese a k-legkozelebbi szomszed szerint
def _train_neigh(value, feauters=['x0', 'y0']):
    #legkozelebbi szomszed
    t=train_neigh(value, first_two=True, features=feauters)
    #train halmaz
    im = image.ProcessImages('train', list_=feauters)
    trainX, trainY = im.make_all()
    im2=image.ProcessImages('test', list_=feauters)
    testX, testY=im2.make_all()
    #test halmaz megjelenitese a train halmazon
    t.test_train_two_features(testX, testY, trainX, trainY)
    predictY=[]
    #kapott ertekek a k szomszed szerint
    predictY=t.predict(testX)

    #vegigiteralas a kepeken: melyiket nem talalta el+megjelenites
    for i, j, z in zip(predictY, testY, testX):
        if i!=j:
            print('coordinates: (',z[0],' ', z[1],') predicted:  ',i, ' original ', j)
            im2.find_pictures(z)
    cv2.waitKey(0)
def train_neigh_():
    t = train_neigh(80)
    im = image.ProcessImages('train')
    X, Y = im.make_all()
    t.train_first_two_features(X, Y)
    max = 0.0
    index = 3
    x_list = []
    y_list = []
    # t3 = train_neigh(4)
    # print('test: ', t3.test())

    for x in range(3, 100, 1):
        # for weight in ['uniform', 'distance']:
        print(' negighbours: ', 'distance', ' ', x)
        t2 = train_neigh(x, first_two=True, features=['x0', 'y0'])
        #t2 = train_neigh(x, first_two=True)
        t2.train(X, Y)
        # t2.load(x, weight)
        v = t2.valid()
        if (v > max):
            max = v
            index = x
            print(v)
        x_list.append(x)
        y_list.append(v)

    plt.plot(x_list, y_list)
    plt.show()

    print('maximum: ', max, ' neighbours: ', index)


def main():

    #SVM:
    t = train()
    #im = image.ProcessImages('train')
    #X, Y = im.make_all()
    #t.train(X, Y)
    #rbf:

    print('SVM clf:')
    t.load(name='clf_rbf.pkl')
    print(t.predict(measure.process('test3.png')))
    print(t.test())

    #linear:
    print('SVM linear:')
    t.load(name='clf_linear.pkl')
    print(t.predict(measure.process('test3.png')))
    print(t.test())

    #k szomszed:
    #osszes feature-ra
    #getBestKValue(400)
    print('osszes feature k neighbours:')
    im = image.ProcessImages('train')
    X, Y = im.make_all()
    #train halmazon elert eredmeny:
    #maximum:  0.7032590051457976  neighbours:  183
    t2 = train_neigh(183)
    t2.train(X,Y)
    #teszt halmazon elert eredmeny:
    #elert eredmeny:  0.5483870967741935
    print('elert eredmeny: ', t2.test())

    #dontesi fa:
    print('dontesi fa: ')
    tree = DT()
    tree.train(X, Y)
    print('elert eredmeny: ',tree.test())
    print(tree.valid())
    tree.show(X, Y)


    #ecc-extent:
    print('ket feature: ecc, extent:')
    #getBestKValue(100, two_feature=True, features=['ecc', 'extent'])
    t3=train_neigh(80, first_two=True, features=['ecc', 'extent'])
    t3.train(X, Y)
    print('elert eredmeny: ', t3.test())
    #elert eredmeny:  0.4946236559139785
    t3.train_first_two_features(X, Y)
    #_train_neigh(80, feauters=['ecc', 'extent'])

    #x0-y0:
    print('ket feature: x0, y0:')
    #getBestKValue(100, two_feature=True, features=['x0', 'y0'])
    t4 = train_neigh(54, first_two=True, features=['x0', 'y0'])
    t4.train(X, Y)
    print('elert eredmeny: ', t4.test())
    #elert eredmeny:  0.3333333333333333
    t4.train_first_two_features(X, Y)
    _train_neigh(54, feauters=['x0', 'y0'])

    #train_neigh_()
    im2=image.ProcessImages('test')
    _train_neigh(80, feauters=['ecc', 'extent'])
    getBestKValue(100, two_feature=True, features=['x0', 'y0'])
    tree=DT()
    #im = image.ProcessImages('train')
    #X, Y = im.make_all()
    tree.train(X, Y)
    print(tree.test())
    print(tree.valid())
    tree.show(X, Y)


if __name__ == '__main__':
    main()