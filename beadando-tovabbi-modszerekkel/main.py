import  measure
import ProcessImage as image
from train import train

def main():
    im=image.ProcessImages('train')
    X, Y=im.make_all()
    t=train()
    t.train(X, Y)
    print(t.predict(measure.process('test1.png')))

if __name__ == '__main__':
    main()