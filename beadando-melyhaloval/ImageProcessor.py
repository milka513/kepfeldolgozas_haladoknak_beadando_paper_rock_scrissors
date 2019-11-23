from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import cv2

class image(object):

    def __init__(self, image_path, INPUT_SIZE_X=300, INPUT_SIZE_Y=200):
        self.image_path=image_path
        #img=load_img(image_path, target_size=(INPUT_SIZE_X, INPUT_SIZE_Y)).convert('RGB')
        self.img=cv2.imread(image_path)
        self.img=cv2.resize(self.img, (INPUT_SIZE_Y, INPUT_SIZE_X))
        #self.img=img_to_array(img)

    def get_image(self):
        #img2=cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Background area using Dialation
        bg = cv2.dilate(closing, kernel, iterations=1)

        # Finding foreground area
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
        ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)
        self.img=fg
        return fg