from skimage.measure import label, regionprops
import numpy as np
import cv2

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

# [ecc,extent,minor_axis_length,major_axis_length, per, x0, y0]
def set(value):
    list=[]
    for v in value:
        list.append(get_value(v))
    return list



def get_value(val):
    if val=='ecc':
        return 0
    if val=='extent':
        return 1
    if val=='minor_axis_length':
        return 2
    if val=='major_axis_length':
        return 3
    if val=='per':
        return 4
    if val=='x0':
        return 5
    if val=='y0':
        return 6

def find_picture(image_path, par,two_values=True, parameters=['ecc', 'extent']):
    Y=process(image_path, two_values=two_values, list=parameters)
    if set(Y)==set(par):
        image=cv2.imread(image_path)
        cv2.imshow('picture', image)



def process(image_path, two_values=False, list=[]):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))
    edged = cv2.Canny(gray, 100, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imshow('Canny Edges After Contouring', edged)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    # dilatacio es erozio

    # Background area using Dialation
    bg = cv2.dilate(closing, kernel, iterations=1)

    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)

    #cv2.imshow('image', fg)

    #cv2.waitKey(0)

    largest = getLargestCC(fg)

    ki = largest * fg
    #cv2.imshow('largest', ki)

    label_img = label(largest)
    regions = regionprops(label_img)

    #cv2.waitKey(0)


    out=[]
    for props in regions:
        ecc = props.eccentricity
        per = props.perimeter
        x0, y0 = props.centroid
        extent=props.extent
        major_axis_length=props.major_axis_length
        minor_axis_length =props.minor_axis_length
		
        #print(ecc)
        #print(per)
        #print(x0, y0)
        out=[ecc,extent,minor_axis_length,major_axis_length, per, x0, y0]
    #cv2.destroyAllWindows()
    out2=[]
    if two_values:
        for v in list:
            out2.append(out[get_value(v)])
    else:
        out2=out
    out=out2
    return out





def run():
    image = cv2.imread('testscissors01-00.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(3,3))
    edged = cv2.Canny(gray, 100, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Canny Edges After Contouring', edged)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    #dilatacio es erozio

    # Background area using Dialation
    bg = cv2.dilate(closing, kernel, iterations=1)

    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)

    cv2.imshow('image', fg)

    cv2.waitKey(0)

    largest = getLargestCC(fg)

    ki = largest * fg
    cv2.imshow('largest', ki)

    label_img = label(largest)
    regions = regionprops(label_img)

    cv2.waitKey(0)

    for props in regions:
        ecc = props.eccentricity
        per = props.perimeter
        x0, y0 = props.centroid

        #print(ecc)
        #print(per)
        #print(x0,y0)


    cv2.destroyAllWindows()
