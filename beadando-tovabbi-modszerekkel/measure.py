from skimage.measure import label, regionprops
import numpy as np
import cv2

#A legnagyobb regio szegmentalasa a kepbol
#-> azert szukseges, mert elofordult par kepnel, hogy maradtak kisebb regiok a zaj miatt
def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # legyen legalabb egy regio
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1

    return largestCC

#eredeti->szurkearnyalat+blur->kuszoboles->closing->otsu + kuszboles->legnagyobb regio->tulajdonsga kinyeres
def process(image_path):
    #eredeti kep
    image = cv2.imread(image_path)

    #szurkearnyalatos kep 3x3-mas szuressel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (3, 3))

    #A kep kuszobolese, minden ertek ami nem fekete (0), az legyen feher, adaptiv otsu algoritmus segitsegevel
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # closing: dilatacio es erozio 3x3-mas kernellel; 2x-szor
    #->dilatacio: befedi a kisebb lyukakat a kepen (vastagit)
    #->erozio: vekonyit
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    #ujboli szures az otsu kuszoberteket hasznalva most mar
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret, fg = cv2.threshold(dist_transform, 0.02 * dist_transform.max(), 255, 0)

    #legnagyobb regio szegmentalasa
    largest = getLargestCC(fg)
    label_img = label(largest)
    regions = regionprops(label_img)

    #Egy regio letezik csak mindossze, implementacios konnyites miatt van for ciklus hasznalva
    #tulajdonsgaok kinyerese vektorba
    out=[]
    for props in regions:
        ecc = props.eccentricity
        per = props.perimeter
        x0, y0 = props.centroid
        extent=props.extent
        axis_length=props.major_axis_length
        out=[ecc, per, x0, y0, extent, axis_length]

    return out
