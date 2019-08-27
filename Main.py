# Main.py

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import FindMactchingChars
import FindMatchingPlates
import ScanForAllPossiblePlates
import pytesseract as tess
import json
import pandas as pd
from pandas.io.json import json_normalize
import urllib.request

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)
showSteps = False


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


def main(all_imgs):
    KNN_is_Trained_successfully = FindMactchingChars.loadKNNDataAndTrainKNN()  # attempt KNN training

    if not KNN_is_Trained_successfully:
        print("\nerror: KNN traning was not successful\n")  # show error message
        exit(0)

    for this_image in all_imgs:

        _, binary = cv2.threshold(this_image, 127, 255, cv2.THRESH_BINARY_INV)
        # ret,binary = cv2.threshold(binary,127,255,cv2.THRESH_OTSU)
        tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        text = tess.image_to_string(binary, lang='eng')
        if len(text) > 0:
            print("Detected Text Raw Image : ", text)

        imgOriginalScene = this_image
        if imgOriginalScene is None:
            print("\nerror: image not read from file \n\n")
            os.system("pause")
            exit(0)

        listOfPossiblePlates, listOfPossiblePlates2 = FindMatchingPlates.detectPlatesInScene(imgOriginalScene)
        listOfPossiblePlates = FindMactchingChars.detectCharsInPlates(listOfPossiblePlates)
        listOfPossiblePlates2 = FindMactchingChars.detectCharsInPlates(listOfPossiblePlates2)

        cv2.imshow("imgOriginalScene", imgOriginalScene)

        if len(listOfPossiblePlates) == 0:
            print("\nno license plates were detected\n")
        else:
            listOfPossiblePlates.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)
            # for plate_no in range(min(1,len(listOfPossiblePlates))):
            licPlate = listOfPossiblePlates[0]  # plate_no]

            # cv2.imshow("sf",licPlate.imgThresh)
            _, a1 = cv2.threshold(licPlate.imgThresh, 127, 255, cv2.THRESH_OTSU)
            # plt.imshow(licPlate.imgThresh)

            text = tess.image_to_string(a1, lang='eng')
            print("TESS SAYS (THRESH_AUTO): ", text)

            cv2.imshow("imgPlate", licPlate.imgPlate)
            #cv2.imshow("imgThresh", licPlate.imgThresh)
            #cv2.imshow("imgThresh2", licPlate.imgThresh2)

            if len(licPlate.strChars) == 0:
                print("\nno characters were detected\n\n")
                return

            drawGreenRectangleAroundPlate(imgOriginalScene, licPlate)

            print("\n Probable license plate read from image = ")
            print("1>  " + licPlate.strChars + "\n")
            text = tess.image_to_string(licPlate.imgPlate, lang='eng')
            print("2>  " + text + "\n")  # write license plate text to std out
            text = tess.image_to_string(licPlate.imgThresh, lang='eng')
            print("3>  " + text + "\n")
            text = tess.image_to_string(licPlate.imgThresh2, lang='eng')
            print("4>  " + text + "\n")

        if len(listOfPossiblePlates2) == 0:
            print("\nno license plates were detected\n")
        else:
            listOfPossiblePlates2.sort(key=lambda possiblePlate: len(possiblePlate.strChars), reverse=True)

            licPlate = listOfPossiblePlates2[0]  # plate_no]

            if len(licPlate.strChars) == 0:
                print("\nno characters were detected\n")
                return
            cv2.imshow("2imgPlate", licPlate.imgPlate)
            #cv2.imshow("2imgThresh", licPlate.imgThresh)
            #cv2.imshow("2imgThresh2", licPlate.imgThresh2)
            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)

            tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
            text = tess.image_to_string(licPlate.imgPlate, lang='eng')
            print("5>  " + text + "\n")
            text = tess.image_to_string(licPlate.imgThresh2, lang='eng')
            print("6>  " + text + "\n")
            print("7>  " + licPlate.strChars + "\n")
            text = tess.image_to_string(licPlate.imgThresh, lang='eng')
            print("8>  " + text + "\n")

            # writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)
        print("----------------------------------------")
        cv2.imshow("imgOriginalScene", imgOriginalScene)
        cv2.imwrite("imgOriginalScene.png", imgOriginalScene)
        cv2.waitKey(0)
    return


def drawRedRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_RED, 2)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_RED, 2)


def drawGreenRectangleAroundPlate(imgOriginalScene, licPlate):
    p2fRectPoints = cv2.boxPoints(licPlate.rrLocationOfPlateInScene)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), SCALAR_GREEN, 3)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), SCALAR_GREEN, 3)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), SCALAR_GREEN, 3)
    cv2.line(imgOriginalScene, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), SCALAR_GREEN, 3)


def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    pt_center_of_text_area_x = 0
    pt_center_of_text_area_y = 0

    pt_lower_left_text_origin_x = 0
    pt_lower_left_text_origin_y = 0

    scene_height, scene_width, scene_num_channels = imgOriginalScene.shape
    plate_height, plate_width, plate_num_channels = licPlate.imgPlate.shape

    int_font_face = cv2.FONT_HERSHEY_SIMPLEX
    flt_font_scale = float(plate_height) / 30.0
    int_font_thickness = int(round(flt_font_scale * 1.5))
    text_size, baseline = cv2.getTextSize(licPlate.strChars, int_font_face, flt_font_scale, int_font_thickness)

    ((int_plate_center_x, int_plate_center_y), (intPlateWidth, intPlateHeight),
     fltCorrectionAngleInDeg) = licPlate.rrLocationOfPlateInScene

    int_plate_center_x = int(int_plate_center_x)
    int_plate_center_y = int(int_plate_center_y)

    pt_center_of_text_area_x = int(int_plate_center_x)

    if int_plate_center_y < (scene_height * 0.75):
        pt_center_of_text_area_y = int(round(int_plate_center_y)) + int(round(plate_height * 1.6))
    else:
        pt_center_of_text_area_y = int(round(int_plate_center_y)) - int(round(plate_height * 1.6))

    text_size_width, text_size_height = text_size

    pt_lower_left_text_origin_x = int(pt_center_of_text_area_x - (text_size_width / 2))
    pt_lower_left_text_origin_y = int(pt_center_of_text_area_y + (text_size_height / 2))
    cv2.putText(imgOriginalScene, licPlate.strChars, (pt_lower_left_text_origin_x, pt_lower_left_text_origin_y), int_font_face,
                flt_font_scale, SCALAR_YELLOW, int_font_thickness)


###################################################################################################
if __name__ == "__main__":
    user_providing_path = str(input("Would you like to input custom image path? Y or N :"))
    user_providing_path = user_providing_path.lower()
    if(user_providing_path == 'y'):
        while True:
            print("The File Should Be in the Same Working Directory")
            path = str(input("Enter the file name : "))
            x_4 = []
            c_img = (cv2.imread(path))
            x_4.append(c_img)
            main(x_4)
    else:
        print("Proceeding with Default config...")
        #img = []
        start_pos = abs(int(input("Enter Starting Pos [for default, enter 0] :")))
        end_pos = abs(int(input("Enter Ending Pos [for default, enter 237] :")))
        end_pos+=1
        for i in range(start_pos,end_pos):
            filename = str(i) + ".jpg"
            path = "training_images/"
            img=[]
            c_img = (cv2.imread(path + filename))
            img.append(c_img)
            main(img)
            #img.append(cv2.imread(path + filename))

            # tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
            # text = tess.image_to_string(cv2.imread(path+filename), lang='eng')
            # print("TESS SAYS : ", text)
        # main(img)

    # with open('INP2.json') as f:
    #    d = json.load(f) 
    # nycphil = json_normalize(d['dum'])
    # a = nycphil.content.head(1)
    # img=[]
    # for link in nycphil.content:
    #    img.append(url_to_image(link))
    # main(img)
