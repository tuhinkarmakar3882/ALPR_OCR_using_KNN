# Main.py

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import DetectChars
import DetectPlates
import PossiblePlate
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

###################################################################################################



def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image


def main(all_imgs):

    
    
    blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training

    if blnKNNTrainingSuccessful == False:                               # if KNN training was not successful
        print("\nerror: KNN traning was not successful\n")  # show error message
        return                                                          # and exit program


    
    
    for i in all_imgs:
        
        ret,binary = cv2.threshold(i,127,255,cv2.THRESH_BINARY_INV)
        #ret,binary = cv2.threshold(binary,127,255,cv2.THRESH_OTSU)
        tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        text = tess.image_to_string(binary, lang='eng')
        if(len(text)>0):
            print("Detected Text Raw Image : ", text)
        
        imgOriginalScene  = i
        if imgOriginalScene is None:
            print("\nerror: image not read from file \n\n")
            os.system("pause")                             
            return                                         

        listOfPossiblePlates,listOfPossiblePlates2 = DetectPlates.detectPlatesInScene(imgOriginalScene)            

        listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)  
        
        listOfPossiblePlates2 = DetectChars.detectCharsInPlates(listOfPossiblePlates2)  

        cv2.imshow("imgOriginalScene", imgOriginalScene)             

        if len(listOfPossiblePlates) == 0:                         
            print("\nno license plates were detected\n")   
        else:                                                       
            listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)            
            #for plate_no in range(min(1,len(listOfPossiblePlates))):
            licPlate = listOfPossiblePlates[0]#plate_no]


           # cv2.imshow("sf",licPlate.imgThresh)
            ret , a1=cv2.threshold(licPlate.imgThresh,127,255,cv2.THRESH_OTSU)
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
            print("2>  "+text +"\n") # write license plate text to std out
            text = tess.image_to_string(licPlate.imgThresh, lang='eng')
            print("3>  "+text +"\n")
            text = tess.image_to_string(licPlate.imgThresh2, lang='eng')
            print("4>  "+text +"\n")
            listOfPossiblePlates2.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)            
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            licPlate = listOfPossiblePlates2[0]#plate_no]

            if len(licPlate.strChars) == 0:                 
                print("\nno characters were detected\n") 
                return                                          
            cv2.imshow("2imgPlate", licPlate.imgPlate)           
            #cv2.imshow("2imgThresh", licPlate.imgThresh)
            #cv2.imshow("2imgThresh2", licPlate.imgThresh2)
            drawRedRectangleAroundPlate(imgOriginalScene, licPlate)
            
            tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
            text = tess.image_to_string(licPlate.imgPlate, lang='eng')
            print("5>  "+text +"\n")
            text = tess.image_to_string(licPlate.imgThresh2, lang='eng')
            print("6>  "+text +"\n")
            print("7>  " + licPlate.strChars + "\n")  # write license plate text to std out
            text = tess.image_to_string(licPlate.imgThresh, lang='eng')
            print("8>  "+text +"\n")
            #writeLicensePlateCharsOnImage(imgOriginalScene, licPlate)          
            print("----------------------------------------")
            cv2.imshow("imgOriginalScene", imgOriginalScene)              

            cv2.imwrite("imgOriginalScene.png", imgOriginalScene)          

        # end if else

            cv2.waitKey(0)
    return


###################################################################################################
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


###################################################################################################
def writeLicensePlateCharsOnImage(imgOriginalScene, licPlate):
    ptCenterOfTextAreaX = 0                             
    ptCenterOfTextAreaY = 0

    ptLowerLeftTextOriginX = 0                          
    ptLowerLeftTextOriginY = 0

    sceneHeight, sceneWidth, sceneNumChannels = imgOriginalScene.shape
    plateHeight, plateWidth, plateNumChannels = licPlate.imgPlate.shape

    intFontFace = cv2.FONT_HERSHEY_SIMPLEX                     
    fltFontScale = float(plateHeight) / 30.0                    
    intFontThickness = int(round(fltFontScale * 1.5))       
    textSize, baseline = cv2.getTextSize(licPlate.strChars, intFontFace, fltFontScale, intFontThickness)      

    ( (intPlateCenterX, intPlateCenterY), (intPlateWidth, intPlateHeight), fltCorrectionAngleInDeg ) = licPlate.rrLocationOfPlateInScene

    intPlateCenterX = int(intPlateCenterX)              
    intPlateCenterY = int(intPlateCenterY)

    ptCenterOfTextAreaX = int(intPlateCenterX)         

    if intPlateCenterY < (sceneHeight * 0.75):                                                  
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) + int(round(plateHeight * 1.6))      
    else:                                                                                      
        ptCenterOfTextAreaY = int(round(intPlateCenterY)) - int(round(plateHeight * 1.6))      

    textSizeWidth, textSizeHeight = textSize               

    ptLowerLeftTextOriginX = int(ptCenterOfTextAreaX - (textSizeWidth / 2))         
    ptLowerLeftTextOriginY = int(ptCenterOfTextAreaY + (textSizeHeight / 2))            
    cv2.putText(imgOriginalScene, licPlate.strChars, (ptLowerLeftTextOriginX, ptLowerLeftTextOriginY), intFontFace, fltFontScale, SCALAR_YELLOW, intFontThickness)


###################################################################################################
if __name__ == "__main__":
    img=[]
    for i in range(140,237):
        filename=str(i)+".jpg"
        path="training_images/"
        x_4=[]
        c_img=(cv2.imread(path+filename))
        x_4.append(c_img)
        main(x_4)
        img.append(cv2.imread(path+filename))
        
        #tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
        #text = tess.image_to_string(cv2.imread(path+filename), lang='eng')
        #print("TESS SAYS : ", text)
    #main(img)
    
    #with open('INP2.json') as f: 
    #    d = json.load(f) 
    #nycphil = json_normalize(d['dum']) 
    #a = nycphil.content.head(1)
    #img=[]
    #for link in nycphil.content:
    #    img.append(url_to_image(link))
    #main(img)
