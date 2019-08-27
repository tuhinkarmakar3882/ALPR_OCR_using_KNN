import cv2
import numpy as np
import math
import Main
import random

import Perform_Preprocess
import FindMactchingChars
import ScanForAllPossiblePlates
import ScanForProbableChars

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def helperFunc(imgOriginalScene,imgThreshScene):
    list_of_possible_plates = []
    list_of_possible_chars_in_scene = findPossibleCharsInScene(imgThreshScene)
    list_of_lists_of_matching_chars_in_scene = FindMactchingChars.findListOfListsOfMatchingChars(list_of_possible_chars_in_scene)
    for listOfMatchingChars in list_of_lists_of_matching_chars_in_scene:
        possible_plate = extract_plate(imgOriginalScene, listOfMatchingChars)
        if possible_plate.imgPlate is not None:
            list_of_possible_plates.append(possible_plate)
    print("\n" + str(len(list_of_possible_plates)) + " possible plates found")
    return list_of_possible_plates


def detectPlatesInScene(img_Original_Scene):
    list_of_possible_plates1 = []
    list_of_possible_plates2 = []

    height, width, num_channels = img_Original_Scene.shape

    img_grayscale_scene = np.zeros((height, width, 1), np.uint8)
    img_thresh_scene = np.zeros((height, width, 1), np.uint8)
    img_contours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    img_grayscale_scene, img_thresh_scene, img_thresh_scene2 = Perform_Preprocess.preprocess(img_Original_Scene)
    
    list_of_possible_plates1 = helperFunc(img_Original_Scene, img_thresh_scene)
    list_of_possible_plates2 = helperFunc(img_Original_Scene, img_thresh_scene2)

    return list_of_possible_plates1,list_of_possible_plates2


def findPossibleCharsInScene(img_Thresh):
    list_of_possible_chars = []

    int_count_of_possible_chars = 0

    img_thresh_copy = img_Thresh.copy()

    contours, npa_hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    height, width = img_Thresh.shape
    img_contours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):                       

        possible_char = ScanForProbableChars.PossibleChar(contours[i])

        if FindMactchingChars.checkIfPossibleChar(possible_char):
            int_count_of_possible_chars = int_count_of_possible_chars + 1
            list_of_possible_chars.append(possible_char)

    return list_of_possible_chars


def extract_plate(imgOriginal, listOfMatchingChars):
    possible_plate = ScanForAllPossiblePlates.PossiblePlate()

    listOfMatchingChars.sort(key = lambda matchingChar: matchingChar.intCenterX)

            # calculate the center point of the plate
    flt_plate_center_x = (listOfMatchingChars[0].intCenterX + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterX) / 2.0
    flt_plate_center_y = (listOfMatchingChars[0].intCenterY + listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY) / 2.0

    pt_plate_center = flt_plate_center_x, flt_plate_center_y

            # calculate plate width and height
    int_plate_width = int((listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectX + listOfMatchingChars[len(listOfMatchingChars) - 1].intBoundingRectWidth - listOfMatchingChars[0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    int_total_of_char_heights = 0

    for matchingChar in listOfMatchingChars:
        int_total_of_char_heights = int_total_of_char_heights + matchingChar.intBoundingRectHeight

    flt_average_char_height = int_total_of_char_heights / len(listOfMatchingChars)

    int_plate_height = int(flt_average_char_height * PLATE_HEIGHT_PADDING_FACTOR)

            # calculate correction angle of plate region
    flt_opposite = listOfMatchingChars[len(listOfMatchingChars) - 1].intCenterY - listOfMatchingChars[0].intCenterY
    flt_hypotenuse = FindMactchingChars.distanceBetweenChars(listOfMatchingChars[0], listOfMatchingChars[len(listOfMatchingChars) - 1])
    flt_correction_angle_in_rad = math.asin(flt_opposite / flt_hypotenuse)
    flt_correction_angle_in_deg = flt_correction_angle_in_rad * (180.0 / math.pi)

            # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possible_plate.rrLocationOfPlateInScene = ( tuple(pt_plate_center), (int_plate_width, int_plate_height), flt_correction_angle_in_deg )

    rotation_matrix = cv2.getRotationMatrix2D(tuple(pt_plate_center), flt_correction_angle_in_deg, 1.0)

    height, width, num_channels = imgOriginal.shape

    img_rotated = cv2.warpAffine(imgOriginal, rotation_matrix, (width, height))
    img_cropped = cv2.getRectSubPix(img_rotated, (int_plate_width, int_plate_height), tuple(pt_plate_center))

    possible_plate.imgPlate = img_cropped
    return possible_plate