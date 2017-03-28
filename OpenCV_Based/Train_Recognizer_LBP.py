#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import Train_Common as tc

if __name__ == "__main__":
    # For face recognition we will the the LBPH Face Recognizer 
    recognizer = cv2.createLBPHFaceRecognizer()

    print "-----------------------------"
    path = "./FaceRecData_LBPH_Test"
    format = ".png"

    tc.Gen_List_All_Sample_Image(path, format)
    tc.Gen_List_Label(path, "Label_test.txt")
    
    images, labels = tc.get_images_and_labels_Assign_Format(path, format)

    recognizer = tc.TrainModel(recognizer, images, labels, 'model_LBPH_test.yml')

