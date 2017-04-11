#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import Train_Common as tc
import sys, getopt

help_message = '''
USAGE: Train_Recognizer_LBP.py [--path <Path>]  [--model-name <ModelName>] [--label-name <LabelName>]
'''

if __name__ == "__main__":
    print (help_message)

    args, opts = getopt.getopt(sys.argv[1:], '', ['path=', 'model-name=', 'label-name='])
    
    # For face recognition we will the the LBPH Face Recognizer 
    recognizer = cv2.createLBPHFaceRecognizer()

    print "-----------------------------"
    
    args = dict(args)
    path = args.get('--path', "./FaceRecData_LBPH_Test")
    ModelName = args.get('--model-name', "Model_default.yml")
    LabelName = args.get('--label-name', "Label_default.txt")

    format = ".png"

    print "Gen list of all sample image.....\n"
    tc.Gen_List_All_Sample_Image(path, format)
    
    print "Gen list of label.....\n"
    tc.Gen_List_Label(path, LabelName)
    
    print "Assign image and label list.....\n"
    images, labels = tc.get_images_and_labels_Assign_Format(path, format)

    print "Train face reconition model.....\n"
    recognizer = tc.TrainModel(recognizer, images, labels, ModelName)

