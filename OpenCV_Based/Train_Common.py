#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image

def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
    
    print  os.listdir(path)
    
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        print  image_path
    
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

    
def get_images_and_labels_Auto(path):
    onlydir = [ f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f)) ]

    print str(len(onlydir)) + " folder"
    
    LabelCnt = 0
 
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []

    for folder in onlydir:
        print folder
        folder_path_tmp = os.path.join(path, folder)

        image_paths = [ f for f in os.listdir(folder_path_tmp) if f.endswith('.jpg') ]
        print str(len(image_paths)) + " image"

        for image in image_paths:
            image_path_tmp = os.path.join(folder_path_tmp, image)
            print image_path_tmp
            img = cv2.imread(image_path_tmp,0)
            images.append(img)
            labels.append(LabelCnt)
            
        LabelCnt = LabelCnt+1
        
    # return the images list and labels list
    return images, labels
    
def get_images_and_labels_Assign_Format(path, format):
    onlydir = [ f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f)) ]

    print str(len(onlydir)) + " folder"
    
    LabelCnt = 0
 
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []

    for folder in onlydir:
        print folder
        folder_path_tmp = os.path.join(path, folder)

        image_paths = [ f for f in os.listdir(folder_path_tmp) if f.endswith(format) ]
        print str(len(image_paths)) + " image"

        for image in image_paths:
            image_path_tmp = os.path.join(folder_path_tmp, image)
            print image_path_tmp
            img = cv2.imread(image_path_tmp,0)
            images.append(img)
            labels.append(LabelCnt)
            
        LabelCnt = LabelCnt+1
        
    # return the images list and labels list
    return images, labels
            
def List_All_Sample_Image(path):
    onlydir = [ f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f)) ]

    print str(len(onlydir)) + " folder"
    #print onlydir

    for folder in onlydir:
        print folder
        folder_path_tmp = os.path.join(path, folder)
        #print folder_path_tmp
        image_paths = [ f for f in os.listdir(folder_path_tmp) if f.endswith('.jpg') ]
        print str(len(image_paths)) + " image"
        #print image_paths
        for image in image_paths:
            image_path_tmp = os.path.join(folder_path_tmp, image)
            print image_path_tmp
            
def Gen_List_All_Sample_Image(path, format, FileName="sample.csv"):
    onlydir = [ f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f)) ]
    LabelCnt = 0
    fo=open(FileName, 'w')

    print str(len(onlydir)) + " folder"
    #print onlydir

    for folder in onlydir:
        print folder
        folder_path_tmp = os.path.join(path, folder)
        #print folder_path_tmp
        image_paths = [ f for f in os.listdir(folder_path_tmp) if f.endswith(format) ]
        print str(len(image_paths)) + " image"
        #print image_paths
        for image in image_paths:
            image_path_tmp = os.path.join(folder_path_tmp, image)
            print image_path_tmp
            fo.write(image_path_tmp)
            fo.write(",")
            fo.write(str(LabelCnt)+"\n")
        LabelCnt = LabelCnt+1
            
    fo.close()
    
def Gen_List_Label(path, FileName="Label_test.txt"):
    onlydir = [ f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f)) ]
    LabelCnt = 0
    fo=open(FileName, 'w')
    for folder in onlydir:
        if LabelCnt == len(onlydir)-1:
            fo.write(folder + " " + str(LabelCnt))
        else:
            fo.write(folder + " " + str(LabelCnt)+"\n")
        LabelCnt = LabelCnt+1     
    fo.close()
    
def Read_List_Label(FileName="Label_test.txt"):
    fo=open(FileName, 'r')
    
    # images will contains face images
    name = []
    # labels will contains the label that is assigned to the image
    labels = []
    
    for line in fo.readlines():
        tmp = line.split()
        name.append(tmp[0])
        labels.append(tmp[1])
    fo.close()
    
    # return the name list and labels list
    return name, labels

def TrainModel(model, img, label, OutName=None):
    if OutName == None:
        OutName = 'model_default.yml'
        
    # Learn the model. Remember our function returns Python lists,
    # so we use np.asarray to turn them into NumPy lists to make
    # the OpenCV wrapper happy:
    model.train(np.asarray(img), np.asarray(label))

    model.save(OutName)
    
    return model
    
if __name__ == "__main__":          
    print "-----------------------------"
    path = "./FaceRecData_LBPH_Test"
    format = ".png"

    #List_All_Sample_Image(path)
    #images, labels = get_images_and_labels_Auto(path)

    images, labels = get_images_and_labels_Assign_Format(path, format)


    size = len(images)

    cnt = 0
    while(cnt < size):
        print
        cv2.namedWindow("traning set",cv2.WINDOW_AUTOSIZE)
        cv2.imshow("traning set", images[cnt])
        #cv2.imwrite("%s.png" % (str(cnt)), images[cnt])
        cv2.waitKey(500)
        cnt = cnt+1
        
        
    name, labels = Read_List_Label()
    
    print name
    print labels



