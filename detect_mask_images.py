


#importing necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import argparse

#creating the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "path to input image")
ap.add_argument("-f", "--face",type = str,
    default = "face_detector",
    help = "path to face detector model directory")
ap.add_argument("-m", "--model", type = str,
    default = "mask_detector.model",
    help = "path to trained model")
ap.add_argument("-c", "--confidence", type = float,
    default = 0.5,
    help = "minimum probability to filter weak detection")
args = vars(ap.parse_args())

# load our serialized face detector model form the disk
print("Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightPath = os.path.sep.join([args["face"],
    "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightPath)

#loading the face mask detector from the disk
print("loading the face mask detector")
model = load_model(args["model"])

#loading and cloning the image, them pickingu up the spatial dimensions
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

#construct a blob from the image
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
    (104.0, 177.0, 123.0))

# pass the blob through the network to obtain face detections
print("computing the face detections")
net.setInput(blob)
detections = net.forward()

#loop over the detections
for i in range(0, detections.shape[2]):
    #extract the confidence (i.e., probability) associated with 
    #the face detections
    confidence = detections[0, 0, i,2]

    #setting a confidence threshold
    if confidence> args["confidence"]:
        #computing the (x,y)-coordinates of the selection box for 
        #the object
        box = detections[0, 0, i, 3:7]* np.array([w, h, w,h])
        (startX, startY, endX, endY) = box.astype("int")

        #ensure the bounding boxes fall within the dimensions
        #of the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h -1, endY))

        #extracting the face Region of Intrest, conveting it from BGR to
        # RGB, and preprocessing it
        face = image[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis = 0)

        #pass the face through model to determin wether it has 
        #mask or not
        (mask, withoutMask) = model.predict(face)[0]

        #determine the class label color we will be using to 
        #draw the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        #adding the probability in the info box
        label = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        #include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask)* 100)

        #display the label and bounding box rectangle
        #on the output frame
        cv2.putText(image, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

#displaying the output
cv2.imshow("Output", image)
cv2.waitkey(0)