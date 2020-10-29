from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
    #constructing a blob from the grabbed frame
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300)),
        (104.0, 177.0, 123.0)

    # pass the blob through our model for face detection
    faceNet.setInput(blob)
    detections = faceNet.forward()

    #initialize our list of faces, their corresponding
    # and the list of predictions from our face mask network
    face = []
    locs = []
    preds = []

    #looping over the detections
    for i in range(0, detections.shape[2]):
        #extracting the probability from face detection
        confidence = detection[0, 0, i, 2]

        #setting a minimum confidence threshold
        if confidence > args["confidence"]:
            #compute the (x,y) coodrinate for the box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #ensure the bounding boxes fall within the dimension
            #of the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            #extract the face ROI, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            #add the face and bounding boxes to the list
            faces.append(face)
            locs.append(startX, startY, endX, endY)

    # only to make a prediction if at least one face was  detected
    if len(faces) > 0:
        #for faster inference we'll make a batch prediction on *all*
        #faces at the same time rather than one-by-one predictions
        #in the above `for` loop
        faces = np.array(faces, dtype = "float32")
        preds = maskNet.predict(faces, batch_size = 32)

    #return a 2-tuple of face loc
    return(locs, preds)

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type = str,
    default= " face_detector",
    help = "path to face detector model directory")
