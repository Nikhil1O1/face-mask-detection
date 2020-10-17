from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

#constructing the argument parser and parsing the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required= True,
    help = "path to input dataset")
ap.add_argument("-p", "--plot", type=str, default= " plot.png",
    help = "path to output loss/accuracy plot")
ap.add_argument("-m", "--model", type = str,
    default= "mask_detector.model",
    help = "path to output face mask detector model")
args = vars(ap.parse_args())

# initialize few ML variables
INIT_LR = 1e-4
EPOCHS = 20
BS = 32  #batch size

#grab the list of images in our dataset, init the data
imagePaths = list(path.list_images(args["dataset"]))
data = []
labels = []

#loop over the image paths
for imgPath in imagePaths:
    # extracting image class labels
    label = imagePath.split(os.path.sep)[-2]

    #load the input image (224 * 224) and preprocess it
    image = load_img(imagePath, target_size = (224* 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    #update the data and labels list
    data.append(image)
    labels.append(label)
print(data, 'next', labels)
