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
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

#loop over the image paths
for imagePath in imagePaths:
    # extracting image class labels
    label = imagePath.split(os.path.sep)[-2]

    #load the input image (224 * 224) and preprocess it
    image = load_img(imagePath, target_size = (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)

    #update the data and labels list
    data.append(image)
    labels.append(label)

#converting data and labels to numoy array
data = np.array(data, dtype = "float32")
labels = np.array(labels)

#doing one hot encoding
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#creating training and testing data sets
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.20, stratify=labels, random_state = 42)

#constructing training image generator

aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
    width_shift_range= 0.2, height_shift_range= 0.2,  shear_range= 0.15,
    horizontal_flip=True, fill_mode="nearest")

#loading MobilNetV2 network, we will leave the fc layerd head#
#This is the base model
baseModel = MobileNetV2(weights="imagenet", include_top=False,
    input_tensor=Input(shape=(224, 224, 3)))

#constructing the head of the model to be placed on the previous base
#model

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

#placing the head over top of the base model
model = Model(inputs= baseModel.input, outputs = headModel)

#freezing all the layers of base model except head layer
for layer in baseModel.layers:
    layer.trainable = False

#compile our model
print("COMPILING MODEL....")
opt = Adam(lr = INIT_LR, decay = INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer = opt,
    metrics = ["accuracy"])

#training the network
#but as the other layers are frozen, we train only the head

H = model.fit(
    aug.flow(trainX,trainY, batch_size = BS),
    steps_per_epoch = len(trainX) // BS,
    validation_data = (testX, testY),
    validation_steps = len(testX) // BS,
    epochs = EPOCHS)

#making predcitons on the test set
print("evaluating network")
predIdxs = model.predict(testX, batch_size = BS)

#finding the label with highest probability
predIdxs = np.argmax(predIdxs, axis = 1)

#printing formatted report
print(classification_report(testY.argmax(axis = 1), predIdxs,
    target_names = lb.classes_))

#saving the model to disk
print("saving the model")
model.save(args["model"], save_format = "h5")

#plot the model details
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label = "train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label = "val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
