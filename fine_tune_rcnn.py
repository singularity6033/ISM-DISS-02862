# import the necessary packages
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from utils.change_cnn_input_size import change_input_size
from utils.iou import compute_iou
from utils.ap import compute_ap
from utils import config
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from progressbar import *
from imutils import paths
import numpy as np
import time
import cv2
import pickle
import os


# load label binarizer
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())
# initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-4
EPOCHS = 5
BS = 32

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

train_generator = aug.flow_from_directory(
    config.COCO_TRAIN_PATH,
    target_size=config.INPUT_DIMS,
    batch_size=BS)
val_generator = aug.flow_from_directory(
    config.COCO_VAL_PATH,
    target_size=config.INPUT_DIMS,
    batch_size=BS)

# labels = [k for k, v in train_generator.class_indices.items()]
# load pre-trained VGG CNN (cifar100) and drop off the head FC layer
# and change the input size to (224, 224, 3)
with open('cifar100vgg.json', 'r') as file:
    model_json = file.read()
origin_model = model_from_json(model_json)
origin_model.load_weights('cifar100vgg.h5')
base_model = Model(inputs=origin_model.input, outputs=origin_model.layers[-8].output)
base_model.summary()
new_model = change_input_size(base_model, 224, 224, 3)
new_model.summary()

# construct the head of the model that will be placed on top of the the base model
headModel = new_model.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5, name='dropout_10')(headModel)
headModel = Dense(79, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become the actual model we will train)
model = Model(inputs=new_model.input, outputs=headModel)
# loop over all layers in the base_model and freeze them so they will
# not be updated during the first training process
for layer in base_model.layers:
    layer.trainable = True
# compile model
opt = Adam(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# fine-tuning the head of the network
print("[INFO] training head...")
H = model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // BS,
    validation_data=val_generator,
    validation_steps=val_generator.n // BS,
    epochs=EPOCHS)
# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save(config.MODEL_PATH, save_format="h5")

print("[INFO] evaluating object detector...")
# make predictions on the testing set and calculate the AP and mAP matrices
print("[INFO] loading testing images...")
# grab all image paths in the test images directory
TestImagePaths = list(paths.list_images(config.TEST_IMAGES))
# using cocoAPI to load coco annotation information
coco = COCO(config.TEST_ANNOTS)
# load COCO categories and super categories
cats = coco.loadCats(coco.getCatIds())
catIds = [cat['id'] for cat in cats]

NUM_CLASS = len(lb.classes_)
# MIN_IOU: minimum threshold of iou to determine whether a roi is TP or NP
MIN_IOU = 0.75
# AP: the area under the precision-recall curve, one class one AP
AP = np.zeros(NUM_CLASS)
# TP_FP: list of TP(1) and NP(0)
TP_FP = [[] for i in range(NUM_CLASS)]
# TP_FN: TP + FN = total num of ground-truth of each class
TP_FN = np.zeros(NUM_CLASS)
# Precision: TP / (TP + FP)
Precision = [[] for i in range(NUM_CLASS)]
# Recall: TP / (TP + FN)
Recall = [[] for i in range(NUM_CLASS)]

# loop over testing image paths
for (i, TestImagePath) in enumerate(TestImagePaths):
    # show a progress report
    print("[INFO] evaluating testing image {}/{}...".format(i + 1, len(TestImagePaths)))
    # extract the filename from the file path and use it to derive
    # the path to the XML annotation file
    filename = TestImagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    img_id = int(filename.lstrip('0'))
    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    annInfos = coco.loadAnns(annIds)
    # initialize our list of ground-truth bounding boxes
    gtBoxes = []
    # coco bounding box format: (x - top left, y - top left, width, height)
    # loop over all ground-truth 'object' elements in testing images
    for annInfo in annInfos:
        # extract the label and bounding box coordinates
        label_id = annInfo['category_id']
        cat = cats[catIds.index(label_id)]['name']
        # record num of ground-truth in each class
        idx_gt = np.where(lb.classes_ == cat)
        TP_FN[idx_gt] += 1
        xMin, yMin, bw, bh = annInfo['bbox']
        xMax = xMin + bw
        yMax = yMin + bh
        # update our list of ground-truth bounding boxes
        gtBoxes.append((xMin, yMin, xMax, yMax, label_id))
    # load the input image from disk
    image = cv2.imread(TestImagePath)
    # run selective search on the image and initialize our list of
    # proposed boxes
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposedRoi = []
    proposedBoxes = []
    # loop over the rectangles generated by selective search
    for (x, y, w, h) in rects:
        roi = image[y:y + h, x:x + w]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi, config.INPUT_DIMS, interpolation=cv2.INTER_CUBIC)
        # further preprocess the ROI
        roi = img_to_array(roi)
        roi = preprocess_input(roi)
        # update our proposedRoi and proposedBoxes lists
        proposedRoi.append(roi)
        proposedBoxes.append((x, y, x + w, y + h))
    # initialize counters used to count the number of positive
    positiveROIs = 0
    proposedRoi = np.array(proposedRoi, dtype="float32")
    label_proba = model.predict(proposedRoi, batch_size=BS)
    proposedLabels = lb.classes_[np.argmax(label_proba, axis=1)]
    # loop over the maximum number of region proposals
    for proposedId, proposedLabel in enumerate(proposedLabels[:config.MAX_PROPOSALS]):
        # loop over the ground-truth bounding boxes
        for gtBox in gtBoxes:
            # compute the intersection over union (iou) between each proposed boxes
            # and the ground-truth bounding box to decide whether it is TP or FP
            iou = compute_iou(gtBox[:-1], proposedBoxes[proposedId])
            (gtStartX, gtStartY, gtEndX, gtEndY, label_id) = gtBox
            cat = cats[catIds.index(label_id)]['name']
            # check to see if the IOU is greater than 80%(P)
            if cat == proposedLabel:
                if iou > MIN_IOU:
                    TP_FP[proposedId].append(1)
                else:
                    TP_FP[proposedId].append(0)
for j in range(len(TP_FP)):
    for k in range(len(TP_FP[j])):
        Precision[j].append(sum(TP_FP[j][0:k]) / (len(TP_FP[j][0:k]) + 1))
        Recall[j].append(sum(TP_FP[j][0:k]) / TP_FN[j])
    AP[j] = compute_ap(Precision[j], Recall[j])
mAP = np.mean(AP, axis=0)

# plot the training loss and accuracy
print("[INFO] plotting results...")
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plotPath = os.path.sep.join([config.PLOTS_PATH, "Training_Loss_and_Accuracy_coco.png"])
plt.savefig(plotPath)

# plot AP and print mAP
plt.style.use("ggplot")
plt.figure()
plt.title("Average Precision (with IoU threshold = " + str(MIN_IOU) + ")")
plt.plot(lb.classes_, AP)
plt.xlabel("Class Labels")
plt.ylabel("AP")
plotPath = os.path.sep.join([config.PLOTS_PATH, "coco_AP.png"])
plt.savefig(plotPath)
plt.close()
print("COCO's mAP is equal to " + str(mAP) + " (with IoU threshold = " + str(MIN_IOU) + ")")
