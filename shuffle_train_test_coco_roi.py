from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from imutils import paths
from utils import config
import numpy as np
import pickle
import os
import cv2


# grab the list of images in our dataset directory,
# then initialize the list of data (i.e., images) and class label
print("[INFO] loading images...")
imagePaths = list(paths.list_images(config.DATASET_BASE_PATH))
labels = []
# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# convert the data and labels to NumPy arrays
labels = np.array(labels)
# perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
print(lb.classes_)
# serialize the label encoder to disk
print("[INFO] saving label encoder...")
f = open(config.ENCODER_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX_dirs, valX_dirs, trainY, valY) = train_test_split(imagePaths, labels, test_size=0.20)
train_label_counter = np.zeros((80, 1), dtype=int)
# construct folders to store train and test images
for train_id, trainX_dir in enumerate(trainX_dirs):
    print("[INFO] shuffling train image {}/{}...".format(train_id + 1, len(trainX_dirs)))
    label = trainX_dir.split(os.path.sep)[-2]
    outputPath = os.path.sep.join([config.COCO_TRAIN_PATH, label])
    train_img = cv2.imread(trainX_dir)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    filename = "{}.png".format(train_label_counter[np.where(lb.classes_ == label), 0])
    filePath = os.path.sep.join([outputPath, filename])
    cv2.imwrite(filePath, train_img)
    train_label_counter[np.where(lb.classes_ == label), 0] += 1

test_label_counter = np.zeros((80, 1), dtype=int)
for test_id, valX_dir in enumerate(valX_dirs):
    print("[INFO] shuffling test image {}/{}...".format(test_id + 1, len(valX_dirs)))
    label = valX_dir.split(os.path.sep)[-2]
    outputPath = os.path.sep.join([config.COCO_VAL_PATH, label])
    test_img = cv2.imread(valX_dir)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    filename = "{}.png".format(test_label_counter[np.where(lb.classes_ == label), 0])
    filePath = os.path.sep.join([outputPath, filename])
    cv2.imwrite(filePath, test_img)
    test_label_counter[np.where(lb.classes_ == label), 0] += 1

# serialize the train label to disk
print("[INFO] saving train label...")
f = open(config.COCO_TRAIN_LABEL_PATH, "wb")
f.write(pickle.dumps(trainY))
f.close()

# serialize the test label to disk
print("[INFO] saving val label...")
f = open(config.COCO_VAL_LABEL_PATH, "wb")
f.write(pickle.dumps(valY))
f.close()