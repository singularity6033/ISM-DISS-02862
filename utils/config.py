# import the necessary packages
import os

# define the base path to the *original* coco dataset and then use
# the base path to derive the coco image and annotations directories
ORIG_BASE_PATH = "./Data/coco"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images/train2017"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations/annotations/instances_train2017.json"])

# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to our
# own dataset with corresponding class label directories
BASE_PATH = "dataset"

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of positive and negative images to be
# generated from each image
MAX_POSITIVE = 30
MAX_NEGATIVE = 10

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)
# define the path to the output model and label binarizer
MODEL_PATH = "object_detector_rcnn.h5"
ENCODER_PATH = "label_encoder.pickle"
# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99
