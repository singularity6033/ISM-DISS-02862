# import the necessary packages
import os

# define the base path to the *original* coco dataset and then use
# the base path to derive the coco image and annotations directories
ORIG_BASE_PATH = "./Data/coco"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images/train2017"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations/annotations/instances_train2017.json"])
TEST_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images/val2017"])
TEST_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations/annotations/instances_val2017.json"])

# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to our
# own dataset with corresponding class label directories
DATASET_BASE_PATH = "dataset"

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label binarizer, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "coco_object_detector_rcnn.h5"])
ENCODER_PATH = os.path.sep.join([BASE_OUTPUT, "coco_label_encoder.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of positive ROI to be generated from each image
MAX_POSITIVE = 20

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99
# import the necessary packages
import os

# define the base path to the *original* coco dataset and then use
# the base path to derive the coco image and annotations directories
ORIG_BASE_PATH = "./Data/coco"
ORIG_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images/train2017"])
ORIG_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations/annotations/instances_train2017.json"])
TEST_IMAGES = os.path.sep.join([ORIG_BASE_PATH, "images/val2017"])
TEST_ANNOTS = os.path.sep.join([ORIG_BASE_PATH, "annotations/annotations/instances_val2017.json"])

# define the base path to the *new* dataset after running our dataset
# builder scripts and then use the base path to derive the paths to our
# own dataset with corresponding class label directories
DATASET_BASE_PATH = "dataset"

# define the base path to the shuffled dataset after running total roi dataset
SHUFFLE_TRAIN_VAL_COCO_ROI = "coco_roi"
COCO_TRAIN_PATH = os.path.sep.join([SHUFFLE_TRAIN_VAL_COCO_ROI, "train"])
COCO_VAL_PATH = os.path.sep.join([SHUFFLE_TRAIN_VAL_COCO_ROI, "val"])
COCO_TRAIN_LABEL_PATH = os.path.sep.join([SHUFFLE_TRAIN_VAL_COCO_ROI, "train_label.pickle"])
COCO_VAL_LABEL_PATH = os.path.sep.join([SHUFFLE_TRAIN_VAL_COCO_ROI, "val_label.pickle"])

# define the path to the base output directory
BASE_OUTPUT = "output"
# define the path to the output model, label binarizer, plots output
# directory, and testing image paths
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "coco_object_detector_rcnn.h5"])
ENCODER_PATH = os.path.sep.join([BASE_OUTPUT, "coco_label_encoder.pickle"])
PLOTS_PATH = os.path.sep.join([BASE_OUTPUT, "plots"])

# define the number of max proposals used when running selective
# search for (1) gathering training data and (2) performing inference
MAX_PROPOSALS = 2000
MAX_PROPOSALS_INFER = 200

# define the maximum number of positive ROI to be generated from each image
MAX_POSITIVE = 20

# initialize the input dimensions to the network
INPUT_DIMS = (224, 224)

# define the minimum probability required for a positive prediction
# (used to filter out false-positive predictions)
MIN_PROBA = 0.99
