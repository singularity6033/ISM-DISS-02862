# import the necessary packages
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

from utils.change_cnn_input_size import change_input_size
from utils import config
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# load label binarizer
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())
# initialize the initial learning rate, number of epochs to train for and batch size
INIT_LR = 1e-4
EPOCHS = 10
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
    target_size=(224, 224),
    batch_size=BS)
val_generator = aug.flow_from_directory(
    config.COCO_VAL_PATH,
    target_size=(224, 224),
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
# base_model = MobileNetV2(weights="imagenet", include_top=False,
#                          input_tensor=Input(shape=(224, 224, 3)))
# construct the head of the model that will be placed on top of the the base model
headModel = new_model.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5, name='dropout_10')(headModel)
headModel = Dense(80, activation="softmax")(headModel)
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
model.save(config.MODEL_TRAIN_HIS_PATH, save_format="h5")
f = open(config.MODEL_TRAIN_HIS_PATH, "wb")
f.write(pickle.dumps(lb))
f.close()

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
