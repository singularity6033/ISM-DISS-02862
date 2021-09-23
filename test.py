import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

coco_data = tfds.load("coco")
coco_train, coco_test = coco_data["train"], coco_data["test"]
# for mnist_example in mnist_train.take(1):  # 只取一个样本
#     image, label = mnist_example["image"], mnist_example["label"]
#     plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
#     print("Label: %d" % label.numpy())