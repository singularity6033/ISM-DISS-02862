import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO

# coco_data = tfds.load("coco")
# coco_train, coco_test = coco_data["train"], coco_data["test"]
# print(len(coco_train))
# a = tf.data.Dataset.from_tensor_slices(list(coco_train))
# for coco_example in coco_train.take(100):  # 只取一个样本
#     image = coco_example["image"]
#     print(image.numpy()[:, :, 0].astype(np.float32).shape)
coco = COCO('/media/user/volume2/students/s121md215_02/Data/coco/annotations/instances_train2017.json')
# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))
