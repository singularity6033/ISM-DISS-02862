import os
import re
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import cv2

img_path = './Data/coco/images/train2017'
dirFiles = os.listdir(img_path)
I = io.imread(os.path.sep.join([img_path, dirFiles[0]]))
name = dirFiles[0]
num = re.sub(r'\D', "", name)
id = int(num.lstrip('0'))
# ret = re.match("[0-9]*", name)
print(id)

# plt.imshow(I)
# plt.show()
dataDir = './Data/coco/annotations'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)
# # display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
imgIds = coco.getImgIds()
img = coco.loadImgs(501693)[0]
# I = io.imread(img['coco_url'])
imgIds = coco.getImgIds(imgIds=[id])
# img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
# # load and display image
# # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# # use url to load image

# plt.axis('off')
# plt.imshow(I)
# plt.show()
#
# # load and display instance annotations
# plt.imshow(I)
# plt.axis('off')
print(img['id'])
annIds = coco.getAnnIds(imgIds=id, iscrowd=None)
# catIds = coco.getCatIds(annIds=annIds)
anns = coco.loadAnns(annIds)
