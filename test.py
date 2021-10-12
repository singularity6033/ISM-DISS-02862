import os
import pickle
import re
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
# from pycocotools.coco import COCO
import cv2

# def isMatch(s1, s2) -> bool:
#     dict = {
#         ')': '(',
#         ']': '[',
#         '}': '{'
#     }
#     if s2 == dict.get(s1, None):
#         return True
#     else:
#         return False
#
#
# print(isMatch( ')', '('))
from utils import config

fuck = [1, 2, 3]
l = ['a', 'b', 'c']
labelNames = np.array(l)
a = np.where(labelNames == 'a')[0]
print(fuck[list(labelNames).index('a')])
# a = np.ones(3)
# dict = {'a': 1, 'b': 2, 'c': 3}
# labels = [k for k, v in dict.items()]
# print(labels)
lb = pickle.loads(open(config.ENCODER_PATH, "rb").read())
print(lb.classes_)
# if '1' is not "train":
#     print(1)
# # f = []
# # f.append((1,1))
# # f.append((2,2))
# # print(f[-1][0])
# # img_path = './Data/coco/images/train2017'
# # print(img_path.split(os.path.sep)[-1])
# l = ['a', 'b', 'c']
# labelNames = np.array(l)
# print(l[l.index('a')])
# a = np.ones(3)
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(labelNames, a)
# plt.xlabel("Class Labels")
# plt.ylabel("AP")
# plt.show()
# a = np.argmax(f, axis=1)
# print(a)
# dirFiles = os.listdir(img_path)
# I = io.imread(os.path.sep.join([img_path, dirFiles[0]]))
# name = dirFiles[0]
# num = re.sub(r'\D', "", name)
# id = int(num.lstrip('0'))
# # ret = re.match("[0-9]*", name)
# print(id)
#
# # plt.imshow(I)
# # plt.show()
print('tr' is not "train")
# dataDir = './Data/coco/annotations'
# dataType = 'train2017'
# annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
# coco = COCO(annFile)
# # # display COCO categories and supercategories
# cats = coco.loadCats(coco.getCatIds())
# nms = [cat['name'] for cat in cats]
# print('COCO categories: \n{}\n'.format(' '.join(nms)))
#
# nms = set([cat['supercategory'] for cat in cats])
# print('COCO supercategories: \n{}'.format(' '.join(nms)))
#
# # get all images containing given categories, select one at random
# catIds = coco.getCatIds(catNms=['person', 'dog', 'skateboard'])
# imgIds = coco.getImgIds()
# img = coco.loadImgs(501693)[0]
# # I = io.imread(img['coco_url'])
# imgIds = coco.getImgIds(imgIds=[id])
# catIds = [cat['id'] for cat in cats]
# cat = cats[catIds.index(5)]['name']
# label_counter = np.zeros((len(cats), 1))
# a = catIds.index(2)
# label_counter[a, 0] += 1
# # img = coco.loadImages(imgIds[np.random.randint(0, len(imgIds))])[0]
# # # load and display image
# # # I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# # # use url to load image
#
# # plt.axis('off')
# # plt.imshow(I)
# # plt.show()
# #
# # # load and display instance annotations
# # plt.imshow(I)
# # plt.axis('off')
# print(img['id'])
# annIds = coco.getAnnIds(imgIds=id, iscrowd=None)
# # catIds = coco.getCatIds(annIds=annIds)
# anns = coco.loadAnns(annIds)
