import pickle

import numpy as np
from matplotlib import pyplot as plt

ap = pickle.loads(open('ap.pickle', "rb").read())
labels = np.arange(len(ap))

plt.figure(figsize=(6, 6), dpi=600)
plt.bar(labels, height=ap, width=0.3)
plt.title('average precision of object detector based on coco')
plt.xlabel('label id')
plt.ylabel('average precision')
plt.ylim([0, 1])
plt.savefig('output/ap_1.png')
plt.show()
