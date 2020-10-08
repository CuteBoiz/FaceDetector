#!/usr/bin/env python
# coding=utf-8

import sys
import cv2
import numpy as np
import torch

sys.path.insert(1, "MTCNN")
from mtcnn import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
imgpath = '2.jpg'
image = cv2.imread(imgpath)
detector = MTCNN(image_size=160, select_largest=False, device=device)

boxes, _, points= detector.detect(image, landmarks=True)
boxes = (np.int32(boxes)).reshape(-1, 2, 2)

for box in boxes:
	cv2.rectangle(image, tuple(box[0]), tuple(box[1]), (200, 100, 0), 2)
for point in points:
	for pt in point:
		cv2.circle(image, tuple(np.int32(pt)), 1, (255, 255, 255), 6)

cv2.imwrite('detected.png', image)