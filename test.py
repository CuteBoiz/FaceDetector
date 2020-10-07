#!/usr/bin/env python
# coding=utf-8

import sys
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(1, "MTCNN")
from mtcnn import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
imgpath = '2.jpg'
img = Image.open(imgpath)
detector = MTCNN(image_size=160, select_largest=False, device=device)

boxes, _, points= detector.detect(img, landmarks=True)
draw = ImageDraw.Draw(img)

for box in boxes:
	draw.rectangle(box.tolist(), outline=(0, 0, 255), width=3)
for pts in points:
	for point in pts:
		draw.point(point, 'red')

img.save('detected.png')



