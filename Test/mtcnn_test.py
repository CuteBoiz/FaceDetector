import torch
import cv2
import numpy as np 
import sys
sys.path.insert(1, "../MTCNN")
from mtcnn import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(device=device)

image_path = '../Images/1.jpg'
image = cv2.imread(image_path)

boxes, points = detector.detect(image, select_largest=False, proba=False, landmarks=True)
if boxes is not None:
	for box, point in zip(boxes, points):
		cv2.rectangle(image, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)
		for pt in point:
			cv2.circle(image, tuple(np.int32(pt)), 1, (255, 255, 255), 6)

cv2.imshow('Preview',image)
cv2.waitKey(0)
cv2.destroyAllWindows()