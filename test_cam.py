import cv2
import time
import sys
import numpy as np
import torch
sys.path.insert(1, "MTCNN")
from mtcnn import MTCNN 


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
detector = MTCNN(image_size=160, select_largest=False, device=device)

cap = cv2.VideoCapture(0)
if not (cap.isOpened()):
	print("Could not open video device")

while(True): 
	ret, frame = cap.read()
	start = time.time()
	image = cv2.flip(frame, 1)
	faces, _ = detector.detect(image, landmarks=False)
	if faces is not None:
		faces = (np.int32(faces)).reshape(-1, 2, 2)
		for face in faces:
			cv2.rectangle(image, tuple(face[0]), tuple(face[1]), (0, 0, 255), 2)
			print("Detect Time: ", time.time() - start)
	cv2.imshow('preview',image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()