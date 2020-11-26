import cv2
import numpy as np
import time 
import sys
sys.path.insert(1, '../Retinaface')
from retinaface import detect


# cap = cv2.VideoCapture("../../workplace/Data/Videos/4.mp4")

# while(True): 
# 	ret, frame = cap.read()
# 	if frame is None:
# 		break
# 	start = time.time()
# 	img = cv2.flip(frame, 1)
# 	boxes, points = detect(img)
# 	if boxes is not None:
# 		for box, point in zip(boxes, points):
# 			cv2.rectangle(img, tuple((np.int32(box[0]), np.int32(box[1]))), tuple((np.int32(box[2]), np.int32(box[3]))), (255, 255, 255), 1)		
# 	print("FPS: ", np.int32(1/(time.time() - start)))
# 	cv2.imshow('preview',img)
# 	if cv2.waitKey(1) & 0xFF == ord('q'):
# 		break
# cap.release()
# cv2.destroyAllWindows()


image_path = "../Images/1.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

faces, points = detect(image)
faces = np.int32(faces)
for face in faces:
	cv2.rectangle(image, (face[0], face[1]), (face[2], face[3]), (255,255,255), 1)

cv2.imshow("preview", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
