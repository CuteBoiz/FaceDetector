import cv2
import dlib

image_path = '../Images/1.jpg'
image = cv2.imread(image_path)

HOG_SVM_detector = dlib.get_frontal_face_detector()
facial_detector = dlib.shape_predictor('../Dlib/shape_predictor_68_face_landmarks.dat')

faces = HOG_SVM_detector(image, 1)

for face  in faces:
	cv2.rectangle(image, (face.left(), face.top()), 
                (face.right(), face.bottom()), 
                (255, 255, 255), 2)
	facial_points = facial_detector(image, face)
	facial_points = list(map(lambda p: (p.x, p.y), facial_points.parts()))
	for point in facial_points:
		cv2.circle(image, point, 1, (255, 255, 255), 1)

cv2.imshow('Preview', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
