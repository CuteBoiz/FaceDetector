import time
import torch
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import dlib 
import sys
sys.path.insert(1, "MTCNN")
from mtcnn import MTCNN
sys.path.insert(1, 'Retinaface')


def timer(detector, detect_fn, images, *args):
	start = time.time()
	boxes = detect_fn(detector, images, *args)
	elapsed = time.time() - start
	h, w = images.shape[1], images.shape[2]
	print('Detecting %d faces in %dx%d in%3.f seconds'%(len(boxes), h, w, elapsed))
	return elapsed

video = cv2.VideoCapture("../Data/Videos/4.mp4")
images_1080_1920 = []
images_720_1280 = []
images_540_960 = []
for i in tqdm(range(int(video.get(cv2.CAP_PROP_FRAME_COUNT)))):
	_, image = video.read()
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	images_1080_1920.append(image)
	images_720_1280.append(cv2.resize(src=image, dsize=(1280, 720)))
	images_540_960.append(cv2.resize(src=image, dsize=(960, 540)))
video.release()

images_1080_1920 = np.stack(images_1080_1920)
images_720_1280 = np.stack(images_720_1280)
images_540_960 = np.stack(images_540_960)

print('Shapes:')
print(images_1080_1920.shape)
print(images_720_1280.shape)
print(images_540_960.shape)

#----------------------------DLIB----------------------------------------------
dlib_face_detector = dlib.get_frontal_face_detector()
def detect_dlib(detector, images):
    faces = []
    for image in images:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = detector(image_gray)
        box = boxes[0]
        face = image[box.top():box.bottom(), box.left():box.right()]
        faces.append(face)
    return faces

times_dlib = []
print("Dlib:")
elapsed = timer(dlib_face_detector, detect_dlib, images_540_960)
times_dlib.append(elapsed)
elapsed = timer(dlib_face_detector, detect_dlib, images_720_1280)
times_dlib.append(elapsed)
elapsed = timer(dlib_face_detector, detect_dlib, images_1080_1920)
times_dlib.append(elapsed)

del dlib_face_detector

#----------------------------MTCNN----------------------------------------------
torch.cuda.empty_cache()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
mtcnn_face_detector = MTCNN(device=device)

def detect_mtcnn(detector, images):
    faces = []
    for image in images:
        boxes = detector.detect(image, select_largest=True, proba=False, landmarks=False)
        for face in boxes:
        	faces.append(face)
    return faces

times_mtcnn = []

print("MTCNN:")
elapsed = timer(mtcnn_face_detector, detect_mtcnn, images_540_960)
times_mtcnn.append(elapsed)
elapsed = timer(mtcnn_face_detector, detect_mtcnn, images_720_1280)
times_mtcnn.append(elapsed)
elapsed = timer(mtcnn_face_detector, detect_mtcnn, images_1080_1920)
times_mtcnn.append(elapsed)

del mtcnn_face_detector
torch.cuda.empty_cache()


#--------------------------RetinaFace-------------------------------------
print("Retinaface:")
from retinaface import detect
def detect_retinaface(detector, images):
    faces = []
    for image in images:
        boxes, points = detector(image)
        for face in boxes:
            faces.append(face)
    return faces
times_retinaface = []

elapsed = timer(detect, detect_retinaface, images_540_960)
times_retinaface.append(elapsed)
elapsed = timer(detect, detect_retinaface, images_720_1280)
times_retinaface.append(elapsed)
elapsed = timer(detect, detect_retinaface, images_1080_1920)
times_retinaface.append(elapsed)

#----------------------------Compare Result--------------------------------------------

fig, ax = plt.subplots(figsize=(10,6))

pos = np.arange(3)
rect1 =plt.bar(pos, times_dlib, 0.2, label='dlib')
rect2 = plt.bar(pos + 0.2, times_mtcnn, 0.2, label='pytorch-mtcnn')
rect3 = plt.bar(pos + 0.4, times_retinaface, 0.2, label='retinaface')

ax.set_ylabel('Elapsed Time (s)')
ax.set_xticks(pos + 0.25)
ax.set_xticklabels(['540x960', '720x1280', '1080x1920'])
plt.legend();

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

autolabel(rect1)
autolabel(rect2)
autolabel(rect3)

plt.savefig('ElapsedTime.png')
cv2.imshow('preview', './ElapsedTime.png')
cv2.waitkey(0)
cv2.destroyAllWindows()