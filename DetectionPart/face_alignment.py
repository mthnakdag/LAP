from imutils.face_utils import FaceAligner,rect_to_bb
import imutils
import dlib
import cv2
import os

img = cv2.imread("/home/pi/LAP/DetectionPart/Faces/align_3.jpg")
img = imutils.resize(img,width=300)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
facedata = os.sep.join([os.getcwd(),"AdditionalFiles","mmod_human_face_detector.dat"])
cnnFaceDetector = dlib.cnn_face_detection_model_v1(facedata) 

predictor = dlib.shape_predictor(os.sep.join([os.getcwd(),"AdditionalFiles","shape_predictor_68_face_landmarks.dat"]))
fa = FaceAligner(predictor, desiredFaceWidth=128)
faces = cnnFaceDetector(img,0)
for face in  faces:
    (x,y,w,h) = rect_to_bb(face.rect)
    faceOrig= imutils.resize(img[y:y+h,x:x+w], width=128)
    faceAlign = fa.align(img,gray,face.rect)
cv2.imshow("ori",img)

cv2.imshow("a",faceAlign)
cv2.waitKey(0)