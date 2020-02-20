import os
import pandas as pd
import cv2
import time
import numpy as np
import dlib

class JpgsFrames:
    
    def __init__(self, df=dict()):
        self.df = pd.DataFrame(df)
        
    def findJpgs(self,rootdir=os.sep.join([os.getcwd(),"Faces"])):
        filelist = []
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".jpg"):
                    filelist.append(filepath)
        return filelist
    
    def createDataFrame(self,files=None):
        if files == None:
            files = self.findJpgs()
        filenames=list()
        filecategories=list()
        filelocs=list()
        for file in files:
            filenames.append(file.split('/')[-1])
            filecategories.append(file.split('/')[-1].split('_')[0])
            filelocs.append(file)
        df = {'filename':filenames , 'filecategory':filecategories ,
                  'fileloc': filelocs}
        self.df = pd.DataFrame(df)
        return self.df
    
    def addactualtoframe(self, *lis):
        self.df["ExpectedFaces"]= lis
        return self.df
    def saveCsv(self,filename):  
        self.df.to_csv(filename+".csv")

    def findFace(self,algorithm_name):
        algorithm_name = algorithm_name.lower()
        path = os.sep.join([os.getcwd(),"DetectedFaces",algorithm_name])+os.sep
        mslist=list()
        findface=list()
        
        if  algorithm_name == "haarcascade":
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
            for imagename,imageloc in zip(self.df["filename"].tolist(),self.df["fileloc"].tolist()):
                img = cv2.imread(imageloc,cv2.IMREAD_GRAYSCALE)
                start = time.time()
                faces = faceCascade.detectMultiScale(img)
                end = time.time()
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                mslist.append((end-start)*100)
                findface.append(len(faces))
                cv2.rectangle(img,(10,10), (120,30), (0,0,0),35)
                cv2.putText(img,str(round((end-start)*100,2))+"ms",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255))
                cv2.imwrite((path+imagename),img)
        
        elif  algorithm_name == "dnn":
            modelfile = os.sep.join([os.getcwd(),"AdditionalFiles","res10_300x300_ssd_iter_140000_fp16 .caffemodel"])
            configfile = os.sep.join([os.getcwd(),"AdditionalFiles","deploy.prototxt"])
            net = cv2.dnn.readNetFromCaffe(configfile,modelfile)
            for imagename,imageloc in zip(self.df["filename"].tolist(),self.df["fileloc"].tolist()):
                img = cv2.imread(imageloc)
                h,w = img.shape[:2]
                start = time.time()
                blob = cv2.dnn.blobFromImage(cv2.resize(img,(300,300)),1.0,(300,300),(104.0,177.0,123.0))
                net.setInput(blob)
                detections = net.forward()
                boxes = list()
                numofface=0
                for i in range(detections.shape[2]):
                    box =detections[0,0,i,3:7] * np.array([w,h,w,h])
                    boxes.append(box)
                end = time.time()
                for i,box in enumerate(boxes):
                    (startX,startY,endX,endY) = box.astype('int')
                    confidence = detections[0,0,i,2]
                    if confidence > 0.5:
                        numofface+=1
                        cv2.rectangle(img,(startX,startY),(endX,endY),(0,0,255),2)
                mslist.append((end-start)*100)
                findface.append(numofface)
                cv2.rectangle(img,(10,10), (120,30), (0,0,0),35)
                cv2.putText(img,str(round((end-start)*100,2))+"ms",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255))
                cv2.imwrite((path+imagename),img)
                
        elif  algorithm_name == "hogdlib":
            hogFaceDetector = dlib.get_frontal_face_detector() 
            for imagename,imageloc in zip(self.df["filename"].tolist(),self.df["fileloc"].tolist()):
                img = cv2.imread(imageloc,cv2.IMREAD_GRAYSCALE)
                start = time.time()
                faces = hogFaceDetector(img)
                end = time.time()
                for face in faces:
                    startX = face.left()
                    startY = face.top()
                    endX = face.right()
                    endY = face.bottom()
                    cv2.rectangle(img,(startX,startY),(endX,endY),(0,0,255),2)
                mslist.append((end-start)*100)
                findface.append(len(faces))
                cv2.rectangle(img,(10,10), (120,30), (0,0,0),35)
                cv2.putText(img,str(round((end-start)*100,2))+"ms",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255))
                cv2.imwrite((path+imagename),img)
                
                
        elif  algorithm_name == "cnndlib":
            facedata = os.sep.join([os.getcwd(),"AdditionalFiles","mmod_human_face_detector.dat"])
            cnnFaceDetector = dlib.cnn_face_detection_model_v1(facedata) 
            for imagename,imageloc in zip(self.df["filename"].tolist(),self.df["fileloc"].tolist()):
                img = cv2.imread(imageloc,cv2.IMREAD_GRAYSCALE)
                start = time.time()
                faces = cnnFaceDetector(img,0)
                end = time.time()
                for face in faces:
                    startX = face.rect.left()
                    startY = face.rect.top()
                    endX = face.rect.right()
                    endY = face.rect.bottom()
                    cv2.rectangle(img,(startX,startY),(endX,endY),(0,0,255),2)
                mslist.append((end-start)*100)
                findface.append(len(faces))
                cv2.rectangle(img,(10,10), (120,30), (0,0,0),35)
                cv2.putText(img,str(round((end-start)*100,2))+"ms",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255))
                cv2.imwrite((path+imagename),img)
        

        self.df[algorithm_name+"-MS"] = mslist
        self.df[algorithm_name+"-FindingFace"] = findface
