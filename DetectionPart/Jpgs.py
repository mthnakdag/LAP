import os
import pandas as pd
import cv2
import time
import numpy as np
import dlib

class JpgsFrames:
    
    def __init__(self, df=dict()):
        self.df = pd.DataFrame(df)
    def findPaths(self,rootdir=os.sep.join([os.path.abspath('..'),'ImgsPart','for_face_detection'])):
        jpgsfilelist = list()
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".jpg"):
                    jpgsfilelist.append(filepath)
        jpgsfilelist.sort()
        return jpgsfilelist
    
    def createDataFrame(self,files=None):
        if files == None:
            files = self.findPaths()
        filenames=list()
        filecategories=list()
        filelocs=list()
        for file in files:
            filenames.append(file.split('/')[-1])
            filecategories.append(file.split('/')[-2])
            filelocs.append(file)
        df = {'filename':filenames , 'filecategory':filecategories ,
                  'fileloc': filelocs}
        self.df = pd.DataFrame(df)
        return self.df
    
    def addactualtoframe(self, rootdir= None):
        if rootdir == None:
            rootdir=os.sep.join([os.path.abspath('..'),'ImgsPart','for_face_detection'])
        txtfilelist = list()
        actualfacelist = list()
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                filepath = subdir + os.sep + file
                if filepath.endswith(".txt"):
                    txtfilelist.append(filepath)
        txtfilelist.sort()
        for txtfile in txtfilelist:
            file = open(txtfile,'r')
            for elem in file:
                elem = elem.split()
                for e in elem:   
                    actualfacelist.append(int(e))
        self.df["ExpectedFaces"]= actualfacelist
        return self.df
    def saveCsv(self,filename):  
        self.df.to_csv(filename+".csv")

    def findFace(self,algorithm_name):
        algorithm_name = algorithm_name.lower()
        path = os.sep.join([os.getcwd(),"DetectedFaces",algorithm_name])+os.sep
        mslist=list()
        findface=list()
        folderlist = list(self.df['filecategory'].unique())
        for folder in folderlist:
            try:
                os.mkdir(path+str(folder))
            except OSError:
                print("This directory existed")            

        if  algorithm_name == "haarcascade":
            faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
            for imagename,imagecat,imageloc in zip(self.df["filename"].tolist(),self.df["filecategory"].tolist(),self.df["fileloc"].tolist()):
                img = cv2.imread(imageloc)
                start = time.time()
                faces = faceCascade.detectMultiScale(img)
                end = time.time()
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
                mslist.append((end-start)*100)
                findface.append(len(faces))
                if img.shape[0]>img.shape[1]:
                    cv2.rectangle(img,(0,0), (img.shape[0]//5,img.shape[1]//8), (0,0,0), thickness =cv2.FILLED)
                else:
                    cv2.rectangle(img,(0,0), (img.shape[1]//5,img.shape[0]//8), (0,0,0), thickness =cv2.FILLED)
                cv2.putText(img,str(round((end-start)*100,2))+"ms",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255))
                cv2.imwrite(os.sep.join([path+imagecat,imagename]),img)
                print(os.sep.join([path+imagecat,imagename]))
                
        elif  algorithm_name == "dnn":
            modelfile = os.sep.join([os.getcwd(),"AdditionalFiles","res10_300x300_ssd_iter_140000_fp16 .caffemodel"])
            configfile = os.sep.join([os.getcwd(),"AdditionalFiles","deploy.prototxt"])
            net = cv2.dnn.readNetFromCaffe(configfile,modelfile)
            for imagename,imagecat,imageloc in zip(self.df["filename"].tolist(),self.df["filecategory"].tolist(),self.df["fileloc"].tolist()):
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
                if img.shape[0]>img.shape[1]:
                    cv2.rectangle(img,(0,0), (img.shape[0]//5,img.shape[1]//8), (0,0,0), thickness =cv2.FILLED)
                else:
                    cv2.rectangle(img,(0,0), (img.shape[1]//5,img.shape[0]//8), (0,0,0), thickness =cv2.FILLED)
                cv2.putText(img,str(round((end-start)*100,2))+"ms",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255))
                cv2.imwrite(os.sep.join([path+imagecat,imagename]),img)
                print(os.sep.join([path+imagecat,imagename]))
                
        elif  algorithm_name == "hogdlib":
            hogFaceDetector = dlib.get_frontal_face_detector() 
            for imagename,imagecat,imageloc in zip(self.df["filename"].tolist(),self.df["filecategory"].tolist(),self.df["fileloc"].tolist()):
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
                if img.shape[0]>img.shape[1]:
                    cv2.rectangle(img,(0,0), (img.shape[0]//5,img.shape[1]//8), (0,0,0), thickness =cv2.FILLED)
                else:
                    cv2.rectangle(img,(0,0), (img.shape[1]//5,img.shape[0]//8), (0,0,0), thickness =cv2.FILLED)
                cv2.putText(img,str(round((end-start)*100,2))+"ms",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255))
                cv2.imwrite(os.sep.join([path+imagecat,imagename]),img)
                print(os.sep.join([path+imagecat,imagename]))
                
        elif  algorithm_name == "cnndlib":
            facedata = os.sep.join([os.getcwd(),"AdditionalFiles","mmod_human_face_detector.dat"])
            cnnFaceDetector = dlib.cnn_face_detection_model_v1(facedata) 
            for imagename,imagecat,imageloc in zip(self.df["filename"].tolist(),self.df["filecategory"].tolist(),self.df["fileloc"].tolist()):
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
                if img.shape[0]>img.shape[1]:
                    cv2.rectangle(img,(0,0), (img.shape[0]//5,img.shape[1]//8), (0,0,0), thickness =cv2.FILLED)
                else:
                    cv2.rectangle(img,(0,0), (img.shape[1]//5,img.shape[0]//8), (0,0,0), thickness =cv2.FILLED)
                cv2.putText(img,str(round((end-start)*100,2))+"ms",(10,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,255,255))
                cv2.imwrite(os.sep.join([path+imagecat,imagename]),img)
                print(os.sep.join([path+imagecat,imagename]))

        self.df[algorithm_name+"-MS"] = mslist
        self.df[algorithm_name+"-FindingFace"] = findface
