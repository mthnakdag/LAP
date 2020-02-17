import os
import pandas as pd
import cv2

class JpgsFrames:
    def __init__(self, df=dict()):
        self.df = pd.DataFrame(df)
        
    def findJpgs(self,rootdir=os.getcwd()):
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
    
    
    def addactualtoframe(self,*lis):
        self.df["ExpectedFaces"]= lis
        return self.df
    
    def saveCsv(self,filename):
        self.df.to_csv(filename+".csv")

