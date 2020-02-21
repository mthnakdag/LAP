#%% LIBRARY PART
from Jpgs import JpgsFrames
import pandas as pd
import cv2
import time

Jpgsframe = JpgsFrames()
Jpgsframe.createDataFrame()
Jpgsframe.addactualtoframe(1,5)


Jpgsframe.findFace("haarcascade")
Jpgsframe.findFace("dnn")
Jpgsframe.findFace("hogdlib")
Jpgsframe.findFace("cnndlib")

Jpgsframe.saveCsv("deneme")

