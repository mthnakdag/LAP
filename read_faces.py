#%% LIBRARY PART
from Jpgs import JpgsFrames
import pandas as pd
import cv2
Jpgsframe = JpgsFrames()
Jpgsframe.createDataFrame()
Jpgsframe.addactualtoframe(1,5)
Jpgsframe.saveCsv("deneme")

