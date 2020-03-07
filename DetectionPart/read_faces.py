#%% LIBRARY PART
from Jpgs import JpgsFrames
import pandas as pd

jpgsframe = JpgsFrames()
jpgsframe.createDataFrame()
jpgsframe.addactualtoframe()

#jpgsframe.findFace("haarcascade")
jpgsframe.findFace("dnn")
#jpgsframe.findFace("hogdlib")
#jpgsframe.findFace("cnndlib")



