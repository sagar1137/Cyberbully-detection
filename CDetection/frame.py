# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 12:05:34 2020

@author: Sagar Patil
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from keras.models import load_model
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import matplotlib.image as mpimg

import cv2
from PIL import ImageTk, Image
im = Image.open('download.jpg')
width, height = im.size
print(width, height)

model = load_model("Final_weights.h5")

import imutils
 
def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image
 
    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
 
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
 
        # yield the next image in the pyramid
        yield image
def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
            
import time
def check(unsave = 0):
    image = cv2.imread("download6.png")
    (winW, winH) = (224, 224)
    maping = {0 : "Neutral", 1 : "Porn", 2 : "Sexy"}
    writer = None
    for resized in pyramid(image, scale=5):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(resized, stepSize=48, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it

            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            # THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
            # MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
            # WINDOW
            output = resized.copy()
            frame = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
            frame = frame/255.0
            preds = model.predict(np.expand_dims(frame, axis=0))[0]
            i = np.argmax(preds)
            label = maping[i]
            print(preds, label)
            
            if unsave:
                if i == 1:
                    return "Porn Found"
                
            
            if not unsave:
                clone = resized.copy()
                image = cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
                cv2.putText(image, label, (x, y+50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)

                cv2.imshow("Window", clone)
                cv2.waitKey(1)
                time.sleep(0.09)

                if writer is None:
                # initialize our video writer
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    writer = cv2.VideoWriter("1.avi", fourcc, 8, (1080, 720), True)


                # write the output frame to disk
                writer.write(clone)
    return "Save to View"

check()
def isUnsave():
    ans = check(1)
    print(ans)

isUnsave()
