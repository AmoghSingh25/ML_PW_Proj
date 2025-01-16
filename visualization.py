import numpy as np
import copy
import cv2 as cv
import matplotlib.pyplot as plt
from config import GROUND_COLOR, SAND_COLOR, WATER_COLOR, DIM_MAP

def display_map(coast, waves_inp=[], waves_pos=[], stream=True, openCV=False):
    cv.namedWindow("Coast")
    
    img = np.zeros((coast.shape[0], coast.shape[1], 3), dtype=int)
    coast_temp = copy.deepcopy(coast)
    coast_temp[:,:,0] = (coast_temp[:,:,0] / np.max(coast_temp[:,:,0])) * 255
    coast_temp[:,:,1] = (coast_temp[:,:,1] / np.max(coast_temp[:,:,1])) * 255
    coast_temp[:,:,2] = (coast_temp[:,:,2] / np.max(coast_temp[:,:,2])) * 255
    coast_temp = coast_temp[...,np.newaxis]
    img = np.array(GROUND_COLOR).reshape(1,1,-1)*coast_temp[:,:,0,:]   # Ground Color
    img+= np.array(SAND_COLOR).reshape(1,1,-1)*coast_temp[:,:,1,:]     # Sand Color
    img+= np.array(WATER_COLOR).reshape(1,1,-1)*coast_temp[:,:,2,:]    # Water Color
    
    if len(waves_inp) > 0:
        for i in range(DIM_MAP):
            img[i,waves_inp[:,i]] = [0,0,255]

    if not openCV:
        plt.imshow(img)
        plt.show()

    if openCV:
        img = img.astype(np.uint8)
        img_cv = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img_cv = cv.resize(img_cv, (1000,1000))
        
        cv.imshow("Coast", img_cv)
        if stream:
            cv.waitKey(2)
        else:
            cv.waitKey(0)
            cv.destroyWindow("Coast")
            cv.waitKey(1)