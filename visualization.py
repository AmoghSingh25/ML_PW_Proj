import numpy as np
import copy
import cv2 as cv
import matplotlib.pyplot as plt
from config import GROUND_COLOR, SAND_COLOR, WATER_COLOR, DIM_MAP

def display_map(coast, waves_inp=[], obstacles=[], stream=True, openCV=False):
    global GROUND_COLOR, SAND_COLOR, WATER_COLOR

    water_color = copy.deepcopy(WATER_COLOR)
    ground_color = copy.deepcopy(GROUND_COLOR)
    sand_color = copy.deepcopy(SAND_COLOR)

    water_color.reverse()
    ground_color.reverse()
    sand_color.reverse()
    
    
    img = np.zeros((coast.shape[0], coast.shape[1],3), dtype=float)
    coast_temp = copy.deepcopy(coast)
    total = coast_temp.sum(axis=-1, keepdims=True) + 1e-8
    coast_temp = coast_temp / total
    
    
    img += np.array(ground_color).reshape(1,1,-1)*coast_temp[:,:,0:1]   ## Ground Color
    img+= np.array(sand_color).reshape(1,1,-1)*coast_temp[:,:,1:2]   ## Sand Color
    img+= np.array(water_color).reshape(1,1,-1)*coast_temp[:,:,2:3]   ## Water Color

    
    for i in range(len(obstacles)):
        img[obstacles[i][0], obstacles[i][1]] = [255,255,0]

    if len(waves_inp) > 0:
        for i in range(DIM_MAP):
            img[i,waves_inp[:,i]] = [0,0,255]

    if not openCV:
        plt.imshow(img.astype(int))
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