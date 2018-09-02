import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import cv2
import os,os.path
INPUTDIR = "D:\VideoAnalytics\opencv-object-tracking\Seat1\\"
OUTPUTDIR = "D:\VideoAnalytics\opencv-object-tracking\Seat\\"
trainfile = "test.txt"
file = open(trainfile,"w")

def write_file(filename,bb):
    file.write(filename+" ")
    [x, y, w, h] = [b for b in bb]
    xmax = x + w
    ymax = y + h
    file.write(str(x) + ',' + str(y) + ',' + str(xmax) + ',' + str(ymax) + ',0\n')

#helper function for hsv augmenting
def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

#hsv augmenting
def light_aug(image,initial,num,bb):
    hue = 0.1
    sat = 2
    val = 2
    for i in range(num):
        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = rgb_to_hsv(np.array(image)/255.)
        x[..., 0] += hue
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x>1] = 1
        x[x<0] = 0
        image_data = hsv_to_rgb(x) # numpy array, 0 to 1
        image_data = image_data*255
        savefile_name = OUTPUTDIR+f.split(".")[0]+"_"+str(i)+".jpg"
        cv2.imwrite(savefile_name,image_data)
        write_file(savefile_name,bb)

for f in os.listdir(INPUTDIR):
    image = cv2.imread(INPUTDIR+f)
    bb = cv2.selectROI("Mark",image)
    write_file(INPUTDIR+f,bb)
    light_aug(image,f,10,bb)
    break
file.close()