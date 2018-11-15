import numpy as np 
from matplotlib import pyplot as plt
import cv2


def imagify(origin_array):
    new_array = np.zeros((100,100))
    for i in range (0,100):
        for j in range (0,100):
            new_array[i][j] = origin_array[i*100+j]
    return new_array

def de_imagify(img):
    new_array = np.zeros((10000))
    for i in range (0,100):
        for j in range (0,100):
            new_array[i*100+j] = img[i][j]
    return np.asarray(new_array)


all_train_img = np.load('KaggleCompetition/all/train_images.npy', encoding='latin1')
all_test_img = np.load('KaggleCompetition/all/test_images.npy', encoding='latin1')



#erosion effect
kernel = np.ones((2,2),np.uint8)

train_erosion = all_train_img.copy()
for i in range(10000):
    train_erosion[i][0] = all_train_img[i][0]
    train_erosion[i][1] = de_imagify( cv2.erode(imagify(all_train_img[i][1]),kernel,iterations= 1))
np.save('KaggleCompetition/all/train_erosion.npy', train_erosion)


test_erosion = all_test_img.copy()
for i in range(10000):
    test_erosion[i][0] = all_test_img[i][0]
    test_erosion[i][1] = de_imagify( cv2.erode(imagify(all_test_img[i][1]),kernel,iterations= 1))
np.save('KaggleCompetition/all/test_erosion.npy', test_erosion)

