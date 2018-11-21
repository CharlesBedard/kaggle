import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from tqdm import tqdm

def imagify(origin_array, size):
    #function to turn a 1d vector into a square matrix
    #origin_aray -> any vector
    #size -> the size of the matrix to create
    new_array = np.zeros((size,size))
    for i in range (0,size):
        for j in range (0,size):
            new_array[i][j] = origin_array[i*size+j]
    return new_array

def de_imagify(img, size):
    #function to turn a square matrix into a vector
    #img -> square matrix
    #size -> the size of the square matrix
    new_array = np.zeros((size ** 2))
    for i in range(size):
        for j in range(size):
            new_array[i*size+j] = img[i][j]
    return np.asarray(new_array)

def crop_all_images(input_file_path, output_file_path):
    #function to take all the images from the input_file_path, and crop them to a uniform size
    #by default, all images are cropped and rescaled to 100,100
    #the resulting images are saved into the output_file_path
    #return: the size of the biggest cropped image, before rescaling
    all_img = np.load(input_file_path, encoding='latin1')

    #make an identical copy of the file, we will only modify the data of the images
    cropped_img = all_img.copy()
    #make a list to store the cropped images temporarily
    cropped_list = []

    #variables storing the size of the biggest image, used to resize all the samples
    max_width = 0
    max_height = 0
    for i in tqdm(range(all_img.shape[0])):
        #get the image in this row
        img = imagify(all_img[i][1],100)
        #make a copy that will remain unaltered
        img_cpy = img.copy()
        #blur the image
        img = cv.GaussianBlur(img,(3,3),0)
        # convert to grayscale
        imgray = np.uint8(img * 255) 
        #convert to binary image
        ret, thresh = cv.threshold(imgray, 20, 255, 0)
        #get the contours in the image
        im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #convert to rgb to have 3 channels
        im2 = cv.cvtColor(im2, cv.COLOR_GRAY2RGB)   
        #now get the biggest contour from the image
        maxArea = 0
        maxIndex = 0
        if len(contours) != 0:
            for i in range(len(contours)):
                if cv.contourArea(contours[i]) > maxArea:
                    maxArea = cv.contourArea(contours[i])
                    maxIndex = i
        #get the coordinates of the rectangle surrounding the shape
        a,b,c,d = cv.boundingRect(contours[maxIndex])
        #draw the rectangle
        #cv.rectangle(img,(a,b),(a+c,b+d),(255),1)
        #crop the original image
        crop = img_cpy[b:b+d, a:a+c]
        temp_max = np.max([c,d])
        crop = cv.resize(crop,(100,100))
        cropped_img[i][1] = de_imagify(crop,100)
        cropped_list.append(crop)
        if (c > max_width):
            max_width = c
        if (d > max_height):
            max_height = d

    #get the max size, i.e. biggest value between width and height
    max_size = np.max([max_height,max_width])
    for i in tqdm(range(cropped_img.shape[0])):
    #for i in tqdm(range(2)):
        #resize the array
        np.resize(cropped_img[i][1],(max_size ** 2))
        #cropped_img[i][1].resize(max_size)
        img = cropped_list[i]
        #crop the image to the max size, as a square
        crop = cv.resize(img,(max_size,max_size))
        #cropped_img[i][1] = de_imagify(crop,max_size)
        cropped_img[i][1] = de_imagify(crop,max_size)
        #crop = cv.resize(crop,(100,100))


    np.save(output_file_path, cropped_img)
    return max_size

def resize_all(input_file_path,output_file_path, current_size, size):
    #function to resize all the images in a file
    #takes the images in input_file_path and puts the resized ones in output_file_path
    #current_size -> the current size of the square matrices representing the images
    #size -> the wanted size of the square matrices
    all_img = np.load(input_file_path, encoding='latin1')
    print(all_img.shape[0])
    all_copy = all_img.copy()
    img_list = []
    for i in tqdm(range(all_img.shape[0])):
        img = imagify(all_img[i][1],current_size)
        #resize the array
        np.resize(all_img[i][1],(size ** 2))
        resized_img = cv.resize(img,(size,size))
        all_img[i][1] = de_imagify(resized_img,size)
    np.save(output_file_path,all_img)

    
#code example taking the images in train_images.npy, cropping them, then rescaling them
size = crop_all_images('comp-551-kaggle-master/all/train_images.npy','comp-551-kaggle-master/all/train_images2.npy')
resize_all('comp-551-kaggle-master/all/train_images2.npy','comp-551-kaggle-master/all/train_images2.npy',size,100)
#code example taking the images in test_images.npy, cropping them, then rescaling them
size = crop_all_images('comp-551-kaggle-master/all/test_images.npy','comp-551-kaggle-master/all/test_images2.npy')
resize_all('comp-551-kaggle-master/all/test_images2.npy','comp-551-kaggle-master/all/test_images2.npy',size,100)
