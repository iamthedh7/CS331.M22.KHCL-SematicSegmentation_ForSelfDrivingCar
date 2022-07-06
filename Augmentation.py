import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

################################ CHANGE BRIGHTNESS FUNCTION ######################################

def change_brightness(image, output_path, value):
    """
    value: do sang thay doi
    """
    ones = np.ones(list(image.shape), dtype=int)
    lighter = image + ones * value
    temp = lighter.ravel()
    for i in range(len(temp)):
        if (temp[i] < 0): 
            temp[i] = 0
        if (temp[i] > 255):
            temp[i] = 255
            
    temp = temp.reshape(list(image.shape))
    cv2.imwrite(output_path, temp)

################################### GET DATA TRAIN ###################################

train_images_0 = []
for i in os.listdir('leftImg8bit\\train'):
    folder_path = 'leftImg8bit\\train\\' + i
    files = os.listdir(folder_path)
    for j in files:
        img_path = 'leftImg8bit\\train\\' + i + '\\' + j
        train_images_0.append(img_path)

train_images_0 = sorted(train_images_0)
print(len(train_images_0))

################################# CREATE FOLDERS FOR AUGMENTED DATA #####################################

folders = os.listdir('leftImg8bit\\train')
for i in folders:
    try:
        os.mkdir('leftImg8bit\\train_light_1\\' + i) 
    except:
        print('Exists')
    
    try:
        os.mkdir('leftImg8bit\\train_light_2\\' + i) 
    except:
        print('Exists')
    
    try:
        os.mkdir('leftImg8bit\\train_night_1\\' + i) 
    except:
        print('Exists')

    try:
        os.mkdir('leftImg8bit\\train_night_2\\' + i) 
    except:
        print('Exists')

######################################################################

path = 'D:\\CV' # edit your path here

################################### AUGMENTATION ###################################

count = 0
for i in train_images_0:
    img = cv2.imread(i)
    dst = path + '\\leftImg8bit\\train_light_1\\' + i.split('\\')[-2] + '\\' + i.split('\\')[-1]
    change_brightness(img, dst, 30)
    count += 1
    if (count % 100 == 0): print(count, end=' ')
print('\n\nFinished 25%')

count = 0
for i in train_images_0:
    img = cv2.imread(i)
    dst = path + '\\leftImg8bit\\train_light_2\\' + i.split('\\')[-2] + '\\' + i.split('\\')[-1]
    change_brightness(img, dst, 60)
    count += 1
    if (count % 100 == 0): print(count, end=' ')
print('\n\nFinished 50%')

count = 0
for i in train_images_0:
    img = cv2.imread(i)
    dst = path + '\\leftImg8bit\\train_night_1\\' + i.split('\\')[-2] + '\\' + i.split('\\')[-1]
    change_brightness(img, dst, -20)
    count += 1
    if (count % 100 == 0): print(count, end=' ')
print('\n\nFinished 75%')

count = 0
for i in train_images_0:
    img = cv2.imread(i)
    dst = path + '\\leftImg8bit\\train_night_2\\' + i.split('\\')[-2] + '\\' + i.split('\\')[-1]
    change_brightness(img, dst, -40)
    count += 1
    if (count % 100 == 0): print(count, end=' ')
print('\n\nFinished 100%')